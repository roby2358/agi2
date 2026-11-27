# GRPO: Fine-Tuning a Small Model with Verifiable Rewards

This document explains how to make a small open model (1–4B parameters) better at one narrow task. The method is reinforcement learning against a **checker written in code**, with no labeled examples and no human ratings. It's called RLVR (Reinforcement Learning with Verifiable Rewards). As of 2026 it's the core recipe for training reasoning models, and here it's scaled down to run on one GPU.

## The Core Idea

Supervised fine-tuning (SFT) shows the model correct answers and says "copy this." RLVR works differently:

1. Give the model a prompt.
2. Let it generate several answers (a *group*, say 8).
3. **Score each answer with a program.** Does the JSON parse? Does it match the schema? Does the puzzle solution actually work? That score is the reward.
4. Push the model toward answers that scored above the group average and away from answers that scored below it.

Step 4 is **GRPO** (Group Relative Policy Optimization, introduced in DeepSeekMath and used in DeepSeek-R1). It needs no separate critic or learned reward model. It compares answers to the same prompt against each other, which is why it's cheap enough for a consumer GPU. GRPO and its variants (DAPO, RLOO) have largely replaced PPO for LLM post-training.

The key insight: **you don't need correct answers, only a way to check answers.** Checking is often much easier than producing, and the check is deterministic.

### The math, briefly

For a prompt `q`, sample `G` completions `o_1..o_G` and score each one to get rewards `r_1..r_G`. The advantage of each completion is its reward normalized within the group:

```
A_i = (r_i - mean(r)) / (std(r) + eps)
```

Every token in completion `o_i` shares the advantage `A_i`. The loss is a clipped policy-gradient objective, as in PPO, plus an optional KL penalty that keeps the model close to a frozen reference copy:

```
ratio_t = π_θ(token_t | context) / π_old(token_t | context)
L = -mean_t[ min(ratio_t * A_i, clip(ratio_t, 1-ε, 1+ε) * A_i) ] + β * KL(π_θ || π_ref)
```

If every completion in a group gets the same reward, all advantages are zero and that prompt teaches nothing. This fact drives most of the practical advice below.

## The Pieces

| Piece | What it is | Tool |
|---|---|---|
| Base model | Small open model, e.g. Qwen3 1.7B/4B, Llama 3.x 3B, Gemma 3 | Hugging Face |
| Environment | Generates prompts and scores outputs | Your own code, or the `verifiers` library / Prime Intellect Environments Hub |
| Trainer | Runs GRPO efficiently on limited VRAM | **Unsloth** (QLoRA + GRPO) or TRL's `GRPOTrainer`; `verl` for larger scale |
| LoRA adapter | Small set of trained weights added on top of the frozen base | Produced by the trainer |
| Serving | Runs the tuned model locally | Ollama or llama.cpp (export to GGUF) |

**QLoRA** keeps the base model frozen in 4-bit precision and trains only a small low-rank adapter. That's what lets a 4–8B model train on a 12–24 GB GPU.

## Worked Example: Always Emit Valid Structured Output

**The task:** given a messy email and a JSON schema, output JSON that validates against the schema and has the correct field values.

### Reward function

The reward is plain Python:

```python
import json
from jsonschema import validate, ValidationError

def reward(completion: str, schema: dict, expected: dict) -> float:
    try:
        obj = json.loads(completion)
    except json.JSONDecodeError:
        return 0.0                      # not even JSON
    try:
        validate(obj, schema)
    except ValidationError:
        return 0.2                      # JSON, but wrong shape
    # partial credit for correct field values
    hits = sum(obj.get(k) == v for k, v in expected.items())
    return 0.5 + 0.5 * hits / len(expected)
```

**Graded rewards (0 / 0.2 / 0.5–1.0) beat pass/fail.** With pass/fail, early training produces groups where every answer scores zero. Their advantages are all zero, so nothing gets learned.

### Training loop (Unsloth + TRL)

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen3-4B", load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[schema_reward],        # wraps reward() above
    train_dataset=prompts,               # ~500–2000 email+schema prompts
    args=GRPOConfig(num_generations=8, max_steps=500, learning_rate=5e-6),
)
trainer.train()
```

### Evaluation

Hold out about 200 prompts that are never used in training. On that set, measure the schema-valid rate and the field accuracy for three models:

- the base model
- the tuned model
- a frontier model (e.g. Claude Sonnet) as the ceiling

## What to Expect

- **Hardware:** one 16–24 GB GPU, or a rented cloud GPU for a few dollars an hour. A run of a few hundred steps takes a few hours.
- **Typical result:** the valid-output rate jumps quickly (e.g. 70% → 98%). Field accuracy improves more slowly. On this *one* task, the tuned small model often gets close to the frontier model while being far cheaper, private, and able to run air-gapped.

### Common problems

- **Reward hacking.** The model finds shortcuts. For example, it will output `{}` if an empty object validates. Make required fields actually required, and read samples regularly rather than only watching the reward curve.
- **Tasks that are too hard** give all-zero groups, so nothing is learned. Start easy and add a curriculum (increasing difficulty over time).
- **Tasks that are too easy** give all-perfect groups, which also teach nothing. Filter out prompts the model already solves every time.
- **Forgetting.** The model can get worse at everything else. That's acceptable for a single-purpose model, but be aware it happens. The KL penalty (`β`) limits it.

## Doing It From Scratch in AGI2

AGI2 already has the parts GRPO needs: `AGI2Model`, `generation.py` for sampling, and the training loop in `training.py`. A from-scratch GRPO is a small addition and a good way to understand the algorithm without Unsloth or TRL hiding the details.

Sketch:

1. **Task.** Pick something a small GPT-2 can plausibly learn and a program can check. Examples: "continue this text so it ends with a period within N tokens", "output a valid arithmetic answer to `a+b=`", or "produce balanced parentheses".
2. **Rollouts.** For each prompt, call the existing sampler `G` times at temperature ~1.0. Keep the token ids of each completion.
3. **Rewards.** Score each completion with the verifier, then compute the group-normalized advantages `A_i`.
4. **Log-probs.** Run a forward pass on `prompt + completion` and take the log-prob of each completion token. Mask out the prompt tokens.
5. **Loss.** Compute `-(A_i * logprob).mean()` over the completion tokens. This is the simplest on-policy form: one gradient step per batch, so the ratio is 1 and clipping is not needed. Add clipping and a frozen reference model for the KL term once that works.
6. **Config.** Put `group_size`, `kl_beta`, `clip_eps`, and the task name in a new `[grpo]` TOML section, following the project convention.

Start from a checkpoint that SFT has already taught the basic format. RL sharpens behavior the model can *sometimes* produce, but it can't find behavior the model never produces.

## Alternative Environment: Game Puzzles

Instead of structured output, use a puzzle from the yuwakisa game prototypes, such as "is this move legal?" or "solve in N moves", with the game engine as the verifier. It's more fun and demos well, though it's less directly useful to clients.

## Why This Is Worth Doing

1. **It shows how modern models are actually made.** Reasoning models and coding agents are trained with this recipe at massive scale. The vocabulary (RLVR, GRPO, environments, verifiers) is worth knowing firsthand.
2. **It's a client offering.** Some clients can't send data to a frontier API. A 4B model tuned for their one extraction task, running on their own hardware, is a concrete deliverable.
3. **The verifier is also an eval.** The same reward function can score other approaches to the same task. For example, you can compare a hand-written prompt, a DSPy GEPA-optimized prompt, and an RL-tuned small model on one measure.

## References

- [Hugging Face: How frontier models train on outcomes in 2026](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026)
- [llm-stats: Post-training in 2026: GRPO, DAPO, RLVR & beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [SitePoint: Fine-tune local LLMs 2026](https://www.sitepoint.com/fine-tune-local-llms-2026/)
- [verl: RL post-training framework](https://github.com/verl-project/verl)
- [Spheron: RL environments: Gymnasium, Prime Intellect & Verifiers](https://www.spheron.network/blog/rl-environments-gpu-cloud-gymnasium-prime-intellect-verifiers/)
- [Anyscale: Open-source RL libraries for LLMs](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)
- DeepSeekMath (Shao et al., 2024): the paper that introduced GRPO
