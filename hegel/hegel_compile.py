"""
Compile a transformer from a Markov chain and report how closely it follows.

Usage (from hegel/):
    uv run python hegel_compile.py resources/corpus20.toml
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from src.chain import Chain, build_chain, count_contexts
from src.compiler import compile_wide
from src.config import load_config
from src.corpus import tokenize
from src.verify import (
    Fidelity,
    compare_chains,
    compare_model_to_chain,
    model_predictor,
    sample,
)


def _report(title: str, fidelity: Fidelity) -> None:
    print(f"\n{title}")
    print(f"  positions compared     {fidelity.positions}")
    print(f"  mean KL (nats)         {fidelity.kl_mean:.3e}")
    print(f"  max KL (nats)          {fidelity.kl_max:.3e}")
    print(f"  top-1 agreement        {fidelity.top1_agreement:.4%}")
    print(f"  max |log-prob diff|    {fidelity.max_abs_diff:.3e}")
    print(f"  reference NLL (nats)   {fidelity.reference_nll:.4f}")
    print(f"  compared NLL (nats)    {fidelity.other_nll:.4f}")


def _describe_chain(chain: Chain) -> None:
    print(f"\nChain: {len(chain.contexts) - 1} contexts kept")
    for length, kept in enumerate(chain.kept_by_length()):
        if length:
            print(f"  length {length}: kept {kept} of {chain.candidates[length]}")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    config = load_config(argv[1])
    started = time.perf_counter()

    text = Path(config.corpus).read_text(encoding="utf-8")
    tokens, tokenizer = tokenize(text, config.vocab_size)
    vocab = int(tokenizer.vocab_size)
    print(f"Corpus: {config.corpus}")
    print(f"  {len(text)} characters, {len(tokens)} tokens, vocabulary {vocab}")

    counts = count_contexts(tokens, config.max_context)
    full = build_chain(counts, vocab, config.max_context, config.floor)
    chain = build_chain(
        counts, vocab, config.max_context, config.floor, config.prune_threshold
    )
    _describe_chain(chain)
    if config.prune_threshold > 0.0:
        _report(
            f"Pruning cost: pruned chain against the unpruned chain "
            f"(threshold {config.prune_threshold} nats)",
            compare_chains(full, chain, tokens),
        )

    model = compile_wide(chain)
    print("\nModel (wide one-hot build, zero gradient steps)")
    print(f"  d_model {model.d_model}, heads {model.n_heads}, d_ff {model.d_ff}")
    print(
        f"  weights {model.parameter_count()}, "
        f"of which nonzero {model.nonzero_count()}"
    )
    if config.checkpoint:
        Path(config.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.checkpoint)
        print(f"  saved to {config.checkpoint}")

    _report(
        "Fidelity: transformer against the chain it was compiled from",
        compare_model_to_chain(model, chain, tokens),
    )

    rng = np.random.default_rng(config.seed)
    predict_model = model_predictor(model)
    print(f"\nSamples ({config.sample_tokens} tokens each, shared random draws)")
    for number in range(config.sample_count):
        start = int(rng.integers(0, len(tokens) - config.max_context))
        prompt = tokens[start : start + config.max_context]
        draws = rng.random(config.sample_tokens)
        from_chain = sample(chain.next_log_probs, prompt, config.sample_tokens, draws)
        from_model = sample(predict_model, prompt, config.sample_tokens, draws)
        same = sum(a == b for a, b in zip(from_chain, from_model))
        print(f"\n[{number + 1}] prompt: {tokenizer.decode(prompt)!r}")
        print(f"  chain: {tokenizer.decode(from_chain)!r}")
        print(f"  model: {tokenizer.decode(from_model)!r}")
        print(f"  tokens identical: {same} of {config.sample_tokens}")

    print(f"\nDone in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
