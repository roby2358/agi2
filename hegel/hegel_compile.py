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
from src.chain import Chain, build_chain
from src.config import load_config
from src.pipeline import compile_from_config
from src.verify import (
    Fidelity,
    compare_chains,
    compare_model_to_chain,
    model_predictor,
    sample,
    unigram_nll,
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

    compiled = compile_from_config(config)
    tokens, tokenizer = compiled.tokens, compiled.tokenizer
    chain, model = compiled.chain, compiled.model
    vocab = chain.vocab_size
    print(f"Corpus: {config.corpus}")
    print(
        f"  {compiled.text_length} characters, {len(tokens)} tokens, "
        f"vocabulary {vocab}"
    )
    held_out = compiled.held_out
    if held_out:
        print(f"  plus {len(held_out)} held-out tokens the chain never sees")

    _describe_chain(chain)
    if config.prune_threshold > 0.0:
        full = build_chain(compiled.counts, vocab, config.max_context, config.floor)
        _report(
            f"Pruning cost: pruned chain against the unpruned chain "
            f"(threshold {config.prune_threshold} nats)",
            compare_chains(full, chain, tokens),
        )

    slots = "one-hot slots" if config.slot_dim == 0 else f"{config.slot_dim}-dim slots"
    logits = (
        "exact logits" if config.logit_dim == 0 else f"{config.logit_dim}-dim logits"
    )
    print(f"\nModel ({slots}, {logits}; zero gradient steps)")
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

    if len(held_out) > 1:
        on_chain = compare_model_to_chain(model, chain, held_out)
        full = build_chain(compiled.counts, vocab, config.max_context, config.floor)
        unpruned_nll = compare_chains(full, chain, held_out).reference_nll
        print(f"\nHeld-out text: NLL in nats per token ({len(held_out)} tokens)")
        print(f"  uniform over the vocabulary   {np.log(vocab):.4f}")
        print(f"  unigram                       {unigram_nll(chain, held_out):.4f}")
        print(f"  unpruned chain                {unpruned_nll:.4f}")
        print(f"  chain                         {on_chain.reference_nll:.4f}")
        print(f"  transformer                   {on_chain.other_nll:.4f}")
        _report("Fidelity on held-out text: transformer against its chain", on_chain)

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
