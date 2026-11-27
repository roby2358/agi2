"""
Try several values of one setting and report how closely each compiled
transformer follows its chain. The chain is built on the config's first
held-out split: once for a setting that changes only the compile, again for
each value of any other setting (max_context, prune_threshold, discount,
vocab_size, smoothing, floor). Fidelity is measured on the first --positions
tokens of the training text and of the held-out text, not all of them, so a
big corpus stays quick.

Usage (from hegel/, with hegel/ on PYTHONPATH):
    python sweep_squeeze.py <config> <setting> <value> [value...] [--positions N]

Compile-only settings: slot_dim, logit_dim, match_threshold, low_weight,
decoder_rounds, code_seed. N defaults to 20000; 0 means every position.
"""

import sys
import time
from typing import List, Optional

from lab import SplitChain, build_split, config_named, peak_gb, table, with_setting
from src.compiler import compile_chain
from src.config import HegelConfig
from src.verify import compare_model_to_chain

COMPILE_ONLY = (
    "slot_dim",
    "logit_dim",
    "match_threshold",
    "low_weight",
    "decoder_rounds",
    "code_seed",
)


def _build(config: HegelConfig) -> SplitChain:
    """Build the chain and say what it holds."""
    started = time.perf_counter()
    built = build_split(config)
    print(
        f"Chain: {len(built.tokens)} training tokens, {len(built.held_out)} held "
        f"out, {len(built.chain.contexts) - 1} contexts kept, built in "
        f"{time.perf_counter() - started:.0f} s"
    )
    print("\n".join(built.by_length()), flush=True)
    return built


def _fidelity(config: HegelConfig, built: SplitChain, positions: int) -> List[str]:
    """Compile one model and compare it with its chain; the table cells."""
    started = time.perf_counter()
    model = compile_chain(
        built.chain,
        slot_dim=config.slot_dim,
        logit_dim=config.logit_dim,
        match_threshold=config.match_threshold,
        low_weight=config.low_weight,
        decoder_rounds=config.decoder_rounds,
        code_seed=config.code_seed,
    )
    seconds = time.perf_counter() - started
    cells = [
        str(len(built.chain.contexts) - 1),
        str(model.d_model),
        f"{model.parameter_count() / 1e6:.1f} M",
        f"{model.nonzero_count() / 1e6:.1f} M",
    ]
    for stream in (built.tokens, built.held_out):
        part = stream[:positions] if positions else stream
        if len(part) < 2:
            cells += ["", "", ""]
            continue
        f = compare_model_to_chain(model, built.chain, part)
        cells += [
            f"{f.kl_mean:.4f}",
            f"{100 * f.top1_agreement:.1f}%",
            f"{f.reference_nll:.3f} / {f.other_nll:.3f}",
        ]
    return cells + [f"{seconds:.0f}"]


def main(argv: List[str]) -> int:
    args = argv[1:]
    positions = 20000
    if "--positions" in args:
        at = args.index("--positions")
        positions = int(args[at + 1])
        del args[at : at + 2]
    if len(args) < 3:
        print(__doc__)
        return 2
    base, key, values = config_named(args[0]), args[1], args[2:]
    sample = f"the first {positions}" if positions else "all"
    print(f"{args[0]}: {key}; fidelity on {sample} positions of each text")
    rows: List[List[str]] = []
    built: Optional[SplitChain] = None
    for raw in values:
        config = with_setting(base, key, raw)
        if built is None or key not in COMPILE_ONLY:
            built = None  # free the last chain before building the next
            built = _build(config)
        rows.append([raw] + _fidelity(config, built, positions))
        print(f"  done {key} = {raw}", file=sys.stderr, flush=True)
    header = [key, "contexts", "d_model", "weights", "nonzero"]
    header += ["train KL", "top-1", "NLL chain / model"]
    header += ["held KL", "top-1", "NLL chain / model", "compile s"]
    print(table(header, rows))
    print(f"Peak memory {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
