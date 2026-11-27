"""
Try several values of one squeeze setting on a fixed chain and report how
closely each compiled transformer follows the chain. The chain is built once
on the config's first held-out split. Fidelity is measured on the first
--positions tokens of the training text and of the held-out text, not all
of them, so a big corpus stays quick.

Usage (from hegel/, with hegel/ on PYTHONPATH):
    python sweep_squeeze.py <config> <setting> <value> [value...] [--positions N]

Settings: slot_dim, logit_dim, match_threshold, low_weight, decoder_rounds,
code_seed. N defaults to 20000; 0 means every position.
"""

import sys
import time
from pathlib import Path
from typing import List

from lab import config_named, peak_gb, table, with_setting
from src.chain import build_chain, count_contexts
from src.compiler import compile_chain
from src.corpus import tokenize_split
from src.verify import compare_model_to_chain

SQUEEZE = (
    "slot_dim",
    "logit_dim",
    "match_threshold",
    "low_weight",
    "decoder_rounds",
    "code_seed",
)


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
    if key not in SQUEEZE:
        raise SystemExit(f"'{key}' is not a squeeze setting: {', '.join(SQUEEZE)}")

    text = Path(base.corpus).read_text(encoding="utf-8")
    tokens, held_out, tokenizer = tokenize_split(
        text,
        base.vocab_size,
        base.holdout_fraction,
        base.holdout_block,
        base.holdout_seed,
    )
    chain = build_chain(
        count_contexts(tokens, base.max_context),
        int(tokenizer.vocab_size),
        base.max_context,
        base.floor,
        base.prune_threshold,
        base.smoothing,
        base.discount,
    )
    train = tokens[:positions] if positions else tokens
    held = held_out[:positions] if positions else held_out
    print(
        f"{args[0]}: {len(chain.contexts) - 1} contexts; fidelity on "
        f"{len(train)} training and {len(held)} held-out positions"
    )
    rows: List[List[str]] = []
    for raw in values:
        config = with_setting(base, key, raw)
        started = time.perf_counter()
        model = compile_chain(
            chain,
            slot_dim=config.slot_dim,
            logit_dim=config.logit_dim,
            match_threshold=config.match_threshold,
            low_weight=config.low_weight,
            decoder_rounds=config.decoder_rounds,
            code_seed=config.code_seed,
        )
        seconds = time.perf_counter() - started
        row = [raw, str(model.d_model), f"{model.parameter_count() / 1e6:.1f} M"]
        for stream in (train, held):
            if len(stream) < 2:
                row += ["", "", ""]
                continue
            f = compare_model_to_chain(model, chain, stream)
            row += [
                f"{f.kl_mean:.4f}",
                f"{100 * f.top1_agreement:.1f}%",
                f"{f.reference_nll:.3f} / {f.other_nll:.3f}",
            ]
        rows.append(row + [f"{seconds:.0f}"])
        del model
        print(f"  done {key} = {raw}", file=sys.stderr, flush=True)
    header = [key, "d_model", "weights", "train KL", "top-1", "NLL chain / model"]
    header += ["held KL", "top-1", "NLL chain / model", "compile s"]
    print(table(header, rows))
    print(f"Peak memory {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
