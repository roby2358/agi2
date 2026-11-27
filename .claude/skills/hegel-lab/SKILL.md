---
name: hegel-lab
description: Scripted routines for the hegel/ experiment (the transformer compiled from a Markov chain). Use when working in hegel/ and about to lint or test, start a compile or build that takes more than a minute, check on a running job, size up a new corpus, sweep a chain or squeeze setting, time saved weights, or write a one-off measurement script. Every routine is a `just cc-*` recipe in hegel/justfile.
---

# hegel-lab

Routines that kept being retyped while working on `hegel/`. Each is a
`cc-` recipe in `hegel/justfile`; anything too long for a recipe is a script
in this skill's `scripts/`. Run everything from `hegel/`.

Use these instead of typing the commands out, and instead of writing a
script into a string and running it. If a routine is missing or wrong, fix
the script or the recipe here; do not work around it inline.

## Recipes

| recipe | what it does |
|---|---|
| `just cc-check` | black and isort (check only), flake8, mypy, pytest. Run before saying work is done. |
| `just cc-fmt` | black and isort, rewriting files. |
| `just cc-timed <name> <command...>` | Runs a command, copies output to `runs/<name>.log` as it comes, ends with time and peak memory. |
| `just cc-status [lines]` | Which timed runs are going, for how long, the tail of each log, free memory. |
| `just cc-stats <config>` | Tokens, distinct contexts by length, contexts kept, rough weight count. No model built. |
| `just cc-sweep <config> <setting> <values...>` | Held-out NLL and top-1 of the chain for each value, over the config's splits. No model built. |
| `just cc-context <config> [max]` | Held-out NLL and top-1 of the unpruned chain at context lengths 1 to max, plus distinct contexts per length. Lean on memory, so it works past context 3 on Hegel (26 s for 6). Needs vocabulary 512. |
| `just cc-smoothing <config> [max]` | The same table under several smoothings: Witten-Bell with a scaled backoff share, absolute discounting, Kneser-Ney. 3 min on Hegel for 6. Edit the `variants` list in `smoothing.py` to change the grid. |
| `just cc-squeeze <config> <setting> <values...>` | Compiles the transformer for each value; fidelity against its chain on a sample of positions, and contexts by length. A squeeze setting reuses one chain; a chain setting (`max_context`, `prune_threshold`, ...) rebuilds it per value. |
| `just cc-throughput <config> [tokens]` | Load time and forward-pass time of saved weights. |
| `just cc-py <script> [args...]` | Runs a scratch script with `src.*` and `lab` importable. |
| `just cc-commit <message-file>` | Stages everything in the repo, commits with that message, pushes. |

`<config>` is a TOML basename in `hegel/resources/` (`corpus20`,
`corpus20_squeezed`, `moby_dick`, `hegel`).

## Long runs

Anything over about a minute (`just compile moby_dick`, `just build hegel`,
`just compile hegel`, sweeps on Hegel) goes through `cc-timed`:

```bash
just cc-timed hegel-build just build hegel
just cc-timed prune-md just cc-sweep moby_dick prune_threshold 0.2 0.5 1 2
```

- `cc-timed` stays in the foreground. Put it in the background with the
  harness (a short timeout on the shell call, or its background option),
  not with `nohup` or `&`, so the user can see and stop it.
- Check on it with `just cc-status`. Do not poll with `ps`, `tail` and `free`
  by hand.
- The log's last line gives exit code, elapsed time and peak memory. Peak
  memory is the largest single process, which is the Python run.
- Logs live in `hegel/runs/`, which is not in git. Reusing a name
  overwrites its log.
- Known durations are in `hegel/CLAUDE.md` (Hegel build 5 min; the full Hegel
  held-out check took 2 h 20 min under the old smoothing). Give the user an estimate from those before
  starting, and say so if a run will take more than a few minutes.

## Sizing a new corpus

1. Add `resources/<name>.toml` (copy `moby_dick.toml`).
2. `just cc-stats <name>`. `d_ff` is the number of contexts kept; memory
   and time scale with it. Hegel (208 k contexts) peaks at 6.6 GB to build.
3. `just cc-squeeze <name> logit_dim 128` gives fidelity on 20,000 positions
   in minutes; the full `just compile <name>` runs the model over all the
   training text (1 to 2 s per 1000 tokens at Hegel's size).

## Sweeps

- `cc-sweep` scores the chain alone, for settings that change it:
  `prune_threshold`, `max_context`, `discount`, `vocab_size`,
  `smoothing`, `floor`. To see the spread across splits on a small corpus,
  use a config with `holdout_seeds` above 1.
- `cc-squeeze` compiles the transformer and checks it against its chain,
  on the first split. For a setting that changes only the compile
  (`slot_dim`, `logit_dim`, `match_threshold`, `low_weight`,
  `decoder_rounds`, `code_seed`) it builds the chain once; for a chain
  setting it rebuilds the chain for each value. Add `--positions N` to
  change the sample (0 = everything).
- `max_context` on Hegel: `cc-sweep hegel max_context 3 4` took 4.7 min
  and peaked at 18.8 GB of a 20.5 GB machine (it also builds the unpruned
  chain; context 3 alone peaks at 8.4 GB), so check `just cc-status` for
  free memory first. `cc-squeeze hegel max_context 3 4` took 11.8 min and
  peaked at 12.0 GB. `src.chain` holds 4 KB per context, so past context
  4 use `cc-context` or `cc-smoothing`.
- Both vary one setting. For a grid, or anything else, write a scratch
  script.

## One-off measurements

Copy `scripts/scratch_template.py` to the session scratchpad as a real
file, edit it there, and run it with `just cc-py <path>`. Never build the
script as a string inside another command: quoting breaks. `lab.py` has
`config_named`, `with_setting`, `build_split` (tokens and chain of the
first split), `peak_gb` and `table`.

When a scratch script gets run a second time for a new corpus or setting,
move it into `scripts/`, give it arguments, add a `cc-` recipe and a row in
the table above.

## After the work

- `just cc-check` must pass.
- Measured numbers go into `hegel/CLAUDE.md`, replacing stale text.
- Do not commit or push without the user's OK each time. With it: write the
  message to a file in the scratchpad (ending with the Co-Authored-By line
  the session gives), look at `git status --short` for anything that should
  not go in, then `just cc-commit <message-file>`. It stages every change in
  the repo that is not ignored, so unstage or ignore strays first.
