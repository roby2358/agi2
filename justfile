# Model names are TOML basenames in semantic/resources/ (e.g. lilwill, lilwill_rwkv).
# Names containing "rwkv" use the RWKV pipeline; everything else the transformer.

_default:
    @just --list --unsorted

# Explain how these recipes work
help:
    @echo 'Usage:'
    @echo '  just train <model>            train a model'
    @echo '  just run <model> ["prompt"]   generate from a trained model'
    @echo ''
    @echo 'The <model> argument is the TOML basename in semantic/resources/; any name'
    @echo 'containing "rwkv" is routed to the RWKV pipeline, everything else to the'
    @echo 'transformer, so one pair of recipes covers both. run takes an optional quoted'
    @echo 'prompt (defaults to "Once upon a time"). Since the justfile lives at the repo'
    @echo 'root, it works from anywhere in the repo -- just searches upward for it.'
    @echo ''
    @echo 'Examples:'
    @echo '  just train lilwill               -> uv run python agi2_train.py resources/lilwill.toml'
    @echo '  just train lilwill_rwkv          -> uv run python rwkv_train.py resources/lilwill_rwkv.toml'
    @echo '  just run lilwill_rwkv "What ho"  -> uv run python rwkv_generate.py resources/lilwill_rwkv.toml "What ho"'

# Train a model: just train lilwill_rwkv
train model:
    cd semantic && uv run python {{ if model =~ "rwkv" { "rwkv_train.py" } else { "agi2_train.py" } }} resources/{{ model }}.toml

# Generate from a trained model: just run lilwill_rwkv "What ho"
run model prompt="Once upon a time":
    cd semantic && uv run python {{ if model =~ "rwkv" { "rwkv_generate.py" } else { "agi2_generate.py" } }} resources/{{ model }}.toml "{{ prompt }}"
