#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=${PY:-python3}
: "${P1_BUNDLE:?Set P1_BUNDLE to the local frozen Hugging Face bundle}"
: "${P1_OUTPUT:?Set P1_OUTPUT to a new experiment directory}"
"$PY" -m p1.pilot --bundle "$P1_BUNDLE" --output-dir "$P1_OUTPUT" \
    --train-limit "${P1_TRAIN_LIMIT:-256}" --validation-limit "${P1_VAL_LIMIT:-256}" \
    --max-concepts "${P1_MAX_CONCEPTS:-24}" --epochs "${P1_EPOCHS:-5}" \
    --seed "${P1_SEED:-42}" --device "${P1_DEVICE:-cpu}"
