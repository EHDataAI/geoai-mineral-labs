#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/c/GeoAI/geoai-mineral-labs"
BRANCH="feat/lab07-spatial-uncertainty-phase-b"
VENV="/home/erickh/.venvs/geoai-mineral-qa"

cd "$REPO"
source "$VENV/bin/activate"

export PYTHONPATH="$PWD/labs/lab05_mineral_prospectivity:$PWD/src"
export PYTHONUNBUFFERED=1
export GIT_PAGER=cat
export PAGER=cat
export GIT_TERMINAL_PROMPT=0
export CI=1

echo "=== ERL V2 | LAB07 PHASE B RUNNER ==="

test "$(git branch --show-current)" = "$BRANCH"
test -f labs/lab07_spatial_uncertainty/results/phase_a_audit.json

echo
echo "=== BUILD ==="

python \
  labs/lab07_spatial_uncertainty/07b_build_score_uncertainty.py

echo
echo "=== RUFF ==="

ruff check \
  labs/lab07_spatial_uncertainty/07b_build_score_uncertainty.py \
  tests/test_lab07_score_uncertainty.py

ruff format --check \
  labs/lab07_spatial_uncertainty/07b_build_score_uncertainty.py \
  tests/test_lab07_score_uncertainty.py

echo
echo "=== TARGETED TESTS ==="

python -m pytest -q \
  tests/test_lab07_score_uncertainty.py

echo
echo "=== FULL REGRESSION ==="

EVIDENCE_DIR="$HOME/.cache/geoai-erl/lab07_phase_b"
mkdir -p "$EVIDENCE_DIR"

python -m pytest -q \
  --override-ini "addopts=" \
  --junitxml="$EVIDENCE_DIR/pytest.xml" \
  -p no:cacheprovider

echo
echo "=== DIFF CHECK ==="

git diff --check

echo
echo "=== PRE-COMMIT ==="

pre-commit run --all-files

echo
echo "=== GITLEAKS ==="

gitleaks detect \
  --source . \
  --no-banner \
  --redact

echo
echo "=== FINAL STATUS ==="

git diff --check

git status --short

echo
echo "LAB07_PHASE_B_RUNNER=PASS"
