#!/usr/bin/env bash
set -euo pipefail

REPO="${LAB07_REPO_OVERRIDE:-/mnt/c/GeoAI/geoai-mineral-labs}"
BRANCH="feat/lab07-spatial-uncertainty-phase-c"
BASELINE="aeeb342aa4fb55de7431023df60d767f374a7d2d"
VENV="${LAB07_VENV_OVERRIDE:-/home/erickh/.venvs/geoai-mineral-qa}"
EVIDENCE_DIR="${LAB07_EVIDENCE_OVERRIDE:-/home/erickh/.cache/geoai-erl/lab07_phase_c}"

cd "$REPO"
source "$VENV/bin/activate"

export PYTHONPATH="$PWD/labs/lab05_mineral_prospectivity:$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export GIT_PAGER=cat
export PAGER=cat
export GIT_TERMINAL_PROMPT=0
export CI=1

echo "=== ERL V2 | LAB07 PHASE C QA ==="

test "$(git branch --show-current)" = "$BRANCH"
test "$(git rev-parse HEAD)" = "$BASELINE"
test "$(git rev-parse refs/remotes/origin/main)" = "$BASELINE"

# Only the five planned Phase C files may differ from the baseline.
python -B - <<'PY'
import subprocess

allowed = {
    "labs/lab07_spatial_uncertainty/07c_build_rank_stability.py",
    "labs/lab07_spatial_uncertainty/results/rank_stability.csv",
    "labs/lab07_spatial_uncertainty/results/phase_c_manifest.json",
    "tests/test_lab07_rank_stability.py",
    "scripts/erl/run_lab07_phase_c.sh",
}
raw = subprocess.check_output(
    ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"]
)
for record in filter(None, raw.decode("utf-8").split("\0")):
    state, path = record[:2], record[3:]
    if state != "??" or path not in allowed:
        raise SystemExit(f"FAIL: Unexpected worktree entry: {state} {path}")
print("PHASE_C_EXACT_WORKTREE_PREFLIGHT=PASS")
PY

mkdir -p "$EVIDENCE_DIR"

echo "=== BUILD ==="
python -B labs/lab07_spatial_uncertainty/07c_build_rank_stability.py

echo "=== RUFF CHECK ==="
ruff check \
  labs/lab07_spatial_uncertainty/07c_build_rank_stability.py \
  tests/test_lab07_rank_stability.py

echo "=== RUFF FORMAT ==="
ruff format --check \
  labs/lab07_spatial_uncertainty/07c_build_rank_stability.py \
  tests/test_lab07_rank_stability.py

echo "=== TARGETED TESTS ==="
python -B -m pytest -q \
  --override-ini "addopts=" \
  --junitxml="$EVIDENCE_DIR/pytest_targeted.xml" \
  -p no:cacheprovider \
  tests/test_lab07_rank_stability.py

echo "=== FULL REGRESSION ==="
python -B -m pytest -q \
  --override-ini "addopts=" \
  --junitxml="$EVIDENCE_DIR/pytest_full.xml" \
  -p no:cacheprovider

echo "=== DIFF CHECK ==="
git diff --check

echo "=== PRE-COMMIT ==="
pre-commit run --all-files
pre-commit run --files \
  labs/lab07_spatial_uncertainty/07c_build_rank_stability.py \
  labs/lab07_spatial_uncertainty/results/rank_stability.csv \
  labs/lab07_spatial_uncertainty/results/phase_c_manifest.json \
  tests/test_lab07_rank_stability.py \
  scripts/erl/run_lab07_phase_c.sh

echo "=== GITLEAKS ==="
gitleaks detect --source . --no-banner --redact
gitleaks dir . --no-banner --redact

echo "=== QA FINALIZATION ==="
python -B labs/lab07_spatial_uncertainty/07c_build_rank_stability.py \
  --finalize "$EVIDENCE_DIR"

echo "=== FINAL STATUS ==="
git diff --check
git status --short
sha256sum \
  labs/lab07_spatial_uncertainty/results/rank_stability.csv \
  labs/lab07_spatial_uncertainty/results/phase_c_manifest.json

echo "LAB07_PHASE_C_RUNNER=PASS"
