"""Bootstrap and structural validation for LAB05.

This script is intentionally conservative:
- creates required directories;
- validates the expected Git branch;
- validates the canonical Gate 0 files;
- patches protected oracle fields in the JSON contract;
- never creates accidental .txt variants;
- never writes tests inside the laboratory directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_BRANCH = "feat/lab05-mineral-prospectivity"

FORBIDDEN_ARTIFACTS = (
    "labs/lab05_mineral_prospectivity/prospectivity_contract.json.txt",
    "labs/lab05_mineral_prospectivity/tests",
    "tests/test_dimensions_v3.py",
)

REQUIRED_FILES = (
    "labs/lab05_mineral_prospectivity/prospectivity_contract.json",
    "labs/lab05_mineral_prospectivity/prospectivity_config.py",
    "tests/test_lab05_contract.py",
)

REQUIRED_DIRECTORIES = (
    "labs/lab05_mineral_prospectivity",
    "labs/lab05_mineral_prospectivity/data",
    "labs/lab05_mineral_prospectivity/results",
    "scripts/lab05",
    "tests",
)

ORACLE_PROTECTED_FIELDS = (
    "true_deposit",
    "latent_favourability",
    "true_deposit_probability",
)


def repo_root() -> Path:
    """Return repository root based on this script location."""
    return Path(__file__).resolve().parents[2]


def current_branch(root: Path) -> str:
    """Read the current Git branch."""
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_branch(root: Path) -> None:
    """Require the dedicated LAB05 feature branch."""
    branch = current_branch(root)

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            "LAB05 bootstrap requires branch "
            f"{EXPECTED_BRANCH!r}; current branch is {branch!r}."
        )

    print(f"BRANCH = {branch}")


def ensure_directories(root: Path) -> None:
    """Create the canonical LAB05 directory structure."""
    for relative in REQUIRED_DIRECTORIES:
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        print(f"DIRECTORY_OK = {relative}")


def validate_required_files(root: Path) -> None:
    """Require the canonical Gate 0 files."""
    missing: list[str] = []

    for relative in REQUIRED_FILES:
        path = root / relative

        if not path.is_file():
            missing.append(relative)
        else:
            print(f"FILE_OK = {relative}")

    if missing:
        raise FileNotFoundError(
            "Missing canonical LAB05 files: "
            + ", ".join(missing)
        )


def validate_forbidden_artifacts(root: Path) -> None:
    """Reject known accidental files and folders."""
    unexpected: list[str] = []

    for relative in FORBIDDEN_ARTIFACTS:
        if (root / relative).exists():
            unexpected.append(relative)

    if unexpected:
        raise RuntimeError(
            "Unexpected LAB05 artifacts detected: "
            + ", ".join(unexpected)
        )

    print("FORBIDDEN_ARTIFACTS = NONE")


def patch_contract(root: Path) -> None:
    """Ensure all synthetic oracle variables are model-forbidden."""
    contract_path = (
        root
        / "labs"
        / "lab05_mineral_prospectivity"
        / "prospectivity_contract.json"
    )

    contract = json.loads(
        contract_path.read_text(encoding="utf-8")
    )

    forbidden = list(
        contract.get("forbidden_model_features", [])
    )

    changed = False

    for field in ORACLE_PROTECTED_FIELDS:
        if field not in forbidden:
            forbidden.append(field)
            changed = True

    contract["forbidden_model_features"] = forbidden

    if changed:
        contract_path.write_text(
            json.dumps(
                contract,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print("CONTRACT_PATCHED = YES")
    else:
        print("CONTRACT_PATCHED = NO")

    for field in ORACLE_PROTECTED_FIELDS:
        if field not in contract["forbidden_model_features"]:
            raise RuntimeError(
                f"Oracle field is not protected: {field}"
            )

    print(
        "FORBIDDEN_MODEL_FEATURES = "
        f"{len(contract['forbidden_model_features'])}"
    )


def validate_json(root: Path) -> None:
    """Validate that the prospectivity contract is valid JSON."""
    contract_path = (
        root
        / "labs"
        / "lab05_mineral_prospectivity"
        / "prospectivity_contract.json"
    )

    json.loads(
        contract_path.read_text(encoding="utf-8")
    )

    print("CONTRACT_JSON = PASS")


def main() -> int:
    """Run LAB05 bootstrap validation."""
    parser = argparse.ArgumentParser()
    parser.parse_args()

    root = repo_root()

    print("LAB05_BOOTSTRAP_START")
    print(f"REPO_ROOT = {root}")

    ensure_branch(root)
    ensure_directories(root)
    validate_required_files(root)
    validate_forbidden_artifacts(root)
    patch_contract(root)
    validate_json(root)

    print("LAB05_STRUCTURE_VALIDATION = PASS")
    print("LAB05_BOOTSTRAP_COMPLETE")

    return 0


if __name__ == "__main__":
    sys.exit(main())
