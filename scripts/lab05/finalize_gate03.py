"""Finalize LAB05 Gate 3 reproducibly and non-interactively.

Engineering Reasoning Loop final gate:
- validate canonical source
- remove transient remediation scripts
- regenerate Gate 3
- verify byte-level reproducibility
- remove Python cache artifacts
- verify Git whitespace
- report repository state
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

SCRIPT_DIR = (
    ROOT
    / "scripts"
    / "lab05"
)

PYTHON = Path(
    "/home/erickh/.venvs/"
    "geoai-mineral-qa/bin/python"
)

EXPECTED_BRANCH = (
    "feat/lab05-mineral-prospectivity"
)

SETUP_PATH = (
    SCRIPT_DIR
    / "setup_and_run_gate03.py"
)

TRANSIENT_SCRIPTS = (
    SCRIPT_DIR
    / "patch_gate03_builder_future.py",
    SCRIPT_DIR
    / "repair_and_rerun_gate03.py",
)

CANONICAL_OUTPUTS = (
    LAB_DIR
    / "05c_build_evidence_layers.py",
    LAB_DIR
    / "data"
    / "prospectivity_evidence.csv",
    LAB_DIR
    / "results"
    / "evidence_manifest.json",
    LAB_DIR
    / "results"
    / "geochemistry_diagnostics.json",
    SCRIPT_DIR
    / "build_gate03.py",
    SCRIPT_DIR
    / "run_gate03.py",
    ROOT
    / "tests"
    / "test_lab05_evidence_layers.py",
)

CACHE_ROOTS = (
    LAB_DIR,
    SCRIPT_DIR,
    ROOT / "tests",
)

REQUIRED_SETUP_MARKERS = (
    '"Cu": 0.040,',
    '"Mo": 0.020,',
    '"Au": 0.012,',
    '"As": 0.030,',
    '"Sb": 0.018,',
    '"censored_fraction"',
    "stdin=subprocess.DEVNULL",
    '"GIT_TERMINAL_PROMPT"',
    '"PYTHONUNBUFFERED"',
    '"CI"',
    "safe_builder_code",
)


def environment() -> dict[str, str]:
    """Return deterministic non-interactive environment."""
    env = os.environ.copy()

    env["PYTHONPATH"] = str(
        ROOT / "src"
    )

    env["PYTHONUNBUFFERED"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["CI"] = "1"

    return env


def run(
    command: list[str],
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute command without interactive stdin."""
    print()
    print(
        "RUN = "
        + " ".join(command)
    )

    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env=environment(),
        stdin=subprocess.DEVNULL,
        capture_output=capture,
        text=True,
    )


def sha256(path: Path) -> str:
    """Calculate SHA256 for one file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def snapshot() -> dict[str, str]:
    """Hash every canonical Gate 3 output."""
    result: dict[str, str] = {}

    for path in CANONICAL_OUTPUTS:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing canonical output: {path}"
            )

        relative = str(
            path.relative_to(ROOT)
        )

        result[relative] = sha256(
            path
        )

    return result


def verify_branch() -> None:
    """Require LAB05 feature branch."""
    result = run(
        [
            "git",
            "branch",
            "--show-current",
        ],
        capture=True,
    )

    branch = result.stdout.strip()

    print(
        f"BRANCH = {branch}"
    )

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            "Gate 3 finalization requires "
            f"{EXPECTED_BRANCH!r}; "
            f"current branch is {branch!r}."
        )


def verify_setup() -> None:
    """Verify repaired behavior lives in canonical setup."""
    if not SETUP_PATH.is_file():
        raise FileNotFoundError(
            SETUP_PATH
        )

    text = SETUP_PATH.read_text(
        encoding="utf-8"
    )

    missing = [
        marker
        for marker in REQUIRED_SETUP_MARKERS
        if marker not in text
    ]

    if missing:
        raise RuntimeError(
            "Canonical Gate 3 setup is missing "
            f"required markers: {missing}"
        )

    print(
        "CANONICAL_SETUP_POLICY = PASS"
    )


def remove_transient_scripts() -> None:
    """Remove one-off remediation scripts."""
    for path in TRANSIENT_SCRIPTS:
        if path.exists():
            path.unlink()

            print(
                "REMOVED_TRANSIENT = "
                f"{path.name}"
            )
        else:
            print(
                "TRANSIENT_ALREADY_ABSENT = "
                f"{path.name}"
            )


def remove_python_caches() -> int:
    """Remove local Python cache directories."""
    removed = 0

    for base in CACHE_ROOTS:
        if not base.exists():
            continue

        caches = sorted(
            (
                path
                for path in base.rglob(
                    "__pycache__"
                )
                if path.is_dir()
            ),
            key=lambda path: len(
                path.parts
            ),
            reverse=True,
        )

        for cache in caches:
            shutil.rmtree(
                cache,
                ignore_errors=False,
            )

            removed += 1

    print(
        f"PYCACHE_REMOVED = {removed}"
    )

    return removed


def verify_reproducibility(
    before: dict[str, str],
    after: dict[str, str],
) -> None:
    """Require exact reproducibility of generated outputs."""
    if before.keys() != after.keys():
        raise RuntimeError(
            "Canonical output set changed "
            "during regeneration."
        )

    changed = [
        path
        for path in before
        if before[path] != after[path]
    ]

    if changed:
        print(
            "NON_REPRODUCIBLE_OUTPUTS:"
        )

        for path in changed:
            print(
                f"  {path}"
            )

        raise RuntimeError(
            "Gate 3 regeneration was not "
            "byte-for-byte reproducible."
        )

    print(
        "BYTE_REPRODUCIBILITY = PASS"
    )


def verify_no_transients() -> None:
    """Require remediation scripts to be absent."""
    remaining = [
        str(path.relative_to(ROOT))
        for path in TRANSIENT_SCRIPTS
        if path.exists()
    ]

    if remaining:
        raise RuntimeError(
            "Transient scripts remain: "
            f"{remaining}"
        )

    print(
        "TRANSIENT_SCRIPT_POLICY = PASS"
    )


def main() -> int:
    """Finalize Gate 3."""
    print(
        "LAB05_GATE03_FINALIZATION_START"
    )

    verify_branch()

    verify_setup()

    before = snapshot()

    print(
        "PRE_REGEN_HASHES = CAPTURED"
    )

    remove_transient_scripts()

    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(SETUP_PATH),
        ]
    )

    run(
        [
            str(PYTHON),
            str(SETUP_PATH),
        ]
    )

    after = snapshot()

    print(
        "POST_REGEN_HASHES = CAPTURED"
    )

    verify_reproducibility(
        before,
        after,
    )

    verify_no_transients()

    remove_python_caches()

    run(
        [
            "git",
            "--no-pager",
            "diff",
            "--check",
        ]
    )

    status = run(
        [
            "git",
            "--no-pager",
            "status",
            "-sb",
        ],
        capture=True,
    )

    print()
    print(
        status.stdout.rstrip()
    )

    print()
    print(
        "NON_INTERACTIVE_POLICY = PASS"
    )

    print(
        "GATE03_REPRODUCIBILITY = PASS"
    )

    print(
        "LAB05_GATE03_FINAL = PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
