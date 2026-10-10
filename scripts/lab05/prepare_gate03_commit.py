"""Prepare LAB05 Gate 3 for a clean Git commit.

Policy:
- Generated heavy datasets remain reproducible but are not versioned.
- Source code, contracts, diagnostics, manifests and tests are versioned.
- No Git hook is bypassed.
- Execution is non-interactive.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

HEAVY_DATASET = (
    LAB_DIR
    / "data"
    / "prospectivity_evidence.csv"
)

LAB_GITIGNORE = (
    LAB_DIR
    / ".gitignore"
)

EXPECTED_BRANCH = (
    "feat/lab05-mineral-prospectivity"
)

MAX_GIT_FILE_BYTES = 5 * 1024 * 1024


def environment() -> dict[str, str]:
    """Return non-interactive execution environment."""
    env = os.environ.copy()

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
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one command without interactive stdin."""
    print()
    print(
        "RUN = "
        + " ".join(command)
    )

    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        env=environment(),
        stdin=subprocess.DEVNULL,
        capture_output=capture,
        text=True,
    )


def current_branch() -> str:
    """Return current Git branch."""
    result = run(
        [
            "git",
            "branch",
            "--show-current",
        ],
        capture=True,
    )

    return result.stdout.strip()


def write_lab_gitignore() -> None:
    """Create a scoped ignore rule for generated heavy evidence."""
    rule = "data/prospectivity_evidence.csv"

    existing: list[str] = []

    if LAB_GITIGNORE.is_file():
        existing = [
            line.rstrip()
            for line in LAB_GITIGNORE.read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    if rule not in existing:
        existing.append(rule)

    content = (
        "\n".join(
            line
            for line in existing
            if line
        )
        + "\n"
    )

    LAB_GITIGNORE.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )

    print(
        "LAB05_GITIGNORE = PASS"
    )


def unstage_heavy_dataset() -> None:
    """Remove generated evidence CSV from index, preserving local file."""
    relative = str(
        HEAVY_DATASET.relative_to(ROOT)
    )

    run(
        [
            "git",
            "restore",
            "--staged",
            "--",
            relative,
        ]
    )

    if not HEAVY_DATASET.is_file():
        raise RuntimeError(
            "Heavy dataset was unexpectedly removed "
            "from the working tree."
        )

    print(
        "HEAVY_DATASET_LOCAL = PRESERVED"
    )


def staged_files() -> list[Path]:
    """Return staged file paths."""
    result = run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
        ],
        capture=True,
    )

    paths: list[Path] = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if line:
            paths.append(
                ROOT / line
            )

    return paths


def validate_staged_sizes() -> None:
    """Reject staged files larger than repository policy."""
    oversized: list[tuple[str, int]] = []

    for path in staged_files():
        if not path.is_file():
            continue

        size = path.stat().st_size

        if size > MAX_GIT_FILE_BYTES:
            oversized.append(
                (
                    str(
                        path.relative_to(ROOT)
                    ),
                    size,
                )
            )

    if oversized:
        print(
            "OVERSIZED_STAGED_FILES:"
        )

        for name, size in oversized:
            print(
                f"  {name} = {size} bytes"
            )

        raise RuntimeError(
            "Oversized files remain staged."
        )

    print(
        "STAGED_FILE_SIZE_POLICY = PASS"
    )


def validate_heavy_dataset_not_staged() -> None:
    """Require generated heavy CSV to stay out of Git index."""
    relative = str(
        HEAVY_DATASET.relative_to(ROOT)
    )

    result = run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
        ],
        capture=True,
    )

    staged = set(
        result.stdout.splitlines()
    )

    if relative in staged:
        raise RuntimeError(
            "Heavy generated dataset remains staged."
        )

    print(
        "HEAVY_DATASET_STAGED = NO"
    )


def main() -> int:
    """Prepare Gate 3 commit."""
    print(
        "LAB05_GATE03_COMMIT_PREP_START"
    )

    branch = current_branch()

    print(
        f"BRANCH = {branch}"
    )

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            "Expected branch "
            f"{EXPECTED_BRANCH!r}; "
            f"found {branch!r}."
        )

    if not HEAVY_DATASET.is_file():
        raise FileNotFoundError(
            HEAVY_DATASET
        )

    print(
        "HEAVY_DATASET_BYTES = "
        f"{HEAVY_DATASET.stat().st_size}"
    )

    unstage_heavy_dataset()

    write_lab_gitignore()

    run(
        [
            "git",
            "add",
            str(
                LAB_GITIGNORE.relative_to(ROOT)
            ),
        ]
    )

    validate_heavy_dataset_not_staged()

    validate_staged_sizes()

    run(
        [
            "git",
            "--no-pager",
            "diff",
            "--cached",
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
        "GENERATED_HEAVY_DATA_POLICY = PASS"
    )

    print(
        "NON_INTERACTIVE_POLICY = PASS"
    )

    print(
        "LAB05_GATE03_COMMIT_PREP = PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
