"""Make Gate 3 commit preparation idempotent.

The heavy generated dataset may already be absent from the Git index.
That condition is valid and must not cause the preparation script to fail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TARGET = (
    ROOT
    / "scripts"
    / "lab05"
    / "prepare_gate03_commit.py"
)

OLD = '''def unstage_heavy_dataset() -> None:
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
'''

NEW = '''def unstage_heavy_dataset() -> None:
    """Remove generated evidence CSV from index when necessary."""
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

    staged = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }

    if relative in staged:
        run(
            [
                "git",
                "restore",
                "--staged",
                "--",
                relative,
            ]
        )

        print(
            "HEAVY_DATASET_UNSTAGED = YES"
        )
    else:
        print(
            "HEAVY_DATASET_UNSTAGED = ALREADY"
        )

    if not HEAVY_DATASET.is_file():
        raise RuntimeError(
            "Heavy dataset is missing from "
            "the working tree."
        )

    print(
        "HEAVY_DATASET_LOCAL = PRESERVED"
    )
'''


def main() -> int:
    """Apply deterministic idempotence fix."""
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(
        encoding="utf-8"
    )

    if NEW in text:
        print(
            "PATCH_ALREADY_APPLIED = YES"
        )
        print(
            "GATE03_COMMIT_PREP_IDEMPOTENCE = PASS"
        )
        return 0

    count = text.count(OLD)

    if count != 1:
        raise RuntimeError(
            "Expected exactly one canonical "
            f"unstage function; found {count}."
        )

    TARGET.write_text(
        text.replace(
            OLD,
            NEW,
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )

    print(
        "PATCHED = prepare_gate03_commit.py"
    )

    print(
        "GATE03_COMMIT_PREP_IDEMPOTENCE = PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
