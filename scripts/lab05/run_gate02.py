"""Execute and validate LAB05 Gate 2."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PYTHON = Path(
    "/home/erickh/.venvs/"
    "geoai-mineral-qa/bin/python"
)

AUDIT = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
    / "05b_audit_labels_and_spatial_cv.py"
)


def run(
    command: list[str],
    *,
    env: dict[str, str],
) -> None:
    """Run one Gate 2 command."""
    print()
    print(
        "RUN = "
        + " ".join(command)
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env=env,
    )


def main() -> int:
    """Execute Gate 2 QA."""
    environment = os.environ.copy()

    environment["PYTHONPATH"] = str(
        ROOT / "src"
    )

    environment["GIT_PAGER"] = "cat"
    environment["PAGER"] = "cat"

    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(AUDIT),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            str(AUDIT),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "tests/test_lab05_spatial_cv_audit.py",
            "-q",
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
        ],
        env=environment,
    )

    run(
        [
            "git",
            "--no-pager",
            "diff",
            "--check",
        ],
        env=environment,
    )

    print()
    print("LAB05_GATE02_QA = PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
