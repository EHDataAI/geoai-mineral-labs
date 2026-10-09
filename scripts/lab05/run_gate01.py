"""Execute and validate LAB05 Gate 1."""

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

GENERATOR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
    / "05a_generate_synthetic_mineral_system.py"
)


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command and fail immediately on error."""
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
    """Run Gate 1 scientific QA."""
    environment = os.environ.copy()

    environment["PYTHONPATH"] = str(
        ROOT / "src"
    )

    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(GENERATOR),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            str(GENERATOR),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "tests/test_lab05_synthetic_system.py",
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
    print("LAB05_GATE01_QA = PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
