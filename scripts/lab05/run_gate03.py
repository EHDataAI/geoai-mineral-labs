"""Execute and validate LAB05 Gate 3."""

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

EVIDENCE = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
    / "05c_build_evidence_layers.py"
)


def run(
    command: list[str],
    *,
    env: dict[str, str],
) -> None:
    """Run one Gate 3 command."""
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
        stdin=subprocess.DEVNULL,
    )


def main() -> int:
    """Execute Gate 3 QA."""
    environment = os.environ.copy()

    environment["PYTHONPATH"] = str(
        ROOT / "src"
    )

    environment["PYTHONUNBUFFERED"] = "1"
    environment["GIT_PAGER"] = "cat"
    environment["PAGER"] = "cat"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["CI"] = "1"

    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(EVIDENCE),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            str(EVIDENCE),
        ],
        env=environment,
    )

    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "tests/test_lab05_evidence_layers.py",
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
    print("LAB05_GATE03_QA = PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
