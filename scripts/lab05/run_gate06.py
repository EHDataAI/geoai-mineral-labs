"""Run LAB05 Gate 6 final hidden-oracle evaluation and QA."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab05_mineral_prospectivity"
)

PYTHON = (
    "/home/erickh/.venvs/"
    "geoai-mineral-qa/bin/python"
)

EVALUATOR = (
    LAB_DIR
    / "05f_evaluate_hidden_oracle.py"
)

TEST_FILE = (
    ROOT
    / "tests"
    / "test_lab05_hidden_oracle_evaluation.py"
)

RESULT_FILES = (
    LAB_DIR
    / "results"
    / "oracle_evaluation_summary.csv",
    LAB_DIR
    / "results"
    / "oracle_hidden_deposit_ranks.csv",
    LAB_DIR
    / "results"
    / "oracle_evaluation_manifest.json",
)


def environment() -> dict[str, str]:
    env = os.environ.copy()

    env["PYTHONPATH"] = (
        str(LAB_DIR)
        + os.pathsep
        + str(
            ROOT
            / "src"
        )
    )

    env["PYTHONUNBUFFERED"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["CI"] = "1"

    return env


def run(
    command: list[str],
) -> None:
    print()
    print(
        "RUN = "
        + " ".join(command)
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env=environment(),
        stdin=subprocess.DEVNULL,
    )


def sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def snapshot() -> dict[str, str]:
    result: dict[str, str] = {}

    for path in RESULT_FILES:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        result[
            str(
                path.relative_to(
                    ROOT
                )
            )
        ] = sha256(
            path
        )

    return result


def main() -> int:
    print(
        "LAB05_GATE06_RUNNER_START"
    )

    run(
        [
            PYTHON,
            "-m",
            "py_compile",
            str(EVALUATOR),
            str(TEST_FILE),
        ]
    )

    run(
        [
            PYTHON,
            "-W",
            "error::RuntimeWarning",
            str(EVALUATOR),
        ]
    )

    first = snapshot()

    run(
        [
            PYTHON,
            "-W",
            "error::RuntimeWarning",
            str(EVALUATOR),
        ]
    )

    second = snapshot()

    if first != second:
        raise RuntimeError(
            "Gate 6 outputs are not byte-reproducible."
        )

    print(
        "BYTE_REPRODUCIBILITY = PASS"
    )

    run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/"
            "test_lab05_hidden_oracle_evaluation.py",
            "-q",
        ]
    )

    run(
        [
            PYTHON,
            "-m",
            "pytest",
            "-q",
        ]
    )

    run(
        [
            "git",
            "--no-pager",
            "diff",
            "--check",
        ]
    )

    print()

    print(
        "NON_INTERACTIVE_POLICY = PASS"
    )

    print(
        "LAB05_GATE06_QA = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
