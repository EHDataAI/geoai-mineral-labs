"""Run LAB05 Gate 5.1 OOS-only scoring and QA."""

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

SCORER = (
    LAB_DIR
    / "05e1_score_full_grid_oos.py"
)

TEST_FILE = (
    ROOT
    / "tests"
    / "test_lab05_oos_full_grid_scoring.py"
)

RESULT_FILES = (
    LAB_DIR
    / "results"
    / "prospectivity_scores_oos.csv",
    LAB_DIR
    / "results"
    / "prospectivity_ranking_summary_oos.csv",
    LAB_DIR
    / "results"
    / "prospectivity_model_manifest_oos.json",
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
        "LAB05_GATE05_1_RUNNER_START"
    )

    run(
        [
            PYTHON,
            "-m",
            "py_compile",
            str(SCORER),
            str(TEST_FILE),
        ]
    )

    run(
        [
            PYTHON,
            "-W",
            "error::RuntimeWarning",
            str(SCORER),
        ]
    )

    first = snapshot()

    run(
        [
            PYTHON,
            "-W",
            "error::RuntimeWarning",
            str(SCORER),
        ]
    )

    second = snapshot()

    if first != second:
        changed = [
            key
            for key in first
            if first[
                key
            ]
            != second[
                key
            ]
        ]

        raise RuntimeError(
            "Gate 5.1 outputs are not "
            "byte-reproducible: "
            f"{changed}"
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
            "test_lab05_oos_full_grid_scoring.py",
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
        "LAB05_GATE05_1_QA = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
