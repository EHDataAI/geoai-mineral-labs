"""Run LAB05 Gate 4.1 pooled OOF benchmark and QA."""

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

BENCHMARK = (
    LAB_DIR
    / "05d1_build_oof_benchmark.py"
)

TEST_FILE = (
    ROOT
    / "tests"
    / "test_lab05_oof_benchmark.py"
)

RESULT_FILES = (
    LAB_DIR
    / "results"
    / "benchmark_oof_metrics.csv",
    LAB_DIR
    / "results"
    / "benchmark_oof_aggregate_metrics.csv",
    LAB_DIR
    / "results"
    / "success_rate_curve_oof.csv",
    LAB_DIR
    / "results"
    / "benchmark_oof_manifest.json",
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


def verify_sizes() -> None:
    limit = (
        5
        * 1024
        * 1024
    )

    for path in RESULT_FILES:
        size = path.stat().st_size

        print(
            f"RESULT_BYTES {path.name} = "
            f"{size}"
        )

        if size > limit:
            raise RuntimeError(
                "Result exceeds 5 MiB: "
                f"{path}"
            )


def main() -> int:
    print(
        "LAB05_GATE04_1_RUNNER_START"
    )

    run(
        [
            PYTHON,
            "-m",
            "py_compile",
            str(BENCHMARK),
            str(TEST_FILE),
        ]
    )

    run(
        [
            PYTHON,
            str(BENCHMARK),
        ]
    )

    first = snapshot()

    run(
        [
            PYTHON,
            str(BENCHMARK),
        ]
    )

    second = snapshot()

    if first != second:
        changed = [
            key
            for key in first
            if first[key]
            != second[key]
        ]

        raise RuntimeError(
            "Gate 4.1 outputs are not "
            "byte-reproducible: "
            f"{changed}"
        )

    print(
        "BYTE_REPRODUCIBILITY = PASS"
    )

    verify_sizes()

    run(
        [
            PYTHON,
            "-m",
            "pytest",
            "tests/"
            "test_lab05_oof_benchmark.py",
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
        "LAB05_GATE04_1_QA = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
