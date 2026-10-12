"""Guard against source drift in the LAB05 Gate 3 builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BUILDER_PATH = (
    ROOT
    / "scripts"
    / "lab05"
    / "build_gate03.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "lab05_build_gate03",
        BUILDER_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def test_gate03_embedded_sources_match_canonical_files() -> None:
    builder = load_builder()

    comparisons = [
        (
            builder.EVIDENCE_CODE,
            ROOT
            / "labs"
            / "lab05_mineral_prospectivity"
            / "05c_build_evidence_layers.py",
        ),
        (
            builder.TEST_CODE,
            ROOT
            / "tests"
            / "test_lab05_evidence_layers.py",
        ),
        (
            builder.RUNNER_CODE,
            ROOT
            / "scripts"
            / "lab05"
            / "run_gate03.py",
        ),
    ]

    for embedded, canonical_path in comparisons:
        canonical = canonical_path.read_text(
            encoding="utf-8"
        )

        assert (
            embedded.rstrip()
            == canonical.rstrip()
        ), (
            "Gate 3 builder source drift detected: "
            f"{canonical_path.relative_to(ROOT)}"
        )
