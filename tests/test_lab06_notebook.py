"""Final notebook tests for LABORATORY 06."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "labs" / "lab06_explainable_geoai"
NOTEBOOK = LAB_DIR / "06_explainable_geoai.ipynb"
RESULTS = LAB_DIR / "results"


def load_notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_lab06_notebook_exists_and_validates():
    assert NOTEBOOK.exists()

    notebook = load_notebook()
    nbformat.validate(notebook)

    assert NOTEBOOK.stat().st_size < 5 * 1024 * 1024


def test_lab06_notebook_has_required_sections():
    notebook = load_notebook()
    markdown = "\n".join(
        cell["source"]
        for cell in notebook.cells
        if cell["cell_type"] == "markdown"
    )

    required = [
        "Frozen explanation contract",
        "Global drivers",
        "Cross-method agreement",
        "Local explanations",
        "Spatial explainability",
        "Stability across background replicates",
        "Scientific interpretation",
        "Publication boundary",
        "Key takeaway",
    ]

    for heading in required:
        assert heading in markdown


def test_lab06_notebook_execution_outputs():
    notebook = load_notebook()

    png_outputs = sum(
        1
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )

    error_outputs = sum(
        1
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    )

    assert png_outputs >= 10
    assert error_outputs == 0


def test_lab06_notebook_qa_manifest():
    manifest = json.loads(
        (RESULTS / "notebook_qa_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["status"] == "PASS"
    assert manifest["phase"] == "F"
    assert manifest["png_outputs"] >= 10
    assert manifest["error_outputs"] == 0
    assert manifest["notebook_bytes"] < 5 * 1024 * 1024
    assert manifest["manual_visual_review_required_before_merge"] is True
    assert manifest["github_rendering_review_required_before_merge"] is True
    assert manifest["oracle_loaded"] is False
    assert manifest["model_reselected"] is False


def test_lab06_visual_contract_is_preserved():
    contract = json.loads(
        (LAB_DIR / "notebook_visual_contract.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["status"] == "design_locked"
    assert contract["figure_budget"] == 10
    assert contract["table_style"]["avoid_raw_dataframe_dump"] is True
    assert "github_rendering_must_be_checked" in contract["visual_rules"]
