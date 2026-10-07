"""Quality checks for the LAB04 generated and executed notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]

LAB_DIR = (
    ROOT
    / "labs"
    / "lab04_geochemical_coda"
)

SOURCE_NOTEBOOK = (
    LAB_DIR
    / "04_geochemical_coda.ipynb"
)

EXECUTED_NOTEBOOK = (
    LAB_DIR
    / "04_geochemical_coda_executed.ipynb"
)


def load_notebook(path: Path) -> nbformat.NotebookNode:
    """Load a notebook using nbformat v4."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return nbformat.read(
            handle,
            as_version=4,
        )


def test_lab04_notebooks_exist() -> None:
    """Both source and executed notebooks must exist."""
    assert SOURCE_NOTEBOOK.is_file()
    assert EXECUTED_NOTEBOOK.is_file()


def test_source_notebook_structure() -> None:
    """The generated notebook must retain the expected cell structure."""
    notebook = load_notebook(SOURCE_NOTEBOOK)

    assert len(notebook.cells) == 45

    markdown_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    ]

    code_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]

    assert len(markdown_cells) == 21
    assert len(code_cells) == 24


def test_executed_notebook_has_no_error_outputs() -> None:
    """Executed notebook must not contain notebook error outputs."""
    notebook = load_notebook(EXECUTED_NOTEBOOK)

    errors = []

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(output)

    assert errors == []


def test_all_nonempty_code_cells_are_executed() -> None:
    """Every non-empty code cell must have an execution count."""
    notebook = load_notebook(EXECUTED_NOTEBOOK)

    unexecuted = []

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue

        if not cell.source.strip():
            continue

        if cell.get("execution_count") is None:
            unexecuted.append(index)

    assert unexecuted == []


def test_executed_notebook_contains_png_outputs() -> None:
    """The executed notebook must contain graphical outputs."""
    notebook = load_notebook(EXECUTED_NOTEBOOK)

    png_outputs = 0

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        for output in cell.get("outputs", []):
            data = output.get("data", {})

            if (
                isinstance(data, dict)
                and "image/png" in data
            ):
                png_outputs += 1

    assert png_outputs >= 10


def test_notebook_unicode_integrity() -> None:
    """Notebook files must remain UTF-8 clean."""
    for path in [
        SOURCE_NOTEBOOK,
        EXECUTED_NOTEBOOK,
    ]:
        raw = path.read_bytes()

        assert not raw.startswith(
            b"\xef\xbb\xbf"
        )

        text = raw.decode("utf-8")

        assert "\ufffd" not in text

        notebook = json.loads(text)

        for cell in notebook["cells"]:
            source = cell.get(
                "source",
                [],
            )

            if isinstance(source, list):
                source = "".join(source)

            assert "?" not in source


def test_executed_notebook_cell_count_matches_source() -> None:
    """Execution must not alter the notebook cell structure."""
    source = load_notebook(SOURCE_NOTEBOOK)
    executed = load_notebook(EXECUTED_NOTEBOOK)

    assert len(source.cells) == len(
        executed.cells
    )
