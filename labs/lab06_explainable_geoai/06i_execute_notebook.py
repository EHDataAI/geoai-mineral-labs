"""Execute and validate the LABORATORY 06 public notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = ROOT / "labs" / "lab06_explainable_geoai"
NOTEBOOK_PATH = LAB_DIR / "06_explainable_geoai.ipynb"
RESULTS_DIR = LAB_DIR / "results"
QA_MANIFEST_PATH = RESULTS_DIR / "notebook_qa_manifest.json"

MIN_PNG_OUTPUTS = 10
MAX_NOTEBOOK_BYTES = 5 * 1024 * 1024


def count_png_outputs(notebook) -> int:
    return sum(
        1
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )


def count_error_outputs(notebook) -> int:
    return sum(
        1
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    )


def main() -> int:
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    for index, cell in enumerate(notebook.cells):
        if not cell.get("id"):
            cell["id"] = f"lab06-{index:03d}"

    code_cells = [
        cell
        for cell in notebook.cells
        if cell.get("cell_type") == "code"
    ]

    markdown_cells = [
        cell
        for cell in notebook.cells
        if cell.get("cell_type") == "markdown"
    ]

    client = NotebookClient(
        notebook,
        timeout=420,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(LAB_DIR)}},
    )

    executed = client.execute()
    nbformat.validate(executed)

    png_outputs = count_png_outputs(executed)
    error_outputs = count_error_outputs(executed)

    if png_outputs < MIN_PNG_OUTPUTS:
        raise RuntimeError(
            "Expected at least "
            f"{MIN_PNG_OUTPUTS} embedded PNG figures; "
            f"found {png_outputs}."
        )

    if error_outputs != 0:
        raise RuntimeError(
            f"Expected zero notebook errors; found {error_outputs}."
        )

    with NOTEBOOK_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        nbformat.write(executed, handle)

    notebook_bytes = NOTEBOOK_PATH.stat().st_size

    if notebook_bytes >= MAX_NOTEBOOK_BYTES:
        raise RuntimeError(
            "Executed notebook exceeds the repository 5 MB "
            f"large-file threshold: {notebook_bytes} bytes."
        )

    digest = hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest()

    manifest = {
        "laboratory": "LABORATORY 06",
        "phase": "F",
        "status": "PASS",
        "notebook": NOTEBOOK_PATH.name,
        "code_cells": len(code_cells),
        "markdown_cells": len(markdown_cells),
        "png_outputs": png_outputs,
        "error_outputs": error_outputs,
        "notebook_bytes": notebook_bytes,
        "sha256": digest,
        "minimum_png_outputs": MIN_PNG_OUTPUTS,
        "maximum_notebook_bytes": MAX_NOTEBOOK_BYTES,
        "manual_visual_review_required_before_merge": True,
        "github_rendering_review_required_before_merge": True,
        "oracle_loaded": False,
        "model_reselected": False,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    QA_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"NOTEBOOK_EXECUTED = {NOTEBOOK_PATH}")
    print(f"NOTEBOOK_CODE_CELLS = {len(code_cells)}")
    print(f"NOTEBOOK_MARKDOWN_CELLS = {len(markdown_cells)}")
    print(f"PNG_OUTPUTS = {png_outputs}")
    print(f"ERROR_OUTPUTS = {error_outputs}")
    print(f"NOTEBOOK_BYTES = {notebook_bytes}")
    print(f"NOTEBOOK_SHA256 = {digest}")
    print("MANUAL_VISUAL_REVIEW_REQUIRED = YES")
    print("LAB06_NOTEBOOK_EXECUTION_QA = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
