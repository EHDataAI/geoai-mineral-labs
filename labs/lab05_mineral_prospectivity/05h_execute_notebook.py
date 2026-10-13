from __future__ import annotations
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = ROOT / "labs" / "lab05_mineral_prospectivity"
NOTEBOOK_PATH = LAB_DIR / "05_mineral_prospectivity.ipynb"

def count_png_outputs(notebook) -> int:
    return sum(
        1
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )

def main() -> int:
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    for index, cell in enumerate(notebook.cells):
        if not cell.get("id"):
            cell["id"] = f"cell-{index:03d}"

    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata":{"path":str(LAB_DIR)}},
    )
    executed = client.execute()
    nbformat.validate(executed)

    png_outputs = count_png_outputs(executed)
    if png_outputs < 4:
        raise RuntimeError(f"Expected at least 4 embedded PNG figures; found {png_outputs}.")

    with NOTEBOOK_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        nbformat.write(executed, handle)

    print(f"NOTEBOOK_EXECUTED={NOTEBOOK_PATH}")
    print(f"PNG_OUTPUTS={png_outputs}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
