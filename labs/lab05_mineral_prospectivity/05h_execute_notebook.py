from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


LAB_DIR = Path(__file__).resolve().parent
NOTEBOOK = LAB_DIR / "05_mineral_prospectivity.ipynb"


def main() -> int:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)

    os.chdir(LAB_DIR)

    with NOTEBOOK.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    for index, cell in enumerate(notebook.cells):
        if not cell.get("id"):
            cell["id"] = f"cell-{index:03d}"

    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(LAB_DIR)}},
    )

    executed = client.execute()

    with NOTEBOOK.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        nbformat.write(executed, handle)

    print(f"NOTEBOOK_EXECUTED={NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
