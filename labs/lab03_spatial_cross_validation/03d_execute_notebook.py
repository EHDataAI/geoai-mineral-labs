from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
NOTEBOOK_PATH = LAB_DIR / "03_spatial_cross_validation.ipynb"

notebook = nbformat.read(
    NOTEBOOK_PATH,
    as_version=4,
)

client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    allow_errors=False,
    resources={
        "metadata": {
            "path": str(REPO_ROOT),
        }
    },
)

executed = client.execute()

executed_text = nbformat.writes(executed)
executed_text = executed_text.rstrip("\r\n") + "\n"

NOTEBOOK_PATH.write_text(
    executed_text,
    encoding="utf-8",
    newline="\n",
)

print("NOTEBOOK_EXECUTION = PASS")
print("OUTPUT =", NOTEBOOK_PATH)
