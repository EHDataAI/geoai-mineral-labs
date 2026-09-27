from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient


NOTEBOOK_NAME = "02_spatial_eda.ipynb"


class SecureAsyncKernelManager(
    AsyncKernelManager
):
    """Kernel manager with encrypted local transport."""

    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault(
            "transport_encryption",
            "required",
        )
        kwargs.setdefault(
            "ip",
            "127.0.0.1",
        )

        super().__init__(**kwargs)


def configure_event_loop() -> None:
    """Use the ZMQ-compatible event loop policy on Windows."""

    if sys.platform != "win32":
        return

    selector_policy = getattr(
        asyncio,
        "WindowsSelectorEventLoopPolicy",
        None,
    )

    if selector_policy is not None:
        asyncio.set_event_loop_policy(
            selector_policy()
        )


def find_repo_root(start: Path) -> Path:
    current = start.resolve()

    for candidate in (current, *current.parents):
        if (
            (candidate / "src" / "geoai").exists()
            and (candidate / "environment.yml").exists()
        ):
            return candidate

    raise RuntimeError(
        "Repository root was not found."
    )


def count_output_errors(
    notebook: nbformat.NotebookNode,
) -> int:
    error_count = 0

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_count += 1

    return error_count


def main() -> None:
    configure_event_loop()

    script_dir = Path(__file__).resolve().parent
    notebook_path = script_dir / NOTEBOOK_NAME
    repo_root = find_repo_root(script_dir)

    notebook = nbformat.read(
        notebook_path,
        as_version=4,
    )

    nbformat.validate(notebook)

    code_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]

    print("=" * 72)
    print("LAB 02 - NOTEBOOK EXECUTION")
    print("=" * 72)

    print("Repository root:")
    print(repo_root)

    print("\nNotebook:")
    print(notebook_path)

    print("\nStructure before execution")
    print("Total cells:", len(notebook.cells))
    print("Code cells:", len(code_cells))

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        kernel_manager_class=(
            SecureAsyncKernelManager
        ),
        allow_errors=False,
        resources={
            "metadata": {
                "path": str(repo_root),
            }
        },
    )

    executed = client.execute()

    nbformat.validate(executed)

    error_count = count_output_errors(executed)

    if error_count != 0:
        raise RuntimeError(
            f"Notebook contains {error_count} error output(s)."
        )

    executed_code_cells = [
        cell
        for cell in executed.cells
        if cell.cell_type == "code"
    ]

    cells_with_execution_count = sum(
        cell.get("execution_count") is not None
        for cell in executed_code_cells
    )

    cells_with_outputs = sum(
        bool(cell.get("outputs"))
        for cell in executed_code_cells
    )

    nbformat.write(
        executed,
        notebook_path,
    )

    print("\nExecution result")
    print("Schema validation: PASS")
    print("Error outputs:", error_count)
    print(
        "Code cells executed:",
        cells_with_execution_count,
        "/",
        len(executed_code_cells),
    )
    print(
        "Code cells with outputs:",
        cells_with_outputs,
    )

    print("\nExecuted notebook saved:")
    print(notebook_path)


if __name__ == "__main__":
    main()
