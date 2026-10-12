"""Disable Git pager in LAB05 Gate 1 automation."""

from __future__ import annotations

import sys
from pathlib import Path


OLD = '"git",\n            "diff",\n            "--check",'
NEW = '"git",\n            "--no-pager",\n            "diff",\n            "--check",'


def patch_file(path: Path) -> None:
    """Patch one Gate 1 automation file idempotently."""
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    if NEW in text:
        print(f"ALREADY_PATCHED = {path.name}")
        return

    count = text.count(OLD)

    if count != 1:
        raise RuntimeError(
            f"{path.name}: expected exactly one Git diff command; "
            f"found {count}."
        )

    path.write_text(
        text.replace(OLD, NEW, 1),
        encoding="utf-8",
    )

    print(f"PATCHED = {path.name}")


def main() -> int:
    """Patch runner and builder."""
    root = Path(__file__).resolve().parents[2]

    files = (
        root / "scripts" / "lab05" / "run_gate01.py",
        root / "scripts" / "lab05" / "build_gate01.py",
    )

    for path in files:
        patch_file(path)

    print("LAB05_GATE01_NO_PAGER_PATCH = PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
