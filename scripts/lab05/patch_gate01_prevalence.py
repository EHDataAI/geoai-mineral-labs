"""Calibrate LAB05 Gate 1 synthetic deposit prevalence."""

from __future__ import annotations

import sys
from pathlib import Path


EXPECTED_OLD = "logit = -7.0 + 9.0 * latent"
EXPECTED_NEW = "logit = -8.5 + 9.0 * latent"


def main() -> int:
    """Patch the Gate 1 generator deterministically."""
    root = Path(__file__).resolve().parents[2]

    generator = (
        root
        / "labs"
        / "lab05_mineral_prospectivity"
        / "05a_generate_synthetic_mineral_system.py"
    )

    if not generator.is_file():
        raise FileNotFoundError(
            f"Generator not found: {generator}"
        )

    text = generator.read_text(
        encoding="utf-8"
    )

    if EXPECTED_NEW in text:
        print("PATCH_ALREADY_APPLIED = YES")
        print("LAB05_GATE01_PREVALENCE_PATCH = PASS")
        return 0

    count = text.count(EXPECTED_OLD)

    if count != 1:
        raise RuntimeError(
            "Expected exactly one original logit expression; "
            f"found {count}."
        )

    patched = text.replace(
        EXPECTED_OLD,
        EXPECTED_NEW,
        1,
    )

    generator.write_text(
        patched,
        encoding="utf-8",
    )

    print("OLD_LOGIT = -7.0 + 9.0 * latent")
    print("NEW_LOGIT = -8.5 + 9.0 * latent")
    print("TARGET_BEHAVIOUR = rare synthetic deposits")
    print("LAB05_GATE01_PREVALENCE_PATCH = PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
