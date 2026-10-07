"""Regression tests for the single-source LAB04 data contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

LAB_DIR = Path(__file__).resolve().parents[1] / "labs" / "lab04_geochemical_coda"


def _load_module():
    import sys

    sys.path.insert(0, str(LAB_DIR))
    spec = importlib.util.spec_from_file_location(
        "lab04_geochemistry_config", LAB_DIR / "geochemistry_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_is_valid():
    cfg = _load_module()
    data = cfg.load_contract()
    assert data["unit"] == "ppm"
    assert data["whole_rock_composition"] is False
    assert data["unit_status"] == "synthetic_convention"


def test_lod_mapping_matches_ordered_array():
    cfg = _load_module()
    names = cfg.get_elements()
    vector = cfg.get_detection_limits()
    mapping = cfg.get_detection_limit_mapping()
    np.testing.assert_array_equal(vector, [mapping[x] for x in names])
    assert vector.shape == (10,)
    assert cfg.get_replacement_fractions()["lod_half"] == 0.5
    assert np.isclose(cfg.get_replacement_fractions()["lod_sqrt2"], 1 / np.sqrt(2))


def test_contract_rejects_unit_mutation(tmp_path, monkeypatch):
    import json

    cfg = _load_module()
    original = cfg.load_contract()
    original["unit"] = "percent"
    bad = tmp_path / "bad_contract.json"
    bad.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(cfg, "CONTRACT_PATH", bad)
    with pytest.raises(ValueError, match="ppm"):
        cfg.load_contract()


def test_utf8_contract_and_scripts():
    for name in (
        "geochemistry_config.py",
        "geochemistry_contract.json",
        "04a_generate_synthetic_geochemistry.py",
        "04b_run_coda_analysis.py",
        "04c_lod_sensitivity.py",
        "04d_audit_data_contract.py",
    ):
        content = (LAB_DIR / name).read_bytes()
        assert not content.startswith(b"\xef\xbb\xbf")
        assert b"\r\n" not in content
        text = content.decode("utf-8")
        assert "\ufffd" not in text
