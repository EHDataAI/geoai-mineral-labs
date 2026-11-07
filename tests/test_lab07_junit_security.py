"""Security regressions for Phase C JUnit evidence parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from defusedxml.common import DTDForbidden

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "labs/lab07_spatial_uncertainty/07c_build_rank_stability.py"


@pytest.fixture(scope="module")
def phase_c():
    spec = importlib.util.spec_from_file_location(
        "lab07_c_xml_security",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_junit_is_accepted(phase_c, tmp_path):
    report = tmp_path / "valid.xml"
    report.write_text(
        '<testsuites><testsuite tests="2" failures="0" '
        'errors="0" skipped="0" /></testsuites>',
        encoding="utf-8",
    )
    assert phase_c.read_junit(report) == 2


def test_junit_dtd_is_rejected(phase_c, tmp_path):
    report = tmp_path / "malicious.xml"
    report.write_text(
        '<!DOCTYPE testsuite [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        '<testsuite tests="1" failures="0" errors="0" '
        'skipped="0">&xxe;</testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(DTDForbidden):
        phase_c.read_junit(report)


def test_junit_skips_remain_blocked(phase_c, tmp_path):
    report = tmp_path / "skipped.xml"
    report.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="1" />',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="JUnit skips"):
        phase_c.read_junit(report)
