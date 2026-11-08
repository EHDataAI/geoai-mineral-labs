"""Security regressions for Phase C JUnit evidence parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

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
        'errors="0" skipped="0">'
        '<testcase name="first"/><testcase name="second"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    assert phase_c.read_junit(report) == 2


def test_forged_junit_summary_cannot_certify_217_tests(phase_c, tmp_path):
    report = tmp_path / "forged.xml"
    report.write_text(
        '<testsuites><testsuite tests="217" failures="0" '
        'errors="0" skipped="0" /></testsuites>',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="case-count mismatch"):
        phase_c.read_junit(report)


def test_hidden_case_failure_is_rejected(phase_c, tmp_path):
    report = tmp_path / "hidden_failure.xml"
    report.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="bad"><failure /></testcase></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="case-level failure"):
        phase_c.read_junit(report)


def test_release_qa_reexecutes_checks(phase_c, tmp_path):
    with patch.object(phase_c.subprocess, "run") as command:
        phase_c.run_release_qa(tmp_path)
    assert command.call_count == 7
    arguments = [call.args[0] for call in command.call_args_list]
    assert any("pytest" in args for args in arguments)
    assert any(args[:2] == ["pre-commit", "run"] for args in arguments)
    assert sum(args[0] == "gitleaks" for args in arguments) == 2


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
        '<testsuite tests="1" failures="0" errors="0" skipped="1">'
        '<testcase name="skipped"/></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="JUnit skips"):
        phase_c.read_junit(report)
