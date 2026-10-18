"""Phase C tests for LABORATORY 06."""
from __future__ import annotations
import importlib.metadata, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"labs"/"lab06_explainable_geoai"/"results"

def test_shap_version_is_pinned():
    assert importlib.metadata.version("shap")=="0.53.0"

def test_permutation_contract():
    m=json.loads((R/"permutation_importance_manifest.json").read_text())
    s=pd.read_csv(R/"permutation_importance_summary.csv")
    assert m["status"]=="PASS"
    assert m["observed_spatial_fits"]==50
    assert m["spatial_group_overlap_max"]==0
    assert m["oracle_loaded"] is False
    assert m["model_reselected"] is False
    assert len(s)==9
    assert s["rank"].tolist()==list(range(1,10))

def test_global_shap_contract():
    m=json.loads((R/"global_shap_manifest.json").read_text())
    assert m["status"]=="PASS"
    assert m["shap_version"]=="0.53.0"
    assert m["total_oos_explanations"]==58000
    assert m["raw_feature_count"]==9
    assert m["transformed_feature_count"]==12
    assert m["maximum_additivity_error"]<=m["additivity_tolerance"]
    assert m["oracle_loaded"] is False
    assert m["model_reselected"] is False
    assert m["output_space"]=="log_odds"

def test_global_shap_artifacts():
    t=pd.read_csv(R/"shap_global_transformed.csv")
    r=pd.read_csv(R/"shap_global_raw.csv")
    s=pd.read_csv(R/"shap_global_sample.csv")
    assert len(t)==12
    assert len(r)==9
    assert len(s)==12000
    assert t["rank"].tolist()==list(range(1,13))
    assert r["rank"].tolist()==list(range(1,10))
    assert not s["shap_value"].isna().any()
