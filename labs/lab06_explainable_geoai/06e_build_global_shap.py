"""LABORATORY 06 Phase C: OOS-aware global SHAP."""
from __future__ import annotations
import importlib.metadata, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap

ROOT=Path(__file__).resolve().parents[2]
LAB05=ROOT/"labs"/"lab05_mineral_prospectivity"
OUT=ROOT/"labs"/"lab06_explainable_geoai"/"results"
G04=LAB05/"05d_run_spatial_benchmark.py"
G051=LAB05/"05e1_score_full_grid_oos.py"
TOL=1e-8

def load_module(path,name):
    if str(LAB05) not in sys.path:
        sys.path.insert(0,str(LAB05))
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load LAB05 module.")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def raw_name(name):
    return "lithology" if name.startswith("lithology_") else name

def main():
    if importlib.metadata.version("shap")!="0.53.0":
        raise RuntimeError("Unexpected SHAP version.")
    g=load_module(G04,"lab05_g04_lab06_c_shap")
    s=load_module(G051,"lab05_g051_lab06_c_shap")
    frame,features=s.load_safe_evidence(g)
    candidate=np.flatnonzero(frame[g.TARGET].to_numpy(dtype=int)==0)
    ids=frame["cell_id"].astype(str).to_numpy()
    summary_rows=[]
    sample_rows=[]
    names0=None
    max_err=0.0
    total=0
    for rep in range(10):
        train=g.sample_background(frame,rep)
        bg=set(train.loc[train[g.TARGET]==0,"cell_id"].astype(str))
        eligible=candidate[np.array([ids[i] not in bg for i in candidate],dtype=bool)]
        if len(eligible)!=5800:
            raise RuntimeError("Unexpected OOS candidate count.")
        pipe=g.build_model_pipeline("logistic_regression",features,g.SEED+rep*1000)
        pipe.fit(train[features],train[g.TARGET].to_numpy(dtype=int))
        pre=pipe.named_steps["preprocess"]
        model=pipe.named_steps["model"]
        names=list(pre.get_feature_names_out())
        if names0 is None:
            names0=names
        elif names!=names0:
            raise RuntimeError("Transformed feature drift.")
        bg_t=pre.transform(train[features])
        oos_t=pre.transform(frame.iloc[eligible][features])
        masker=shap.maskers.Independent(
            bg_t,
            max_samples=len(bg_t),
        )
        exp=shap.LinearExplainer(model,masker)(oos_t)
        vals=np.asarray(exp.values,dtype=float)
        base=np.asarray(exp.base_values,dtype=float)
        if base.ndim==0:
            base=np.full(len(eligible),float(base))
        elif base.ndim>1:
            base=base.reshape(len(eligible),-1)[:,0]
        err=float(np.max(np.abs(base+vals.sum(axis=1)-model.decision_function(oos_t))))
        max_err=max(max_err,err)
        if err>TOL:
            raise RuntimeError(f"SHAP additivity failed: {err}")
        for j,name in enumerate(names):
            summary_rows.append({
              "replicate":rep,"transformed_feature":name,
              "raw_feature":raw_name(name),
              "mean_abs_shap":float(np.mean(np.abs(vals[:,j]))),
              "mean_signed_shap":float(np.mean(vals[:,j])),
            })
        rng=np.random.default_rng(g.SEED+rep)
        pick=rng.choice(len(eligible),size=100,replace=False)
        for pos in pick:
            cell=str(frame.iloc[int(eligible[pos])]["cell_id"])
            for j,name in enumerate(names):
                sample_rows.append({
                  "replicate":rep,"cell_id":cell,
                  "transformed_feature":name,"raw_feature":raw_name(name),
                  "shap_value":float(vals[pos,j]),
                  "transformed_value":float(oos_t[pos,j]),
                })
        total+=len(eligible)
        print(f"GLOBAL_SHAP_REPLICATE_COMPLETE = {rep+1}/10")
    x=pd.DataFrame(summary_rows)
    t=(x.groupby(["transformed_feature","raw_feature"],as_index=False)
       .agg(mean_abs_shap=("mean_abs_shap","mean"),
            median_abs_shap=("mean_abs_shap","median"),
            std_abs_shap=("mean_abs_shap","std"),
            mean_signed_shap=("mean_signed_shap","mean"))
       .sort_values(["mean_abs_shap","transformed_feature"],ascending=[False,True],kind="mergesort")
       .reset_index(drop=True))
    t["rank"]=np.arange(1,len(t)+1,dtype=int)
    r=(x.groupby(["replicate","raw_feature"],as_index=False)["mean_abs_shap"].sum()
       .groupby("raw_feature",as_index=False)
       .agg(mean_abs_shap=("mean_abs_shap","mean"),
            median_abs_shap=("mean_abs_shap","median"),
            std_abs_shap=("mean_abs_shap","std"))
       .sort_values(["mean_abs_shap","raw_feature"],ascending=[False,True],kind="mergesort")
       .reset_index(drop=True))
    r["rank"]=np.arange(1,len(r)+1,dtype=int)
    sample=pd.DataFrame(sample_rows)
    t.to_csv(OUT/"shap_global_transformed.csv",index=False)
    r.to_csv(OUT/"shap_global_raw.csv",index=False)
    sample.to_csv(OUT/"shap_global_sample.csv",index=False)
    meta={
      "laboratory":"LABORATORY 06","phase":"C","status":"PASS",
      "method":"linear_shap_oos_operational_ensemble","shap_version":"0.53.0",
      "background_replicates":10,"oos_candidates_per_replicate":5800,
      "total_oos_explanations":total,"sample_per_replicate":100,
      "sample_rows_long":len(sample),"transformed_feature_count":len(t),
      "raw_feature_count":len(r),"maximum_additivity_error":max_err,
      "additivity_tolerance":TOL,"output_space":"log_odds",
      "oracle_loaded":False,"model_reselected":False,
      "causal_interpretation_allowed":False,
      "correlated_predictor_attribution_requires_caution":True,
    }
    (OUT/"global_shap_manifest.json").write_text(
        json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("GLOBAL_SHAP = PASS")
    print("SHAP_VERSION = 0.53.0")
    print(f"TOTAL_OOS_EXPLANATIONS = {total}")
    print(f"MAXIMUM_ADDITIVITY_ERROR = {max_err:.16g}")
    print("SHAP_OUTPUT_SPACE = LOG_ODDS")
    print("ORACLE_USED = NO")
    print("MODEL_RESELECTED = NO")
    print("LAB06_PHASE_C_SHAP = PASS")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
