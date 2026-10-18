"""LABORATORY 06 Phase C: spatial held-out permutation importance."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parents[2]
LAB05=ROOT/"labs"/"lab05_mineral_prospectivity"
OUT=ROOT/"labs"/"lab06_explainable_geoai"/"results"
GATE04=LAB05/"05d_run_spatial_benchmark.py"

def load_module(path,name):
    if str(LAB05) not in sys.path:
        sys.path.insert(0,str(LAB05))
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load LAB05 Gate 4.")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    g=load_module(GATE04,"lab05_gate04_lab06_c_perm")
    evidence=pd.read_csv(g.EVIDENCE_PATH)
    manifest=g.load_json(g.MANIFEST_PATH)
    features=list(g.feature_sets_from_manifest(manifest)["A"])
    if len(features)!=9:
        raise RuntimeError("Expected nine raw features.")
    rows=[]
    fits=0
    for rep in range(10):
        sampled=g.sample_background(evidence,rep)
        y=sampled[g.TARGET].to_numpy(dtype=int)
        splits=list(g.get_splits(sampled,"spatial",rep))
        if len(splits)!=5:
            raise RuntimeError("Expected five spatial folds.")
        for fold,(tr,te) in enumerate(splits,start=1):
            _,_,overlap=g.validate_fold(sampled,tr,te,"spatial")
            if overlap!=0:
                raise RuntimeError("Spatial group leakage.")
            seed=g.SEED+rep*1000+fold*10
            pipe=g.build_model_pipeline("logistic_regression",features,seed)
            pipe.fit(sampled.iloc[tr][features],y[tr])
            p=permutation_importance(
                pipe,sampled.iloc[te][features],y[te],
                scoring="average_precision",n_repeats=8,
                random_state=seed,n_jobs=1,
            )
            for j,feature in enumerate(features):
                for repeat,value in enumerate(p.importances[j]):
                    rows.append({
                        "replicate":rep,"fold":fold,"feature":feature,
                        "repeat":repeat,"importance_ap_drop":float(value),
                        "group_overlap":overlap,
                    })
            fits+=1
        print(f"PERMUTATION_REPLICATE_COMPLETE = {rep+1}/10")
    d=pd.DataFrame(rows)
    if len(d)!=3600 or int(d["group_overlap"].max())!=0:
        raise RuntimeError("Permutation artifact contract failed.")
    s=(d.groupby("feature",as_index=False)
       .agg(
          importance_mean=("importance_ap_drop","mean"),
          importance_median=("importance_ap_drop","median"),
          importance_std=("importance_ap_drop","std"),
          positive_fraction=("importance_ap_drop",lambda x:float((x>0).mean())),
       )
       .sort_values(["importance_mean","feature"],ascending=[False,True],kind="mergesort")
       .reset_index(drop=True))
    s["rank"]=np.arange(1,len(s)+1,dtype=int)
    d.to_csv(OUT/"permutation_importance_spatial.csv",index=False)
    s.to_csv(OUT/"permutation_importance_summary.csv",index=False)
    meta={
      "laboratory":"LABORATORY 06","phase":"C","status":"PASS",
      "method":"spatial_held_out_permutation_importance",
      "feature_set":"A","model":"logistic_regression",
      "scoring":"average_precision","observed_spatial_fits":fits,
      "permutation_repeats":8,"spatial_group_overlap_max":0,
      "oracle_loaded":False,"model_reselected":False,
      "correlated_predictor_caution":True,
    }
    (OUT/"permutation_importance_manifest.json").write_text(
        json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("PERMUTATION_IMPORTANCE = PASS")
    print("SPATIAL_FITS = 50")
    print("PERMUTATION_ROWS = 3600")
    print("SPATIAL_GROUP_OVERLAP_MAX = 0")
    print("ORACLE_USED = NO")
    print("MODEL_RESELECTED = NO")
    print("LAB06_PHASE_C_PERMUTATION = PASS")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
