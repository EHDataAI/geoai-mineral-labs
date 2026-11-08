"""LAB07-D: descriptive regimes of conditional out-of-sample uncertainty.

This stage never reads oracle artifacts, refits models or treats rank/score
outputs as calibrated mineral-deposit probabilities.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "labs/lab07_spatial_uncertainty/results"
PHASE_A = RESULTS / "phase_a_audit.json"
PHASE_B = RESULTS / "phase_b_manifest.json"
PHASE_C = RESULTS / "phase_c_manifest.json"
SCORE_CSV = RESULTS / "score_uncertainty.csv"
RANK_CSV = RESULTS / "rank_stability.csv"
OUTPUT_CSV = RESULTS / "spatial_uncertainty_regimes.csv"
OUTPUT_MANIFEST = RESULTS / "phase_d_manifest.json"

N_CANDIDATES = 6300
GRID_STEP = 1000
GRID_AXIS_COUNT = 80
MASKED_POSITIONS = 100
CUTOFFS = {"q70": 0.70, "q75": 0.75, "q80": 0.80}
ORACLE_FIELDS = frozenset(
    {
        "true_deposit",
        "latent_favourability",
        "true_deposit_probability",
        "distance_to_known_deposit",
        "deposit_id",
        "known_positive",
    }
)

SCORE_COLUMNS = [
    "cell_id",
    "Easting",
    "Northing",
    "spatial_block_id",
    "oos_model_count",
    "prospectivity_score_oos_mean",
    "prospectivity_score_oos_median",
    "prospectivity_score_oos_std",
    "prospectivity_score_oos_p25",
    "prospectivity_score_oos_p75",
    "prospectivity_score_oos_iqr",
    "oos_support_fraction",
    "score_mean_median_abs_gap",
    "score_std_percentile",
    "score_iqr_percentile",
]
RANK_COLUMNS = [
    "cell_id",
    "Easting",
    "Northing",
    "spatial_block_id",
    "oos_model_count",
    "candidate_rank",
    "oos_replicate_rank_mean",
    "oos_replicate_rank_std",
    "oos_replicate_rank_median",
    "oos_replicate_rank_p25",
    "oos_replicate_rank_p75",
    "oos_replicate_rank_iqr",
    "oos_priority_mean",
    "oos_priority_median",
    "oos_priority_std",
    "oos_priority_iqr",
    "oos_top_1pct_frequency",
    "oos_top_5pct_frequency",
    "oos_top_10pct_frequency",
]
ROOK_OFFSETS = ((GRID_STEP, 0), (-GRID_STEP, 0), (0, GRID_STEP), (0, -GRID_STEP))
QUEEN_OFFSETS = tuple(
    (dx, dy)
    for dx in (-GRID_STEP, 0, GRID_STEP)
    for dy in (-GRID_STEP, 0, GRID_STEP)
    if dx or dy
)
EXPECTED_ROOK_DEGREES = {1: 1, 2: 30, 3: 637, 4: 5632}
EXPECTED_QUEEN_DEGREES = {3: 4, 4: 9, 5: 316, 6: 75, 7: 558, 8: 5338}
EXPECTED_SUPPORT = {4: 1, 5: 4, 6: 32, 7: 204, 8: 968, 9: 2298, 10: 2793}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(f"LAB07 Phase D FAIL: {reason}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_sha256(path: Path) -> str:
    """Check canonical LF hash, accepting consistent CRLF but not mixed endings."""
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    require(b"\r" not in lf, f"Unsupported CR: {path.name}")
    require(
        not (b"\r\n" in raw and b"\n" in raw.replace(b"\r\n", b"")),
        f"Mixed line endings: {path.name}",
    )
    return hashlib.sha256(lf).hexdigest()


def checksum_variants(path: Path) -> set[str]:
    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n")
    normalized_sha256(path)
    return {
        hashlib.sha256(payload).hexdigest()
        for payload in (raw, canonical, canonical.replace(b"\n", b"\r\n"))
    }


def validate_upstream() -> None:
    """Refuse drift from finalized upstream without reading any oracle data."""
    a, b, c = (read_json(path) for path in (PHASE_A, PHASE_B, PHASE_C))
    require(a.get("status") == "PASS", "Phase A status")
    require(a.get("ready_for_phase_b") is True, "Phase A authorization")
    require(a.get("gates", {}).get("LAB07_PHASE_A") == "READY", "Phase A gate")
    require(b.get("status") == "PASS", "Phase B status")
    require(b.get("ready_for_phase_c") is True, "Phase B authorization")
    require(c.get("status") == "PASS", "Phase C status")
    require(c.get("ready_for_phase_d") is True, "Phase C authorization")
    require(c.get("gates", {}).get("LAB07_PHASE_C") == "PASS", "Phase C gate")
    require(
        all(value == "PASS" for value in c.get("gates", {}).values()), "Phase C gates"
    )
    qa = c.get("qa", {})
    require(qa.get("status") == "PASS", "Phase C QA")
    require(qa.get("targeted_tests", 0) >= 17, "Phase C targeted QA")
    require(qa.get("full_regression_tests", 0) >= 217, "Phase C full QA")
    for stage, obj in (("B", b), ("C", c)):
        require(
            obj.get("candidate_rows") == N_CANDIDATES, f"Phase {stage} candidate count"
        )
        require(obj.get("background_replicates") == 10, f"Phase {stage} replicates")
        require(
            obj.get("score_semantics")
            == "prospectivity_score_not_calibrated_deposit_probability",
            f"Phase {stage} score meaning",
        )
        require(obj.get("oracle_loaded") is False, f"Phase {stage} oracle loaded")
        require(obj.get("oracle_used") is False, f"Phase {stage} oracle used")
    require(a["oracle_boundary"].get("oracle_loaded") is False, "Phase A oracle loaded")
    require(a["oracle_boundary"].get("oracle_used") is False, "Phase A oracle used")
    require(c.get("oos_candidates_per_replicate") == 5800, "Phase C OOS size")
    require(c.get("rank_scale") == "replicate_oos_1_to_5800", "Phase C rank scale")
    require(
        c.get("frequency_denominator") == "cell_specific_oos_model_count",
        "Phase C top-k denominator",
    )
    require(
        b["source_sha256"]["phase_a_manifest"] in checksum_variants(PHASE_A),
        "Phase A manifest SHA drift",
    )
    for key, path in (("phase_a_manifest", PHASE_A), ("phase_b_manifest", PHASE_B)):
        require(
            c["source_sha256_lf"][key] == normalized_sha256(path),
            f"Phase C upstream SHA drift: {key}",
        )
    require(
        b["output_sha256"]["score_uncertainty_csv"] in checksum_variants(SCORE_CSV),
        "Phase B CSV SHA drift",
    )
    require(
        c["output_sha256_lf"]["rank_stability_csv"] == normalized_sha256(RANK_CSV),
        "Phase C CSV SHA drift",
    )


def load_inputs() -> pd.DataFrame:
    """Enforce B/C schemas, exact spatial identity and support accounting."""
    score = pd.read_csv(SCORE_CSV)
    rank = pd.read_csv(RANK_CSV)
    for name, frame, expected in (
        ("B", score, SCORE_COLUMNS),
        ("C", rank, RANK_COLUMNS),
    ):
        require(frame.columns.tolist() == expected, f"Phase {name} column contract")
        require(len(frame) == N_CANDIDATES, f"Phase {name} candidate count")
        require(frame["cell_id"].notna().all(), f"Phase {name} missing cell IDs")
        require(frame["cell_id"].is_unique, f"Phase {name} duplicate cell IDs")
        require(
            not ORACLE_FIELDS.intersection(frame.columns), f"Phase {name} oracle field"
        )

    result = score.merge(
        rank,
        on="cell_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_score", "_rank"),
    )
    require(len(result) == N_CANDIDATES, "Phase B/C cell identity")
    for key in ("Easting", "Northing", "spatial_block_id", "oos_model_count"):
        require(
            result[f"{key}_score"].equals(result[f"{key}_rank"]),
            f"Phase B/C mismatch: {key}",
        )
        result[key] = result.pop(f"{key}_score")
        result = result.drop(columns=f"{key}_rank")

    require(
        result["oos_model_count"].value_counts().sort_index().to_dict()
        == EXPECTED_SUPPORT,
        "Frozen OOS support histogram",
    )
    require(result["spatial_block_id"].nunique() == 64, "Spatial block count")
    for key in (
        "prospectivity_score_oos_std",
        "score_std_percentile",
        "oos_priority_std",
        "oos_replicate_rank_std",
        "oos_support_fraction",
        "oos_top_5pct_frequency",
    ):
        require(
            np.isfinite(result[key].to_numpy(dtype=float)).all(), f"Nonfinite: {key}"
        )
    require((result["prospectivity_score_oos_std"] >= 0).all(), "Negative score STD")
    require((result["oos_priority_std"] >= 0).all(), "Negative priority STD")
    require(
        result["score_std_percentile"].between(0, 1).all(), "Score percentile range"
    )
    require(
        result["oos_top_5pct_frequency"].between(0, 1).all(), "Top5 frequency range"
    )
    require(
        np.allclose(
            result["oos_support_fraction"],
            result["oos_model_count"] / 10.0,
            rtol=0,
            atol=1e-13,
        ),
        "OOS support fraction drift",
    )
    require(
        np.allclose(
            result["score_std_percentile"],
            result["prospectivity_score_oos_std"].rank(method="average", pct=True),
            rtol=0,
            atol=1e-13,
        ),
        "Phase B score percentile drift",
    )
    require(
        np.allclose(
            result["oos_priority_std"],
            result["oos_replicate_rank_std"] / 5799.0,
            rtol=0,
            atol=1e-13,
        ),
        "Phase C priority scale drift",
    )
    return result.sort_values("cell_id", kind="mergesort").reset_index(drop=True)


def create_neighbors(
    coordinates: np.ndarray, *, require_frozen_grid: bool = True
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], dict]:
    """Derive symmetric rook/queen graphs from observed cells, without imputation."""
    require(np.isfinite(coordinates).all(), "Nonfinite coordinates")
    require(np.array_equal(coordinates, np.rint(coordinates)), "Noninteger coordinates")
    points = [tuple(map(int, row)) for row in coordinates]
    lookup = {point: i for i, point in enumerate(points)}
    require(len(lookup) == len(points), "Duplicate coordinates")
    xs = sorted({point[0] for point in points})
    ys = sorted({point[1] for point in points})
    require(len(xs) >= 2 and len(ys) >= 2, "Degenerate coordinates")
    require(all(np.diff(xs) == GRID_STEP), "Easting grid spacing")
    require(all(np.diff(ys) == GRID_STEP), "Northing grid spacing")
    missing = len(xs) * len(ys) - len(points)
    if require_frozen_grid:
        require(len(points) == N_CANDIDATES, "Frozen candidate count")
        require((len(xs), len(ys)) == (GRID_AXIS_COUNT, GRID_AXIS_COUNT), "Grid axes")
        require(missing == MASKED_POSITIONS, "Masked positions")
        require((xs[0], xs[-1]) == (350500, 429500), "Easting bounds")
        require((ys[0], ys[-1]) == (7100500, 7179500), "Northing bounds")

    def adjacency(offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, ...]]:
        return [
            tuple(
                lookup[(x + dx, y + dy)]
                for dx, dy in offsets
                if (x + dx, y + dy) in lookup
            )
            for x, y in points
        ]

    rook = adjacency(ROOK_OFFSETS)
    queen = adjacency(QUEEN_OFFSETS)
    require(all(rook) and all(queen), "Isolated cell")
    rook_degree = dict(sorted(Counter(map(len, rook)).items()))
    queen_degree = dict(sorted(Counter(map(len, queen)).items()))
    if require_frozen_grid:
        require(rook_degree == EXPECTED_ROOK_DEGREES, "Rook topology drift")
        require(queen_degree == EXPECTED_QUEEN_DEGREES, "Queen topology drift")

    visited: set[int] = set()
    sizes = []
    for start in range(len(points)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in rook[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    if require_frozen_grid:
        require(sizes == [N_CANDIDATES], "Disconnected frozen grid")
    return (
        rook,
        queen,
        {
            "grid_axes": [len(xs), len(ys)],
            "grid_step_coordinate_units": GRID_STEP,
            "observed_candidates": len(points),
            "masked_positions": missing,
            "rook_degree_histogram": rook_degree,
            "queen_degree_histogram": queen_degree,
            "rook_connected_component_sizes": sorted(sizes, reverse=True),
        },
    )


def moran_i_descriptive(values: np.ndarray, neighbors: list[tuple[int, ...]]) -> float:
    """Global Moran I with row-standardized weights; no p-value computed."""
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    require(denominator > 0, "Undefined Moran I for a constant surface")
    spatial_lag = np.array([np.mean(centered[list(items)]) for items in neighbors])
    return float(np.dot(centered, spatial_lag) / denominator)


def regime(score_high: bool, priority_high: bool) -> str:
    if score_high and priority_high:
        return "joint_high"
    if score_high:
        return "score_high_only"
    if priority_high:
        return "priority_high_only"
    return "neither_high"


def connected_regimes(
    labels: list[str], rook: list[tuple[int, ...]]
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Label all rook-connected same-regime components, retaining singletons."""
    visited: set[int] = set()
    identifiers = np.zeros(len(labels), dtype=int)
    sizes = np.zeros(len(labels), dtype=int)
    counts: Counter[str] = Counter()
    component_id = 0
    for start, label in enumerate(labels):
        if start in visited:
            continue
        component_id += 1
        members = []
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in rook[node]:
                if neighbor not in visited and labels[neighbor] == label:
                    visited.add(neighbor)
                    stack.append(neighbor)
        for node in members:
            identifiers[node] = component_id
            sizes[node] = len(members)
        counts[label] += 1
    return identifiers, sizes, dict(sorted(counts.items()))


def build_spatial_regimes(
    frame: pd.DataFrame,
    rook: list[tuple[int, ...]],
    queen: list[tuple[int, ...]],
) -> tuple[pd.DataFrame, dict]:
    """Local dispersion contrasts and relative two-dimensional regime labels."""
    require(len(frame) == len(rook) == len(queen), "Neighborhood alignment")
    score = frame["prospectivity_score_oos_std"].to_numpy(dtype=float)
    priority = frame["oos_priority_std"].to_numpy(dtype=float)
    score_pct = frame["score_std_percentile"].to_numpy(dtype=float)
    priority_pct = pd.Series(priority).rank(method="average", pct=True).to_numpy()

    def lag(values: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(values[list(items)])) for items in rook])

    score_lag = lag(score)
    priority_lag = lag(priority)
    output = frame[
        ["cell_id", "Easting", "Northing", "spatial_block_id", "oos_model_count"]
    ].copy()
    output["prospectivity_score_oos_std"] = score
    output["oos_priority_std"] = priority
    output["score_std_percentile"] = score_pct
    output["priority_std_percentile"] = priority_pct
    output["rook_neighbor_count"] = [len(items) for items in rook]
    output["queen_neighbor_count"] = [len(items) for items in queen]
    output["rook_neighbor_score_std_mean"] = score_lag
    output["rook_neighbor_priority_std_mean"] = priority_lag
    output["rook_score_std_abs_contrast"] = np.abs(score - score_lag)
    output["rook_priority_std_abs_contrast"] = np.abs(priority - priority_lag)
    output["oos_top_5pct_frequency"] = frame["oos_top_5pct_frequency"].to_numpy()

    sensitivity = {}
    for label, cutoff in CUTOFFS.items():
        categories = [
            regime(bool(x >= cutoff), bool(y >= cutoff))
            for x, y in zip(score_pct, priority_pct, strict=True)
        ]
        output[f"regime_{label}"] = categories
        sensitivity[label] = sum(value == "joint_high" for value in categories)
    components, sizes, counts = connected_regimes(output["regime_q75"].tolist(), rook)
    output["rook_regime_q75_component_id"] = components
    output["rook_regime_q75_component_size"] = sizes
    require(output.notna().all().all(), "Missing spatial output values")
    support = pd.crosstab(output["regime_q75"], output["oos_model_count"])
    summary = {
        "cutoff_sensitivity_joint_high_counts": sensitivity,
        "regime_q75_counts": dict(sorted(Counter(output["regime_q75"]).items())),
        "oos_support_by_regime_q75": {
            str(label): {
                str(key): int(value) for key, value in support.loc[label].items()
            }
            for label in sorted(support.index)
        },
        "rook_regime_q75_component_counts": counts,
        "moran_i_descriptive": {
            "score_std_rook": moran_i_descriptive(score, rook),
            "score_std_queen": moran_i_descriptive(score, queen),
            "priority_std_rook": moran_i_descriptive(priority, rook),
            "priority_std_queen": moran_i_descriptive(priority, queen),
        },
    }
    return output, summary


def build_manifest(result: pd.DataFrame, geometry: dict, summary: dict) -> dict:
    """Pending manifest cannot authorize Phase E before full external QA."""
    return {
        "schema_version": "1.0.0",
        "laboratory": "LABORATORY 07",
        "phase": "D",
        "status": "QA_PENDING",
        "ready_for_phase_e": False,
        "candidate_rows": len(result),
        "masked_positions": MASKED_POSITIONS,
        "background_replicates": 10,
        "score_semantics": "prospectivity_score_not_calibrated_deposit_probability",
        "scope": "descriptive_spatial_structure_of_conditional_oos_uncertainty",
        "oracle_loaded": False,
        "oracle_used": False,
        "spatial_weights": {
            "primary": "rook_cell_adjacency",
            "sensitivity": "queen_cell_adjacency",
            "weights": "row_standardized",
            "self_neighbors": False,
            "masked_cells_interpolated": False,
        },
        "regime_policy": {
            "type": "descriptive_joint_percentile_partition",
            "primary": "q75",
            "sensitivity": CUTOFFS,
            "dimensions": ["score_oos_std", "oos_priority_std"],
            "support_used_as_regime_weight": False,
            "component_minimum_size_filter": None,
        },
        "spatial_topology": geometry,
        "descriptive_results": summary,
        "interpretation_limitations": [
            "Regimes use sample-relative percentiles, not physical risk thresholds.",
            "Moran I values are descriptive; no permutation p-values are claimed.",
            "Neighborhood averages are not new model predictions.",
            "Neighbor counts vary at boundaries and masked cells remain missing.",
            "Overlapping OOS training replicates are not independent trials.",
            "OOS support is 4-10; no confidence intervals are claimed.",
            "Priorities and top-k frequencies are not deposit probabilities.",
        ],
        "source_sha256_lf": {
            "phase_d_builder_code": normalized_sha256(Path(__file__)),
            "phase_a_manifest": normalized_sha256(PHASE_A),
            "phase_b_manifest": normalized_sha256(PHASE_B),
            "phase_c_manifest": normalized_sha256(PHASE_C),
            "score_uncertainty_csv": normalized_sha256(SCORE_CSV),
            "rank_stability_csv": normalized_sha256(RANK_CSV),
        },
        "gates": {
            "UPSTREAM_CONTRACT": "PASS",
            "CANDIDATE_IDENTITY": "PASS",
            "MASKED_GRID_TOPOLOGY": "PASS",
            "NEIGHBORHOOD_ACCOUNTING": "PASS",
            "SCORE_SEMANTICS": "PASS",
            "ORACLE_ISOLATION": "PASS",
            "RUFF": "PENDING",
            "TARGETED_TESTS": "PENDING",
            "FULL_REGRESSION": "PENDING",
            "PRE_COMMIT": "PENDING",
            "GITLEAKS": "PENDING",
            "LAB07_PHASE_D": "PENDING",
        },
    }


def validate_existing_manifest(previous: dict, expected: dict) -> None:
    """Preserve the scientific contract and fail closed on release-state drift."""
    for key in (
        "schema_version",
        "laboratory",
        "phase",
        "candidate_rows",
        "masked_positions",
        "background_replicates",
        "score_semantics",
        "scope",
        "spatial_weights",
        "regime_policy",
        "spatial_topology",
        "descriptive_results",
        "interpretation_limitations",
        "oracle_loaded",
        "oracle_used",
        "source_sha256_lf",
        "output_sha256_lf",
    ):
        require(previous.get(key) == expected.get(key), f"Existing Phase D {key} drift")

    require(previous.get("ready_for_phase_e") is False, "Phase E must remain closed")
    require(
        previous.get("phase_e_authorization", "CLOSED") == "CLOSED",
        "Phase E authorization changed",
    )
    status = previous.get("status")
    require(status in {"QA_PENDING", "PASS"}, "Phase D status")
    if status == "QA_PENDING":
        require(previous.get("gates") == expected["gates"], "Pending QA gate drift")
        require(
            previous.get("qa", {}).get("status", "PENDING") == "PENDING",
            "Pending QA state drift",
        )
    else:
        qa = previous.get("qa", {})
        require(qa.get("status") == "PASS", "Released QA status missing")
        require(qa.get("targeted_tests", 0) >= 25, "Released targeted QA missing")
        require(qa.get("full_regression_tests", 0) >= 242, "Released full QA missing")
        require(
            qa.get("independent_scientific_audit") == "PASS",
            "Released scientific audit missing",
        )
        require(
            set(previous.get("gates", {})) == set(expected["gates"])
            and all(value == "PASS" for value in previous["gates"].values()),
            "Released QA gates incomplete",
        )


def main() -> None:
    validate_upstream()
    frame = load_inputs()
    coordinates = frame[["Easting", "Northing"]].to_numpy(dtype=float)
    rook, queen, geometry = create_neighbors(coordinates)
    result, summary = build_spatial_regimes(frame, rook, queen)
    manifest = build_manifest(result, geometry, summary)
    csv_payload = result.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    if OUTPUT_CSV.exists():
        require(OUTPUT_CSV.read_bytes() == csv_payload, "Existing Phase D CSV drift")
    else:
        with OUTPUT_CSV.open("xb") as handle:
            handle.write(csv_payload)
    manifest["output_sha256_lf"] = {
        "spatial_uncertainty_regimes_csv": normalized_sha256(OUTPUT_CSV)
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if OUTPUT_MANIFEST.exists():
        previous = read_json(OUTPUT_MANIFEST)
        validate_existing_manifest(previous, manifest)
    else:
        with OUTPUT_MANIFEST.open("xb") as handle:
            handle.write(manifest_payload)
    print(f"LAB07_PHASE_D_BUILD=PASS; QA=PENDING; CANDIDATES={len(result)}")
    print(f"PHASE_D_CSV={OUTPUT_CSV.relative_to(ROOT)}")
    print(f"PHASE_D_MANIFEST={OUTPUT_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
