"""Synthetic, oracle-free regression tests for LAB07 Phase D."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "labs/lab07_spatial_uncertainty/07d_build_spatial_regimes.py"
SPEC = importlib.util.spec_from_file_location("lab07_phase_d", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("LAB07 Phase D module unavailable")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def synthetic_frame() -> pd.DataFrame:
    """A 3x3 lattice with a masked center and eight observed cells."""
    coordinates = [(x, y) for x in (0, 1000, 2000) for y in (0, 1000, 2000)]
    coordinates.remove((1000, 1000))
    n = len(coordinates)
    score_std = np.array([0.01 + 0.005 * i for i in range(n)])
    priority_std = np.array([0.02 + 0.004 * ((i * 3) % n) for i in range(n)])
    return pd.DataFrame(
        {
            "cell_id": np.arange(1, n + 1),
            "Easting": [point[0] for point in coordinates],
            "Northing": [point[1] for point in coordinates],
            "spatial_block_id": np.repeat("synthetic", n),
            "oos_model_count": [4, 5, 6, 7, 8, 9, 10, 10],
            "prospectivity_score_oos_std": score_std,
            "oos_priority_std": priority_std,
            "score_std_percentile": pd.Series(score_std).rank(pct=True),
            "oos_top_5pct_frequency": np.zeros(n),
        }
    )


def synthetic_frozen_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matching 6300-row source schemas to test join integrity in isolation."""
    n = module.N_CANDIDATES
    index = np.arange(n)
    support = np.concatenate(
        [np.full(amount, count) for count, amount in module.EXPECTED_SUPPORT.items()]
    )
    std = 0.01 + index * 1e-6
    rank_std = 100.0 + index * 1e-4
    shared = {
        "cell_id": index + 1,
        "Easting": 350500 + 1000 * (index % 80),
        "Northing": 7100500 + 1000 * (index // 80),
        "spatial_block_id": index % 64,
        "oos_model_count": support,
    }
    score = pd.DataFrame(
        {
            **shared,
            "prospectivity_score_oos_mean": 0.5,
            "prospectivity_score_oos_median": 0.5,
            "prospectivity_score_oos_std": std,
            "prospectivity_score_oos_p25": 0.4,
            "prospectivity_score_oos_p75": 0.6,
            "prospectivity_score_oos_iqr": 0.2,
            "oos_support_fraction": support / 10.0,
            "score_mean_median_abs_gap": 0.0,
            "score_std_percentile": pd.Series(std).rank(method="average", pct=True),
            "score_iqr_percentile": 0.5,
        }
    )[module.SCORE_COLUMNS]
    rank = pd.DataFrame(
        {
            **shared,
            "candidate_rank": index + 1,
            "oos_replicate_rank_mean": 500.0,
            "oos_replicate_rank_std": rank_std,
            "oos_replicate_rank_median": 500.0,
            "oos_replicate_rank_p25": 490.0,
            "oos_replicate_rank_p75": 510.0,
            "oos_replicate_rank_iqr": 20.0,
            "oos_priority_mean": 0.5,
            "oos_priority_median": 0.5,
            "oos_priority_std": rank_std / 5799.0,
            "oos_priority_iqr": 20.0 / 5799.0,
            "oos_top_1pct_frequency": 0.0,
            "oos_top_5pct_frequency": 0.0,
            "oos_top_10pct_frequency": 0.0,
        }
    )[module.RANK_COLUMNS]
    return score, rank


class TestTopology(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_frame()
        self.coordinates = self.frame[["Easting", "Northing"]].to_numpy(dtype=float)

    def test_masked_center_is_not_filled(self) -> None:
        rook, queen, audit = module.create_neighbors(
            self.coordinates, require_frozen_grid=False
        )
        self.assertEqual(audit["masked_positions"], 1)
        self.assertEqual(audit["rook_connected_component_sizes"], [8])
        self.assertEqual(len(rook), 8)
        self.assertEqual(len(queen), 8)

    def test_rook_and_queen_symmetric(self) -> None:
        rook, queen, _ = module.create_neighbors(
            self.coordinates, require_frozen_grid=False
        )
        for graph in (rook, queen):
            for origin, destinations in enumerate(graph):
                for destination in destinations:
                    self.assertIn(origin, graph[destination])
                    self.assertNotEqual(origin, destination)

    def test_rook_subset_queen(self) -> None:
        rook, queen, _ = module.create_neighbors(
            self.coordinates, require_frozen_grid=False
        )
        for four, eight in zip(rook, queen, strict=True):
            self.assertTrue(set(four).issubset(eight))

    def test_duplicate_coordinates_rejected(self) -> None:
        damaged = self.coordinates.copy()
        damaged[0] = damaged[1]
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            module.create_neighbors(damaged, require_frozen_grid=False)

    def test_irregular_coordinates_rejected(self) -> None:
        damaged = self.coordinates.copy()
        damaged[0, 0] = -777
        with self.assertRaisesRegex(RuntimeError, "spacing"):
            module.create_neighbors(damaged, require_frozen_grid=False)

    def test_nonfinite_coordinates_rejected(self) -> None:
        damaged = self.coordinates.copy()
        damaged[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            module.create_neighbors(damaged, require_frozen_grid=False)

    def test_frozen_grid_rejects_toy_data(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "candidate count"):
            module.create_neighbors(self.coordinates)


class TestRegimes(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_frame()
        self.rook, self.queen, _ = module.create_neighbors(
            self.frame[["Easting", "Northing"]].to_numpy(dtype=float),
            require_frozen_grid=False,
        )

    def test_repeated_build_is_deterministic(self) -> None:
        first, details1 = module.build_spatial_regimes(
            self.frame, self.rook, self.queen
        )
        second, details2 = module.build_spatial_regimes(
            self.frame, self.rook, self.queen
        )
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        self.assertEqual(details1, details2)

    def test_population_and_oracle_exclusion(self) -> None:
        result, _ = module.build_spatial_regimes(self.frame, self.rook, self.queen)
        self.assertEqual(len(result), 8)
        self.assertListEqual(result["cell_id"].tolist(), self.frame["cell_id"].tolist())
        self.assertFalse(module.ORACLE_FIELDS.intersection(result.columns))

    def test_local_mean_excludes_self(self) -> None:
        result, _ = module.build_spatial_regimes(self.frame, self.rook, self.queen)
        source = self.frame["prospectivity_score_oos_std"].to_numpy()
        for i, neighbors in enumerate(self.rook):
            self.assertAlmostEqual(
                result.loc[i, "rook_neighbor_score_std_mean"],
                float(np.mean(source[list(neighbors)])),
            )

    def test_cutoffs_nested(self) -> None:
        result, summary = module.build_spatial_regimes(
            self.frame, self.rook, self.queen
        )
        sizes = summary["cutoff_sensitivity_joint_high_counts"]
        self.assertGreaterEqual(sizes["q70"], sizes["q75"])
        self.assertGreaterEqual(sizes["q75"], sizes["q80"])
        self.assertTrue(
            result["regime_q75"]
            .isin(
                ("joint_high", "score_high_only", "priority_high_only", "neither_high")
            )
            .all()
        )

    def test_regime_components_retain_singletons(self) -> None:
        result, _ = module.build_spatial_regimes(self.frame, self.rook, self.queen)
        ids = result["rook_regime_q75_component_id"]
        sizes = result["rook_regime_q75_component_size"]
        self.assertTrue((ids > 0).all())
        self.assertTrue((sizes > 0).all())
        for component in ids.unique():
            cells = result.loc[ids == component]
            self.assertEqual(len(cells), cells["rook_regime_q75_component_size"].iat[0])
            self.assertEqual(cells["regime_q75"].nunique(), 1)

    def test_moran_finite_on_nonconstant_surface(self) -> None:
        _, summary = module.build_spatial_regimes(self.frame, self.rook, self.queen)
        self.assertTrue(
            all(np.isfinite(x) for x in summary["moran_i_descriptive"].values())
        )

    def test_constant_moran_undefined(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "constant surface"):
            module.moran_i_descriptive(np.ones(8), self.rook)


class TestIntegrity(unittest.TestCase):
    def test_full_candidate_join(self) -> None:
        score, rank = synthetic_frozen_tables()
        with tempfile.TemporaryDirectory() as directory:
            score_path = Path(directory) / "score.csv"
            rank_path = Path(directory) / "rank.csv"
            score.to_csv(score_path, index=False)
            rank.to_csv(rank_path, index=False)
            with (
                patch.object(module, "SCORE_CSV", score_path),
                patch.object(module, "RANK_CSV", rank_path),
            ):
                result = module.load_inputs()
            self.assertEqual(len(result), module.N_CANDIDATES)
            self.assertTrue(result["cell_id"].is_unique)

    def test_coordinate_drift_rejected(self) -> None:
        score, rank = synthetic_frozen_tables()
        rank.loc[0, "Easting"] += 1000
        with tempfile.TemporaryDirectory() as directory:
            score_path = Path(directory) / "score.csv"
            rank_path = Path(directory) / "rank.csv"
            score.to_csv(score_path, index=False)
            rank.to_csv(rank_path, index=False)
            with (
                patch.object(module, "SCORE_CSV", score_path),
                patch.object(module, "RANK_CSV", rank_path),
                self.assertRaisesRegex(RuntimeError, "mismatch: Easting"),
            ):
                module.load_inputs()

    def test_crlf_and_lf_share_normalized_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_bytes(b"a,b\n1,2\n")
            first = module.normalized_sha256(path)
            path.write_bytes(b"a,b\r\n1,2\r\n")
            self.assertEqual(first, module.normalized_sha256(path))

    def test_mixed_newlines_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_bytes(b"a,b\r\n1,2\n")
            with self.assertRaisesRegex(RuntimeError, "Mixed"):
                module.normalized_sha256(path)

    def test_phase_c_gate_is_finalized(self) -> None:
        manifest = json.loads(module.PHASE_C.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "PASS")
        self.assertTrue(manifest["ready_for_phase_d"])
        self.assertFalse(manifest["oracle_loaded"])
        self.assertFalse(manifest["oracle_used"])

    def test_new_manifest_cannot_claim_qa_pass(self) -> None:
        with patch.object(module, "normalized_sha256", return_value="test-digest"):
            manifest = module.build_manifest(
                synthetic_frame(), {"grid_axes": [3, 3]}, {"moran_i_descriptive": {}}
            )
        self.assertEqual(manifest["status"], "QA_PENDING")
        self.assertFalse(manifest["ready_for_phase_e"])
        self.assertEqual(manifest["gates"]["LAB07_PHASE_D"], "PENDING")
        self.assertFalse(manifest["oracle_loaded"])

    def test_exported_regimes_survive_csv_round_trip(self) -> None:
        exported = pd.read_csv(module.OUTPUT_CSV, float_precision="round_trip")
        self.assertEqual(len(exported), module.N_CANDIDATES)
        self.assertTrue(exported["cell_id"].is_unique)
        for label, threshold in module.CUTOFFS.items():
            expected = [
                module.regime(score >= threshold, priority >= threshold)
                for score, priority in zip(
                    exported["score_std_percentile"],
                    exported["priority_std_percentile"],
                    strict=True,
                )
            ]
            self.assertEqual(exported[f"regime_{label}"].tolist(), expected)

    def test_q70_boundary_cell_638_is_exact_after_csv_parse(self) -> None:
        exported = pd.read_csv(module.OUTPUT_CSV, float_precision="round_trip")
        boundary = exported.loc[exported["cell_id"] == 638]
        self.assertEqual(len(boundary), 1)
        row = boundary.iloc[0]
        self.assertEqual(
            format(row["priority_std_percentile"], ".17g"),
            "0.69999999999999996",
        )
        self.assertEqual(row["regime_q70"], "priority_high_only")


if __name__ == "__main__":
    unittest.main()
