import json
import tempfile
import unittest
from pathlib import Path

import pop909_conditioned_reconstruction as p


class OutputTests(unittest.TestCase):
    def _parsed(self):
        cfg = {
            "run_id": "test",
            "run_role": "dry_run",
            "split": "validation",
            "sample_count": 2,
            "selection_seed": 3345,
            "ordering_rule": "fixed",
            "fallback_used": True,
            "fallback_reason": "test subset",
            "checkpoints": {
                "authors": {"role": "authors_reference", "path": "a.pt", "provenance_note": "authors"},
                "ours": {"role": "ours_dry_run", "path": "o.pt", "provenance_note": "ours"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg))
            return p.parse_config(path)

    def _row(self, cfg, idx, delta):
        authors = {name: 10.0 for name in p.LOSS_COMPONENTS}
        ours = {name: 10.0 + delta for name in p.LOSS_COMPONENTS}
        ident = p.build_segment_identity("validation", idx, idx, cfg, npz_path=f"{idx:03d}.npz", sorted_file_index=idx)
        return p.make_row(cfg, ident, authors, ours)

    def test_manifest_record_validates(self):
        cfg = self._parsed()
        row = self._row(cfg, 1, -0.5)
        row["stratum"] = "near_tie"
        rec = p.manifest_record(cfg, row)
        p.validate_manifest_record(rec)
        self.assertEqual(rec["deltas"]["loss_signed"], -0.5)

    def test_rank_rows_orders_strata(self):
        cfg = self._parsed()
        rows = [self._row(cfg, i, delta) for i, delta in enumerate([0.1, 5.0, -4.0, 0.01, 1.0, -1.0])]
        strata = p.rank_rows(rows, per_stratum=2)
        self.assertEqual(strata["near_tie"][0]["compound_id"], rows[3]["compound_id"])
        self.assertEqual(strata["authors_much_better"][0]["compound_id"], rows[1]["compound_id"])
        self.assertEqual(strata["ours_much_better"][0]["compound_id"], rows[2]["compound_id"])

    def test_summary_marks_dry_run_non_final(self):
        cfg = self._parsed()
        rows = [self._row(cfg, 0, 0.1)]
        strata = p.rank_rows(rows, per_stratum=1)
        summary = p.build_summary(cfg, rows, {"csv": "x.csv"}, strata)
        self.assertTrue(summary["non_final_warning"])
        self.assertIn("Phase 9", summary["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
