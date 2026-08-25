import json
import tempfile
import unittest
from pathlib import Path

import pop909_conditioned_reconstruction as p


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def _config(self):
        return {
            "run_role": "dry_run",
            "split": "validation",
            "sample_count": 2,
            "selection_seed": 3345,
            "ordering_rule": "fixed",
            "fallback_used": True,
            "fallback_reason": "test subset",
            "checkpoints": {
                "authors": {"role": "authors_reference", "path": "model_param/polydis-v1.pt", "provenance_note": "authors"},
                "ours": {"role": "ours_dry_run", "path": "fake.pt", "provenance_note": "ours"},
            },
        }

    def test_official_final_requires_accepted_final_role(self):
        cfg = self._config()
        cfg["run_role"] = "official_final"
        cfg["checkpoints"]["ours"]["role"] = "ours_dry_run"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaises(p.ConfigError):
                p.parse_config(path)

    def test_config_defaults_and_manifest_role(self):
        cfg = self._config()
        cfg["run_role"] = "smoke"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg))
            parsed = p.parse_config(path)
        self.assertEqual(parsed.manifest_run_role, "dry_run")
        self.assertEqual(parsed.loader["num_bar"], 2)
        self.assertTrue(parsed.split_policy["fallback_used"])

    def test_extract_model_state_from_training_payload(self):
        ref = p.CheckpointRef(role="ours_dry_run", path="x", provenance_note="x")
        payload = {"model_state_dict": {"a": 1}, "epoch": 1}
        self.assertEqual(p.extract_model_state(payload, ref), {"a": 1})

    def test_segment_identity_has_required_fields(self):
        cfg = self._config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg))
            parsed = p.parse_config(path)
        ident = p.build_segment_identity("validation", 3, 4, parsed, npz_path="001.npz", sorted_file_index=1)
        for key in ["compound_id", "npz_path", "sorted_file_index", "dataset_index", "loader_index", "shift", "num_bar", "loader_seed"]:
            self.assertIn(key, ident)

    def test_official_final_requires_epoch4_and_epoch6_candidates(self):
        cfg = self._config()
        cfg["run_role"] = "official_final"
        cfg["checkpoints"] = {
            "authors": {"role": "authors_reference", "path": "a.pt", "provenance_note": "authors"},
            "ours": {"role": "ours_official_final", "path": "o.pt", "provenance_note": "ours", "accepted": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaises(p.ConfigError):
                p.parse_config(path)

    def test_official_final_accepts_named_epoch_candidates(self):
        cfg = self._config()
        cfg["run_role"] = "official_final"
        cfg["checkpoints"] = {
            "authors": {"role": "authors_reference", "path": "a.pt", "provenance_note": "authors", "accepted": True},
            "ours_epoch4": {"role": "ours_best_validation_epoch4", "path": "e4.pt", "provenance_note": "epoch4", "accepted": True, "epoch": 4},
            "ours_epoch6": {"role": "ours_protocol_final_epoch6", "path": "e6.pt", "provenance_note": "epoch6", "accepted": True, "epoch": 6},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg))
            parsed = p.parse_config(path)
        self.assertEqual(parsed.checkpoints["ours_epoch4"].epoch, 4)
        self.assertEqual(parsed.checkpoints["ours_epoch6"].epoch, 6)

    def test_phase9_cluster_full_config_is_unbounded_cuda_evidence(self):
        parsed = p.parse_config(ROOT / "configs" / "pop909_conditioned_reconstruction_cluster_full_validation.json")
        meta = p.validate_config_files(parsed, check_files=False)
        self.assertEqual(parsed.run_role, "official_final")
        self.assertEqual(parsed.split, "validation")
        self.assertIsNone(parsed.sample_count)
        self.assertEqual(parsed.device, "cuda")
        self.assertTrue(parsed.split_policy["full_split_target"])
        self.assertFalse(parsed.split_policy["fallback_used"])
        self.assertEqual(set(parsed.checkpoints), {"authors", "ours_epoch4", "ours_epoch6"})
        self.assertEqual(parsed.checkpoints["ours_epoch4"].epoch, 4)
        self.assertEqual(parsed.checkpoints["ours_epoch6"].epoch, 6)
        self.assertIn("cluster-full-validation", meta["run_id"])

    def test_phase9_cluster_smoke_config_is_bounded_non_final_gate(self):
        parsed = p.parse_config(ROOT / "configs" / "pop909_conditioned_reconstruction_cluster_smoke.json")
        self.assertEqual(parsed.run_role, "official_final")
        self.assertEqual(parsed.sample_count, 2)
        self.assertEqual(parsed.device, "cuda")
        self.assertFalse(parsed.split_policy["full_split_target"])
        self.assertTrue(parsed.split_policy["fallback_used"])
        self.assertIn("smoke", parsed.run_id)

    def test_authors_decoder_pack_lengths_are_cpu_safe_for_cuda_smoke(self):
        decoder = (ROOT / "dl_modules" / "pnotree_decoder.py").read_text()
        self.assertIn("lengths.view(-1).cpu()", decoder)
        self.assertIn("predicted_lengths.view(-1).cpu()", decoder)


if __name__ == "__main__":
    unittest.main()
