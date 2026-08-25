import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from commu_pipeline import audit_dataset, inventory
from commu_pipeline.forward_loss_probe import LOSS_KEYS


class CommuPipelineContractTests(unittest.TestCase):
    def test_inventory_contract_has_required_keys_and_bounded_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "COMMUDataset" / "npzFiles").mkdir(parents=True)
            for i in range(3):
                (repo / "COMMUDataset" / "npzFiles" / f"commu{i:05d}.npz").write_bytes(b"x")
            data = inventory.build_inventory(repo, limit=1)
            for key in ["sources", "artifact_directories", "classifications", "quarantine_candidates"]:
                self.assertIn(key, data)
            self.assertIn("generated artifact", data["classification_labels"])
            npz_dir = next(item for item in data["artifact_directories"] if item["path"] == "COMMUDataset/npzFiles")
            self.assertEqual(npz_dir["file_count"], 3)
            self.assertEqual(len(npz_dir["examples"]), 1)

    def test_audit_summary_schema_on_tiny_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "npz"
            p.mkdir()
            np.savez(p / "commu00001.npz", beat=np.zeros((2, 6), dtype=np.int32), chord=np.zeros((2, 14)), melody=np.empty((0, 8), dtype=np.int32), bridge=np.empty((0, 8), dtype=np.int32), piano=np.zeros((1, 8), dtype=np.int32), track_role="bass", time_signature="4/4", audio_key="c", chord_progressions="[['C','C','C','C']]", pitch_range="mid", num_measures="1", bpm="120", genre="x", inst="piano", sample_rhythm="standard")
            summary = audit_dataset.audit(p, max_files=None, sample_per_role=1)
            self.assertIn("role_counts", summary)
            self.assertIn("anomalies", summary)
            self.assertEqual(summary["role_counts"].get("bass"), 1)

    def test_forward_loss_probe_declares_required_loss_keys(self):
        for key in ["recon_loss", "chord_loss", "kl_loss", "final_loss"]:
            self.assertIn(key, LOSS_KEYS)


if __name__ == "__main__":
    unittest.main()
