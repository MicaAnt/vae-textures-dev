import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from commu_pipeline.harmony_trace import trace


class HarmonyTraceTests(unittest.TestCase):
    def _fixture(self, mismatch=False):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        csv_path = root / "meta.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "time_signature", "chord_progressions", "num_measures"])
            writer.writeheader()
            writer.writerow({"id": "commu00001", "time_signature": "4/4", "chord_progressions": "[['C', 'C', 'G', 'G', 'Am', 'Am', 'F', 'F']]", "num_measures": "1"})
        chord = np.zeros((4, 14))
        chord[:, 0] = [0, 7, 9, 5]
        if mismatch:
            chord[1, 0] = 1
        npz_path = root / "commu00001.npz"
        np.savez(npz_path, chord=chord)
        return tmp, csv_path, npz_path

    def test_trace_matches_fundamental_column(self):
        tmp, csv_path, npz_path = self._fixture()
        self.addCleanup(tmp.cleanup)
        data = trace(csv_path, npz_path, "commu00001")
        self.assertTrue(data["fundamental_column_matches"])
        self.assertEqual(data["fundamentals"], [0, 7, 9, 5])

    def test_trace_detects_mismatch(self):
        tmp, csv_path, npz_path = self._fixture(mismatch=True)
        self.addCleanup(tmp.cleanup)
        data = trace(csv_path, npz_path, "commu00001")
        self.assertFalse(data["fundamental_column_matches"])


if __name__ == "__main__":
    unittest.main()
