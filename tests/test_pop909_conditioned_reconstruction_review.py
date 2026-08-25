import tempfile
import unittest
from pathlib import Path

from pop909_conditioned_reconstruction_review import ReviewError, ReviewRun, VALID_LABELS

SMOKE = Path('_artefatos/pop909-conditioned-reconstruction/smoke-local')


class ReviewRunTests(unittest.TestCase):
    def test_smoke_file_map_and_stats(self):
        run = ReviewRun.from_run_dir(SMOKE)
        fmap = run.file_map()
        self.assertIn('csv', fmap)
        self.assertIn('assets_dir', fmap)
        stats = run.global_stats()
        self.assertEqual(stats['row_count'], 2)
        self.assertIn('delta_loss', stats)
        self.assertTrue(stats['non_final_warning'])

    def test_case_selection(self):
        run = ReviewRun.from_run_dir(SMOKE)
        ids = run.selected_case_ids(['near_tie'], max_cases=1)
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids[0].startswith('validation|'))

    def test_fixed_selection_refuses_smoke_without_override(self):
        run = ReviewRun.from_run_dir(SMOKE)
        with self.assertRaises(ReviewError):
            run.select_fixed_review_cases(write=False)

    def test_fixed_selection_manifest_with_override(self):
        run = ReviewRun.from_run_dir(SMOKE)
        manifest = run.select_fixed_review_cases(diagnostic_override=True, write=False)
        self.assertLess(manifest['selection_count'], 24)
        self.assertEqual(manifest['target_count'], 24)
        self.assertIn('selected_cases', manifest)
        if manifest['selected_cases']:
            case = manifest['selected_cases'][0]
            self.assertIn('requested_stratum', case)
            self.assertIn('actual_source_stratum', case)
            self.assertIn('selection_reason', case)

    def test_notes_label_validation(self):
        run = ReviewRun.from_run_dir(SMOKE)
        cid = run.selected_case_ids(['near_tie'], max_cases=1)[0]
        before = len(run.notes())
        self.assertIn('conflict', VALID_LABELS)
        rec = run.write_note(cid, 'conflict', 'test note', reviewer='unit-test')
        self.assertEqual(rec['label'], 'conflict')
        self.assertGreaterEqual(len(run.notes()), before + 1)
        with self.assertRaises(ValueError):
            run.write_note(cid, 'automatic comparable')


if __name__ == '__main__':
    unittest.main()
