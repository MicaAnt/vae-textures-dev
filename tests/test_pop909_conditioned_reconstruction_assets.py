import shutil
import tempfile
import unittest
from pathlib import Path

from pop909_conditioned_reconstruction_assets import _case_ids_from_selection, safe_case_dir_name, synthesize_wav
from pop909_conditioned_reconstruction_review import ReviewRun


class AssetTests(unittest.TestCase):
    def test_safe_case_dir_name(self):
        name = safe_case_dir_name('validation|seed=3345|file=0')
        self.assertNotIn('|', name)
        self.assertIn('validation', name)

    def test_audio_fallback_without_midi_dependency(self):
        result = synthesize_wav(Path('missing.mid'), Path('missing.wav'), soundfont='definitely-missing.sf2')
        self.assertIn(result['status'], {'fallback', 'ok'})
        if result['status'] == 'fallback':
            self.assertTrue(result['reason'])

    def test_case_ids_from_selection_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path('_artefatos/pop909-conditioned-reconstruction/smoke-local')
            target = Path(tmp) / 'smoke-local'
            shutil.copytree(source, target)
            run = ReviewRun.from_run_dir(target)
            cid = run.selected_case_ids(['near_tie'], max_cases=1)[0]
            manifest = {
                'selected_cases': [{'compound_id': cid}],
                'selection_manifest_path': str(run.selection_manifest_path()),
            }
            run.selection_manifest_path().write_text(__import__('json').dumps(manifest))
            ids, loaded = _case_ids_from_selection(run)
            self.assertEqual(ids, [cid])
            self.assertEqual(loaded['selected_cases'][0]['compound_id'], cid)


if __name__ == '__main__':
    unittest.main()
