import unittest

import pop909_conditioned_reconstruction as p


class LossTests(unittest.TestCase):
    def test_delta_sign_is_ours_minus_authors(self):
        authors = {name: 10.0 for name in p.LOSS_COMPONENTS}
        ours = {name: 7.0 for name in p.LOSS_COMPONENTS}
        deltas = p.compute_deltas(authors, ours)
        self.assertEqual(deltas["loss_signed"], -3.0)
        self.assertEqual(deltas["loss_abs"], 3.0)

    def test_loss_recon_conflict_detection(self):
        self.assertTrue(p.loss_recon_conflict(-1.0, 1.0))
        self.assertTrue(p.loss_recon_conflict(1.0, -1.0))
        self.assertFalse(p.loss_recon_conflict(-1.0, -0.5))
        self.assertFalse(p.loss_recon_conflict(0.0, 1.0))

    def test_csv_header_covers_all_loss_components(self):
        fields = p.csv_fieldnames()
        for name in p.LOSS_COMPONENTS:
            self.assertIn(f"authors_{name}", fields)
            self.assertIn(f"ours_{name}", fields)
            self.assertIn(f"delta_{name}", fields)
            self.assertIn(f"abs_delta_{name}", fields)

    def test_named_pairwise_delta_signs_are_candidate_minus_baseline(self):
        authors = {name: 10.0 for name in p.LOSS_COMPONENTS}
        epoch4 = {name: 8.0 for name in p.LOSS_COMPONENTS}
        epoch6 = {name: 11.0 for name in p.LOSS_COMPONENTS}
        self.assertEqual(p.compute_deltas(authors, epoch4)["loss_signed"], -2.0)
        self.assertEqual(p.compute_deltas(authors, epoch6)["loss_signed"], 1.0)
        self.assertEqual(p.compute_deltas(epoch4, epoch6)["loss_signed"], 3.0)


if __name__ == "__main__":
    unittest.main()
