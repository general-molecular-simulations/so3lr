"""Regression checks for trajectory assignments, cap handling, and rigid alignment."""

import json
from pathlib import Path
import unittest

import numpy as np

from peptide_analysis import aligned_heavy_rmsd, analyze_trajectory, backbone


class TrajectoryAnalysisTest(unittest.TestCase):
    def test_fractions_change_over_time_and_count_all_residues(self):
        dist = Path(__file__).parent / "web"
        sample = json.loads((Path(__file__).parent / "test_data" / "ala15-10ps.json").read_text())
        structures = json.loads((dist / "molecules.json").read_text())
        elements = next(item["elements"] for item in structures if item["id"] == sample["id"])
        result = sample["result"]
        # The folded peptide at 0, 9.65 and 10 ps has 13, 8 and 12 helical residues.
        frames = [result["frames"][result["times_fs"].index(t)] for t in (0, 9650, 10000)]
        peptide = analyze_trajectory(elements, frames)["peptide"]
        self.assertEqual(peptide["residue_count"], 15)
        np.testing.assert_allclose(peptide["fractions"]["H"], np.array([13, 8, 12]) / 15)
        np.testing.assert_allclose(peptide["fractions"]["T"], np.array([0, 4, 1]) / 15)
        np.testing.assert_allclose(peptide["fractions"]["C"], np.array([2, 3, 2]) / 15)
        np.testing.assert_allclose(sum(map(np.array, peptide["fractions"].values())), 1)

    def test_ala4_and_multichain_inputs_only_have_general_analysis(self):
        structures = json.loads((Path(__file__).parent / "web" / "molecules.json").read_text())
        folded = next(item for item in structures if item["id"] == "ala15_folded")
        coil = next(item for item in structures if item["id"] == "ala4")
        positions = np.concatenate([folded["coordinates"], np.asarray(coil["coordinates"]) + 100.])
        peptide = analyze_trajectory(folded["elements"] + coil["elements"], [positions])["peptide"]
        self.assertIsNone(peptide)
        self.assertIsNone(analyze_trajectory(coil["elements"], [coil["coordinates"]])["peptide"])

    def test_fifteen_residues_must_all_be_alanine(self):
        library = json.loads((Path(__file__).parent / "web" / "molecules.json").read_text())
        ala15 = next(item for item in library if item["id"] == "ala15_folded")
        elements = list(ala15["elements"])
        xyz = np.array(ala15["coordinates"])
        chains, heavy = backbone(elements, xyz)
        n, ca, c, _ = chains[0][7]
        beta, = heavy[ca] - {n, c}
        # Extend the methyl side chain into serine while preserving the 15-residue backbone.
        h = min((i for i, el in enumerate(elements) if el == "H"),
                key=lambda i: np.linalg.norm(xyz[i] - xyz[beta]))
        direction = xyz[h] - xyz[beta]
        xyz[h] = xyz[beta] + 1.42 * direction / np.linalg.norm(direction)
        elements[h] = "O"
        self.assertEqual(len(backbone(elements, xyz)[0][0]), 15)
        self.assertIsNone(analyze_trajectory(elements, [xyz])["peptide"])

    def test_alignment_removes_translation_rotation_and_hydrogen_motion(self):
        ref = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1.]])
        rotated = ref[:, [1, 0, 2]] * [-1, 1, 1] + [2, -3, 4]
        rotated[-1] += 10
        mirrored = ref * [-1, 1, 1]
        rmsd, count = aligned_heavy_rmsd(["C", "O", "N", "F", "H"], np.array([ref, rotated, mirrored]))
        self.assertEqual(count, 4)
        self.assertEqual(rmsd[:2], [0, 0])
        self.assertGreater(rmsd[2], 0.4)

    def test_nonpeptide_has_no_assignments(self):
        result = analyze_trajectory(["O", "H", "H"], [[[0, 0, 0], [0.96, 0, 0], [-0.25, 0.93, 0]]])
        self.assertIsNone(result["peptide"])
        self.assertEqual(result["heavy_atom_count"], 1)
        self.assertEqual(result["rmsd"], [0])


if __name__ == "__main__":
    unittest.main()
