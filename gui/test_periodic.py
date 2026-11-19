import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
from ase.io import read

import serve
from web.periodic_job import run_periodic, validate_periodic_geometry, write_periodic_result


class PeriodicTest(unittest.TestCase):
    def setUp(self):
        self.cell = [[10., 0., 0.], [1., 9., 0.], [0., 0.5, 11.]]
        self.xyz = [[2., 2., 2.], [2.96, 2., 2.], [1.76, 2.93, 2.]]
        self.request = dict(elements=["O", "H", "H"], coordinates=self.xyz, charge=0, cell=self.cell,
                            pbc=[True]*3, task="relax", model="so3lr-tauri-m", precision="mixed",
                            steps=2, fmax=.05, device="cpu")

    def test_periodic_flags_and_images(self):
        cpu = jax.local_devices(backend="cpu")[0]
        with patch.object(serve, "choose_device", return_value=("cpu", cpu)):
            self.assertEqual(serve.validate(self.request)["cell"], self.cell)
            for invalid in [dict(cell=None), dict(cell=[[1,0,0]]*3), dict(pbc=[True,True,False]),
                            dict(charge=1), dict(task="md")]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    serve.validate({**self.request, **invalid})
        with self.assertRaisesRegex(ValueError,"coincide"):
            validate_periodic_geometry(self.cell, [[0.,0.,0.],[1.,9.,0.]])
        validate_periodic_geometry(self.cell, self.xyz)

    def test_replay_cells_and_export_are_real(self):
        fixtures = Path(__file__).parent / "test_data"
        s = json.loads((fixtures / "mil53-al.json").read_text())
        r = json.loads((fixtures / "mil53-al-relax.json").read_text())["result"]
        for k in ('cells','energies','frame_steps','volumes_A3'):
            self.assertEqual(len(r[k]),len(r['frames']))
        self.assertGreater(len(r['frames']),2)
        np.testing.assert_allclose(r['frames'][0],s['coordinates'],atol=1e-6)
        np.testing.assert_allclose(r['volumes_A3'],np.linalg.det(r['cells']))
        self.assertLess(r['energies'][-1],r['energies'][0])
        self.assertTrue(r['converged'])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "opt.extxyz"
            write_periodic_result(path,s["elements"],r,"so3lr-tauri-m","mixed")
            a=read(path,index=":")
            self.assertEqual(len(a),len(r['frames']))
            self.assertTrue(all(x.pbc.all() for x in a))
            self.assertTrue(all(x.info["task"] == "relax" for x in a))
            self.assertTrue(all(x.info["optimizer"] == r["periodic"]["optimizer"] for x in a))
            np.testing.assert_allclose(a[-1].cell,r['cells'][-1])
            np.testing.assert_allclose(a[-1].get_forces(),r['forces'],atol=1e-7)

    def test_real_periodic_energy_is_lattice_translation_invariant(self):
        c = jax.local_devices(backend="cpu")[0]
        with jax.default_device(c), jax.enable_x64(True):
            options=dict(model="so3lr-tauri-s", precision="mixed", task="energy")
            first=run_periodic(self.request["elements"],self.xyz,self.cell,**options)
            second_xyz=np.array(self.xyz)
            second_xyz[1]+=self.cell[1]
            second=run_periodic(self.request["elements"],second_xyz,self.cell,**options)
        np.testing.assert_allclose(first['energy'],second['energy'],atol=2e-5,rtol=0)
        np.testing.assert_allclose(first['forces'],second['forces'],atol=4e-5,rtol=0)
