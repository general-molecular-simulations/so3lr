#!/usr/bin/env python
"""Shared validation utilities for SO3LR integration tests."""
import pathlib
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import yaml
from ase.io import read


class Validator:
    """Collects errors from the checks below; report() prints them and returns success."""

    def __init__(self, workdir: pathlib.Path = None):
        self.workdir = workdir or pathlib.Path('.')
        self.errors = []

    def check_file_exists(self, filepath: Union[str, pathlib.Path]) -> bool:
        exists = (self.workdir / filepath).exists()
        if not exists:
            self.errors.append(f"File not found: {filepath}")
        return exists

    def check_files_exist(self, filepaths: List[Union[str, pathlib.Path]]) -> bool:
        return all([self.check_file_exists(f) for f in filepaths])

    def check_log_contains(self, log_file: Union[str, pathlib.Path],
                           required_strings: List[str]) -> bool:
        """Case-insensitive check that every required string is in the log."""
        path = self.workdir / log_file
        if not path.exists():
            self.errors.append(f"Log file not found: {log_file}")
            return False
        content = path.read_text().lower()
        missing = [s for s in required_strings if s.lower() not in content]
        for s in missing:
            self.errors.append(f"Required string not found in {log_file}: '{s}'")
        return not missing

    def validate_hdf5(self, filepath: Union[str, pathlib.Path],
                      required_datasets: List[str],
                      check_shapes: Optional[Dict[str, tuple]] = None) -> bool:
        """Check that the datasets exist and, where given, have the expected shapes."""
        path = self.workdir / filepath
        if not path.exists():
            self.errors.append(f"HDF5 file not found: {filepath}")
            return False
        n_errors = len(self.errors)
        with h5py.File(path, 'r') as f:
            print(f"HDF5 datasets in {filepath}: {list(f.keys())}")
            for dataset in required_datasets:
                if dataset not in f:
                    self.errors.append(f"Required dataset '{dataset}' not found in {filepath}")
                elif check_shapes and check_shapes.get(dataset) not in (None, f[dataset].shape):
                    self.errors.append(
                        f"Dataset '{dataset}' has wrong shape: "
                        f"expected {check_shapes[dataset]}, got {f[dataset].shape}"
                    )
        return len(self.errors) == n_errors

    def validate_xyz(self, filepath: Union[str, pathlib.Path], expected_atoms: int) -> bool:
        path = self.workdir / filepath
        if not path.exists():
            self.errors.append(f"XYZ file not found: {filepath}")
            return False
        n_atoms = len(read(path))
        if n_atoms != expected_atoms:
            self.errors.append(f"Expected {expected_atoms} atoms in {filepath}, got {n_atoms}")
            return False
        return True

    def load_settings(self, settings_file: str = 'settings.yaml') -> dict:
        with open(self.workdir / settings_file) as f:
            return yaml.safe_load(f)

    def count_atoms(self, input_file: str) -> int:
        return len(read(self.workdir / input_file))

    def check_box_changed(self, trajectory: str = 'trajectory.hdf5',
                          tolerance: float = 1e-9) -> bool:
        """The barostat must actually move the cell between the first and last saved
        frames. Both are stored at the run's precision, so the comparison does
        not depend on it (unlike comparing with the input file's cell)."""
        path = self.workdir / trajectory
        if not path.exists():
            return False  # check_file_exists already reported it
        with h5py.File(path, 'r') as f:
            if 'box' not in f or len(f['box']) < 2:
                self.errors.append(f"Need a 'box' dataset with at least 2 frames in {trajectory}")
                return False
            first, last = np.asarray(f['box'][0]), np.asarray(f['box'][-1])
        if np.max(np.abs(last - first)) <= tolerance:
            self.errors.append(f"Box did not change: first frame {first}, last frame {last}")
            return False
        return True

    def check_settings_in_logs(self, settings: dict) -> bool:
        """
        Check that every explicitly set value appears verbatim in stderr.log and,
        if set, in settings['log_file'].
        """
        # Values the log transforms (units, Enabled/Disabled, NVT/NPT) or does not print.
        skip_keys = {
            'log_file',
            'md_dt',              # logged in fs with :.4g formatting
            'relax_before_run',   # logged as Enabled/Disabled
            'save_buffer',
            'observables',
            'min_start_dt',
            'min_max_dt',
            'min_n_min',
            'restart_load_path',
            'restart_save_path',
            'restart_reset',
            'ensemble',           # logged as NVT/NPT/etc
            'nhc_thermo',         # logged as nhc_thermo * dt_fs (fs)
            'nhc_baro',           # logged as nhc_baro * dt_fs (fs)
            'langevin_thermo',    # logged as langevin_thermo * dt_fs (fs)
            'kspace_electrostatics',
            'kspace_smearing',
            'kspace_spacing',
            'kspace_interp_nodes',
        }
        log_contents = {}
        for lf in ['stderr.log'] + ([settings['log_file']] if 'log_file' in settings else []):
            path = self.workdir / lf
            if path.exists():
                log_contents[lf] = path.read_text()
            else:
                self.errors.append(f"Log file not found: {lf}")
        if not log_contents:
            return False

        n_errors = len(self.errors)
        for key, value in settings.items():
            if key in skip_keys or value is None:
                continue
            for lf, content in log_contents.items():
                if str(value) not in content:
                    self.errors.append(f"Setting '{key}: {value}' not found in {lf}")
        return len(self.errors) == n_errors

    def report(self) -> bool:
        """Print the errors (to stdout) and return True if there were none."""
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  ERROR: {error}")
            print("\nValidation FAILED")
            return False
        print("\nValidation PASSED")
        return True
