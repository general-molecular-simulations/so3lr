#!/usr/bin/env python
"""Shared validation utilities for SO3LR integration tests."""
import pathlib
from typing import List, Optional, Union, Dict, Any


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class Validator:
    """Base validator class for integration tests."""

    def __init__(self, workdir: pathlib.Path = None):
        """Initialize validator with working directory."""
        self.workdir = workdir or pathlib.Path('.')
        self.errors = []
        self.warnings = []

    def check_file_exists(self, filepath: Union[str, pathlib.Path],
                          required: bool = True) -> bool:
        """
        Check if a file exists.

        Args:
            filepath: Path to file (relative to workdir or absolute)
            required: If True, add to errors; if False, add to warnings

        Returns:
            True if file exists, False otherwise
        """
        path = self.workdir / filepath if not pathlib.Path(filepath).is_absolute() else pathlib.Path(filepath)
        exists = path.exists()

        if not exists:
            msg = f"File not found: {filepath}"
            if required:
                self.errors.append(msg)
            else:
                self.warnings.append(msg)

        return exists

    def check_files_exist(self, filepaths: List[Union[str, pathlib.Path]],
                          required: bool = True) -> bool:
        """Check multiple files exist."""
        all_exist = True
        for filepath in filepaths:
            if not self.check_file_exists(filepath, required):
                all_exist = False
        return all_exist

    def check_log_contains(self, log_file: Union[str, pathlib.Path],
                          required_strings: List[str] = None,
                          forbidden_strings: List[str] = None,
                          case_sensitive: bool = False) -> bool:
        """
        Check that log file contains (or doesn't contain) specific strings.

        Args:
            log_file: Path to log file
            required_strings: Strings that must be present
            forbidden_strings: Strings that must not be present
            case_sensitive: Whether to perform case-sensitive search

        Returns:
            True if all checks pass, False otherwise
        """
        path = self.workdir / log_file
        if not path.exists():
            self.errors.append(f"Log file not found: {log_file}")
            return False

        content = path.read_text()
        if not case_sensitive:
            content = content.lower()

        all_passed = True

        # Check required strings
        if required_strings:
            for string in required_strings:
                search_string = string if case_sensitive else string.lower()
                if search_string not in content:
                    self.errors.append(f"Required string not found in {log_file}: '{string}'")
                    all_passed = False

        # Check forbidden strings
        if forbidden_strings:
            for string in forbidden_strings:
                search_string = string if case_sensitive else string.lower()
                if search_string in content:
                    self.errors.append(f"Forbidden string found in {log_file}: '{string}'")
                    all_passed = False

        return all_passed

    def validate_hdf5(self, filepath: Union[str, pathlib.Path],
                     required_datasets: List[str] = None,
                     optional_datasets: List[str] = None,
                     check_shapes: Dict[str, tuple] = None,
                     require_any_of: List[str] = None) -> bool:
        """
        Validate HDF5 file structure and content.

        Args:
            filepath: Path to HDF5 file
            required_datasets: Dataset names that must exist
            optional_datasets: Dataset names to check for (warnings only)
            check_shapes: Dict mapping dataset names to expected shapes (None for any)
            require_any_of: List of dataset names where at least one must exist

        Returns:
            True if validation passes, False otherwise
        """
        path = self.workdir / filepath
        if not path.exists():
            self.errors.append(f"HDF5 file not found: {filepath}")
            return False

        try:
            import h5py
        except ImportError:
            self.warnings.append("h5py not available, skipping HDF5 validation")
            return True

        try:
            with h5py.File(path, 'r') as f:
                datasets = list(f.keys())
                print(f"HDF5 datasets in {filepath}: {datasets}")

                all_passed = True

                # Check require_any_of datasets
                if require_any_of:
                    found = False
                    for dataset in require_any_of:
                        if dataset in f:
                            found = True
                            break
                    if not found:
                        self.errors.append(
                            f"None of the required datasets found in {filepath}: {require_any_of}"
                        )
                        all_passed = False

                # Check required datasets
                if required_datasets:
                    for dataset in required_datasets:
                        if dataset not in f:
                            self.errors.append(f"Required dataset '{dataset}' not found in {filepath}")
                            all_passed = False
                        elif check_shapes and dataset in check_shapes:
                            expected_shape = check_shapes[dataset]
                            actual_shape = f[dataset].shape
                            if expected_shape is not None and actual_shape != expected_shape:
                                self.errors.append(
                                    f"Dataset '{dataset}' has wrong shape: "
                                    f"expected {expected_shape}, got {actual_shape}"
                                )
                                all_passed = False

                # Check optional datasets
                if optional_datasets:
                    for dataset in optional_datasets:
                        if dataset not in f:
                            self.warnings.append(f"Optional dataset '{dataset}' not found in {filepath}")
                        else:
                            print(f"Found optional dataset: {dataset}")

                return all_passed

        except Exception as e:
            self.errors.append(f"Failed to validate HDF5 {filepath}: {e}")
            return False

    def validate_xyz(self, filepath: Union[str, pathlib.Path],
                    expected_atoms: Optional[int] = None,
                    expected_elements: Optional[List[str]] = None) -> bool:
        """
        Validate XYZ structure file.

        Args:
            filepath: Path to XYZ/ExtXYZ file
            expected_atoms: Expected number of atoms
            expected_elements: Expected element symbols

        Returns:
            True if validation passes, False otherwise
        """
        path = self.workdir / filepath
        if not path.exists():
            self.errors.append(f"XYZ file not found: {filepath}")
            return False

        try:
            from ase.io import read
        except ImportError:
            self.warnings.append("ASE not available, skipping XYZ validation")
            return True

        try:
            atoms = read(path)
            n_atoms = len(atoms)
            print(f"XYZ file {filepath} has {n_atoms} atoms")

            all_passed = True

            if expected_atoms is not None and n_atoms != expected_atoms:
                self.errors.append(
                    f"Expected {expected_atoms} atoms in {filepath}, got {n_atoms}"
                )
                all_passed = False

            if expected_elements is not None:
                actual_elements = atoms.get_chemical_symbols()
                if sorted(actual_elements) != sorted(expected_elements):
                    self.errors.append(
                        f"Element mismatch in {filepath}: "
                        f"expected {expected_elements}, got {actual_elements}"
                    )
                    all_passed = False

            return all_passed

        except Exception as e:
            self.errors.append(f"Failed to read XYZ {filepath}: {e}")
            return False

    def validate_npz(self, filepath: Union[str, pathlib.Path],
                    required_keys: List[str] = None) -> bool:
        """
        Validate NumPy .npz file.

        Args:
            filepath: Path to .npz file
            required_keys: Keys that must exist in the npz file

        Returns:
            True if validation passes, False otherwise
        """
        path = self.workdir / filepath
        if not path.exists():
            self.errors.append(f"NPZ file not found: {filepath}")
            return False

        try:
            import numpy as np
        except ImportError:
            self.warnings.append("NumPy not available, skipping NPZ validation")
            return True

        try:
            data = np.load(path)
            keys = list(data.keys())
            print(f"NPZ file {filepath} contains keys: {keys}")

            all_passed = True

            if required_keys:
                for key in required_keys:
                    if key not in data:
                        self.errors.append(f"Required key '{key}' not found in {filepath}")
                        all_passed = False

            return all_passed

        except Exception as e:
            self.errors.append(f"Failed to read NPZ {filepath}: {e}")
            return False

    def load_settings(self, settings_file: str = 'settings.yaml') -> dict:
        """Load and return a YAML settings file as a dict."""
        import yaml
        path = self.workdir / settings_file
        with open(path) as f:
            return yaml.safe_load(f)

    def count_atoms(self, input_file: str) -> int:
        """Count atoms in an input structure file using ASE."""
        from ase.io import read
        atoms = read(self.workdir / input_file)
        return len(atoms)

    def check_settings_in_logs(self, settings: dict,
                               skip_keys: Optional[List[str]] = None) -> bool:
        """
        Check that all explicitly defined settings values appear in the logs.

        Checks stderr.log always, and also checks the file given by
        settings['log_file'] if that key is present.

        Args:
            settings: Dict of settings (from load_settings).
            skip_keys: Setting keys to skip (e.g. transformed values).

        Returns:
            True if all checks pass, False otherwise.
        """
        # Keys whose values are transformed or not logged directly
        default_skip = {
            'log_file',
            'md_dt',              # logged as md_dt * 1000 (fs)
            'relax_before_run',   # logged as Enabled/Disabled
            'save_buffer',        # not logged
            'observables',        # list, not logged as-is
            'min_start_dt',       # not logged
            'min_max_dt',         # not logged
            'min_n_min',          # not logged
            'restart_load_path',  # not logged as-is
            'restart_save_path',  # not logged as-is
            'restart_reset',      # not logged as-is
            'ensemble',           # logged as NVT/NPT/etc, not as-is
            'nhc_thermo',         # logged as nhc_thermo * dt_fs (fs)
            'nhc_baro',           # logged as nhc_baro * dt_fs (fs)
            'langevin_thermo',      # logged as langevin_thermo * dt_fs (fs)
            'kspace_electrostatics',
            'kspace_smearing',
            'kspace_spacing',
            'kspace_interp_nodes',
        }
        transformed_keys = {
            'md_dt': lambda v: str(int(v * 1000)),  # Convert ps to fs
            'relax_before_run': lambda v: 'Enabled' if v else 'Disabled',
            'ensemble': lambda v: v.upper() if isinstance(v, str) else str(v),
            'nhc_thermo': lambda v: str(int(v * 1000)) if isinstance(v, (int, float)) else str(v),
            'nhc_baro': lambda v: str(int(v * 1000)) if isinstance(v, (int, float)) else str(v),
            'langevin_thermo': lambda v: str(int(v * 1000)) if isinstance(v, (int, float)) else str(v),
        }
        skip_keys = default_skip | set(skip_keys or [])

        # Determine which log files to check
        log_files = ['stderr.log']
        if 'log_file' in settings:
            log_files.append(settings['log_file'])

        # Read all log contents once
        log_contents = {}
        for lf in log_files:
            path = self.workdir / lf
            if not path.exists():
                self.errors.append(f"Log file not found: {lf}")
                continue
            log_contents[lf] = path.read_text()

        if not log_contents:
            return False

        all_passed = True
        for key, value in settings.items():
            if key in skip_keys or value is None:
                continue
            value_str = str(value)
            if key in transformed_keys:
                value_str = transformed_keys[key](value)
            for lf, content in log_contents.items():
                if value_str not in content:
                    self.errors.append(
                        f"Setting '{key}: {value_str}' not found in {lf}"
                    )
                    all_passed = False

        return all_passed

    def report(self) -> bool:
        """
        Print validation report and return overall success status.

        Returns:
            True if no errors, False if errors occurred
        """
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  WARNING: {warning}")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  ERROR: {error}")
            print("\nValidation FAILED")
            return False

        print("\nValidation PASSED")
        return True
