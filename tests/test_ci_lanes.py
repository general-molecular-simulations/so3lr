"""Every test file runs in a CI lane: the lanes in .github/workflows/CI.yml list their files
by name, so a new test file that is not added to one never runs in CI."""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CI = ROOT / '.github' / 'workflows' / 'CI.yml'


def test_every_test_file_is_in_a_ci_lane():
    if not CI.exists():
        pytest.skip('no CI workflow next to the tests (installed package)')
    targets = set(re.findall(r'\btests/[\w/.-]*', CI.read_text()))
    directories = {t.rstrip('/') + '/' for t in targets if not t.endswith('.py')}

    def covered(path):
        return path in targets or any(path.startswith(d) for d in directories)

    missing = [p for p in (str(f.relative_to(ROOT)) for f in sorted((ROOT / 'tests').rglob('test_*.py')))
               if not covered(p)]
    assert not missing, f'not run by any lane of .github/workflows/CI.yml: {missing}'
