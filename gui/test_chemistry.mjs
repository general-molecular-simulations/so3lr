import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {makePython, parseXYZ} from './web/chemistry.mjs';

const atoms = 'H 0 0 0\nF 0 0 0.917';
const molecule = parseXYZ('2\nHF\n' + atoms, 'HF');
const dynamics = readFileSync(new URL('./web/native_md.py', import.meta.url), 'utf8');

test('single-structure import rejects explicit coordinate-unit metadata', () => {
  for (const key of ['length_unit', 'coordinate_unit', 'units']) {
    assert.throws(() => parseXYZ('2\n' + key + '=bohr\n' + atoms, 'HF'), /plain XYZ/);
  }
});

test('MD scripts identify saved frames and include CUDA for automatic or GPU selection', () => {
  for (const device of ['auto', 'gpu', 'cpu']) {
    const script = makePython(molecule, 'so3lr-tauri-m', 'md', {device}, dynamics);
    assert.match(script, /frame\.info\.update\(task="md"/);
    assert.equal(script.includes('jax[cuda12]'), device !== 'cpu');
  }
});
