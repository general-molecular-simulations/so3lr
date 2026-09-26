"""Small-file overwrite safeguards, without a multi-file publication system."""
import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from so3lr.cli import so3lr_eval as evaluation


def test_predictions_replace_instead_of_append(tmp_path, monkeypatch):
    destination = tmp_path / 'predictions.xyz'
    write(destination, Atoms('He', positions=[[0, 0, 0]]))
    monkeypatch.setattr(evaluation, 'jraph_to_ase_atoms', lambda atoms: atoms)
    frames = [Atoms('H', positions=[[0, 0, 0]]), Atoms('Li', positions=[[1, 0, 0]])]
    evaluation.save_to_file(frames, destination, len(frames), overwrite=True)
    assert [frame.get_chemical_formula() for frame in read(destination, ':')] == ['H', 'Li']


@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt])
@pytest.mark.parametrize('fail_after', [0, 1])
def test_failed_serialization_preserves_existing_predictions(tmp_path, monkeypatch, failure, fail_after):
    destination = tmp_path / 'predictions.xyz'
    destination.write_text('previous results')
    converted = []
    def broken_conversion(_):
        if len(converted) == fail_after:
            raise failure('interrupted serialization')
        converted.append(True)
        return Atoms('H', positions=[[0, 0, 0]])
    monkeypatch.setattr(evaluation, 'jraph_to_ase_atoms', broken_conversion)
    with pytest.raises(failure):
        evaluation.save_to_file([object(), object()], destination, 2, overwrite=True)
    assert destination.read_text() == 'previous results'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['predictions.xyz']


def test_empty_predictions_do_not_replace_existing_file(tmp_path):
    destination = tmp_path / 'predictions.xyz'
    destination.write_text('previous results')
    with pytest.raises(ValueError, match='empty'):
        evaluation.save_to_file([], destination, 0)
    assert destination.read_text() == 'previous results'


@pytest.mark.parametrize('kind', ['directory', 'symlink', 'dangling_symlink'])
def test_unsafe_destination_is_rejected(tmp_path, kind):
    destination = tmp_path / 'predictions.xyz'
    if kind == 'directory':
        destination.mkdir()
    else:
        target = tmp_path / 'target.xyz'
        if kind == 'symlink':
            target.write_text('must survive')
        destination.symlink_to(target)
    with pytest.raises(RuntimeError, match='not a regular file'):
        evaluation._check_output_destination(destination, overwrite=True)


def test_no_overwrite_is_enforced_at_final_creation(tmp_path, monkeypatch):
    destination = tmp_path / 'predictions.xyz'
    original_link = os.link
    def concurrent_creation(source, target):
        destination.write_text('another job')
        return original_link(source, target)
    monkeypatch.setattr(evaluation.os, 'link', concurrent_creation)
    with pytest.raises(FileExistsError):
        with evaluation._staged_output(destination, overwrite=False) as temporary:
            temporary.write_text('our results')
    assert destination.read_text() == 'another job'


def test_symlink_created_during_serialization_is_not_followed(tmp_path):
    destination = tmp_path / 'predictions.xyz'
    target = tmp_path / 'other.xyz'
    target.write_text('must survive')
    with pytest.raises(RuntimeError, match='not a regular file'):
        with evaluation._staged_output(destination, overwrite=True) as temporary:
            temporary.write_text('our results')
            destination.symlink_to(target)
    assert target.read_text() == 'must survive'
    assert destination.is_symlink()


def test_embedding_archive_retains_mode_and_is_pickle_free(tmp_path):
    destination = tmp_path / 'predictions_x.npz'
    destination.write_bytes(b'previous')
    destination.chmod(0o640)
    with evaluation._staged_output(destination, overwrite=True) as temporary:
        np.savez_compressed(temporary, embeddings=np.ones((2, 3)), n_atoms=np.array([2]))
    assert destination.stat().st_mode & 0o777 == 0o640
    with np.load(destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['n_atoms'], [2])
        assert archive['embeddings'].shape == (2, 3)


@pytest.mark.parametrize('save_predictions', [False, True])
def test_existing_embeddings_rejected_before_model_load(tmp_path, save_predictions):
    datafile = tmp_path / 'input.xyz'
    datafile.write_text('input is not read before output validation')
    save_to = tmp_path / 'predictions.xyz' if save_predictions else None
    base = save_to if save_to is not None else datafile
    embedding = base.with_name(base.stem + '_x.npz')
    embedding.write_bytes(b'previous embeddings')
    with pytest.raises(RuntimeError, match='already exists'):
        evaluation.evaluate_so3lr_on(datafile=str(datafile), save_to=save_to, targets='x')
    assert embedding.read_bytes() == b'previous embeddings'
