"""Small-file overwrite safeguards, without a multi-file publication system."""
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


def test_directory_destination_is_rejected(tmp_path):
    destination = tmp_path / 'predictions.xyz'
    destination.mkdir()
    with pytest.raises(RuntimeError, match='not a regular file'):
        evaluation._check_output_destination(destination, overwrite=True)


@pytest.mark.parametrize('dangling', [False, True])
def test_symlinked_destination_writes_through_the_link(tmp_path, dangling):
    target = tmp_path / 'scratch' / 'predictions.xyz'
    target.parent.mkdir()
    if not dangling:
        target.write_text('previous results')
    destination = tmp_path / 'predictions.xyz'
    destination.symlink_to(target)
    with evaluation._staged_output(destination, overwrite=True) as temporary:
        temporary.write_text('our results')
    assert destination.is_symlink()
    assert target.read_text() == 'our results'


def test_no_overwrite_is_rechecked_before_publishing(tmp_path):
    destination = tmp_path / 'predictions.xyz'
    with pytest.raises(RuntimeError, match='already exists'):
        with evaluation._staged_output(destination, overwrite=False) as temporary:
            temporary.write_text('our results')
            destination.write_text('another job')
    assert destination.read_text() == 'another job'


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
