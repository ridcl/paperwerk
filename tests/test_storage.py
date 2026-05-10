import hashlib
import os

import pytest

from pile.storage import LocalStorageBackend


@pytest.fixture
def backend(tmp_path):
    return LocalStorageBackend(str(tmp_path / "storage"))


def test_init_creates_base_path(tmp_path):
    base = tmp_path / "new_dir"
    assert not base.exists()
    LocalStorageBackend(str(base))
    assert base.is_dir()


def test_add_writes_file_and_returns_path(backend):
    content = b"hello world"
    path = backend.add("notes.txt", content)

    digest = hashlib.md5(content).hexdigest()[:6]
    assert path == f"{digest}_notes.txt"
    assert os.path.isfile(os.path.join(backend.base_path, path))


def test_add_strips_directory_from_filename(backend):
    path = backend.add("/some/dir/report.pdf", b"data")
    assert "/" not in path
    assert path.endswith("_report.pdf")


def test_add_is_deterministic_for_same_content(backend):
    p1 = backend.add("a.txt", b"same")
    p2 = backend.add("a.txt", b"same")
    assert p1 == p2
    entries = os.listdir(backend.base_path)
    assert entries == [p1]


def test_add_different_content_produces_different_paths(backend):
    p1 = backend.add("a.txt", b"foo")
    p2 = backend.add("a.txt", b"bar")
    assert p1 != p2


def test_get_returns_original_content(backend):
    content = b"\x00\x01binary\xffdata"
    path = backend.add("blob.bin", content)
    assert backend.get(path) == content


def test_get_missing_path_raises(backend):
    with pytest.raises(FileNotFoundError):
        backend.get("does_not_exist.txt")


def test_remove_deletes_file(backend):
    path = backend.add("x.txt", b"bye")
    full = os.path.join(backend.base_path, path)
    assert os.path.exists(full)

    backend.remove(path)
    assert not os.path.exists(full)


def test_remove_missing_path_raises(backend):
    with pytest.raises(FileNotFoundError):
        backend.remove("does_not_exist.txt")
