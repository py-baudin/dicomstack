""" header cache (phase 3.4) """

import os
import copy
import gzip
import json
import shutil
import zipfile

import pydicom
import pytest
from pydicom.uid import generate_uid

import golden_lib
from dicomstack import DicomStack, DICOM
from dicomstack import cache
from dicomstack import dicomstack as core

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def parses(monkeypatch):
    """count header parses"""
    calls = []
    read_dataset = core.read_dataset

    def counting(data, stop_before_pixels=True):
        calls.append(stop_before_pixels)
        return read_dataset(data, stop_before_pixels)

    monkeypatch.setattr(core, "read_dataset", counting)
    return calls


@pytest.fixture()
def series(tmp_path):
    src = pydicom.dcmread(os.path.join(HERE, "MRLEGS.DCM"))
    folder = tmp_path / "data"
    for k in range(4):
        ds = copy.deepcopy(src)
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [0, 0, float(k)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        os.makedirs(folder / "sub", exist_ok=True)
        ds.save_as(str(folder / "sub" / f"im{k}.dcm"), enforce_file_format=True)
    (folder / "notes.txt").write_text("not DICOM")
    return folder


def values(stack, fields=("InstanceNumber", "ImageType", "PixelSpacing", "StudyDate", "[CSAImageHeaderType]")):
    return [[frame.get(field) for field in fields] for frame in stack]


@pytest.mark.parametrize("name", golden_lib.FIXTURES)
def test_golden_cached(tmp_path, name):
    """second load from the cache: same values and types as the golden snapshot"""
    if not os.path.isfile(golden_lib.golden_path(name)):
        pytest.skip("no golden snapshot: run test/make_golden.py")
    path = golden_lib.make_fixtures(str(tmp_path / "fixtures"))[name]
    if os.path.dirname(path) == HERE:
        # do not write a cache file in the test folder
        os.makedirs(tmp_path / name)
        path = shutil.copy(path, tmp_path / name)
    DicomStack(path, cache=True)  # writes the cache
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    assert os.path.isfile(os.path.join(folder, cache.CACHE_NAME))
    actual = golden_lib.snapshot({name: path}, cache=True)[name]
    actual = json.loads(json.dumps(actual))
    differences = golden_lib.diff(golden_lib.load(name), actual)
    assert not differences, "\n".join(differences)


def test_second_load_no_parse(series, parses):
    stack = DicomStack(series, cache=True)
    assert len(parses) == 5  # 4 DICOM + 1 non-DICOM
    assert (series / cache.CACHE_NAME).is_file()

    parses.clear()
    stack2 = DicomStack(series, cache=True)
    assert parses == []
    assert stack2.filenames == stack.filenames
    assert stack2.non_dicom == stack.non_dicom == ["notes.txt"]
    assert stack2.frames == stack.frames
    assert values(stack2) == values(stack)
    assert parses == []  # cached values

    # not cached: sequences, bytes, nested selectors, dataset, pixels -> header read
    frame = stack2.frames[0]
    assert frame["IconImageSequence.1.Rows"] == 64
    assert frame[DICOM.IconImageSequence[0].Columns] == 64
    assert frame.dataset["Modality"].value == "MR"
    assert frame.pixels.shape == (192, 256)
    with pytest.raises(KeyError):
        frame["UnknownField"]
    with pytest.raises(TypeError):
        frame[5]


def test_cache_not_rewritten(series):
    DicomStack(series, cache=True)
    path = series / cache.CACHE_NAME
    mtime = path.stat().st_mtime_ns
    DicomStack(series, cache=True)
    assert path.stat().st_mtime_ns == mtime


def test_changed_file(series, parses):
    DicomStack(series, cache=True)
    # change one file (content and mtime)
    path = series / "sub" / "im2.dcm"
    ds = pydicom.dcmread(str(path))
    ds.InstanceNumber = 30
    ds.save_as(str(path), enforce_file_format=True)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    # remove one file
    os.remove(series / "sub" / "im3.dcm")

    parses.clear()
    stack = DicomStack(series, cache=True)
    assert len(parses) == 1
    assert sorted(stack["InstanceNumber"]) == [1, 2, 30]
    entries = cache.load(cache.locations(str(series), True), "uid")
    assert sorted(entries) == ["notes.txt", "sub/im0.dcm", "sub/im1.dcm", "sub/im2.dcm"]


def test_corrupt_cache(series):
    DicomStack(series, cache=True)
    (series / cache.CACHE_NAME).write_bytes(b"garbage")
    with pytest.warns(UserWarning, match="Invalid cache"):
        stack = DicomStack(series, cache=True)
    assert sorted(stack["InstanceNumber"]) == [1, 2, 3, 4]
    # rebuilt
    assert cache.load(cache.locations(str(series), True), "uid")


def test_outdated_cache(series, parses):
    DicomStack(series, cache=True)
    parses.clear()
    DicomStack(series, cache=True, dedup="content")  # other mode: rebuilt
    assert len(parses) == 5
    parses.clear()
    DicomStack(series, cache=True, dedup="content")
    assert parses == []


def test_read_only_folder(series, tmp_path, monkeypatch, parses):
    user = tmp_path / "user-cache"
    monkeypatch.setattr(cache, "user_cache_dir", lambda: str(user))
    write_atomic = cache.write_atomic

    def read_only(path, data):
        if os.path.dirname(path) == str(series):
            raise PermissionError("read-only")
        return write_atomic(path, data)

    monkeypatch.setattr(cache, "write_atomic", read_only)
    DicomStack(series, cache=True)
    assert not (series / cache.CACHE_NAME).exists()
    assert len(list(user.glob("*.json.gz"))) == 1
    parses.clear()
    DicomStack(series, cache=True)
    assert parses == []


def test_refresh_and_explicit_path(series, tmp_path, parses):
    DicomStack(series, cache=True)
    parses.clear()
    DicomStack(series, cache="refresh")
    assert len(parses) == 5

    path = tmp_path / "my-cache.json.gz"
    files = [str(series / "sub" / f"im{k}.dcm") for k in range(4)]
    DicomStack(filenames=files, cache=str(path))
    assert path.is_file()
    parses.clear()
    stack = DicomStack(filenames=files, cache=str(path))
    assert parses == [] and len(stack) == 4


def test_cache_file_not_loaded(series):
    DicomStack(series, cache=True)
    stack = DicomStack(series)  # cache disabled: the cache file is still skipped
    assert stack.non_dicom == ["notes.txt"]


def test_zip(series, tmp_path, parses):
    zpath = tmp_path / "zipped" / "a.zip"
    zpath.parent.mkdir()
    with zipfile.ZipFile(zpath, "w") as zf:
        for k in range(4):
            zf.write(series / "sub" / f"im{k}.dcm", f"sub/im{k}.dcm")
        zf.write(series / "notes.txt", "notes.txt")
    stack = DicomStack(zpath, cache=True)
    parses.clear()
    stack2 = DicomStack(zpath, cache=True)
    assert parses == []
    assert stack2.filenames == stack.filenames
    assert len(stack2.non_dicom) == 1
    assert values(stack2) == values(stack)
    assert stack2.frames[0].pixels.shape == (192, 256)


def test_multiframe(multi, parses):
    stack = DicomStack(multi, cache=True)
    parses.clear()
    stack2 = DicomStack(multi, cache=True)
    assert parses == []
    assert stack2["EchoTime"] == stack["EchoTime"]
    assert len(set(stack2.frames)) == 90


def test_cache_size(series):
    DicomStack(series, cache=True)
    with gzip.open(series / cache.CACHE_NAME, "rt") as fp:
        data = json.load(fp)
    assert data["header"]["format"] == cache.FORMAT_VERSION
    assert (series / cache.CACHE_NAME).stat().st_size < 50_000
