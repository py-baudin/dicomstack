""" pixel reading (phase 3.1): fast path and fallbacks equal pydicom's pixel_array """

import io
import os
import copy
import zipfile

import numpy as np
import pydicom
import pytest
from pydicom.uid import (
    generate_uid,
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    DeflatedExplicitVRLittleEndian,
    RLELossless,
)

from dicomstack import DicomStack, DICOM
from dicomstack.dicomstack import DicomFile, read_native_pixels

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def legs():
    return pydicom.dcmread(os.path.join(HERE, "MRLEGS.DCM"))


def expected_pixels(path):
    return pydicom.dcmread(str(path)).pixel_array


def check(path, fast=True, monkeypatch=None):
    """DicomFile(path).pixels == pydicom's pixel_array; header parsed once"""
    dcmfile = DicomFile(str(path))
    dcmfile.dataset  # read the header
    if monkeypatch is not None:
        # pixels must not re-parse the whole file
        def fail(*args, **kwargs):
            raise AssertionError("whole file parsed again")

        monkeypatch.setattr(dcmfile, "_read", fail)
    pixels = dcmfile.pixels
    expected = expected_pixels(path)
    assert pixels.dtype == expected.dtype
    assert pixels.shape == expected.shape
    assert np.array_equal(pixels, expected)
    assert pixels.flags.writeable

    # fast path used?
    with open(path, "rb") as fp:
        fp.seek(dcmfile.pixel_offset)
        syntax = dcmfile.dataset.file_meta.TransferSyntaxUID
        native = read_native_pixels(fp, dcmfile.dataset, syntax.is_implicit_VR)
    assert (native is not None) == fast
    return pixels


def write(ds, path, syntax=ExplicitVRLittleEndian, **values):
    ds = copy.deepcopy(ds)
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    for name, value in values.items():
        setattr(ds, name, value)
    if syntax == RLELossless:
        ds.compress(RLELossless)
    else:
        ds.file_meta.TransferSyntaxUID = syntax
    ds.save_as(str(path), enforce_file_format=True)
    return path


def test_test_data(monkeypatch, brainfile, legsfile, multi):
    for path in [brainfile, legsfile, multi]:
        check(path, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    "bits, stored, signed",
    [(8, 8, False), (16, 16, True), (16, 12, False), (16, 12, True), (32, 32, False), (32, 20, True)],
)
def test_bit_depths(tmp_path, legs, monkeypatch, bits, stored, signed):
    rng = np.random.default_rng(0)
    dtype = np.dtype(f"<{'i' if signed else 'u'}{bits // 8}")
    # random values, including garbage in the unused high bits
    raw = rng.integers(np.iinfo(dtype).min, np.iinfo(dtype).max, size=(5, 7), dtype=dtype, endpoint=True)
    path = write(
        legs, tmp_path / "im.dcm",
        Rows=5, Columns=7, BitsAllocated=bits, BitsStored=stored, HighBit=stored - 1,
        PixelRepresentation=int(signed), PixelData=raw.tobytes(),
    )
    pixels = check(path, monkeypatch=monkeypatch)
    if stored < bits and signed:
        assert pixels.min() >= -(2 ** (stored - 1))


def test_implicit_vr(tmp_path, legs, monkeypatch):
    check(write(legs, tmp_path / "im.dcm", syntax=ImplicitVRLittleEndian), monkeypatch=monkeypatch)


def test_multiframe_native(tmp_path, legs, monkeypatch):
    frames = np.arange(3 * 4 * 5, dtype="<u2").reshape(3, 4, 5)
    path = write(
        legs, tmp_path / "im.dcm",
        Rows=4, Columns=5, NumberOfFrames=3, BitsStored=16, HighBit=15, PixelData=frames.tobytes(),
    )
    assert np.array_equal(check(path, monkeypatch=monkeypatch), frames)


def test_unsupported_fall_back(tmp_path, legs, monkeypatch):
    # HighBit not BitsStored - 1: decoded by pydicom, header not parsed again
    check(write(legs, tmp_path / "hb.dcm", HighBit=15), fast=False, monkeypatch=monkeypatch)


def test_rle(tmp_path, legs, monkeypatch):
    # compressed: pixel data element read alone, decoded by pydicom
    check(write(legs, tmp_path / "rle.dcm", syntax=RLELossless), fast=False, monkeypatch=monkeypatch)


def test_deflated(tmp_path, legs):
    path = write(legs, tmp_path / "deflated.dcm", syntax=DeflatedExplicitVRLittleEndian)
    dcmfile = DicomFile(str(path))
    assert dcmfile.dataset and dcmfile.pixel_offset is None
    assert np.array_equal(dcmfile.pixels, expected_pixels(path))


#
# 3.2 workers, 3.3 dtype


@pytest.fixture()
def series(tmp_path, legs):
    """5 slices, 2 echoes, and a non-DICOM file"""
    for echo in range(2):
        for k in range(5):
            write(
                legs, tmp_path / f"im{echo}_{k}.dcm",
                EchoTime=10.0 * (echo + 1), ImagePositionPatient=[0, 0, float(k)],
                ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            )
    (tmp_path / "notes.txt").write_text("not DICOM")
    return tmp_path


def test_load_workers(series):
    stack = DicomStack(series)
    stack2 = DicomStack(series, workers=4)
    assert stack2.filenames == stack.filenames
    assert stack2.non_dicom == stack.non_dicom == ["notes.txt"]
    assert stack2.frames == stack.frames
    assert stack2.frames[0].dicomfile.pixel_offset == stack.frames[0].dicomfile.pixel_offset


def test_iter_pixels_workers(series, multi, monkeypatch):
    stack = DicomStack(series)
    expected = list(DicomStack(series).iter_pixels())
    assert all(np.array_equal(a, b) for a, b in zip(stack.iter_pixels(workers=4), expected))

    # multi-frame: the file is read once
    reads = []
    read_bytes = DicomFile._read_pixel_bytes
    monkeypatch.setattr(
        DicomFile, "_read_pixel_bytes", lambda self: reads.append(1) or read_bytes(self)
    )
    pixels = list(DicomStack(multi).iter_pixels(workers=4))
    assert len(pixels) == 90 and len(reads) == 1
    assert np.array_equal(np.stack(pixels), expected_pixels(multi))


def test_as_volume_workers_dtype(series):
    stack = DicomStack(series)
    indices, volumes = stack.as_volume(by="EchoTime")
    indices2, volumes2 = DicomStack(series).as_volume(by="EchoTime", workers=4)
    assert indices2 == indices
    assert all(np.array_equal(a, b) for a, b in zip(volumes, volumes2))

    volume = stack(EchoTime=10.0).as_volume(dtype="float32")
    assert volume.dtype == np.float32
    assert np.allclose(volume, volumes[0])
    assert volume.spacing == volumes[0].spacing


def test_warning_location(series):
    """warnings point to the user's code"""
    stack = DicomStack(series)(DICOM.ImagePositionPatient[2] != 3)
    with pytest.warns(UserWarning) as record:
        stack.as_volume()  # duplicated positions (2 echoes) and missing slice
    messages = [str(warning.message) for warning in record]
    assert any("duplicated" in message for message in messages)
    assert any("slice spacing" in message for message in messages)
    assert all(warning.filename == __file__ for warning in record)


def test_zip_member(tmp_path, legsfile):
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as zf:
        zf.write(legsfile, "LEGS.DCM")
    frame = DicomStack(tmp_path / "a.zip").frames[0]
    assert frame.dicomfile.pixel_offset is None
    assert np.array_equal(frame.pixels, expected_pixels(legsfile))
