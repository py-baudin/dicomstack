""" regression tests for the bugs fixed in phase 1 (PLAN.md, appendix A) """

import os
import sys
import copy
import math
import zipfile
import warnings
import subprocess

import numpy as np
import pydicom
import pytest
from pydicom.uid import generate_uid

import dicomstack
from dicomstack import DicomStack, DICOM, utils, pixeldata
from dicomstack.dicomstack import get_zip_path
from dicomstack.query import Selector

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def src():
    return pydicom.dcmread(os.path.join(HERE, "MRLEGS.DCM"))


@pytest.fixture()
def make(src):
    """write a copy of MRLEGS.DCM with a new SOPInstanceUID and given values"""

    def make(path, **kw):
        ds = copy.deepcopy(src)
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        for k, v in kw.items():
            setattr(ds, k, v)
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        utils.save_as(ds, str(path), enforce_file_format=True)
        return ds

    return make


#
# 1.1


def test_sort_several_fields(tmp_path, make):
    for i, (a, b) in enumerate([(1, 3), (1, 1), (1, 2)]):
        make(tmp_path / f"f{i}.dcm", SeriesNumber=a, InstanceNumber=b)
    stack = DicomStack(tmp_path).sort("SeriesNumber", "InstanceNumber")
    assert stack["InstanceNumber"] == [1, 2, 3]

    # three fields
    stack = DicomStack(tmp_path).sort("SeriesNumber", "Modality", "InstanceNumber")
    assert stack["InstanceNumber"] == [1, 2, 3]


#
# 1.2


def test_studies_same_study_id(tmp_path, make):
    for k in range(2):
        make(
            tmp_path / f"s{k}.dcm",
            StudyInstanceUID=generate_uid(),
            StudyID="1",
            StudyDescription=f"study {k}",
            SeriesNumber=k + 1,
        )
    stack = DicomStack(tmp_path)
    tree = stack.dicomtree
    assert len(tree) == 2
    assert "1" in tree
    description = stack.describe()
    assert "study 0" in description
    assert "study 1" in description


def test_studies_non_numeric_study_id(tmp_path, make):
    make(tmp_path / "a.dcm", StudyInstanceUID="1.2.3.1", StudyID="ABC")
    make(tmp_path / "b.dcm", StudyInstanceUID="1.2.3.2", StudyID="")
    tree = DicomStack(tmp_path).dicomtree
    assert len(tree) == 2
    assert "ABC" in tree


def test_describe_empty_series_number(tmp_path, make):
    make(tmp_path / "a.dcm", SeriesNumber=None, SeriesDescription="empty")
    make(tmp_path / "b.dcm", SeriesNumber=2, SeriesDescription="two")
    description = DicomStack(tmp_path).describe()
    assert "empty" in description
    assert "two" in description
    # numeric series first
    assert description.index("two") < description.index("empty")


#
# 1.3


@pytest.fixture()
def zipped(tmp_path, make):
    """zip file with members: IM1, IM10, sub/IM2, sub2/IM3"""
    src = tmp_path / "src"
    members = {"IM1": 1, "IM10": 10, "sub/IM2": 2, "sub2/IM3": 3}
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for member, number in members.items():
            make(src / member, SeriesNumber=number)
            zf.write(src / member, member)
    return path


def test_zip_member_exact(zipped):
    stack = DicomStack(filenames=[os.path.join(zipped, "IM1")])
    assert stack["SeriesNumber"] == [1]


def test_zip_member_folder(zipped):
    stack = DicomStack(filenames=[os.path.join(zipped, "sub")])
    assert stack["SeriesNumber"] == [2]


def test_zip_several_members(zipped):
    names = ["IM10", os.path.join("sub", "IM2")]
    stack = DicomStack(filenames=[os.path.join(zipped, name) for name in names])
    assert sorted(stack["SeriesNumber"]) == [2, 10]


def test_zip_whole(zipped):
    stack = DicomStack(zipped)
    assert sorted(stack["SeriesNumber"]) == [1, 2, 3, 10]


def test_zip_opened_once(zipped, monkeypatch):
    opened = []
    ZipFile = zipfile.ZipFile

    def counting(*args, **kwargs):
        opened.append(args[0])
        return ZipFile(*args, **kwargs)

    monkeypatch.setattr(dicomstack.dicomstack.zipfile, "ZipFile", counting)
    names = ["IM1", "IM10", "sub"]
    stack = DicomStack(filenames=[os.path.join(zipped, name) for name in names])
    assert len(stack) == 3
    assert len(opened) == 1


def test_zip_upper_case(tmp_path, zipped):
    upper = tmp_path / "upper" / "B.ZIP"
    upper.parent.mkdir()
    upper.write_bytes(zipped.read_bytes())
    stack = DicomStack(upper.parent)
    assert sorted(stack["SeriesNumber"]) == [1, 2, 3, 10]


def test_get_zip_path():
    join = os.path.join
    assert get_zip_path(join("a", "b.zip", "c")) == join("a", "b.zip")
    assert get_zip_path(join("a", "B.ZIP")) == join("a", "B.ZIP")
    assert get_zip_path("a/b.zip/c") == "a/b.zip"
    assert get_zip_path(join("a.zip.bak", "c")) is None
    assert get_zip_path(join("a", "b.zipper")) is None


#
# 1.4


def test_fake_zip(tmp_path):
    (tmp_path / "x.zip").write_text("hello")
    stack = DicomStack(tmp_path)
    assert len(stack) == 0
    assert stack.non_dicom == ["x.zip"]


#
# 1.5


def test_no_global_warning_hook():
    # in a fresh interpreter: pytest replaces warnings.showwarning during tests
    code = (
        "import warnings; hook = warnings.showwarning; import dicomstack; "
        "print(warnings.showwarning is hook)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "True", out.stderr


#
# 1.6


def test_isin_alias():
    getter = lambda selector: 2
    assert Selector("a").isin([1, 2]).execute(getter)
    assert not Selector("a").isin([1, 3]).execute(getter)
    # string queries are not implemented
    assert 'stack("SeriesNumber' not in dicomstack.query.__doc__


#
# 1.7


@pytest.fixture()
def gap(tmp_path, make):
    for z in [0, 1, 2, 4]:
        make(
            tmp_path / f"s{z}.dcm",
            ImagePositionPatient=[0, 0, float(z)],
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
        )
    return DicomStack(tmp_path)


def test_volume_missing_slice(gap):
    with pytest.warns(UserWarning, match="spacing"):
        gap.as_volume()
    with pytest.raises(ValueError, match="spacing"):
        gap.as_volume(strict=True)


def test_volume_uniform_no_warning(gap):
    stack = gap(DICOM.ImagePositionPatient[2] < 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        volume = stack.as_volume(strict=True)
    assert volume.spacing[2] == 1


#
# 1.8


def test_single_errors(tmp_path, make):
    for i in range(2):
        make(tmp_path / f"f{i}.dcm", InstanceNumber=i)
    stack = DicomStack(tmp_path)
    with pytest.raises(dicomstack.MultipleValuesError):
        stack.single("InstanceNumber")
    with pytest.raises(dicomstack.NoValueError):
        stack.single("StudyID")
    # backward compatible
    assert issubclass(dicomstack.MultipleValuesError, ValueError)
    assert issubclass(dicomstack.NoValueError, ValueError)


#
# 1.9


@pytest.fixture()
def oblique(tmp_path, make):
    """oblique series, non-square pixels, rows != columns"""
    a, b = math.radians(30), math.radians(20)
    X = np.array([math.cos(a), math.sin(a), 0])
    Y = np.array([-math.sin(a) * math.cos(b), math.cos(a) * math.cos(b), math.sin(b)])
    N = np.cross(X, Y)
    origin = np.array([10.0, -20.0, 30.0])
    dr, dc, dz = 0.8, 0.5, 2.0
    pixels = []
    for k in range(3):
        ds = make(
            tmp_path / f"s{k}.dcm",
            ImageOrientationPatient=list(X) + list(Y),
            ImagePositionPatient=list(origin + k * dz * N),
            PixelSpacing=[dr, dc],
        )
        pixels.append(ds.pixel_array)
    geometry = dict(X=X, Y=Y, origin=origin, dr=dr, dc=dc, dz=dz, N=N)
    return DicomStack(tmp_path), pixels, geometry


def test_volume_spacing_non_square(oblique):
    stack, pixels, geo = oblique
    volume = stack.as_volume(rescale=False)
    rows, cols = pixels[0].shape
    assert rows != cols
    assert volume.shape == (cols, rows, 3)
    assert np.allclose(volume.spacing, (geo["dc"], geo["dr"], geo["dz"]))

    for i, j, k in [(0, 0, 0), (7, 11, 2), (cols - 1, rows - 1, 1)]:
        # voxel position from the volume's geometry
        pos = np.array(volume.origin)
        for axis, index in enumerate((i, j, k)):
            pos = pos + index * volume.spacing[axis] * np.array(volume.transform[axis])
        # DICOM position
        expected = geo["origin"] + k * geo["dz"] * geo["N"]
        expected = expected + i * geo["dc"] * geo["X"] + j * geo["dr"] * geo["Y"]
        assert np.allclose(pos, expected, atol=1e-9)
        assert volume[i, j, k] == pixels[k][j, i]


def test_dicomvolume_attributes():
    geometry = dict(origin=(1, 2, 3), spacing=(0.5, 0.8, 2), transform=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    vol = pixeldata.DicomVolume(np.zeros((4, 3, 2)), metadata={"EchoTime": 10.0}, **geometry)
    assert (vol.origin, vol.spacing, vol.transform) == tuple(geometry.values())
    assert vol.metadata == {"EchoTime": "10.0"}
    with pytest.deprecated_call():
        assert vol.tags == geometry

    # kept by views and elementwise operations, dropped by reductions
    assert (vol + 1).spacing == geometry["spacing"]
    assert vol.T.metadata == vol.metadata
    assert not isinstance(vol.sum(axis=0), pixeldata.DicomVolume)

    # deprecated `tags` keyword
    with pytest.deprecated_call():
        vol = pixeldata.DicomVolume(np.zeros((2, 2, 2)), tags=geometry)
    assert (vol.origin, vol.spacing, vol.transform) == tuple(geometry.values())
    assert vol.metadata == {}


def test_dicomvolume_slicing(oblique):
    stack, pixels, geo = oblique
    volume = stack.as_volume(rescale=False)

    def coords(vol, index):
        pos = np.array(vol.origin)
        for axis, i in enumerate(index):
            pos = pos + i * vol.spacing[axis] * np.array(vol.transform[axis])
        return pos

    # slices: geometry updated
    nx = volume.shape[0]
    for key, start, step in [
        (np.s_[:, :, 1:2], (0, 0, 1), (1, 1, 1)),
        (np.s_[5:, ::2], (5, 0, 0), (1, 2, 1)),
        (np.s_[..., 2:3], (0, 0, 2), (1, 1, 1)),
        (np.s_[::-1], (nx - 1, 0, 0), (-1, 1, 1)),
    ]:
        sub = volume[key]
        assert isinstance(sub, pixeldata.DicomVolume)
        assert np.allclose(sub.origin, coords(volume, start))
        # voxel (1, 1, 0) of sub: same value and position as in volume
        index = tuple(a + b for a, b in zip(start, np.multiply(step, (1, 1, 0))))
        assert sub[1, 1, 0] == volume[index]
        assert np.allclose(coords(sub, (1, 1, 0)), coords(volume, index))

    # integer index on a spatial axis: plain array
    assert type(volume[:, :, 0]) is np.ndarray
    assert type(volume[0]) is np.ndarray
    assert type(volume[[0, 1]]) is np.ndarray


def test_write_volume_slices(tmp_path, oblique):
    """write a volume slice by slice, read it back"""
    stack, pixels, geo = oblique
    volume = stack.as_volume(rescale=False).astype("uint16")
    study_uid, series_uid = generate_uid(), generate_uid()
    (tmp_path / "out").mkdir()
    for k in range(volume.shape[2]):
        utils.write_dataset(
            volume[:, :, k : k + 1],
            str(tmp_path / "out" / f"im{k}.dcm"),
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_uid,
            InstanceNumber=k + 1,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stack2 = DicomStack(tmp_path / "out")
        volume2 = stack2.as_volume(rescale=False)

    for k, frame in enumerate(stack2.sort("InstanceNumber")):
        assert np.array_equal(frame.pixels, pixels[k])
    assert np.array_equal(volume2, volume)
    assert np.allclose(volume2.origin, volume.origin, atol=1e-6)
    assert np.allclose(volume2.spacing, volume.spacing, atol=1e-6)
    assert np.allclose(volume2.transform, volume.transform, atol=1e-6)


def test_format_pixels_plain_array():
    data = np.arange(3 * 4, dtype="uint16").reshape((3, 4))
    tags, raw = pixeldata.format_pixels(data, dtype="uint16")
    assert (tags["Rows"], tags["Columns"]) == (3, 4)
    assert raw == data.tobytes()
    assert not set(tags) & set(utils.GEOMETRY_TAGS)

    # explicit geometry is kept
    ds = utils.make_dataset(data, ImagePositionPatient=[0, 0, 5], PixelSpacing=[0.8, 0.5])
    assert list(ds.ImagePositionPatient) == [0, 0, 5]
    assert list(ds.PixelSpacing) == [0.8, 0.5]


def test_update_dataset_volume_geometry(oblique, legsfile):
    stack, pixels, geo = oblique
    volume = stack.as_volume(rescale=False).astype("uint16")
    ref = pydicom.dcmread(legsfile)
    ds = utils.update_dataset(ref, data=volume[:, :, 2:3])
    expected = geo["origin"] + 2 * geo["dz"] * geo["N"]
    assert np.allclose(list(ds.ImagePositionPatient), expected, atol=1e-6)
    assert np.allclose(list(ds.PixelSpacing), [geo["dr"], geo["dc"]])


def test_tomio_format_pixels(oblique):
    tomio = pytest.importorskip("tomio")
    stack, pixels, geo = oblique
    vol = tomio.asvolume(stack.as_volume(rescale=False).astype("uint16"))
    tags, raw = pixeldata.format_pixels(vol[:, :, 1:2], dtype="uint16")
    expected = geo["origin"] + 1 * geo["dz"] * geo["N"]
    assert np.allclose(tags["ImagePositionPatient"], expected, atol=1e-6)
    assert np.allclose(tags["PixelSpacing"], [geo["dr"], geo["dc"]])
    assert np.allclose(tags["ImageOrientationPatient"], list(geo["X"]) + list(geo["Y"]), atol=1e-6)
    assert (tags["Rows"], tags["Columns"]) == pixels[1].shape
    assert raw == pixels[1].astype("uint16").tobytes()


def test_tomio_asvolume(oblique):
    """DicomVolume converted to tomio.volume.Volume keeps its geometry"""
    tomio = pytest.importorskip("tomio")
    stack, pixels, geo = oblique
    volume = stack.as_volume(rescale=False)
    vol = tomio.asvolume(volume)
    assert isinstance(vol, tomio.Volume)
    assert vol.shape == volume.shape
    assert np.array_equal(np.asarray(vol), np.asarray(volume))
    assert np.allclose(vol.spacing, volume.spacing)
    assert np.allclose(vol.origin, volume.origin)
    assert np.allclose(vol.transform, volume.transform)
    assert vol.metadata == {}

    # tomio's affine maps voxel indices to the DICOM (LPS) positions
    rows, cols = pixels[0].shape
    for i, j, k in [(0, 0, 0), (7, 11, 2), (cols - 1, rows - 1, 1)]:
        expected = geo["origin"] + k * geo["dz"] * geo["N"]
        expected = expected + i * geo["dc"] * geo["X"] + j * geo["dr"] * geo["Y"]
        assert np.allclose(vol.get_coords((i, j, k)), expected, atol=1e-9)


#
# 1.10


def test_write_dataset_valid(tmp_path):
    study_uid, series_uid = generate_uid(), generate_uid()
    data = np.arange(3 * 4, dtype="uint16").reshape((3, 4))
    for i in range(3):
        utils.write_dataset(
            data + i,
            str(tmp_path / f"im{i}.dcm"),
            StudyInstanceUID=study_uid,
            SeriesInstanceUID=series_uid,
            InstanceNumber=i + 1,
            ImagePositionPatient=[0, 0, i],
        )

    datasets = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i in range(3):
            datasets.append(pydicom.dcmread(str(tmp_path / f"im{i}.dcm")))

    uids = set()
    for ds in datasets:
        assert ds.SOPInstanceUID
        assert ds.SOPClassUID == pydicom.uid.MRImageStorage
        assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
        assert ds.file_meta.MediaStorageSOPClassUID == ds.SOPClassUID
        assert ds.file_meta.ImplementationClassUID == pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        assert ds.file_meta.TransferSyntaxUID == pydicom.uid.ExplicitVRLittleEndian
        assert ds.PhotometricInterpretation == "MONOCHROME2"
        assert len(ds.ImageOrientationPatient) == 6
        assert ds.FrameOfReferenceUID
        uids.add(ds.SOPInstanceUID)
    assert len(uids) == 3
    # same series: same frame of reference
    assert len({ds.FrameOfReferenceUID for ds in datasets}) == 1

    stack = DicomStack(tmp_path)
    assert len(stack) == 3
    assert stack.unique("SeriesInstanceUID") == [series_uid]
    tree = stack.dicomtree
    assert len(tree) == 1
    assert [len(series) for series in next(iter(tree.values())).values()] == [3]


def test_write_dataset_from_reference(tmp_path, legsfile):
    ref = pydicom.dcmread(legsfile)
    data = ref.pixel_array[:20, :20]
    utils.write_dataset(data, str(tmp_path / "copy.dcm"), dataset=ref)
    ds = pydicom.dcmread(str(tmp_path / "copy.dcm"))
    # derived image: new instance UID
    assert ds.SOPInstanceUID != ref.SOPInstanceUID
    assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID


def test_write_dataset_storage_class_uid(tmp_path):
    data = np.zeros((2, 2), dtype="uint16")
    uid = pydicom.uid.CTImageStorage
    utils.write_dataset(data, str(tmp_path / "ct.dcm"), media_storage_class=uid)
    ds = pydicom.dcmread(str(tmp_path / "ct.dcm"))
    assert ds.SOPClassUID == uid
    assert ds.file_meta.MediaStorageSOPClassUID == uid
