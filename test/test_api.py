""" API additions (phase 5) """

import os
import copy
import zipfile

import numpy as np
import pydicom
import pytest
from pydicom.uid import generate_uid

from dicomstack import DicomStack, NoValueError
from dicomstack.files import DicomFile
from dicomstack.pixeldata import DicomVolume

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def src():
    return pydicom.dcmread(os.path.join(HERE, "MRLEGS.DCM"))


@pytest.fixture()
def folder(tmp_path, src):
    """2 series x 2 echoes x 3 slices (12 files), one with ImageComments"""
    study = generate_uid()
    for series in (1, 2):
        series_uid = generate_uid()
        for echo in (1, 2):
            for k in range(3):
                ds = copy.deepcopy(src)
                ds.StudyInstanceUID = study
                ds.SeriesInstanceUID = series_uid
                ds.SeriesNumber = series
                ds.EchoTime = 10.0 * echo
                ds.InstanceNumber = 10 * echo + k
                ds.ImagePositionPatient = [0.0, 0.0, float(k)]
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.SOPInstanceUID = generate_uid()
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                if (series, echo, k) == (2, 2, 2):
                    ds.ImageComments = "last"
                path = tmp_path / "data" / f"s{series}" / f"im{echo}_{k}.dcm"
                path.parent.mkdir(parents=True, exist_ok=True)
                ds.save_as(str(path), enforce_file_format=True)
    return tmp_path / "data"


@pytest.fixture()
def stack(folder):
    return DicomStack(folder)


#
# exploring


def test_fields(stack):
    all_fields, any_fields = stack.fields(), stack.fields("any")
    assert "EchoTime" in all_fields and "[CSAImageHeaderType]" in all_fields
    assert "ImageComments" in any_fields and not "ImageComments" in all_fields
    assert set(all_fields) < set(any_fields)
    assert DicomStack().fields() == []
    with pytest.raises(ValueError):
        stack.fields("some")


def test_varying(stack):
    assert stack.varying() == ["EchoTime", "ImageComments", "SeriesInstanceUID", "SeriesNumber"]
    assert stack.varying(within="SeriesNumber") == ["EchoTime", "ImageComments"]
    assert stack.varying(within=["SeriesNumber", "EchoTime"]) == ["ImageComments"]
    assert "InstanceNumber" in stack.varying(exclude=())


def test_table(stack):
    table = stack.table("SeriesNumber", "EchoTime")
    assert len(table) == 12
    assert table[0] == {"SeriesNumber": 1, "EchoTime": 10}
    assert set(stack.table()[0]) == set(stack.fields())


def test_to_dataframe(stack):
    pandas = pytest.importorskip("pandas")
    frame = stack.to_dataframe("SeriesNumber", "EchoTime")
    assert isinstance(frame, pandas.DataFrame)
    assert frame.shape == (12, 2)
    assert list(frame.columns) == ["SeriesNumber", "EchoTime"]


def test_repr(stack, legsfile):
    assert repr(stack) == "DICOM(12 frames | 1 study | 2 series)"
    assert repr(DicomStack(legsfile)) == "DICOM(1 frame | 1 study | 1 series)"
    assert repr(DicomStack()) == "DICOM(0 frames)"
    html = stack._repr_html_()
    assert html.startswith("<pre>DICOM(12 frames") and "n=6" in html


#
# selecting


def test_groupby(stack):
    groups = list(stack.groupby("EchoTime"))
    assert [values for values, _ in groups] == [10, 20]
    assert all(len(substack) == 6 for _, substack in groups)
    groups = list(stack.groupby("SeriesNumber", "EchoTime"))
    assert [values for values, _ in groups] == [(1, 10), (1, 20), (2, 10), (2, 20)]
    assert [len(substack) for _, substack in stack.series()] == [6, 6]
    # missing values: skipped
    assert [len(substack) for _, substack in stack.groupby("ImageComments")] == [1]


def test_indexing(stack):
    assert stack[0] is stack.frames[0]
    assert stack[-1] is stack.frames[-1]
    sub = stack[2:5]
    assert isinstance(sub, DicomStack) and sub.frames == stack.frames[2:5]
    assert sub.filenames == stack.filenames[2:5]
    # fields
    assert stack["SeriesNumber"][0] == 1
    assert stack["SeriesNumber", "EchoTime"][0] == (1, 10)
    assert len(stack[(0x0018, 0x0081)]) == 12


def test_set_operations(stack, legsfile):
    echo1, series1 = stack(EchoTime=10), stack(SeriesNumber=1)
    union = echo1 + series1
    assert len(union) == 9
    assert union.frames[:6] == echo1.frames  # order of the left operand
    assert len(echo1 - series1) == 3
    assert (echo1 - series1)["SeriesNumber"] == [2, 2, 2]
    inter = echo1 & series1
    assert len(inter) == 3 and inter.filenames == [f for f in echo1.filenames if f.startswith("s1")]
    assert (echo1 + series1).root == stack.root
    # different roots: absolute filenames
    other = DicomStack(legsfile)
    mixed = echo1 + other
    assert len(mixed) == 7 and mixed.root is None
    assert os.path.isabs(mixed.filenames[0])
    with pytest.raises(TypeError):
        stack + 1


def test_sort_missing(stack):
    assert len(stack.sort("ImageComments")) == 1
    last = stack.sort("ImageComments", missing="last")
    assert len(last) == 12 and last[0]["ImageComments"] == "last"
    assert last.frames[1:] == [f for f in stack.frames if f.get("ImageComments") is None]
    with pytest.raises(NoValueError):
        stack.sort("ImageComments", missing="error")
    assert len(stack.sort("InstanceNumber", missing="error")) == 12


#
# volumes


def test_affine(stack):
    volume = stack(SeriesNumber=1, EchoTime=10).as_volume()
    affine = volume.affine
    for index in [(0, 0, 0), (5, 7, 2)]:
        expected = np.asarray(volume.origin, dtype=float)
        for axis, i in enumerate(index):
            expected = expected + i * volume.spacing[axis] * np.asarray(volume.transform[axis])
        assert np.allclose(affine @ np.r_[index, 1], np.r_[expected, 1])
    with pytest.raises(ValueError):
        DicomVolume(np.zeros((2, 2, 2))).affine


def test_as_volume_stack(stack):
    series1 = stack(SeriesNumber=1)
    indices, volumes = series1.as_volume(by="EchoTime")
    volume = series1.as_volume(by="EchoTime", stack=True)
    assert volume.shape == volumes[0].shape + (2,)
    assert np.array_equal(volume[..., 1], volumes[1])
    assert volume.metadata == {"EchoTime": "[10, 20]"}
    assert volume.spacing == volumes[0].spacing

    volume = stack.as_volume(by=["SeriesNumber", "EchoTime"], stack=True)
    assert volume.shape[3] == 4
    assert volume.metadata == {"SeriesNumber": "[1, 1, 2, 2]", "EchoTime": "[10, 20, 10, 20]"}
    assert stack.as_volume(by="UnknownField", stack=True) is None


#
# loading


def test_include_exclude_recursive(folder):
    (folder / "notes.txt").write_text("not DICOM")
    assert len(DicomStack(folder, include="*.dcm").non_dicom) == 0
    assert len(DicomStack(folder, include="*.dcm")) == 12
    assert len(DicomStack(folder, exclude=["*.txt", "s2/*"])) == 6
    assert len(DicomStack(folder, include="s1/im1_*")) == 3
    assert DicomStack(folder, recursive=False).non_dicom == ["notes.txt"]
    assert len(DicomStack(folder, recursive=False)) == 0
    files = [str(folder / "s1" / "im1_0.dcm"), str(folder / "notes.txt")]
    assert len(DicomStack(filenames=files, exclude="*.txt").non_dicom) == 0


def test_progress(folder):
    calls = []
    DicomStack(folder, progress=lambda done, total: calls.append((done, total)))
    assert calls[-1] == (12, 12) and len(calls) == 12
    # text progress bar (stderr)
    assert len(DicomStack(folder, progress=True)) == 12


def test_progress_bar():
    import io
    from dicomstack.utils import ProgressBar

    out = io.StringIO()
    bar = ProgressBar(4, width=8, file=out, interval=0)
    for done in range(1, 5):
        bar.update(done)
    bar.update(4)  # unchanged: not redrawn
    bar.close()
    assert out.getvalue() == (
        "\r[##------] 1/4\r[####----] 2/4\r[######--] 3/4\r[########] 4/4\n"
    )

    # redrawn at most every `interval` seconds, last update always shown
    out = io.StringIO()
    bar = ProgressBar(1000, width=10, file=out, interval=60)
    for done in range(1, 1001):
        bar.update(done)
    assert out.getvalue() == "\r[----------]    1/1000\r[##########] 1000/1000"


def test_progress_auto(folder, capsys):
    # default: no progress bar when stderr is not a terminal
    DicomStack(folder)
    assert capsys.readouterr().err == ""
    DicomStack(folder, progress=True)
    assert capsys.readouterr().err.endswith("12/12\n")


def test_from_datasets(src, legsfile):
    datasets = []
    for k in range(3):
        ds = copy.deepcopy(src)
        ds.filename = None  # (copied from the file dataset)
        ds.SOPInstanceUID = generate_uid()
        ds.ImagePositionPatient = [0, 0, float(k)]
        datasets.append(ds)
    stack = DicomStack.from_datasets(datasets + [datasets[0]])  # duplicate removed
    assert len(stack) == 3
    assert stack.filenames == ["dataset_1.dcm", "dataset_2.dcm", "dataset_3.dcm"]
    assert stack["ImagePositionPatient_3"] == [0, 1, 2]
    assert stack[0].pixels.shape == (192, 256)
    assert stack.as_volume().shape == (256, 192, 3)
    assert list(stack.iter_pixels(workers=2))[1].shape == (192, 256)
    assert stack[0].dataset["Modality"].value == "MR"
    # datasets read from files keep their name
    assert DicomStack.from_datasets([pydicom.dcmread(legsfile)]).filenames == [legsfile]
    # no file contents to compare
    with pytest.raises(ValueError):
        DicomFile.from_dataset(datasets[0], "x.dcm", dedup="content")


#
# other


def test_export(stack, tmp_path, legsfile):
    sub = stack(SeriesNumber=2, EchoTime=20)
    exported = sub.export(tmp_path / "out")
    assert len(exported) == 3
    out = DicomStack(tmp_path / "out")
    assert out.filenames == sub.filenames
    assert out["PatientName"] == [None] * 3  # emptied (empty values read as None)
    assert out["SeriesNumber"] == [2] * 3

    # zip members and in-memory datasets
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as zf:
        zf.write(legsfile, "LEGS.DCM")
    zipped = DicomStack(tmp_path / "a.zip").export(tmp_path / "out_zip", anonymize=False)
    source, copy_ = pydicom.dcmread(legsfile), pydicom.dcmread(str(zipped[0]))
    assert set(copy_.keys()) == set(source.keys())  # nothing removed (anonymize=False)
    memory = DicomStack.from_datasets([pydicom.dcmread(legsfile)])
    assert memory.export(tmp_path / "out_memory")[0].name == "MRLEGS.DCM"
