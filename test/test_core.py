""" tests of the thin core (phase 2): lazy access, duplicates, pickling """

import os
import copy
import pickle
import zipfile

import pydicom
import pytest
from pydicom.uid import generate_uid

from dicomstack import DicomStack, DICOM
from dicomstack import dicomstack as ds_module
from dicomstack.dicomstack import DicomFile, DicomDataset

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def src():
    return pydicom.dcmread(os.path.join(HERE, "MRLEGS.DCM"))


def write(src, path, uid=None, **values):
    ds = copy.deepcopy(src)
    if uid is not False:
        ds.SOPInstanceUID = uid or generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    else:
        del ds.SOPInstanceUID
    for name, value in values.items():
        setattr(ds, name, value)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return ds


#
# lazy access


def test_header_only(legsfile):
    dcmfile = DicomFile(legsfile)
    assert dcmfile._dataset is None
    frame = dcmfile.frames[0]
    assert "PixelData" not in dcmfile.dataset
    assert not hasattr(dcmfile, "bytes")
    # offset of the pixel data element
    with open(legsfile, "rb") as fp:
        fp.seek(dcmfile.pixel_offset)
        assert fp.read(4) == b"\xe0\x7f\x10\x00"
    # field access does not convert the whole dataset
    assert frame["Modality"] == "MR"
    assert frame._dataset is None
    assert frame.pixels.ndim == 2


def test_same_values_as_full_conversion(tmp_path, src):
    """frame[key] == DicomDataset path for keys not covered by the golden data"""
    ds = copy.deepcopy(src)
    # repeating group (overlay), in two groups
    ds.add_new(0x60003000, "OW", b"\x00\x01" * 8)
    ds.add_new(0x60023000, "OW", b"\x02\x03" * 8)
    ds.add_new(0x60020010, "US", 4)
    path = tmp_path / "overlay.dcm"
    ds.save_as(str(path), enforce_file_format=True)

    frame = DicomStack(path).frames[0]
    full = DicomDataset.from_dataset(frame.dicomfile.dataset)
    # repeating groups: no keyword in pydicom, accessible by tag only
    for key in ["OverlayData", "OverlayRows"]:
        with pytest.raises(KeyError):
            full[key]
        with pytest.raises(KeyError):
            frame[key]
    keys = ["TransferSyntaxUID", "MediaStorageSOPInstanceUID", "IconImageSequence",
            (0x6000, 0x3000), (0x6002, 0x3000), (0x6002, 0x0010), (0x0002, 0x0010)]
    for key in keys:
        expected = full[key].value
        value = frame[key]
        if key == "IconImageSequence":
            assert [d.elements_by_tag.keys() for d in value] == [d.elements_by_tag.keys() for d in expected]
        else:
            assert value == expected, key

    # icon pixel data is skipped, as in the full conversion
    with pytest.raises(KeyError):
        frame["IconImageSequence.1.PixelData"]
    with pytest.raises(TypeError):
        frame[5]


def test_value_cache(legsfile):
    frame = DicomStack(legsfile).frames[0]
    assert frame["ImageType_1"] == frame[DICOM.ImageType[0]] == "ORIGINAL"
    assert frame._values  # cached
    # unhashable keys are not cached
    assert frame[DICOM.ImageType[slice(0, 1)]] == ("ORIGINAL",)


def test_dicom_selector_lazy():
    assert "EchoTime" in dir(DICOM)
    assert DICOM.EchoTime.keys == ("EchoTime",)
    assert "EchoTime" not in vars(DICOM)
    with pytest.raises(AttributeError):
        DICOM.__deepcopy__


#
# deferred private values


def deferred_tags(header):
    from pydicom.dataelem import RawDataElement

    return [
        tag for tag in header.keys()
        if isinstance(header.get_item(tag, keep_deferred=True), RawDataElement)
        and header.get_item(tag, keep_deferred=True).value is None
    ]


def test_private_values_deferred(legsfile, monkeypatch):
    frame = DicomStack(legsfile).frames[0]
    header = frame.dicomfile.dataset
    deferred = deferred_tags(header)
    private = [tag for tag in header.keys() if tag.is_private and not tag.is_private_creator]
    assert deferred and set(deferred) == set(private)  # private creators are kept

    # "[Name]" lookups do not read the deferred values
    reads = []
    import pydicom.filereader

    read = pydicom.filereader.read_deferred_data_element
    monkeypatch.setattr(
        pydicom.filereader, "read_deferred_data_element",
        lambda *args: reads.append(args[-1].tag) or read(*args),
    )
    with pytest.raises(KeyError):
        frame["[UnknownPrivateName]"]
    assert "[CSAImageHeaderType]" in frame.keywords()
    assert reads == []

    # values read on access, same as without deferral
    full = DicomFile(legsfile, defer_private=False)
    full.dataset
    reference = full.frames[0]
    for key in ["[CSAImageHeaderType]", "[TablePositionOrigin]", "[CSAImageHeaderInfo]"]:
        assert frame[key] == reference[key]
    assert len(reads) == 3
    assert len(deferred_tags(header)) == len(deferred) - 3
    # full conversion (sequences: lists of DicomDataset, compared by their elements' values)
    def values(element):
        if element.sequence:
            return [[e.value for e in item.elements if not e.sequence] for item in element.value]
        return element.value

    keywords = frame.dataset.elements_by_keyword
    expected = reference.dataset.elements_by_keyword
    assert set(keywords) == set(expected)
    assert all(values(keywords[k]) == values(v) for k, v in expected.items())


def test_private_values_not_deferred(tmp_path, legsfile):
    # zip members: no file to read them from
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as zf:
        zf.write(legsfile, "LEGS.DCM")
    header = DicomStack(tmp_path / "a.zip").frames[0].dicomfile.dataset
    assert deferred_tags(header) == []
    # cache: values cached, then deferred
    stack = DicomStack(legsfile, cache=str(tmp_path / "cache.json.gz"))
    assert deferred_tags(stack.frames[0].dicomfile.dataset)
    stack = DicomStack(legsfile, cache=str(tmp_path / "cache.json.gz"))
    assert stack.frames[0]["[CSAImageHeaderType]"] == "IMAGE NUM 4"


def test_private_values_pickle(legsfile):
    stack = pickle.loads(pickle.dumps(DicomStack(legsfile)))
    assert stack.frames[0]["[CSAImageHeaderType]"] == "IMAGE NUM 4"


#
# duplicates


def test_dedup_uid_vs_content(tmp_path, src):
    uid = generate_uid()
    # same UID, different contents (changed private value)
    write(src, tmp_path / "a" / "1.dcm", uid=uid, SeriesDescription="one")
    write(src, tmp_path / "a" / "2.dcm", uid=uid, SeriesDescription="two")
    assert len(DicomStack(tmp_path / "a")) == 1
    assert len(DicomStack(tmp_path / "a", dedup="content")) == 2
    with pytest.raises(ds_module.DuplicatedFramesError):
        DicomStack(tmp_path / "a", duplicates="error")

    # identical files
    write(src, tmp_path / "b" / "1.dcm", uid=uid)
    with open(tmp_path / "b" / "1.dcm", "rb") as fp:
        (tmp_path / "b" / "2.dcm").write_bytes(fp.read())
    assert len(DicomStack(tmp_path / "b")) == 1
    assert len(DicomStack(tmp_path / "b", dedup="content")) == 1

    with pytest.raises(ValueError):
        DicomStack(tmp_path / "b", dedup="unknown")


def test_dedup_no_uid(tmp_path, src):
    write(src, tmp_path / "1.dcm", uid=False)
    write(src, tmp_path / "2.dcm", uid=False)
    stack = DicomStack(tmp_path)
    assert len(stack) == 2
    assert stack.frames[0] != stack.frames[1]


def test_dedup_zip(tmp_path, src):
    uid = generate_uid()
    write(src, tmp_path / "src" / "1.dcm", uid=uid, SeriesDescription="one")
    write(src, tmp_path / "src" / "2.dcm", uid=uid, SeriesDescription="two")
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as zf:
        for name in ["1.dcm", "2.dcm"]:
            zf.write(tmp_path / "src" / name, name)
    assert len(DicomStack(tmp_path / "a.zip")) == 1
    assert len(DicomStack(tmp_path / "a.zip", dedup="content")) == 2
    # zip members: pixels read from the archive
    assert DicomStack(tmp_path / "a.zip").frames[0].pixels.ndim == 2


def test_multiframe_identity(multi):
    stack = DicomStack(multi)
    assert len(stack) == 90
    assert len(set(stack.frames)) == 90


def test_modes_not_equal(legsfile):
    frame_uid = DicomStack(legsfile).frames[0]
    frame_content = DicomStack(legsfile, dedup="content").frames[0]
    assert frame_uid != frame_content
    assert frame_uid == DicomStack(legsfile).frames[0]


#
# pickle


def test_pickle(legsfile, multi):
    for path in [legsfile, multi]:
        stack = DicomStack(path)
        stack["EchoTime"]
        stack2 = pickle.loads(pickle.dumps(stack))
        # values cache and pixels are not pickled
        assert stack2.frames[0]._values == {}
        assert stack2.frames[0]._pixels is None
        assert stack2.frames == stack.frames
        assert stack2["EchoTime"] == stack["EchoTime"]
        assert stack2.frames[0].pixels.shape == stack.frames[0].pixels.shape
