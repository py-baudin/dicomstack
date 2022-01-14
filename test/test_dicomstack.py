""" test dicomstack """
# -*- coding: utf-8 -*-

import os
import pydicom
import pytest
import pickle
import json
import numpy as np

from os.path import dirname, join
from dicomstack import dicomstack

DATA_DIR = join(dirname(dirname(dicomstack.__file__)), "data")


def test_parse_keys():
    """test _parse_field"""
    assert dicomstack.parse_keys("Field") == ["Field"]
    assert dicomstack.parse_keys("Field_1") == ["Field", 0]
    assert dicomstack.parse_keys("Field.1.Subfield") == ["Field", 0, "Subfield"]

    with pytest.raises(ValueError):
        print(dicomstack.parse_keys("Invalid Two"))

    with pytest.raises(ValueError):
        print(dicomstack.parse_keys("Name_"))

    with pytest.raises(ValueError):
        dicomstack.parse_keys("Name_a")


def test_dicomfile_single(legsfile):
    # single slice DICOM
    path = legsfile
    dcmfile = dicomstack.DicomFile(path)
    assert dcmfile._pixels is None
    assert dcmfile._dataset is None

    # load data and pixels
    assert "ImageType" in dcmfile.dataset
    assert dcmfile.pixels.ndim == 2

    # get frames
    frames = dcmfile.frames
    assert len(frames) == 1

    frame = frames[0]
    assert frame._dataset is None  # elements not loaded yet
    assert frame._pixels is None  # pixels not loaded yet
    assert frame.dataset  # this is a property
    assert frame.pixels.ndim == 2

    # test element
    # single value
    element = frame.dataset["Modality"]
    assert element.name == "Modality"
    assert element.VR == "CS"
    assert element.tag == (0x08, 0x60)
    assert element.value == "MR"

    # multi value
    element = frame.dataset["ImageType"]
    assert isinstance(element.value, tuple)
    assert element.value[0] == "ORIGINAL"

    # sequence
    element = frame.dataset["IconImageSequence"]
    assert element.sequence
    assert isinstance(element.value, list)
    assert isinstance(element.value[0], dicomstack.DicomDataset)
    with pytest.raises(IndexError):
        element.value[1]

    # test frame getitem
    assert frame["Modality"] == "MR"
    assert frame[(0x08, 0x60)] == "MR"
    assert frame[("0x08", "0x60")] == "MR"
    assert frame["Manufacturer"] == "SIEMENS"
    assert frame["ImageType"][0] == "ORIGINAL"
    assert frame["ImageType_1"] == "ORIGINAL"
    assert frame["IconImageSequence.1.Rows"] == 64

    with pytest.raises(KeyError):
        assert frame["UnknownField"]
    assert frame.get("UnknownField") is None

    with pytest.raises(IndexError):
        assert frame["IconImageSequence.10.Rows"]
    assert frame.get("IconImageSequence.10.Rows") is None

    # pickle
    tmp = pickle.dumps(dcmfile)
    assert hash(dcmfile) == hash(pickle.loads(tmp))

    tmp = pickle.dumps(frame)
    assert frame == pickle.loads(tmp)


def test_dicomfile_multi(multi):
    # enhanced DICOM
    path = multi
    dcmfile = dicomstack.DicomFile(path)
    assert dcmfile._pixels is None
    assert dcmfile._dataset is None

    # load data and pixels
    assert dcmfile.nframe == 90
    assert "ImageType" in dcmfile.dataset
    assert dcmfile.pixels.ndim == 3

    # get frames
    frames = dcmfile.frames
    assert len(frames) == 90

    frame = frames[-1]

    assert frame._dataset is None
    assert frame._pixels is None
    assert frame.index == 89
    assert "EchoTime" in frame.dataset
    assert frame.pixels.ndim == 2


def test_get_zip_path():
    """test get zippath"""
    assert dicomstack.get_zip_path(join("some", "simple", "path")) == None
    assert dicomstack.get_zip_path(join("some", "zipped.zip", "path")) == join(
        "some", "zipped.zip"
    )


def test_dicom_tag_class():
    DicomTag = dicomstack.DicomTag
    assert DicomTag(10, 45) == (10, 45)
    assert DicomTag(10, "0x2d") == ("0x0a", 45)
    assert DicomTag(*DicomTag(10, 45)) == (0x0A, 0x2D)


def test_dicomstack_empty(tmpdir):
    """test dicomstack class"""

    # invalid path
    with pytest.raises(FileNotFoundError):
        dicomstack.DicomStack("unknown")

    empty = tmpdir.mkdir("empty")

    # empty path
    stack = dicomstack.DicomStack(empty)
    assert len(stack) == 0
    assert not stack


def test_dicomstack_nondicom(tmpdir):
    """test dicomstack class"""

    # create non dicom data
    file1 = tmpdir.join("file1.txt")
    with open(file1, "w") as fp:
        fp.write("foobar")
    other = tmpdir.mkdir("other")
    file2 = other.join("file2.txt")
    with open(file2, "w") as fp:
        fp.write("foobaz")

    # non dicom data
    stack = dicomstack.DicomStack(tmpdir)
    assert len(stack) == 0
    assert not stack
    assert set(stack.non_dicom) == {"file1.txt", join("other", "file2.txt")}

    # using filenames
    stack = dicomstack.DicomStack(filenames=[file1])
    assert stack.non_dicom == [file1]


def test_dicomstack_single(legsfile):
    """test DicomStack with single file"""
    DicomStack = dicomstack.DicomStack

    path = legsfile
    stack = DicomStack(path)

    # check attributes
    assert len(stack) == 1
    assert stack
    assert stack.filenames == [os.path.basename(path)]
    assert stack.frames == list(stack)

    assert stack.frames[0]["Manufacturer"] == "SIEMENS"
    assert stack.frames[0].dataset["Manufacturer"].tag == (0x08, 0x70)
    assert stack.frames[0].dataset["Manufacturer"].name == "Manufacturer"
    assert stack.frames[0].dataset["Manufacturer"].VR == "LO"

    # get field values
    assert stack.get_field_values("Modality") == ["MR"]
    assert stack["Modality"] == ["MR"]
    assert stack["Manufacturer", "Modality"] == [("SIEMENS", "MR")]
    assert stack["ImageType_1"] == ["ORIGINAL"]
    assert stack["StudyID"] == [None]
    with pytest.raises(KeyError):
        stack["UnknownField"]

    # unique
    assert stack.single("Modality") == "MR"
    assert stack.unique("Modality") == ["MR"]
    assert stack.unique("Manufacturer", "ManufacturerModelName", "Modality") == [
        ("SIEMENS", "Avanto", "MR")
    ]
    assert stack.unique("StudyID") == []
    assert stack.unique("UnknownField") == []

    # filter by fields
    assert not stack.filter_by_field(Modality="FOOBAR")
    filtered = stack(Modality="MR")
    assert filtered
    assert filtered.filenames == stack.filenames

    # convert to volume
    volume = stack.as_volume()
    assert volume.ndim == 3
    assert volume.size > 1
    assert "origin" in volume.tags
    assert "spacing" in volume.tags
    assert "transform" in volume.tags
    spacing = stack.frames[0]["PixelSpacing"]
    origin = tuple(stack.frames[0]["ImagePositionPatient"])
    assert volume.tags["spacing"] == tuple(spacing + (1,))
    assert volume.tags["transform"] == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert volume.tags["origin"] == origin

    # check for duplicates
    with pytest.raises(dicomstack.DuplicatedFramesError):
        stack2 = DicomStack(filenames=[path, path])

    # test remove duplicates
    stack2 = DicomStack.from_frames(stack.frames + stack.frames)
    assert len(stack2) == 2 * len(stack)
    assert stack2.remove_duplicates()["SOPInstanceUID"] == stack["SOPInstanceUID"]

    # from files
    stack_ = DicomStack(filenames=[legsfile])
    assert stack_.root is None
    assert stack_.filenames == [legsfile]
    assert stack_(Modality="MR").filenames == stack_.filenames

    # test pickle
    pi = pickle.dumps(stack)
    stack2 = pickle.loads(pi)
    assert stack2

    # dicomtree
    tree = stack.dicomtree
    # is serializable
    tree2 = json.loads(json.dumps(tree))
    assert tree2 == tree

    # queries
    assert stack[stack.Modality] == ["MR"]
    assert stack[stack.Manufacturer, stack.Modality] == [("SIEMENS", "MR")]
    assert stack[stack.ImageType[0]] == ["ORIGINAL"]
    assert stack[stack.StudyID] == [None]
    with pytest.raises(KeyError):
        stack[stack.UnknownField]

    assert not stack(stack.Modality == "FOOBAR")
    assert stack(stack.Modality == "MR")


def test_dicomstack_zipped(legszip):
    path = legszip
    stack = dicomstack.DicomStack(path)
    assert stack
    assert len(stack) == 1
    assert stack["Manufacturer", "Modality"] == [("SIEMENS", "MR")]
    assert stack.filenames == [join("legs.zip", "LEGS.DCM")]
    assert stack.root == os.path.dirname(legszip)


def test_dicomstack_multi(multi):
    # ingenia Multi echo
    path = multi
    stack = dicomstack.DicomStack(path)
    assert stack
    assert len(stack) == 90
    assert set(stack["EchoNumbers"]) == set(range(1, 18))

    # echo times
    assert stack.has_field("EchoTime")
    assert "EchoTime" in stack

    # multiple echos
    with pytest.raises(ValueError):
        stack.single("EchoTime")

    # the number of unique echo times is 18
    echo_times = stack.unique("EchoTime")
    assert len(echo_times) == 18

    # test filter
    filtered = stack(EchoTime=echo_times[0])
    assert filtered
    assert all(value == echo_times[0] for value in filtered["EchoTime"])

    # test as_volume
    assert stack(UnknownField=1).as_volume() is None
    assert stack(ImageType=1).as_volume() is None
    assert stack.as_volume(by="UnknownField") == ([], [])

    # convert to volumes
    echo_times, volumes = stack.as_volume(by="EchoTime")
    assert len(volumes) == len(echo_times) == 18
    assert set(echo_times) == set(stack.unique("EchoTime"))
    assert all(volume.shape == volumes[0].shape for volume in volumes[1:])
    assert all(dim > 1 for dim in volumes[0].shape)

    volume = volumes[0]
    spacing = stack.frames[0]["PixelSpacing"]
    slice_spacing = stack.frames[0]["SpacingBetweenSlices"]
    origin = tuple(stack.frames[0]["ImagePositionPatient"])
    assert np.all(np.isclose(volume.tags["spacing"], spacing + (slice_spacing,)))
    assert volume.tags["origin"] == origin

    # queries
    filtered = stack(stack.EchoTime == echo_times[0])
    assert filtered
    assert all(value == echo_times[0] for value in filtered[stack.EchoTime])

    assert not stack(stack.EchoTime < 0)
    assert stack(stack.EchoTime < 10)
    assert stack(stack.EchoTime.is_in(echo_times[:2]))
    assert stack((stack.EchoTime < 10) | (stack.EchoTime > 30))
    assert not stack((stack.EchoTime < 10) & (stack.EchoTime > 30))
