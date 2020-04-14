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
    """ test _parse_field """
    assert dicomstack.parse_keys("Field") == (["Field"], None)
    assert dicomstack.parse_keys("Field_1") == (["Field"], 0)
    assert dicomstack.parse_keys("Field.1.Subfield") == (["Field", 0, "Subfield"], None)

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
    frames = dcmfile.get_frames()
    assert len(frames) == 1

    frame = frames[0]
    assert frame._elements is None  # elements not loaded yet
    assert frame._pixels is None  # pixels not loaded yet
    assert frame.elements  # this is a property
    assert frame.pixels.ndim == 2

    # test element
    # single value
    element = frame.elements["Modality"]
    assert element.name == "Modality"
    assert element.VR == "CS"
    assert element.tag == (0x0008, 0x0060)
    assert element.value == "MR"

    # multi value
    element = frame.elements["ImageType"]
    assert isinstance(element.value, tuple)
    assert element.value[0] == "ORIGINAL"

    # sequence
    element = frame.elements["IconImageSequence"]
    assert isinstance(element.value, list)
    assert element[0]
    with pytest.raises(IndexError):
        element[1]
    assert isinstance(element.value[0], dicomstack.OrderedDict)

    # test frame getitem
    assert frame["Modality"] == "MR"
    assert frame["Manufacturer"] == "SIEMENS"
    assert frame[("Modality", "Manufacturer")] == ("MR", "SIEMENS")
    assert frame["ImageType"][0] == "ORIGINAL"
    assert frame["ImageType_1"] == "ORIGINAL"
    assert frame["IconImageSequence.1.Rows"] == 64

    with pytest.raises(KeyError):
        assert frame["UnknownField"]
    assert frame.get("UnknownField") is None

    with pytest.raises(IndexError):
        assert frame["IconImageSequence.10.Rows"]
    assert frame.get("IconImageSequence.10.Rows") is None


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
    frames = dcmfile.get_frames()
    assert len(frames) == 90

    frame = frames[-1]

    assert frame._elements is None
    assert frame._pixels is None
    assert frame.index == 89
    assert "EchoTime" in frame.elements
    assert frame.pixels.ndim == 2


def test_get_zip_path():
    """ test get zippath """
    assert dicomstack.get_zip_path(join("some", "simple", "path")) == None
    assert dicomstack.get_zip_path(join("some", "zipped.zip", "path")) == join(
        "some", "zipped.zip"
    )


def test_dicomstack_empty():
    """ test dicomstack class """

    # empty path
    stack = dicomstack.DicomStack("unknown")
    assert len(stack) == 0
    assert not stack


def test_dicomstack_nondicom(tmpdir):
    """ test dicomstack class """

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
    assert set(stack.non_dicom) == {"file1.txt", "other/file2.txt"}

    # using filenames
    stack = dicomstack.DicomStack(filenames=[file1])
    assert stack.non_dicom == [file1]


def test_dicomstack_single(legsfile):
    """ test DicomStack with single file """

    path = legsfile
    stack = dicomstack.DicomStack(path)

    # check attributes
    assert len(stack) == 1
    assert stack
    assert stack.filenames == [os.path.basename(path)]
    assert stack.frames == list(stack)

    assert stack.frames[0]["Manufacturer"] == "SIEMENS"
    assert stack.frames[0].elements["Manufacturer"].tag == (0x8, 0x70)
    assert stack.frames[0].elements["Manufacturer"].name == "Manufacturer"
    assert stack.frames[0].elements["Manufacturer"].VR == "LO"

    # get field values
    assert stack.get_field_values("Modality") == ["MR"]
    assert stack["Modality"] == ["MR"]
    assert stack["Manufacturer", "Modality"] == [("SIEMENS", "MR")]
    assert stack["ImageType_1"] == ["ORIGINAL"]

    # unique
    assert stack.single("Modality") == "MR"
    assert stack.unique("Modality") == ["MR"]

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

    # from files
    stack_ = dicomstack.DicomStack(filenames=[legsfile])
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


def test_dicomstack_zipped(legszip):
    path = legszip
    stack = dicomstack.DicomStack(path)
    assert stack
    assert len(stack) == 1
    assert stack["Manufacturer", "Modality"] == [("SIEMENS", "MR")]
    assert stack.filenames == ["legs.zip/LEGS.DCM"]
    assert stack.root == os.path.dirname(legszip)


def test_dicomstack_multi(multi):
    # ingenia Multi echo
    path = multi
    stack = dicomstack.DicomStack(path)
    assert stack
    assert len(stack) == 90
    assert set(stack["EchoNumbers"]) == set(range(1, 18))

    # multiple echos
    with pytest.raises(ValueError):
        stack.single("EchoTime")

    # some of the frames have no echo times
    assert None in stack["EchoTime"]
    assert len(set(stack["EchoTime"])) == 18

    # the number of unique echo times is 17
    echo_times = stack.unique("EchoTime")
    assert len(echo_times) == 17

    # test filter
    filtered = stack(EchoTime=echo_times[0])
    assert filtered
    assert all(value == echo_times[0] for value in filtered["EchoTime"])

    # convert to volumes
    echo_times, volumes = stack.as_volume(by="EchoTime")
    assert len(volumes) == len(echo_times) == 17
    assert set(echo_times) == set(stack.unique("EchoTime"))
    assert all(volume.shape == volumes[0].shape for volume in volumes[1:])
    assert all(dim > 1 for dim in volumes[0].shape)

    volume = volumes[0]
    spacing = stack.frames[0]["PixelSpacing"]
    slice_spacing = stack.frames[0]["SpacingBetweenSlices"]
    origin = tuple(stack.frames[0]["ImagePositionPatient"])
    assert np.all(np.isclose(volume.tags["spacing"], spacing + (slice_spacing,)))
    assert volume.tags["origin"] == origin
