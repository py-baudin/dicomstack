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

""" TODO
    synthetic dummy data
"""


def test_dicomfile_single():
    # single slice DICOM
    path = join(DATA_DIR, "avanto_T1w", "file1")
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
    assert frame._elements is None
    assert frame._pixels is None
    assert frame.elements
    assert frame.pixels.ndim == 2

    # test element
    element = frame.elements["ManufacturerModelName"]
    assert element.name == "Manufacturer's Model Name"
    assert element.VR == "LO"
    assert element.tag == (0x0008, 0x1090)
    assert element.get() == "Avanto"

    element = frame.elements["ImageType"]
    assert element.get()[0] == "ORIGINAL"
    assert element.get(0) == "ORIGINAL"

    # test frame.get
    assert frame.get("Manufacturer") == "SIEMENS"
    assert frame.get("Manufacturer", "ManufacturerModelName") == ("SIEMENS", "Avanto")
    assert frame.get("ImageType")[0] == "ORIGINAL"
    assert frame.get("ImageType_0") == "ORIGINAL"


def test_dicomfile_multi():
    # enhanced DICOM
    path = join(DATA_DIR, "ingenia_multiecho_enhanced", "file2")
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


def test_parse_field():
    """ test _parse_field """
    assert dicomstack.parse_field("Name") == ("Name", None)
    assert dicomstack.parse_field("Name_0") == ("Name", 0)

    with pytest.raises(ValueError):
        print(dicomstack.parse_field("Invalid1"))

    with pytest.raises(ValueError):
        print(dicomstack.parse_field("Invalid Two"))

    with pytest.raises(ValueError):
        print(dicomstack.parse_field("Name_"))

    with pytest.raises(ValueError):
        dicomstack.parse_field("Name_a")


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


def test_dicomstack_single():
    """ test DicomStack with single file """
    # Avanto T1w
    path = join(DATA_DIR, "avanto_T1w")
    stack = dicomstack.DicomStack(path)

    # check attributes
    assert len(stack) == 1
    assert stack
    assert all(os.path.isfile(filename) for filename in stack.filenames)
    assert stack.frames == list(stack)

    assert stack.frames[0]["Manufacturer"] == "SIEMENS"
    assert stack.frames[0].elements["Manufacturer"].tag == (0x8, 0x70)
    assert stack.frames[0].elements["Manufacturer"].name == "Manufacturer"
    assert stack.frames[0].elements["Manufacturer"].VR == "LO"

    # get field values
    assert stack.get_field_values("MagneticFieldStrength") == [1.5]
    assert stack["MagneticFieldStrength"] == [1.5]
    assert stack["Manufacturer", "ManufacturerModelName"] == [("SIEMENS", "Avanto")]
    assert stack["ImageType_0"] == ["ORIGINAL"]

    # unique
    assert stack.single("Manufacturer") == "SIEMENS"
    assert stack.unique("Manufacturer") == ["SIEMENS"]

    # filter by fields
    assert not stack.filter_by_field(MagneticFieldStrength=3)
    assert stack(MagneticFieldStrength=1.5)

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

    # test pickle
    pi = pickle.dumps(stack)
    stack2 = pickle.loads(pi)
    assert stack2

    # dicomtree
    tree = stack.dicomtree
    # is serializable
    tree2 = json.loads(json.dumps(tree))
    assert tree2 == tree


def test_dicomstack_zipped():
    # zipped Signa T1w
    path = join(DATA_DIR, "signa_T1w.zip")
    stack = dicomstack.DicomStack(path)
    assert stack
    assert len(stack) == 1
    assert stack["Manufacturer", "ManufacturerModelName"] == [
        ("GE MEDICAL SYSTEMS", "Signa HDxt")
    ]


def test_dicomstack_multi():
    # ingenia Multi echo
    path = join(DATA_DIR, "ingenia_multiecho_enhanced")
    stack = dicomstack.DicomStack(path)
    assert stack
    assert len(stack) == 90
    assert set(stack["EchoNumbers"]) == set(range(1, 18))

    # multiple echos
    with pytest.raises(ValueError):
        stack.single("EchoTime")

    # convert to volumes
    echo_times, volumes = stack.as_volume(by="EchoTime")
    assert echo_times == stack.unique("EchoTime")
    assert len(echo_times) == 18
    assert set(echo_times) == set(stack["EchoTime"])
    assert all(volume.shape == volumes[0].shape for volume in volumes[1:])
    assert all(dim > 1 for dim in volumes[0].shape)

    volume = volumes[0]
    spacing = stack.frames[0]["PixelSpacing"]
    slice_spacing = stack.frames[0]["SpacingBetweenSlices"]
    origin = tuple(stack.frames[0]["ImagePositionPatient"])
    assert np.all(np.isclose(volume.tags["spacing"], spacing + (slice_spacing,)))
    assert volume.tags["origin"] == origin
