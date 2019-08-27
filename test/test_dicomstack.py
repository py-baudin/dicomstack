# -*- coding: utf-8 -*-
""" test dicomstack """

import os
import argparse
import tempfile
import json
import pytest

import numpy as np
import pydicom

from os.path import dirname, join
from dicomstack import dicomstack

DATA_DIR = join(dirname(dirname(dicomstack.__file__)), "data")


def test_load_dicom_frames():
    """ test dicom loader """
    path = join(DATA_DIR, "avanto_T1w", "file1")
    dataset = pydicom.dcmread(path)
    frames = dicomstack.load_dicom_frames(dataset)
    assert len(frames) == 1
    assert "ImageType" in frames[0]["elements"]
    assert frames[0]["elements"]["ImageType"]["value"][:2] == ("ORIGINAL", "PRIMARY")

    # enhanced-dicom
    path = join(DATA_DIR, "ingenia_multiecho_enhanced", "file2")
    dataset = pydicom.dcmread(path)
    frames = dicomstack.load_dicom_frames(dataset)
    assert len(frames) == 90
    assert all(frame["dataset"] is dataset for frame in frames)
    assert all("EchoTime" in frame["elements"] for frame in frames)


def test_parse_field():
    """ test _parse_field """
    assert dicomstack._parse_field("Name") == ("Name", None)
    assert dicomstack._parse_field("Name_0") == ("Name", 0)

    with pytest.raises(ValueError):
        print(dicomstack._parse_field("Invalid1"))

    with pytest.raises(ValueError):
        print(dicomstack._parse_field("Invalid Two"))

    with pytest.raises(ValueError):
        print(dicomstack._parse_field("Name_"))

    with pytest.raises(ValueError):
        dicomstack._parse_field("Name_a")


def test_get_zip_path():
    """ test get zippat """
    assert dicomstack.get_zip_path(join("some", "simple", "path")) == None
    assert dicomstack.get_zip_path(join("some", "zipped.zip", "path")) == join(
        "some", "zipped.zip"
    )


def test_dicomstack_class(tmpdir):
    """ test dicomstack class """

    # empty path
    stack = dicomstack.DicomStack("unknown")
    assert len(stack) == 0
    assert not stack

    # Avanto T1w

    path = join(DATA_DIR, "avanto_T1w")
    stack = dicomstack.DicomStack(path)

    # check attributes
    assert len(stack) == 1
    assert stack
    assert stack.filenames == [join(path, filename) for filename in os.listdir(path)]
    assert stack.frames == list(stack)

    assert stack[0]["Manufacturer"]["value"] == "SIEMENS"
    assert stack[0]["Manufacturer"]["tag"] == (0x8, 0x70)
    assert stack[0]["Manufacturer"]["name"] == "Manufacturer"
    assert stack[0]["Manufacturer"]["VR"] == "LO"

    # has field
    assert "ImageType" in stack
    assert not "UnknownField" in stack

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
    spacing = stack[0]["PixelSpacing"]["value"]
    origin = tuple(stack[0]["ImagePositionPatient"]["value"])
    assert volume.tags["spacing"] == tuple(spacing + (1,))
    assert volume.tags["transform"] == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert volume.tags["origin"] == origin

    # save
    dir1 = tmpdir.mkdir("dir1")
    stack.save(dir1)

    stack2 = dicomstack.DicomStack(dir1)
    assert stack2[0] == stack[0]


def test_dicomstack_zipped():

    # zipped Signa T1w

    path = join(DATA_DIR, "signa_T1w.zip")
    stack = dicomstack.DicomStack(path)
    assert stack
    assert len(stack) == 1
    assert stack["Manufacturer", "ManufacturerModelName"] == [
        ("GE MEDICAL SYSTEMS", "Signa HDxt")
    ]


def test_dicomstack_multi(tmpdir):
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
    spacing = stack[0]["PixelSpacing"]["value"]
    slice_spacing = stack[0]["SpacingBetweenSlices"]["value"]
    origin = tuple(stack[0]["ImagePositionPatient"]["value"])
    assert np.all(np.isclose(volume.tags["spacing"], spacing + (slice_spacing,)))
    assert volume.tags["origin"] == origin

    # test serialization
    substack = stack(EchoTime=echo_times[:2])
    stack2 = dicomstack.DicomStack(filenames=substack.dicomtree())
    assert stack2
    assert stack2.unique("EchoTime") == echo_times[:2]

    # with a json file
    substack = stack(EchoTime=echo_times[:3])
    with open(tmpdir.join("dicomtree.json"), "w") as f:
        json.dump(substack.dicomtree(), f)

    stack3 = dicomstack.DicomStack(tmpdir.join("dicomtree.json").strpath)
    assert stack3
    assert stack3.unique("EchoTime") == echo_times[:3]
