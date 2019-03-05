# -*- coding: utf-8 -*-
""" test dicomstack """

import os
import argparse
import tempfile
import pytest

import numpy as np
import pydicom

from os.path import dirname, join
from dicomstack import dicomstack

DATA_DIR = join(dirname(dirname(dicomstack.__file__)), "data")


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


# def test_load_dicom_frames():
#     """ test dicom loader """
#     path = join(DATA_DIR, "ingenia_multiecho_enhanced", "IM_0016")
#     dicom_obj = pydicom.dcmread(path)
#     dicomdict = dicomstack.load_dicom_dataset(dicom_obj)
#     1/0


def test_dicomstack_class():
    """ test dicomstack class """

    # empty path
    stack = dicomstack.DICOMStack("unknown")
    assert len(stack) == 0
    assert not stack

    # Avanto T1w

    path = join(DATA_DIR, "avanto_T1w")
    stack = dicomstack.DICOMStack(path)

    # check attributes
    assert len(stack) == 1
    assert stack
    assert stack.filenames == [join(path, filename) for filename in os.listdir(path)]
    assert stack.elements == list(stack)

    # has field
    assert "ImageType" in stack
    assert not "UnknownField" in stack

    # get field values
    assert stack.get_field_values("MagneticFieldStrength") == [1.5]
    assert stack["MagneticFieldStrength"] == [1.5]
    assert stack["Manufacturer", "ManufacturerModelName"] == [("SIEMENS", "Avanto")]
    assert stack["ImageType_0"] == ["ORIGINAL"]

    # filter by fields
    assert not stack.filter_by_field(MagneticFieldStrength=3)
    assert stack(MagneticFieldStrength=1.5)

    # convert to volume
    volume = stack.as_volume()
    ndarray = np.asarray(volume)
    assert ndarray.ndim == 3
    assert ndarray.size > 1

    # zipped Signa T1w

    path = join(DATA_DIR, "signa_T1w.zip")
    stack = dicomstack.DICOMStack(path)
    assert stack
    assert len(stack) == 1
    assert stack["Manufacturer", "ManufacturerModelName"] == [
        ("GE MEDICAL SYSTEMS", "Signa HDxt")
    ]

    # ingenia Multi echo

    path = join(DATA_DIR, "ingenia_multiecho_enhanced")
    stack = dicomstack.DICOMStack(path)
    assert stack
    assert len(stack) == 90
    assert set(stack["EchoNumbers"]) == set(range(1, 18))

    # convert to volumes
    echo_times, volumes = stack.as_volume(by="EchoTime")
    ndarrays = [np.asarray(vol) for vol in volumes]
    assert len(echo_times) == 18
    assert set(echo_times) == set(stack["EchoTime"])
    assert all(volume.shape == volumes[0].shape for volume in ndarrays[1:])
    assert all(dim > 1 for dim in ndarrays[0].shape)
