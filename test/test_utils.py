""" test DICOM utils """
# coding=utf-8

import os
import numpy as np
import pydicom
import pytest

from os.path import join, dirname
from dicomstack import utils, dicomstack


def test_make_dataset():
    """test creating dataset"""
    data = np.arange(3 * 4, dtype="uint8").reshape((3, 4))
    ds = utils.make_dataset(data, PatientName="Foo Bar")
    assert ds.PatientName == "Foo Bar"

    with pytest.raises(ValueError):
        utils.make_dataset(data, BadTag="Foo Bar")


def test_write_dataset(tmpdir):
    """test writing dicom file"""
    filename = tmpdir.join("test.dcm")
    data = np.arange(3 * 4, dtype="uint8").reshape((3, 4))
    utils.write_dataset(data, filename)

    # load for testing
    ds = pydicom.dcmread(str(filename))
    assert ds.PatientName == "Anonymous"
    assert np.all(ds.pixel_array == data)


def test_update_dataset(brain, tmpdir):
    # test copy brain image
    data = brain.pixel_array[:20, :20]
    filename = tmpdir.join("test.dcm")
    utils.write_dataset(data, filename, dataset=brain, PatientName="Foo^Bar")

    # check values
    ds = pydicom.dcmread(str(filename))
    assert ds.StudyDescription == "BRAIN"  # original tag
    assert ds.PatientName == "Foo^Bar"  # updated tag
    assert np.all(ds.pixel_array == data)


def test_anonymize_file(tmpdir, legsfile):
    path = legsfile

    dest = tmpdir.mkdir("test")
    utils.export_file(path, dest / "MRLEGS.DCM", mapper={"PatientName": "Anonymous"})
    assert "MRLEGS.DCM" in os.listdir(dest)
    stack = dicomstack.DicomStack(dest)
    assert len(stack) == 1
    assert stack.single("PatientName") == "Anonymous"
