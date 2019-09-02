""" test DICOM utils """
# coding=utf-8

import os
import numpy as np
import pydicom
import pytest

from os.path import join, dirname
from dicomstack import utils, dicomstack


def test_write_dataset(tmpdir):
    """ test writing dicom file """
    filename = tmpdir.join("test.dcm")
    data = np.arange(3 * 4, dtype="uint8").reshape((3, 4))
    utils.write_dataset(data, filename)

    # load for testing
    dcm = pydicom.dcmread(str(filename))
    assert dcm.PatientName == "Anonymous"
    assert np.all(dcm.pixel_array == data)


def test_update_dataset(brain, tmpdir):
    # test copy brain image
    data = brain.pixel_array[:20, :20]
    filename = tmpdir.join("test.dcm")
    utils.write_dataset(data, filename, dataset=brain, PatientName="Foo^Bar")

    # check values
    dcm = pydicom.dcmread(str(filename))
    assert dcm.StudyDescription == "BRAIN"  # original tag
    assert dcm.PatientName == "Foo^Bar"  # updated tag
    assert np.all(dcm.pixel_array == data)


def test_anonymize_file(tmpdir, legsfile):
    path = legsfile

    dest = tmpdir.mkdir("test")
    utils.anonymize_file(path, dest)
    assert "MRLEGS.DCM" in os.listdir(dest)
    stack = dicomstack.DicomStack(dest)
    assert len(stack) == 1
    assert stack.single("PatientName") == "Anonymous"

    utils.anonymize_file(path, dest, filename="test.dcm")
    assert "test.dcm" in os.listdir(dest)
    stack = dicomstack.DicomStack(dest)
    assert len(stack) == 2
    assert stack.single("PatientName") == "Anonymous"


# to fix
# def test_anonymize_stack(tmpdir, legsfile):
#     path = os.path.dirname(legsfile)
#     dest = tmpdir.mkdir("test1")
#
#     utils.anonymize_stack(path, dest)
#     assert "avanto_T1w" in os.listdir(dest)
#     assert "file1" in os.listdir(dest.join("avanto_T1w"))
#     assert "file2" in os.listdir(dest.join("ingenia_multiecho_enhanced"))
#
#     dest = tmpdir.mkdir("test2")
#     utils.anonymize_stack(join(path, "avanto_T1w"), dest, prefix="foobar_")
#     assert "foobar_1" in os.listdir(dest)
