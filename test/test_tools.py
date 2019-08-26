""" test DICOM tools """
# coding=utf-8

import os
import pydicom
import pytest

from os.path import join, dirname
import dicomstack
from dicomstack import tools


DATA_DIR = join(dirname(dirname(dicomstack.__file__)), "data")


def test_anonymize_file(tmpdir):
    path = join(DATA_DIR, "avanto_T1w", "file1")

    dest = tmpdir.mkdir("test")
    tools.anonymize_file(path, dest)
    assert "file1" in os.listdir(dest)
    stack = dicomstack.DicomStack(dest)
    assert len(stack) == 1
    assert stack.unique("PatientName") == "Anonymous"

    tools.anonymize_file(path, dest, filename="test.dcm")
    assert "test.dcm" in os.listdir(dest)
    stack = dicomstack.DicomStack(dest)
    assert len(stack) == 2
    assert stack.unique("PatientName") == "Anonymous"


def test_anonymize_stack(tmpdir):
    path = DATA_DIR
    dest = tmpdir.mkdir("test1")

    tools.anonymize_stack(path, dest)
    assert "avanto_T1w" in os.listdir(dest)
    assert "file1" in os.listdir(dest.join("avanto_T1w"))
    assert "file2" in os.listdir(dest.join("ingenia_multiecho_enhanced"))

    dest = tmpdir.mkdir("test2")
    tools.anonymize_stack(join(path, "avanto_T1w"), dest, prefix="foobar_")
    assert "foobar_1" in os.listdir(dest)
