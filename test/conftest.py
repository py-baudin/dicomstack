""" pytest configuration """
# coding=utf-8

import os
import zipfile
import pytest
import pydicom


@pytest.fixture(scope="session")
def brainfile():
    """ brain filename """
    return os.path.join(os.path.dirname(__file__), "MRBRAIN.DCM")


@pytest.fixture()
def brain(brainfile):
    """ load brain dataset """
    return pydicom.dcmread(brainfile)


@pytest.fixture()
def legsfile():
    """ brain filename """
    return os.path.join(os.path.dirname(__file__), "MRLEGS.DCM")


@pytest.fixture()
def legs(legsfile):
    """ load brain dataset """
    return pydicom.dcmread(brainfile)


@pytest.fixture()
def legszip(legsfile, tmpdir):
    dest = tmpdir.join("legs.zip")
    with zipfile.ZipFile(dest, "w") as zf:
        zf.write(legsfile, arcname="LEGS.DCM")
    return dest


@pytest.fixture()
def multizip():
    return os.path.join(os.path.dirname(__file__), "MRMULTI.zip")


@pytest.fixture()
def multi(multizip, tmpdir):
    with zipfile.ZipFile(multizip, "r") as zf:
        zf.extract("MRMULTI.DCM", path=tmpdir)
    return tmpdir.join("MRMULTI.DCM")
