""" the Python examples of README.md must run

Each ```python block is executed in order, in a shared namespace, in a folder containing:
    data/series1/IM{1..4}.dcm         T1 axial, 4 slices (SeriesNumber 1)
    data/series2/IM{echo}_{1..4}.dcm  T2 multi-echo, 2 echoes x 4 slices (SeriesNumber 2)
    data/notes.txt                    not DICOM
    data.zip                          series1/ and series2/ zipped
    multiframe.dcm                    enhanced (multi-frame) MR, 90 frames
Blocks preceded by `<!-- skip-test -->` are not executed.
"""

import os
import re
import copy
import zipfile

import pydicom
import pytest
from pydicom.uid import generate_uid

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(os.path.dirname(HERE), "README.md")


def make_readme_data(dest):
    """example data of the README"""
    src = pydicom.dcmread(os.path.join(HERE, "MRLEGS.DCM"))
    study = generate_uid()
    series = [
        ("series1", 1, "T1 axial", [5.0]),
        ("series2", 2, "T2 multi-echo", [10.0, 20.0]),
    ]
    for folder, number, description, echo_times in series:
        series_uid = generate_uid()
        for echo, echo_time in enumerate(echo_times, start=1):
            for k in range(1, 5):
                ds = copy.deepcopy(src)
                ds.PatientName = "Doe^John"
                ds.PatientID = "P0001"
                ds.StudyInstanceUID = study
                ds.StudyID = "1"
                ds.StudyDescription = "Legs"
                ds.SeriesInstanceUID = series_uid
                ds.SeriesNumber = number
                ds.SeriesDescription = description
                ds.EchoTime = echo_time
                ds.InstanceNumber = (echo - 1) * 4 + k
                ds.ImagePositionPatient = [-190.0, -142.5, 2.0 * k]
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.SOPInstanceUID = generate_uid()
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                name = f"IM{k}.dcm" if len(echo_times) == 1 else f"IM{echo}_{k}.dcm"
                path = os.path.join(dest, "data", folder, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                ds.save_as(path, enforce_file_format=True)
    with open(os.path.join(dest, "data", "notes.txt"), "w") as fp:
        fp.write("not a DICOM file")

    with zipfile.ZipFile(os.path.join(dest, "data.zip"), "w") as zf:
        for root, _, files in os.walk(os.path.join(dest, "data")):
            for file in files:
                if file.endswith(".dcm"):
                    path = os.path.join(root, file)
                    zf.write(path, os.path.relpath(path, os.path.join(dest, "data")))

    with zipfile.ZipFile(os.path.join(HERE, "MRMULTI.zip")) as zf:
        with open(os.path.join(dest, "multiframe.dcm"), "wb") as fp:
            fp.write(zf.read("MRMULTI.DCM"))


def python_blocks(text):
    """(line number, code) of the README's python blocks, except skipped ones"""
    blocks = []
    pattern = re.compile(r"(<!-- skip-test -->\s*)?```python\n(.*?)```", re.S)
    for match in pattern.finditer(text):
        if match.group(1) is None:
            line = text[: match.start(2)].count("\n") + 1
            blocks.append((line, match.group(2)))
    return blocks


def test_readme(tmp_path, monkeypatch):
    with open(README, encoding="utf-8") as fp:
        blocks = python_blocks(fp.read())
    assert len(blocks) > 10
    make_readme_data(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for line, code in blocks:
        try:
            exec(compile(code, f"README.md:{line}", "exec"), namespace)
        except Exception as exc:
            pytest.fail(f"README.md, block at line {line}: {type(exc).__name__}: {exc}")
