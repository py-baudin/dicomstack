""" generate the benchmark dataset (see PLAN.md, appendix B)

    python benchmarks/make_data.py [dest]   (default: benchmarks/data)

dest/
    stack/sNN/IM{echo}_{slice}.dcm   2,000 single-frame files built from test/MRLEGS.DCM:
                                     1 study, 10 series, 2 echoes, 100 slices
    stack.zip                        same files, ZIP_STORED
    multi/MRMULTI.DCM                multi-frame file (90 frames) from test/MRMULTI.zip
"""

import os
import sys
import copy
import shutil
import zipfile
import pydicom
from pydicom.uid import generate_uid

from dicomstack.utils import save_as

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(os.path.dirname(HERE), "test")

NSERIES = 10
ECHOES = [10.0, 20.0]
NSLICES = 100


def make_stack(dest):
    src = pydicom.dcmread(os.path.join(TEST_DIR, "MRLEGS.DCM"))
    study_uid = generate_uid()
    for series in range(NSERIES):
        series_uid = generate_uid()
        folder = os.path.join(dest, f"s{series + 1:02d}")
        os.makedirs(folder, exist_ok=True)
        for echo, echo_time in enumerate(ECHOES):
            for slice in range(NSLICES):
                ds = copy.deepcopy(src)
                ds.StudyInstanceUID = study_uid
                ds.SeriesInstanceUID = series_uid
                ds.SeriesNumber = series + 1
                ds.EchoTime = echo_time
                ds.InstanceNumber = slice + 1
                ds.ImagePositionPatient = [0, 0, slice]
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.SOPInstanceUID = generate_uid()
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                filename = os.path.join(folder, f"IM{echo + 1}_{slice + 1}.dcm")
                save_as(ds, filename, enforce_file_format=True)
        print(f"series {series + 1}/{NSERIES}")


def make_zip(src, dest):
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(src):
            for file in sorted(files):
                path = os.path.join(root, file)
                zf.write(path, arcname=os.path.relpath(path, os.path.dirname(src)))


def make_multi(dest):
    with zipfile.ZipFile(os.path.join(TEST_DIR, "MRMULTI.zip"), "r") as zf:
        zf.extract("MRMULTI.DCM", path=dest)


def main(dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    make_stack(os.path.join(dest, "stack"))
    make_zip(os.path.join(dest, "stack"), os.path.join(dest, "stack.zip"))
    make_multi(os.path.join(dest, "multi"))
    print(f"data written to: {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data"))
