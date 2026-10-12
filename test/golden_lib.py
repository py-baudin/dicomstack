""" golden snapshot of dicomstack's outputs (shared by make_golden.py and test_golden.py)

Fixtures:
    brain: MRBRAIN.DCM
    legs: MRLEGS.DCM
    legszip: MRLEGS.DCM in legs.zip
    multi: MRMULTI.DCM (extracted from MRMULTI.zip)
    generated: 2 series x 2 echoes x 5 slices built from MRLEGS.DCM with fixed UIDs,
        plus one non-DICOM file
"""

import os
import copy
import gzip
import json
import shutil
import hashlib
import zipfile

import numpy as np
import pydicom

from dicomstack import DicomStack, DICOM
from dicomstack.dicomstack import DicomDataset

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")

FIXTURES = ["brain", "legs", "legszip", "multi", "generated"]

# nested selectors evaluated on every frame
SELECTORS = {
    "ImageType_1": "ImageType_1",
    "DICOM.ImageType[0]": DICOM.ImageType[0],
    "ImagePositionPatient_3": "ImagePositionPatient_3",
    "DICOM.PixelSpacing[1]": DICOM.PixelSpacing[1],
    "IconImageSequence.1.Rows": "IconImageSequence.1.Rows",
    "DICOM.IconImageSequence[0].Columns": DICOM.IconImageSequence[0].Columns,
    "(0x0008, 0x0060)": (0x0008, 0x0060),
    "(0x0002, 0x0010)": (0x0002, 0x0010),
}

# fixed UIDs for the generated stack
UID_ROOT = "1.2.826.0.1.3680043.8.498.4242."


#
# fixtures


def make_fixtures(dest):
    """create all fixture paths in folder dest, return {name: path}"""
    paths = {
        "brain": os.path.join(HERE, "MRBRAIN.DCM"),
        "legs": os.path.join(HERE, "MRLEGS.DCM"),
    }

    # zipped legs
    zdir = os.path.join(dest, "legszip")
    os.makedirs(zdir, exist_ok=True)
    zpath = os.path.join(zdir, "legs.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(paths["legs"], arcname="LEGS.DCM")
    paths["legszip"] = zpath

    # multi-frame
    mdir = os.path.join(dest, "multi")
    with zipfile.ZipFile(os.path.join(HERE, "MRMULTI.zip"), "r") as zf:
        zf.extract("MRMULTI.DCM", path=mdir)
    paths["multi"] = os.path.join(mdir, "MRMULTI.DCM")

    # generated
    paths["generated"] = make_generated(os.path.join(dest, "generated"))
    return paths


def make_generated(dest, nseries=2, nechoes=2, nslices=5):
    """deterministic generated stack"""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    src = pydicom.dcmread(os.path.join(HERE, "MRLEGS.DCM"))
    native = not src.file_meta.TransferSyntaxUID.is_compressed
    if native:
        pixels = src.pixel_array

    index = 0
    for series in range(nseries):
        for echo in range(nechoes):
            for slice in range(nslices):
                index += 1
                ds = copy.deepcopy(src)
                ds.StudyInstanceUID = UID_ROOT + "1"
                ds.StudyID = "1"
                ds.SeriesInstanceUID = UID_ROOT + f"2.{series + 1}"
                ds.SeriesNumber = series + 1
                ds.SeriesDescription = f"series {series + 1}"
                ds.EchoTime = 10.0 * (echo + 1)
                ds.EchoNumbers = echo + 1
                ds.InstanceNumber = echo * nslices + slice + 1
                ds.ImagePositionPatient = [0.0, 0.0, float(slice)]
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.SOPInstanceUID = UID_ROOT + f"3.{index}"
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                if native:
                    # distinct pixel values for each file
                    ds.PixelData = (pixels + index).astype(pixels.dtype).tobytes()
                folder = os.path.join(dest, f"s{series + 1}")
                os.makedirs(folder, exist_ok=True)
                filename = os.path.join(folder, f"IM{echo + 1}_{slice + 1}.dcm")
                ds.save_as(filename, enforce_file_format=True)

    with open(os.path.join(dest, "notes.txt"), "w") as fp:
        fp.write("not a DICOM file")
    return dest


#
# snapshot


def snapshot(paths):
    """return {fixture: snapshot}"""
    return {name: snapshot_stack(DicomStack(path)) for name, path in paths.items()}


def snapshot_stack(stack):
    """json-able snapshot of a stack"""
    snap = {
        "len": len(stack),
        "filenames": [normpath(f) for f in stack.filenames],
        "non_dicom": [normpath(f) for f in stack.non_dicom],
        "describe": normpath(stack.describe()),
        "dicomtree": json.loads(normpath(json.dumps(stack.dicomtree, sort_keys=True))),
        "frames": [snapshot_frame(frame) for frame in stack],
        "as_volume": capture(lambda: encode_volume(stack.as_volume())),
    }

    for by in ["EchoTime", ["SeriesNumber", "EchoTime"]]:
        key = f"as_volume(by={by})"
        snap[key] = capture(lambda: encode_volumes(*stack.as_volume(by=by)))
    return snap


def snapshot_frame(frame):
    """values and types of every element in a frame"""
    dataset = frame.dataset
    snap = {"index": frame.index, "keywords": {}, "tags": {}, "selectors": {}}
    for keyword in dataset.elements_by_keyword:
        snap["keywords"][keyword] = capture(lambda: encode_typed(frame[keyword]))
    for tag, element in dataset.elements_by_tag.items():
        if not element.keyword:
            key = f"{tag.group:04x},{tag.element:04x}"
            snap["tags"][key] = capture(lambda: encode_typed(frame[tuple(tag)]))
    for name, selector in SELECTORS.items():
        snap["selectors"][name] = capture(lambda: encode_typed(frame[selector]))
    return snap


#
# encoding


def capture(func):
    """return func's value or the raised exception's type"""
    try:
        return {"value": func()}
    except Exception as exc:
        return {"error": type(exc).__name__}


def encode_typed(value):
    return {"type": type(value).__name__, "value": encode(value)}


def encode(value):
    """encode value in json, keeping track of types"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, tuple):
        return {"tuple": [encode_typed(v) for v in value]}
    elif isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    elif isinstance(value, list) and all(isinstance(v, DicomDataset) for v in value):
        return {"sequence": [encode_dataset(v) for v in value]}
    raise TypeError(f"Cannot encode value of type {type(value)}")


def encode_dataset(dataset):
    items = {}
    for element in dataset.elements:
        key = element.keyword or f"{element.tag.group:04x},{element.tag.element:04x}"
        items[key] = encode_typed(element.value)
    return items


def encode_array(array):
    array = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "min": float(array.min()),
        "max": float(array.max()),
        "sum": float(array.sum()),
    }


def encode_volume(volume):
    if volume is None:
        return None
    return {
        "array": encode_array(volume),
        "origin": encode_typed(tuple(volume.origin)),
        "spacing": encode_typed(tuple(volume.spacing)),
        "transform": encode_typed(tuple(tuple(ax) for ax in volume.transform)),
    }


def encode_volumes(indices, volumes):
    return {
        "indices": [encode_typed(index) for index in indices],
        "volumes": [encode_volume(volume) for volume in volumes],
    }


def normpath(string):
    return string.replace("\\", "/")


#
# io / comparison


def golden_path(name):
    return os.path.join(GOLDEN_DIR, f"{name}.json.gz")


def save(snapshots):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name, snap in snapshots.items():
        # mtime=0: identical snapshots give identical files
        with open(golden_path(name), "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fp:
                fp.write(json.dumps(snap, indent=0, sort_keys=True).encode("utf-8"))


def load(name):
    with gzip.open(golden_path(name), "rt", encoding="utf-8") as fp:
        return json.load(fp)


def diff(expected, actual, path="", out=None, limit=20):
    """list differences between two json-like objects"""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                out.append(f"{path}/{key}: missing")
            elif key not in expected:
                out.append(f"{path}/{key}: unexpected")
            else:
                diff(expected[key], actual[key], f"{path}/{key}", out, limit)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(expected)} != {len(actual)}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            diff(a, b, f"{path}[{i}]", out, limit)
    elif expected != actual:
        out.append(f"{path}: {expected!r} != {actual!r}")
    return out[:limit]
