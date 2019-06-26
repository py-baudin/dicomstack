# coding=utf-8
""" pydicom wrapper class for simple manipulation of DICOM data """

import os
import math
import json
import glob
import functools
import zipfile
import struct
import logging
from io import BytesIO
from operator import mul
from functools import reduce
from collections import defaultdict

import pydicom
import tinydb

try:
    import numpy

    HAS_NUMPY = True
except ImportError:
    # no pixel array support
    HAS_NUMPY = False


LOGGER = logging.getLogger(__name__)


class DicomStack(object):
    """ load, sort and filter DICOM images
    """

    def __init__(self, path=None, filenames=None):
        """ path can be:
            * a directory (or a list of),
            * a file (or a list of)
            * a zip file
            * a dicomtree JSON file
        """
        self.db = tinydb.TinyDB(storage=tinydb.storages.MemoryStorage)
        self.non_dicom = []

        if not filenames and not path:
            # empty stack
            return

        if not filenames:
            # search path
            pathes = glob.glob(path)
            filenames = []
            for path in pathes:
                if os.path.isdir(path):
                    filenames.extend(list_files(path))
                else:
                    filenames.append(path)

        elif isinstance(filenames, str):
            filenames = [filenames]

        # load dicom files
        self.load_files(filenames)

    @property
    def elements(self):
        return self.db.all()

    @property
    def filenames(self):
        return [element["filename"] for element in self.elements]

    def __len__(self):
        """ number of DICOM images """
        return len(self.db)

    def __bool__(self):
        """ return True if stack is not empty """
        return len(self) > 0

    def __contains__(self, field):
        """ test if field is present """
        return self.has_field(field)

    def __iter__(self):
        """ iterates dicom elements """
        return iter(self.db)

    def __repr__(self):
        return "DICOM(%d)" % len(self.db)

    def __getitem__(self, fields):
        """ short for get_field_values or _index"""
        if isinstance(fields, int):
            # return item #i
            return self._index(fields)
        elif not isinstance(fields, tuple):
            fields = (fields,)
        return self.get_field_values(*fields)

    def __call__(self, **filters):
        """ shortcut for filter_by_field """
        return self.filter_by_field(**filters)

    def single(self, *fields):
        """ return single value for field """
        values = list(set(self.get_field_values(*fields)))
        if len(values) > 1:
            raise ValueError("Multiple values found for %s" % fields)
        elif not values:
            raise ValueError("No value found for %s" % fields)
        return values[0]

    def unique(self, *fields):
        """ return unique values for field """
        return sorted(set(self.get_field_values(*fields)))

    @classmethod
    def from_elements(cls, elements):
        """ create a new stack from a db object """
        if not all(
            [isinstance(element, tinydb.database.Document) for element in elements]
        ):
            raise TypeError
        stack = cls()
        stack.db.insert_multiple(elements)
        return stack

    def filter_by_field(self, **filters):
        """ return a sub stack with matching values for the given field """
        elements = self._filter(**filters)
        return self.from_elements(elements)

    def get_field_values(self, *fields):
        """ return a list a values for the given fields """
        elements = self._existing(*fields)
        return [_get_values(element, fields=fields) for element in elements]

    def sort(self, *fields):
        """ reindex database using field values """
        elements = self._existing(*fields)
        sort_key = functools.partial(_get_values, fields=fields)
        sorted_elements = sorted(elements, key=sort_key)
        for i, element in enumerate(sorted_elements):
            element["index"] = i
        return self.from_elements(sorted_elements)

    def as_volume(self, by=None, rescale=True):
        """ as volume """
        if not HAS_NUMPY:
            raise ImportError("numpy required")

        # sort by position
        if self.has_field("InStackPositionNumber"):
            stack = self.sort("InStackPositionNumber")
        elif self.has_field("SliceLocation"):
            stack = self.sort("SliceLocation")
        else:
            raise NotImplementedError("Could not defined sorting method")

        if not by:
            # single non-indexed volume
            return _make_volume(stack.elements, rescale=rescale)

        # else: indexed volumes

        # unique values
        if isinstance(by, str):
            by = [by]
        indices = sorted(set(stack.get_field_values(*by)))

        # index values for each volume
        if len(by) == 1:
            filters = [{by[0]: value} for value in indices]
        else:
            filters = [
                dict((field, value) for field, value in zip(by, values))
                for values in indices
            ]

        # create volumes
        volumes = []
        for filter in filters:
            substack = stack.filter_by_field(**filter)
            volumes.append(_make_volume(substack.elements, rescale=rescale))
        return indices, volumes

    def load_files(self, filenames):
        """ load filenames """

        if isinstance(filenames, dict):
            filters = filenames
        else:
            filenames = sorted(set(filenames))
            filters = {}

        for filename in filenames:

            # check if file is zipped
            zip_path = get_zip_path(filename)
            if zip_path:
                self.load_zipfile(filename, filters.get(filename))
                continue

            try:
                # read dicom object
                dicom_obj = pydicom.dcmread(filename)

            except (IOError, pydicom.errors.InvalidDicomError):
                try:
                    # load json dicomtree
                    with open(filename, "r") as f:
                        dicomtree = json.load(f)
                    self.load_files(dicomtree)
                except:
                    # other files
                    self.non_dicom.append(filename)
                continue

            # retrieve DICOM data
            self._load_frames(filename, dicom_obj, filters.get(filename))

    def load_zipfile(self, filename, filters=None):
        """ load files in zipfile"""
        zip_path = get_zip_path(filename)
        assert zipfile.is_zipfile(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for zfile in zf.filelist:

                if zfile.filename.endswith("/"):
                    # is a directory: skip
                    continue
                zfilename = os.path.normpath(zfile.filename)
                full_zfilename = os.path.join(zip_path, zfilename)
                if not filename in full_zfilename:
                    continue

                # read file
                rawfile = zf.read(zfile)
                dicom_bytes = BytesIO(rawfile)

                try:
                    # read dicom object
                    dicom_obj = pydicom.read_file(dicom_bytes)
                except IOError:
                    self.non_dicom.append(zfilename)

                # retrieve DICOM data
                self._load_frames(full_zfilename, dicom_obj, filters)

    def _load_frames(self, filename, dicom_obj, filters=None):
        """ retrieve and store DICOM data in dicom_obj"""
        frames = load_dicom_frames(dicom_obj)

        for frame in frames:
            if filters:
                uid = [f["instance_uid"] for f in filters]
                if not frame["SOPInstanceUID"]["value"] in uid:
                    continue

            index = len(self.db)
            frame["filename"] = os.path.normpath(filename)
            frame["index"] = index
            self.db.insert(frame)

    def has_field(self, field):
        """ check whether a field is present """
        query = tinydb.Query()
        cond = query[field].exists()
        return bool(self.db.search(cond))

    def _existing(self, *items):
        """ return elements with existing values"""
        query = tinydb.Query()
        condition = None
        for item in items:
            if not isinstance(item, str):
                raise TypeError

            elif isinstance(item, str):
                # fields
                field, index = _parse_field(item)
                if index:
                    cond = query[field].value[index].exists()
                else:
                    cond = query[field].value.exists()
            else:
                raise TypeErrorload_files
            condition = cond if not condition else (cond & condition)
        return self.db.search(condition)

    def _filter(self, **filters):
        """ return elements filtered by fields """
        query = tinydb.Query()
        condition = None
        for name, value in filters.items():
            field, index = _parse_field(name)
            cond = query[field].value

            if index is not None:
                cond = cond[index]
            if not isinstance(value, list):
                value = [value]
            cond = cond.one_of(value)
            condition = cond if condition is None else (cond & condition)

        return self.db.search(condition)

    def _index(self, index):
        """ return element matching index """
        query = tinydb.Query()
        return self.db.get(query.index == index)

    def dicomtree(self, root="."):
        """ return serializable summary of DicomStack """
        dicomtree = defaultdict(list)

        def get(element, field):
            field = element.get(field)
            if not field:
                return None
            return field["value"]

        for element in self.elements:
            filename = os.path.relpath(element["filename"], start=root)
            dicomtree[filename].append(
                {
                    "study_id": get(element, "StudyID"),
                    "study_description": get(element, "StudyDescription"),
                    "study_uid": element["StudyInstanceUID"]["value"],
                    "series_number": get(element, "SeriesNumber"),
                    "series_description": get(element, "SeriesDescription"),
                    "series_uid": element["SeriesInstanceUID"]["value"],
                    "instance_uid": element["SOPInstanceUID"]["value"],
                }
            )
        # add non dicom:
        for filename in self.non_dicom:
            dicomtree["filename"].append({})

        return dicomtree


def get_zip_path(path):
    """ return the zip-file root of path """
    if not ".zip" in path:
        return None
    return path[: path.find(".zip") + 4]


def _parse_field(string):
    """ parse string with optional index suffix
        syntax: "txt" or "txt_num"
    """
    split = string.split("_")
    try:
        field = split[0]
        assert field.isalpha()
    except AssertionError:
        raise ValueError('Invalid field name in: "%s"' % string)

    try:
        index = None if len(split) == 1 else int(split[1])
    except TypeError:
        raise ValueError('Cannot parse index in: "%s"' % string)

    return field, index


def _get_values(element, fields):
    """ get DICOM element value """
    values = []
    for name in fields:
        field, index = _parse_field(name)
        if index is not None:
            values.append(element[field]["value"][index])
            continue
        values.append(element[field]["value"])
    if len(fields) == 1:
        return values[0]
    return tuple(values)


if HAS_NUMPY:

    class DicomVolume(numpy.ndarray):
        """ simple layer over numpy ndarray to add attribute: volume.info
        """

        def __new__(cls, input_array, tags=None):
            """ create Volume object """
            # copy the data
            obj = numpy.asarray(input_array).view(cls)
            obj.tags = tags
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self.tags = getattr(obj, "tags", {})

        def __array_wrap__(self, out_arr, context=None):
            """ propagate metadata if wrap is called """
            return super().__array_wrap__(self, out_arr, context)

        def __array_wrap__(self, out_arr, context=None):
            if self.shape != out_arr.shape:
                # if not same shape: drop metadata
                return out_arr
            # else wrap out_array
            return numpy.ndarray.__array_wrap__(self, out_arr, context)

    def _make_volume(frames, rescale=True):
        """ return volume from a sequence of frames"""

        # find geometry
        nframe = len(frames)
        first = frames[0]
        last = frames[-1]

        origin = first["ImagePositionPatient"]["value"]
        end = last["ImagePositionPatient"]["value"]
        ax1 = tuple(first["ImageOrientationPatient"]["value"][:3])
        ax2 = tuple(first["ImageOrientationPatient"]["value"][3:])
        vec3 = [b - a for a, b in zip(origin, end)]
        norm3 = math.sqrt(sum(value ** 2 for value in vec3))
        if nframe == 1:
            ax3 = (
                ax1[1] * ax2[2] - ax1[2] * ax2[1],
                ax1[2] * ax2[0] - ax1[0] * ax2[2],
                ax1[0] * ax2[1] - ax1[1] * ax2[0],
            )
            spacing3 = 1
        else:
            ax3 = tuple(value / norm3 for value in vec3)
            spacing3 = norm3 / (nframe - 1)
        transform = (ax1, ax2, ax3)
        spacing = first["PixelSpacing"]["value"] + (spacing3,)

        tags = {
            "origin": tuple(origin),
            "spacing": tuple(spacing),
            "transform": tuple(transform),
        }

        # make volume
        slices = []
        for frame in frames:
            slope, intercept = 1, 0
            pixels = frame["pixels"]
            if rescale:
                slope = frame.get("RescaleSlope", {}).get("value", 1)
                intercept = frame.get("RescaleSlope", {}).get("value", 0)
                pixels = pixels * slope + intercept
            slices.append(pixels)
        return DicomVolume(slices, tags).T


def list_files(dirpath):
    """ list all files in dirpath and its sub folders """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(dirpath):
        all_files.extend([os.path.join(dirpath, name) for name in filenames])
    return all_files


def load_dicom_frames(dataset):
    """ load all dicom fields in dataset"""

    def _list_dicom_elements(dataset, root=False):
        elements, sequences = [], []
        frame_items = None
        for item in dataset:
            if item.keyword == "PixelData":
                # skip
                continue
            elif item.keyword == "PerFrameFunctionalGroupsSequence":
                # multi-frame DICOM
                frame_items = item
            elif item.VR == "SQ":
                # other sequences
                sequences.append(item)
            else:
                elements.append(item)

        # flatten sequences
        for sequence in sequences:
            for item in sequence:
                elements.extend(_list_dicom_elements(item))

        if not root:
            return elements

        # else: solve frames (root level only)
        if not frame_items:
            return [elements]
        frames = []
        for item in frame_items:
            frame = list(elements)  # copy elements
            frame.extend(_list_dicom_elements(item))
            frames.append(frame)
        return frames

    raw_frames = _list_dicom_elements(dataset, root=True)

    def _get_element_info(element):
        def _cast_element_value(value):
            if isinstance(value, pydicom.multival.MultiValue):
                # if value is an array
                return tuple([_cast_element_value(v) for v in value])
            elif isinstance(value, pydicom.valuerep.DSfloat):
                # if value is defined as float
                if value.is_integer():
                    return int(value)
                return value.real
            # else try force casting to int
            try:
                return int(value)
            except (TypeError, ValueError):
                return str(value)

        return {
            "name": str(element.name),
            "tag": (element.tag.group, element.tag.elem),
            "value": _cast_element_value(element.value),
            "VR": str(element.VR),
        }

    # pixel data
    pixel_array = None
    if HAS_NUMPY and "PixelData" in dataset:
        shape = dataset.pixel_array.shape
        # add 1st dimension if needed
        pixel_array = dataset.pixel_array.reshape((-1, shape[-2], shape[-1]))

    # put frame into dicts
    frames = []
    for i, items in enumerate(raw_frames):
        frame = dict((element.keyword, _get_element_info(element)) for element in items)
        frame["pixels"] = pixel_array[i]
        frames.append(frame)
    return frames


# class PixelArray:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.index = None
#     def __getitem__(self, key):
#         """ return obj with """
#          pa = PixelArray(self.dataset)
#          pa.index = key
#          return pa
#     @property
#     def data(self):
#         shape = self.dataset.pixel_array.shape
#         return self.dataset.pixel_array.reshape((-1, shape[-2], shape[-1]))
