""" pydicom wrapper class for easy manipulation of DICOM data """
# coding=utf-8

import os
import math
import glob
import datetime
import functools
from io import BytesIO
from collections import OrderedDict
import zipfile
import logging
import pydicom

from . import pixeldata

LOGGER = logging.getLogger(__name__)

# exceptions
InvalidDicomError = pydicom.errors.InvalidDicomError


class DicomStack(object):
    """ load, sort and filter DICOM images
    """

    def __init__(self, path=None, filenames=None):
        """ path can be:
            * a directory (or a list of),
            * a file (or a list of)
            * a zip file
        """

        self.frames = []
        self.non_dicom = []
        self.root = None

        if not filenames and not path:
            # empty stack
            LOGGER.info("New DICOM stack (empty)")
            return

        elif not filenames:
            # search path
            root, filenames = list_files(path)
            self.root = root

        elif not isinstance(filenames, (list, tuple)):
            filenames = [filenames]

        # load dicom files
        LOGGER.info("New DICOM stack (from %s files)" % len(filenames))
        self._load_files(filenames)

    @classmethod
    def from_frames(cls, frames, root=None):
        """ create a new stack from list of frames """
        LOGGER.debug("New DICOM stack (from %s frames)" % len(frames))
        if not all([isinstance(frame, DicomFrame) for frame in frames]):
            raise TypeError
        stack = cls.__new__(cls)
        stack.frames = list(frames)
        stack.non_dicom = []
        stack.root = root
        return stack

    def __len__(self):
        """ number of DICOM images """
        return len(self.frames)

    def __bool__(self):
        """ return True if stack is not empty """
        return len(self) > 0

    def __iter__(self):
        """ iterates dicom frames """
        return iter(self.frames)

    def __repr__(self):
        return "DICOM(%d)" % len(self)

    @property
    def filenames(self):
        """ return list of dicom files """
        if not self.root:
            return [frame.dicomfile.filename for frame in self.frames]
        return [
            os.path.relpath(frame.dicomfile.filename, self.root)
            for frame in self.frames
        ]

    @property
    def dicomtree(self):
        LOGGER.info("Make dicom tree")

        def _describe(frame):
            return {
                # UID and number
                "StudyInstanceUID": frame["StudyInstanceUID"],
                "SeriesInstanceUID": frame["SeriesInstanceUID"],
                "StudyID": frame["StudyID"],
                "SeriesNumber": frame["SeriesNumber"],
                # dates and time
                "StudyDate": frame["StudyDate"],
                "StudyTime": frame["StudyTime"],
                # patient
                "PatientID": frame["PatientID"],
                # description
                "StudyDescription": frame.get("StudyDescription"),
                "SeriesDescription": frame.get("SeriesDescription"),
                # file
                "filename": frame.dicomfile.filename,
                "pixel_data": frame.dicomfile.pixels is not None,
            }

        return {
            "DICOM": [_describe(frame) for frame in self.frames],
            "NON_DICOM": self.non_dicom,
        }

    def __call__(self, **filters):
        """ short for filter_by_field """
        return self.filter_by_field(**filters)

    def __getitem__(self, items):
        """ short for get_field_values"""
        if not isinstance(items, (tuple, list)):
            items = [items]
        return self.get_field_values(*items)

    def single(self, *fields, default=...):
        """ return single value for field """
        values = list(set(self.get_field_values(*fields)))
        if len(values) > 1:
            raise ValueError("Multiple values found for %s" % fields)
        elif not values and default is not ...:
            return default
        elif not values:
            raise ValueError("No value found for %s" % fields)
        return values[0]

    def unique(self, *fields):
        """ return unique values for field """
        return sorted(set(self.get_field_values(*fields)))

    def filter_by_field(self, **filters):
        """ return a sub stack with matching values for the given field """
        LOGGER.debug("Filter by fields: %s" % str(filters))
        frames = self._filter(**filters)
        return self.from_frames(frames, root=self.root)

    def get_field_values(self, *fields):
        """ return a list a values for the given fields """
        LOGGER.debug("Get fields' values: %s" % str(fields))
        frames = self._existing(*fields)
        return [frame.get(*fields) for frame in frames]

    def sort(self, *fields):
        """ reindex database using field values """
        LOGGER.debug("Sort by fields: %s" % str(fields))
        frames = self._existing(*fields)
        sorted_frames = sorted(frames, key=lambda f: f.get(*fields))
        return self.from_frames(sorted_frames, root=self.root)

    def has_field(self, field):
        """ return True if all frame have the given field """
        return all(frame.get(field) is not None for frame in self.frames)

    @pixeldata.available
    def as_volume(self, by=None, rescale=True):
        """ as volume """
        LOGGER.debug("Make volume (use fields: %s)" % str(by))

        # sort by position
        if len(self) == 0:
            return None
        elif len(self) == 1:
            stack = self
        elif self.has_field("InStackPositionNumber"):
            stack = self.sort("InStackPositionNumber")
        elif self.has_field("SliceLocation"):
            stack = self.sort("SliceLocation")
        elif self.has_field("AcquisitionNumber"):
            stack = self.sort("AcquisitionNumber")
        else:
            raise NotImplementedError("Could not defined sorting method")

        if not by:
            # single non-indexed volume
            return pixeldata.make_volume(stack.frames, rescale=rescale)

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
            volume = pixeldata.make_volume(substack.frames, rescale=rescale)
            volumes.append(volume)
        return indices, volumes

    def _existing(self, *fields):
        """ return frames with existing values"""
        filtered = []
        for frame in self.frames:
            if frame.get(*fields) is None:
                # check index exists
                continue
            filtered.append(frame)
        return filtered

    def _filter(self, **filters):
        """ return frames filtered by fields """
        fields = list(filters)

        matchlist = []
        for field in fields:
            match = filters[field]
            if not isinstance(match, list):
                matchlist.append([match])
            else:
                matchlist.append(match)

        filtered = []
        for frame in self.frames:
            values = frame.get(*filters)
            if values is None:
                continue
            elif len(filters) == 1:
                values = [values]
            if not all(v in m for v, m in zip(values, matchlist)):
                continue
            filtered.append(frame)
        return filtered

    def _load_files(self, filenames):
        """ load filenames """
        for filename in filenames:
            zip_path = get_zip_path(filename)
            if zip_path:
                self._load_zipfile(filename)
            else:
                self._load_file(filename)

    def _load_file(self, filename):
        """ load single Dicom file """
        root = self.root
        if root:
            filepath = os.path.join(root, filename)
        else:
            filepath = str(filename)

        try:
            # read dicom object
            dicomfile = DicomFile(filepath)
            frames = dicomfile.get_frames()
        except (IOError, InvalidDicomError):
            # other files
            self.non_dicom.append(filename)
        else:
            self.frames.extend(frames)

    def _load_zipfile(self, filename):
        """ load files in zipfile"""
        root = self.root
        if root:
            filepath = os.path.join(root, filename)
        else:
            filepath = str(filename)

        zip_path = get_zip_path(filepath)
        if not zipfile.is_zipfile(zip_path):
            self.non_dicom.append(filename)

        with zipfile.ZipFile(zip_path, "r") as zf:
            for zfile in zf.filelist:

                if zfile.filename.endswith("/"):
                    # is a directory: skip
                    continue
                zfilename = os.path.normpath(zfile.filename)
                full_zfilename = os.path.join(zip_path, zfilename)
                if not filename in full_zfilename:
                    continue

                try:
                    # read dicom object
                    dicomfile = DicomFile(full_zfilename, bytes=zf.read(zfile))
                    frames = dicomfile.get_frames()
                except (IOError, InvalidDicomError):
                    # other files
                    self.non_dicom.append(full_zfilename)
                else:
                    self.frames.extend(frames)


class DicomFile:
    """ pickable DICOM file """

    _dataset = None
    _pixels = None
    _nframe = None  # number of frames

    def __init__(self, filename, bytes=None):
        """ init DicomFile object """
        if not bytes:
            # load filename
            if not pydicom.misc.is_dicom(filename):
                raise InvalidDicomError("Invalid DICOM file: %s" % filename)
            with open(filename, "rb") as fp:
                bytes = fp.read()

        self.bytes = bytes
        self.filename = filename

    # pickle
    def __getstate__(self):
        return (self.filename, self.bytes)

    def __setstate__(self, state):
        self.filename, self.bytes = state

    @property
    def pixels(self):
        """ retrieve pixel data """
        if self._pixels is None:
            if not "PixelData" in self.dataset:
                self._load_dataset(load_pixels=True)
            self._pixels = self.dataset.pixel_array
        return self._pixels

    @property
    def nframe(self):
        if not self._nframe:
            self._load_dataset()
        return self._nframe

    @property
    def dataset(self):
        """ retrieve DICOM dataset """
        if self._dataset is None:
            self._load_dataset()
        return self._dataset

    def _load_dataset(self, load_pixels=False):
        """ parse DICOM object stored in bytes """
        dataset = pydicom.dcmread(
            BytesIO(self.bytes), stop_before_pixels=not load_pixels
        )
        self._nframe = get_nframe(dataset)
        self._dataset = dataset

    def get_frames(self):
        """ return list of frames """
        if not self.nframe:
            return [DicomFrame(self)]
        return [DicomFrame(self, index=i) for i in range(self.nframe)]

    def __repr__(self):
        repr = f"DicomFile({self.filename}"
        if self.nframe:
            repr += f" ({self.nframe})"
        return repr + ")"


class DicomFrame:
    """ pickable DICOM frame """

    def __init__(self, dicomfile, elements=None, pixels=None, index=None):
        """
            index: frame index in multi-frame (enhanced) DICOM
        """
        self.dicomfile = dicomfile
        self.index = index
        self._elements = elements
        self._pixels = pixels

    def __repr__(self):
        """ represent DICOM frame """
        if not self._elements:
            return "DICOM frame (pending)"

        repr = "DICOM frame\n"
        for name in self.elements:
            element = self.elements[name]
            repr += f"{str(element):100}\n"
        return repr

    def get(self, *fields, default=None):
        """ get values of corresponding elements """
        values = []
        for field in fields:
            # parse field into name, index
            name, index = parse_field(field)

            element = self.elements.get(name)
            if element is None:
                return default
            value = element.get(index)
            if value is None:
                return default
            values.append(value)

        if len(fields) == 1:
            return values[0]
        return tuple(values)

    def __getitem__(self, field):
        """ get field value """
        value = self.get(field)
        if value is None:
            raise KeyError("Invalid field: %s" % field)
        return value

    @property
    def elements(self):
        """ retrieve DICOM field values """
        if self._elements is None:
            # retrieve elements from dataset
            elements = parse_dataset(self.dicomfile.dataset, self.index)
            self._elements = OrderedDict((e.keyword, e) for e in elements)
        return self._elements

    @property
    def pixels(self):
        """ retrieve pixel data """
        if self._pixels is None:
            # load dataset
            pixels = self.dicomfile.pixels
            if pixels is not None and self.index is not None:
                pixels = pixels[self.index]
            self._pixels = pixels
        return self._pixels


class DicomElement:
    """ pickable DICOM element """

    def __init__(self, element):
        """ init DICOM element """
        self.name = str(element.name)
        self.keyword = str(element.keyword)
        self.value = parse_element(element)
        self.tag = (element.tag.group, element.tag.elem)
        self.VR = str(element.VR)
        self.repr = str(element)

    @property
    def sequence(self):
        return self.VR == "SQ"

    def get(self, index=None, default=None):
        """ return value or value[index] """
        if index is None:
            return self.value

        elif not isinstance(self.value, tuple):
            return default
        try:
            return self.value[index]
        except KeyError:
            return default

    def __getitem__(self, index):
        """ get field value """
        value = self.get(index)
        if value is None:
            raise KeyError("Invalid index: %s" % index)
        return value

    def __repr__(self):
        """ represent DICOM element """
        string = self.repr
        if self.sequence:
            for element in self.value:
                repr_element = repr(element)
                string += f"\n  {repr_element}"
        return string


def get_zip_path(path):
    """ return the zip-file root of path """
    path = str(path)
    if not ".zip" in path:
        return None
    return path[: path.find(".zip") + 4]


def list_files(path):
    """ list all files in path and its sub folders """
    if os.path.isfile(path):
        return os.path.dirname(path), [os.path.basename(path)]
    # pathes = glob.glob(str(path))
    filenames = []
    # walk path
    for root, _, files in os.walk(path):
        root = os.path.relpath(root, path)
        if root == ".":
            root = ""
        filenames.extend([os.path.join(root, name) for name in files])
    return path, filenames


def parse_field(string):
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


def parse_element(element):
    """ cast raw value """
    if not element.value:
        return element.value

    elif element.VR == "SQ":
        sequence = []
        for d in element:
            sequence.extend(parse_dataset(d))
        return sequence

    elif element.VR in ["UI", "SH", "LT", "PN", "UT", "OW"]:
        return str(element.value)

    elif element.VR == "DA":
        # date
        fmt = "%Y%m%d"
        return datetime.datetime.strptime(element.value, fmt).date().isoformat()

    elif element.VR == "TM":
        # time
        if "." in element.value:
            fmt = "%H%M%S.%f"
        else:
            fmt = "%H%M%S"
        return datetime.datetime.strptime(element.value, fmt).time().isoformat()
    else:
        # other: string, int or float
        return cast(element.value)


def cast(value):
    """ cast DICOM value type """

    if isinstance(value, pydicom.multival.MultiValue):
        # if value is an array
        return tuple([cast(v) for v in value])

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


def parse_dataset(dataset, index=None, flatten=False):
    """ parse dataset to retrieve elements """
    elements = []
    for element in dataset:
        if element.keyword == "PixelData":
            # skip pixel data
            continue
        elif element.keyword == "PerFrameFunctionalGroupsSequence":
            # multi-frame DICOM
            elements.extend(parse_dataset(element[index], flatten=True))
        elif flatten and element.VR == "SQ":
            # extend elements
            for _dataset in element:
                elements.extend(parse_dataset(_dataset, flatten=True))
        else:
            # append to elements
            elements.append(DicomElement(element))
    return elements


def get_nframe(dataset):
    """ return number of frames if several """
    frames = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if frames:
        return len(frames)
    # else
    return None
