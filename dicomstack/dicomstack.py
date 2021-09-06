""" pydicom wrapper class for easy manipulation of DICOM data """
# coding=utf-8

import os
import pathlib
import re
import math
import glob
import datetime
import functools
from io import BytesIO, StringIO
from collections import OrderedDict
import zipfile
import logging
import pydicom

from . import pixeldata
from .query import Selector, Query

LOGGER = logging.getLogger(__name__)

# exceptions
InvalidDicomError = pydicom.errors.InvalidDicomError


class DicomStack(object):
    """load, sort and filter DICOM images"""

    def __init__(self, path=None, filenames=None):
        """path can be:
        * a directory (or a list of),
        * a file (or a list of)
        * a zip file
        """

        self.frames = []
        self.non_dicom = []
        self.root = None

        if not filenames and not path:
            filenames = []

        elif not filenames:
            # search path
            root, filenames = list_files(path)
            self.root = root

        elif not isinstance(filenames, (list, tuple)):
            filenames = [filenames]

        if not filenames:
            # empty stack
            LOGGER.info("New DICOM stack (empty)")
            return

        # load dicom files
        LOGGER.info("New DICOM stack (from %s files)" % len(filenames))
        self._load_files(filenames)

    @classmethod
    def from_frames(cls, frames, root=None):
        """create a new stack from list of frames"""
        LOGGER.debug("New DICOM stack (from %s frames)" % len(frames))
        if not all([isinstance(frame, DicomFrame) for frame in frames]):
            raise TypeError("Invalid DicomFrame in %s" % frames)
        stack = cls.__new__(cls)
        stack.frames = list(frames)
        stack.non_dicom = []
        stack.root = root
        return stack

    def __len__(self):
        """number of DICOM images"""
        return len(self.frames)

    def __bool__(self):
        """return True if stack is not empty"""
        return len(self) > 0

    def __iter__(self):
        """iterates dicom frames"""
        return iter(self.frames)

    def __repr__(self):
        return "DICOM(%d)" % len(self)

    def __contains__(self, field):
        """short for has_field"""
        return self.has_field(field)

    @property
    def filenames(self):
        """return list of dicom files"""
        if not self.root:
            return [frame.dicomfile.filename for frame in self.frames]
        return [
            os.path.relpath(frame.dicomfile.filename, self.root)
            for frame in self.frames
        ]

    @property
    def zipfiles(self):
        """return path to zip files"""
        zipfiles = {get_zip_path(file) for file in self.filenames}
        return list(zipfiles)

    @property
    def dicomtree(self):
        """return DICOM-tree"""
        LOGGER.info("Make dicom tree")

        def _describe(frame):
            filename = frame.dicomfile.filename
            if self.root:
                filename = os.path.relpath(filename, self.root)
            return {
                # UID and number
                "StudyInstanceUID": frame["StudyInstanceUID"],
                "SeriesInstanceUID": frame["SeriesInstanceUID"],
                "StudyID": frame.get("StudyID"),
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
                "filename": filename,
            }

        tree = {}
        if self.non_dicom:
            # put non dicom data in study: None
            tree["0"] = {i: file for i, file in enumerate(self.non_dicom)}

        uidtree = {}
        studies = {}
        for frame in self.frames:
            info = _describe(frame)
            uid = info["StudyInstanceUID"]
            seriesnumber = str(info["SeriesNumber"])

            # study
            if not uid in uidtree:
                studies[uid] = info.get("StudyID")
                uidtree[uid] = {}

            # series
            if not seriesnumber in uidtree[uid]:
                uidtree[uid][seriesnumber] = []
            uidtree[uid][seriesnumber].append(info)

        # fill tree using study ids
        for uid in sorted(studies):
            if studies[uid]:
                id = studies[uid]
            elif not tree:
                id = 1
            else:
                id = max([int(i) for i in tree]) + 1
            tree[str(id)] = uidtree[uid]

        return tree

    def describe(self, **kwargs):
        """method for self description"""
        return describe(self.dicomtree, **kwargs)

    def __call__(self, query=None, **filters):
        """short for filter_by_field or filter_by_query"""
        if query is not None:
            return self.filter_by_query(query)
        return self.filter_by_field(**filters)

    def __getitem__(self, items):
        """short for get_field_values"""
        if not isinstance(items, (tuple, list)):
            items = [items]
        if not all(isinstance(item, (str, Selector)) for item in items):
            raise TypeError(f"Invalid Dicom tag type: {items}")
        return self.get_field_values(*items)

    def __getattr__(self, name):
        """get stack attribute or init selector"""
        if name[0].isalpha() and name[0].upper() == name[0]:
            return self._select(name)
        return getattr(super(), name)

    def single(self, *fields, default=..., precision=None):
        """return single value for field"""
        values = self.get_field_values(*fields, ignore_missing=True)
        if precision is not None:
            try:
                if isinstance(values[0], tuple):
                    # sequence
                    values = [tuple(round(value, precision) for value in seq) for seq in values]
                else:
                    # single value
                    values = [round(item, precision) for item in values]
            except TypeError:
                pass
        values = set(values)

        if len(values) > 1:
            raise ValueError("Multiple values found for %s" % fields)
        elif not values and default is not ...:
            return default
        elif not values:
            raise ValueError("No value found for %s" % fields)
        return values.pop()

    def unique(self, *fields):
        """return unique values for field"""
        return sorted(set(self.get_field_values(*fields, ignore_missing=True)))

    def filter_by_field(self, **filters):
        """return a sub stack with matching values for the given field"""
        LOGGER.debug("Filter by fields: %s" % str(filters))
        frames = self._filter(**filters)
        return self.from_frames(frames, root=self.root)

    def filter_by_query(self, query):
        """return a sub stack with from frames with validated queries"""
        LOGGER.debug("Filter by query: %s" % str(query))
        frames = self._query(query)
        return self.from_frames(frames, root=self.root)

    def get_field_values(self, *fields, ignore_missing=False):
        """return a list a values for the given fields"""
        LOGGER.debug("Get fields' values: %s" % str(fields))
        if ignore_missing:
            # skip values
            frames = self._existing(*fields)
        else:
            frames = self.frames
        if len(fields) == 1:
            fields = fields[0]
        return [frame[fields] for frame in frames]

    def remove_duplicates(self):
        """remove duplicated frames"""
        unique_uids = set()
        uids = self.get_field_values("SOPInstanceUID")
        unique_frames = [
            frame
            for uid, frame in zip(uids, self.frames)
            if not (uid in unique_uids or unique_uids.add(uid))
        ]
        return self.from_frames(unique_frames, root=self.root)

    def sort(self, *fields):
        """reindex database using field values (skip frames with missing values)"""
        LOGGER.debug("Sort by fields: %s" % str(fields))
        frames = self._existing(*fields)
        sorted_frames = sorted(frames, key=lambda f: f.get(*fields))
        return self.from_frames(sorted_frames, root=self.root)

    def getaxis(self):
        """return the image acquisition axis with respect to the patient"""
        orient = self.single("ImageOrientationPatient", precision=3)
        vec = (
            abs(orient[1] * orient[5] - orient[2] * orient[4]),
            abs(orient[0] * orient[5] - orient[2] * orient[3]),
            abs(orient[0] * orient[4] - orient[1] * orient[3]),
        )
        return vec.index(max(vec))

    def reorder(self):
        """reindex stack based on spatial information"""
        if len(self) <= 1:
            return self
        # use ImagePositionPatient
        axis = self.getaxis()
        field = f"ImagePositionPatient_{axis + 1}"
        if len(self.unique(field)) == len(self):
            return self.sort(field)
        else:
            raise ValueError(
                "Could not sort stack spatially: non-unique slice locations"
            )

    def has_field(self, field, how="all"):
        """return True if all frame have the given field"""
        if how == "all":
            func = all
        else:
            fun == any
        try:
            return func(frame.get(field) is not None for frame in self.frames)
        except KeyError:
            return False

    @pixeldata.available
    def as_volume(self, by=None, rescale=True, reorder=True):
        """as volume"""
        LOGGER.debug("Make volume (use fields: %s)" % str(by))

        if len(self) == 0:
            return None

        # else continue
        stack = self.remove_duplicates()

        if not by:
            # single volume

            if reorder:
                # sort by location
                stack = stack.reorder()
            # single non-indexed volume
            return pixeldata.make_volume(stack.frames, rescale=rescale)

        # else: indexed volumes

        # unique values
        if isinstance(by, str):
            by = [by]
        indices = sorted(set(stack.get_field_values(*by, ignore_missing=True)))

        # index values for each volume
        if not indices:
            # empty stack
            return [], []
        elif len(by) == 1:
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

            if reorder:
                # sort by location
                substack = substack.reorder()

            volume = pixeldata.make_volume(substack.frames, rescale=rescale)
            volumes.append(volume)
        return indices, volumes

    def _existing(self, *fields):
        """return frames with existing values"""
        filtered = []
        for frame in self.frames:
            values = frame.get(fields)
            if values is None or None in values:
                # check index exists
                continue
            filtered.append(frame)
        return filtered

    def _filter(self, **filters):
        """return frames filtered by fields"""
        fields = tuple(filters)

        matchlist = []
        for field in fields:
            match = filters[field]
            if not isinstance(match, (list, tuple)):
                matchlist.append([match])
            else:
                matchlist.append(match)

        filtered = []
        for frame in self.frames:
            values = frame.get(fields)
            if values is None:
                continue
            if not all(v in m for v, m in zip(values, matchlist)):
                continue
            filtered.append(frame)
        return filtered

    def _select(self, name):
        """return query Selector"""
        return Selector(name)

    def _query(self, query: Query):
        """return subset of frames based on query object"""
        return [frame for frame in self.frames if query.execute(frame.get)]

    def _load_files(self, filenames):
        """load filenames"""
        for filename in filenames:
            # skip DICOMDIR
            if str(filename).endswith("DICOMDIR"):
                continue
            zip_path = get_zip_path(filename)
            if zip_path:
                self._load_zipfile(filename)
            else:
                self._load_file(filename)

        # duplicates
        nframes = len(self.unique("SOPInstanceUID"))
        if len(self) != nframes:
            LOGGER.warning(f"Duplicated frames were found ({len(self) - nframes})")

    def _load_file(self, filename):
        """load single Dicom file"""
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
        """load files in zipfile"""
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


def describe(dicomtree, regex=None):
    """return string description of dicomtree"""

    info = StringIO()

    # DICOM
    for study_id in sorted(dicomtree):
        if study_id == "0":
            # NON-DICOM
            print("0: (non-DICOM data)", file=info)
            for i, file in dicomtree[study_id].items():
                if not regex or re.search(regex, file):
                    print(f"\t{i}: {file}", file=info)
            continue

        # else: DICOM data
        study = dicomtree[study_id]
        first_series = next(iter(study))
        study_description = study[first_series][0]["StudyDescription"]
        if not study_description:
            study_description = "(no study description)"
        print(f"{study_id}: {study_description}", file=info)
        for seriesnumber in sorted([int(v) for v in study]):
            frames = study[str(seriesnumber)]
            len_series = len(frames)
            series_description = frames[0]["SeriesDescription"]
            if regex and not re.search(regex, series_description):
                # skip
                continue
            desc = f"{series_description} (n={len_series})"
            print(f"\t{seriesnumber}: {desc}", file=info)

    return info.getvalue()


class DicomFile:
    """pickable DICOM file"""

    _dataset = None
    _pixels = None
    _nframe = None  # number of frames

    def __init__(self, filename, bytes=None):
        """init DicomFile object"""
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
        """retrieve pixel data"""
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
        """retrieve DICOM dataset"""
        if self._dataset is None:
            self._load_dataset()
        return self._dataset

    def _load_dataset(self, load_pixels=False):
        """parse DICOM object stored in bytes"""
        dataset = pydicom.dcmread(
            BytesIO(self.bytes), stop_before_pixels=not load_pixels
        )
        self._nframe = get_nframe(dataset)
        self._dataset = dataset

    def get_frames(self):
        """return list of frames"""
        if not self.nframe:
            return [DicomFrame(self)]
        return [DicomFrame(self, index=i) for i in range(self.nframe)]

    def __repr__(self):
        repr = f"DicomFile({self.filename}"
        if self.nframe:
            repr += f" ({self.nframe})"
        return repr + ")"


class DicomFrame:
    """pickable DICOM frame"""

    def __init__(self, dicomfile, elements=None, pixels=None, index=None):
        """
        index: frame index in multi-frame (enhanced) DICOM
        """
        self.dicomfile = dicomfile
        self.index = index
        self._elements = elements
        self._pixels = pixels

    def __repr__(self):
        """represent DICOM frame"""
        if not self._elements:
            return "DICOM frame (pending)"

        repr = "DICOM frame\n"
        for name in self.elements:
            element = self.elements[name]
            repr += f"{str(element):100}\n"
        return repr

    def __getitem__(self, fields):
        """get field(s) value"""

        if isinstance(fields, (str, Selector)):
            field_list = [fields]
        elif isinstance(fields, tuple) and all(
            isinstance(field, (str, Selector)) for field in fields
        ):
            field_list = list(fields)
        else:
            raise ValueError(f"Invalid field value(s): {fields}")

        # return one value per field
        values = [get_element_value(self.elements, field) for field in field_list]

        if isinstance(fields, (str, Selector)):
            return values[0]
        return tuple(values)

    def get(self, fields, default=None):
        try:
            values = self.__getitem__(fields)
        except (KeyError, IndexError):
            return default
        return values

    @property
    def elements(self):
        """retrieve DICOM field values"""
        if self._elements is None:
            # retrieve elements from dataset
            self._elements = parse_dataset(self.dicomfile.dataset, self.index)
        return self._elements

    @property
    def pixels(self):
        """retrieve pixel data"""
        if self._pixels is None:
            # load dataset
            pixels = self.dicomfile.pixels
            if pixels is not None and self.index is not None:
                pixels = pixels[self.index]
            self._pixels = pixels
        return self._pixels


class DicomElement:
    """pickable DICOM element"""

    @property
    def sequence(self):
        return self.VR == "SQ"

    def __init__(self, element):
        """init DICOM element
        TODO: use only properties to save memory?
        """
        self.name = str(element.name)
        self.keyword = str(element.keyword)
        self.tag = (element.tag.group, element.tag.elem)
        self.VR = str(element.VR)
        self.value = parse_element(element)
        self.repr = str(element)

    def __getitem__(self, item):
        """get item if sequence"""
        if not self.sequence:
            raise AttributeError("Cannot get item of non-sequence DICOM element")
        return self.value[item]

    def __repr__(self):
        """represent DICOM element"""
        string = self.repr
        if self.sequence:
            for i, elements in enumerate(self.value):
                string += f"\n  item: {i}"
                for element in elements.values():
                    repr_element = repr(element)
                    string += f"\n    {repr_element}"
        return string


def get_zip_path(path):
    """return the zip-file root of path"""
    path = str(path)
    if not ".zip" in path:
        return None
    return path[: path.find(".zip") + 4]


def list_files(path):
    """list all files in path and its sub folders"""
    files = []
    if "*" in str(path):
        # glob relative paths
        subpaths = list(pathlib.Path().glob(str(path)))
    else:
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        subpaths = [path]

    for subpath in subpaths:
        if subpath.is_file():
            files.append(subpath)
        else:
            files.extend([path for path in subpath.rglob("*") if path.is_file()])
    if not subpaths:
        # empty
        return None, []

    # find root directory
    root = os.path.commonpath(subpaths)
    if os.path.isfile(root):
        root = os.path.dirname(root)
    filenames = [str(file.relative_to(root)) for file in files]
    return root, filenames


def parse_element(element):
    """cast raw value"""

    if element.VR == "SQ":
        return [parse_dataset(d) for d in element]

    elif element.value is None or element.repval == "''":
        return None  # ""

    elif element.VR in ["UI", "SH", "LT", "PN", "UT", "OW"]:
        return str(element.value)

    else:
        # other: string, int or float or date
        return cast(element.value, element.VR)


def parse_dataset(dataset, index=None, flatten=False):
    """parse dataset to retrieve elements"""
    elements = OrderedDict()
    for element in dataset:
        if element.keyword == "PixelData":
            # skip pixel data
            continue

        elif element.keyword == "PerFrameFunctionalGroupsSequence":
            # multi-frame DICOM: flatten sub-dataset
            elements.update(parse_dataset(element[index], flatten=True))

        elif element.keyword == "SharedFunctionalGroupsSequence":
            # multi-frame DICOM: flatten sub-dataset
            for _dataset in element:
                elements.update(parse_dataset(_dataset, flatten=True))

        elif flatten and element.VR == "SQ":
            # flatten  sequence
            for _dataset in element:
                elements.update(parse_dataset(_dataset))  # , flatten=True))
        else:
            # append to elements
            elements[element.keyword] = DicomElement(element)
    return elements


def cast(value, VR):
    """cast DICOM value type"""

    if isinstance(value, pydicom.multival.MultiValue):
        # if value is an array
        return tuple([cast(v, VR) for v in value])

    elif VR == "DA":
        # date
        try:
            return datetime.datetime.strptime(value, "%Y%m%d").date().isoformat()
        except ValueError:
            # invalid date
            return None

    elif VR == "TM":
        # time
        try:
            if "." in value:
                return datetime.datetime.strptime(value, "%H%M%S.%f").time().isoformat()
            else:
                return datetime.datetime.strptime(value, "%H%M%S").time().isoformat()
        except ValueError:
            # invalid time
            return None

    elif isinstance(value, (pydicom.valuerep.DSfloat, float)):
        # if value is defined as float
        if value.is_integer():
            return int(value)
        return value.real

    elif isinstance(value, bytes):
        # if value is defined as bytes
        return value
    # else try force casting to int
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def get_nframe(dataset):
    """return number of frames if several"""
    frames = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if frames:
        return len(frames)
    # else
    return None


def get_element_value(elements, key):
    """get element's value by key"""

    # get element's value recursively
    def _get(elements, keys):
        key = keys[0]
        element = elements[keys[0]]
        if len(keys) > 1:
            return _get(element, keys[1:])
        return element.value

    if isinstance(key, str):
        keys, index = parse_keys(key)
    elif isinstance(key, Selector):
        keys, index = parse_selector(key)
    else:
        raise TypeError(f"Invalid key type: {key}")
    value = _get(elements, keys)

    # return value
    if index is not None:
        return value[index]
    return value


def parse_selector(sel):
    keys = sel.key
    index = None
    if isinstance(keys[-1], int):
        index = keys[-1]
        keys = keys[:-1]
    return keys, index


def parse_keys(string):
    """parse string with optional index suffix
    syntax: "txt" or "txt_num"
    """
    # parse key
    if "_" in string:
        keys, index = string.split("_")
        index = int(index) - 1
    else:
        keys, index = string, None

    def cast_key(value):
        if value.isdigit():
            return int(value) - 1
        elif value.isalnum():
            return value
        else:
            raise ValueError(f"Invalid field value: {value}")

    # split into sub-fields/index
    keys = [cast_key(key) for key in keys.split(".")]
    if isinstance(keys[-1], int):
        raise ValueError(f"Last item in key must not be an int: {keys}")

    return keys, index
