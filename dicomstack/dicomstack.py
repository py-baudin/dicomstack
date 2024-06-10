""" pydicom wrapper class for easy manipulation of DICOM data """
# coding=utf-8

import os
import pathlib
import re
import datetime
import itertools
from io import BytesIO, StringIO
from collections import OrderedDict
import zipfile
import logging
import warnings
import pydicom
from pydicom import config

from . import pixeldata
from .query import Selector, Query

LOGGER = logging.getLogger(__name__)

# exceptions
InvalidDicomError = pydicom.errors.InvalidDicomError


class DuplicatedFramesError(Exception):
    """raise when duplicated frames are found"""


class DicomStack(object):
    """load, sort and filter DICOM images"""

    def __init__(self, path=None, filenames=None, *, duplicates="remove"):
        """path can be:
        * a directory (or a list of),
        * a file (or a list of)
        * a zip file

        duplicates: 'remove', 'error', 'ignore'
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
        self._load_files(filenames, duplicates)

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
                "StudyInstanceUID": frame.get("StudyInstanceUID", "UNKNOWN"),
                "SeriesInstanceUID": frame.get("SeriesInstanceUID", "UNKNOWN"),
                "StudyID": frame.get("StudyID", -1),
                "SeriesNumber": frame.get("SeriesNumber", -1),
                # dates and time
                "StudyDate": frame.get("StudyDate"),
                "StudyTime": frame.get("StudyTime"),
                # patient
                "PatientID": frame.get("PatientID"),
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
        if DicomTag.is_tag(items) or not isinstance(items, (tuple, list)):
            items = [items]
        return self.get_field_values(*items)

    def __getattr__(self, name):
        """get stack attribute or init selector"""
        if name[0].isalpha() and name[0].upper() == name[0]:
            return self.select(name)
        return getattr(super(), name)

    def single(self, *fields, default=..., precision=None):
        """return single value for field"""
        values = self.get_field_values(*fields, ignore_missing=True)
        if not values and default is not ...:
            return default

        if precision is not None:
            try:
                if isinstance(values[0], tuple):
                    # sequence
                    values = [
                        tuple(round(value, precision) for value in seq)
                        for seq in values
                    ]
                else:
                    # single value
                    values = [round(item, precision) for item in values]
            except TypeError:
                pass
        values = set(values)

        if len(values) > 1:
            raise ValueError("Multiple values found for %s" % fields)
        elif not values:
            raise ValueError("No value found for %s" % fields)
        return values.pop()

    def unique(self, *fields):
        """return unique values for field"""
        return sorted(set(self.get_field_values(*fields, ignore_missing=True)))

    def filter_by_field(self, **filters):
        """return a sub stack with matching values for the given field"""
        LOGGER.debug("Filter by fields: %s" % str(filters))
        frames = self._filter(filters)
        return self.from_frames(frames, root=self.root)

    def filter_by_query(self, query):
        """return a sub stack with from frames with validated queries"""
        LOGGER.debug("Filter by query: %s" % str(query))
        frames = self._query(query)
        return self.from_frames(frames, root=self.root)

    def get_field_values(self, field, *fields, ignore_missing=False):
        """return a list a values for the given fields"""
        _fields = (field,) + fields
        LOGGER.debug("Get fields' values: %s" % str(_fields))

        if ignore_missing:
            # skip missing values
            frames = self._existing(_fields)
        else:
            frames = self.frames

        if not fields:
            # single field
            return [frame[field] for frame in frames]

        # multiple fields
        return [tuple(frame[field] for field in _fields) for frame in frames]

    def remove_duplicates(self, *, keys=['SOPInstanceUID', 'ImagePositionPatient'], inplace=False, warn=False):
        """remove duplicated frames (incl. from different files)

            default keys: use both uid and position for uniqueness
        """
        indices = set()
        files = set()
        frames = []
        for frame in self.frames:
            index = tuple(frame[key] for key in keys)
            if index in indices:  # and not frame.dicomfile in files:
                continue  # skip
            elif not index in indices:
                indices.add(index)
                files.add(frame.dicomfile)
            frames.append(frame)
            continue

        diff = len(self) - len(frames)
        if warn and diff > 0:
            warnings.warn(f'Removed {diff} duplicated frames.')

        if inplace:
            self.frames = frames
            return self
        else:
            return self.from_frames(frames, root=self.root)

    def sort(self, *fields):
        """reindex database using field values (skip frames with missing values)"""
        LOGGER.debug("Sort by fields: %s" % str(fields))
        frames = self._existing(fields)
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
        # field = f"ImagePositionPatient_{axis + 1}"
        field = Selector("ImagePositionPatient")[axis]
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
            func = any
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
        # stack = self.remove_duplicates()

        if not by:
            # single volume
            stack = self.remove_duplicates(keys=['ImagePositionPatient'], warn=True)

            if reorder:
                # sort by location
                stack = stack.reorder()
            # single non-indexed volume
            return pixeldata.make_volume(stack.frames, rescale=rescale)

        # else: indexed volumes

        # unique values
        if isinstance(by, str):
            by = [by]
        indices = sorted(set(self.get_field_values(*by, ignore_missing=True)))

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
            # substack = stack.filter_by_field(**filter)
            substack = self.filter_by_field(**filter)
            substack = substack.remove_duplicates(keys=['ImagePositionPatient'], warn=True)

            if reorder:
                # sort by location
                substack = substack.reorder()

            volume = pixeldata.make_volume(substack.frames, rescale=rescale)
            volumes.append(volume)
        return indices, volumes

    def _existing(self, fields):
        """return frames with existing values"""
        filtered = []
        for frame in self.frames:
            if all([frame.get(field) is not None for field in fields]):
                # all fields exist
                filtered.append(frame)
        return filtered

    def _filter(self, filters):
        """return frames filtered by fields"""
        # field names
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
            try:
                values = [frame[field] for field in fields]
            except KeyError:
                continue
            if not all(v in m for v, m in zip(values, matchlist)):
                continue
            filtered.append(frame)
        return filtered

    def select(self, sel):
        """return query Selector"""
        if DicomTag.is_tag(sel):
            return Selector(DicomTag(*sel))
        return Selector(sel)

    def _query(self, query: Query):
        """return subset of frames based on query object"""
        return [frame for frame in self.frames if query.execute(frame.get)]

    def _load_files(self, filenames, duplicates):
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
        unique_frames = set(self.frames)
        nframes = len(unique_frames)
        if len(self) != nframes:
            msg = f"{len(self) - nframes} duplicate frames were found"
            if duplicates == "error":
                raise DuplicatedFramesError(msg)
            elif duplicates == "ignore":
                LOGGER.warning(msg + " (ignoring)")
            elif duplicates == "remove":
                LOGGER.warning(msg + " (removing)")
                duplicated = set()
                self.frames = [
                    frame
                    for frame in self.frames
                    if not (frame in duplicated or duplicated.add(frame))
                ]

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
            frames = dicomfile.frames
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
                elif "DICOMDIR" in zfile.filename:
                    # skip DICOMDIR for now
                    continue
                zfilename = os.path.normpath(zfile.filename)
                full_zfilename = os.path.join(zip_path, zfilename)
                if not filename in full_zfilename:
                    continue

                try:
                    # read dicom object
                    dicomfile = DicomFile(full_zfilename, bytes=zf.read(zfile))
                    frames = dicomfile.frames
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
    _frames = None

    def __init__(self, filename, bytes=None):
        """init DicomFile object"""
        logging.debug(f"DicomFile {filename}: init")
        if not bytes:
            # load filename
            if not pydicom.misc.is_dicom(filename):
                raise InvalidDicomError("Invalid DICOM file: %s" % filename)
            with open(filename, "rb") as fp:
                bytes = fp.read()

        self.bytes = bytes
        self.filename = filename

    def __hash__(self):
        # return hash(self.bytes[:1000])
        return hash(self.bytes)

    # pickle
    def __getstate__(self):
        return (self.filename, self.bytes)

    def __setstate__(self, state):
        self.filename, self.bytes = state

    def __repr__(self):
        repr = f"DicomFile({self.filename}"
        if self._dataset:
            nframe = get_nframe(self._dataset)
            if nframe is not None:
                repr += f" [{nframe}]"
        else:
            repr += " <not loaded>"
        return repr + ")"

    @property
    def dataset(self):
        """retrieve DICOM dataset"""
        if self._dataset is None:
            self._load_dataset()
        return self._dataset

    @property
    def nframe(self):
        """number of frames in multi-frame DICOM"""
        return get_nframe(self.dataset)

    @property
    def frames(self):
        """return list of frames"""
        if self._frames is None:
            nframe = get_nframe(self.dataset)
            if nframe is None:
                self._frames = [DicomFrame(self)]
            else:
                self._frames = [DicomFrame(self, index=i) for i in range(nframe)]
        return self._frames

    @property
    def pixels(self):
        """retrieve pixel data"""
        if self._pixels is None:
            if not "PixelData" in self.dataset:
                # TODO read only Pixels ? cf specific_tags in dcmread
                self._load_dataset(load_pixels=True)
            self._pixels = self.dataset.pixel_array
        return self._pixels

    def _load_dataset(self, load_pixels=False):
        """parse DICOM object stored in bytes"""
        logging.debug(f"DicomFile {self.filename}: load dataset")

        with config.strict_reading():
            try:
                dataset = pydicom.dcmread(
                    BytesIO(self.bytes), stop_before_pixels=not load_pixels
                )
            except EOFError as exc:
                warnings.warn(str(exc))
                delimiter = b'\xfe\xff\xdd\xe0' # sequence delimiter
                dataset = pydicom.dcmread(
                    BytesIO(self.bytes + delimiter), stop_before_pixels=not load_pixels
                )
        self._dataset = dataset


class DicomDataset:
    """collection of DICOM elements"""

    def __init__(self, elements):
        """elements is a list of DicomElement"""
        logging.debug(f"DicomDataset {len(elements)}: init")
        self.elements = elements
        self.elements_by_tag = {elmt.tag: elmt for elmt in self.elements}
        self.elements_by_keyword = {
            elmt.keyword: elmt for elmt in self.elements if elmt.keyword
        }

    @classmethod
    def from_dataset(cls, dataset, index=None):
        elements = parse_dataset(dataset, index)
        return cls(elements)

    def __repr__(self):
        # TODO: add nesting indentation
        repr = ""
        for element in self.elements:
            repr += f"{str(element):100}"
            if element is not self.elements[-1]:
                repr += "\n"
        return repr

    def __contains__(self, key):
        if isinstance(key, DicomTag):
            # get value by tag
            return key in self.elements_by_tag
        elif isinstance(key, str):
            # get value by keyword
            return key in self.elements_by_keyword
        else:
            raise TypeError(f"Invalid key type: {key}")

    def get(self, field, default=None):
        """get element"""
        try:
            return self.__getitem__(field)
        except (KeyError, IndexError):
            return default

    def __getitem__(self, key):
        """get element"""
        item = None

        if DicomTag.is_tag(key):
            # get value by tag
            tag = DicomTag(*key)
            element = self.elements_by_tag[tag]

        elif isinstance(key, str):
            # get value by keyword
            element = self.elements_by_keyword[key]
        else:
            raise TypeError(f"Invalid key type: {key}")

        # return value
        return element


class DicomFrame:
    """pickable DICOM frame"""

    def __init__(self, dicomfile, pixels=None, index=None):
        """
        index: frame index in multi-frame (enhanced DICOM)
        """
        logging.debug(f"DicomFrame {dicomfile.filename} ({index}): init")
        self.dicomfile = dicomfile
        self.index = index
        self._dataset = None
        self._pixels = pixels

    def __hash__(self):
        return hash(self.dicomfile) ^ hash(self.index)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        """represent DICOM frame"""
        if not self._dataset:
            return "DICOM frame (pending)"
        return "DICOM frame\n" + str(self._dataset)

    @property
    def dataset(self):
        if not self._dataset:
            # load dataset
            logging.debug(
                f"DicomFrame {self.dicomfile.filename} ({self.index}): parse dataset"
            )
            self._dataset = DicomDataset.from_dataset(
                self.dicomfile.dataset, index=self.index
            )
        return self._dataset

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

    def __getitem__(self, field):
        """get field value"""
        if isinstance(field, Selector):
            keys = field.keys
        elif isinstance(field, str):
            keys = parse_keys(field)
        else:
            keys = [field]

        container = self.dataset
        for _key in keys:
            item = container[_key]
            container = item

        if isinstance(item, DicomElement):
            return item.value
        return item

    def get(self, field, default=None):
        """similar to dict.get"""
        try:
            values = self.__getitem__(field)
        except (KeyError, IndexError):
            return default
        return values


class DicomTag:
    """DICOM tag"""

    def __init__(self, group, element):
        self.group = int(group, 0) if isinstance(group, str) else int(group)
        self.element = int(element, 0) if isinstance(element, str) else int(element)

    def __repr__(self):
        return f"({self.group:04x}, {self.element:04x})"

    def __iter__(self):
        return iter((self.group, self.element))

    def __eq__(self, other):
        other = type(self)(*other) if isinstance(other, tuple) else other
        return (self.group, self.element) == (other.group, other.element)

    def __hash__(self):
        return hash((self.group, self.element))

    @classmethod
    def is_tag(cls, obj):
        if isinstance(obj, cls):
            return True
        elif not isinstance(obj, tuple):
            return False
        try:
            cls(*obj)
        except (ValueError, TypeError):
            return False
        return True


# helper for creating queries
class DicomTagSelector:
    def __init__(self):
        for tag in pydicom.datadict.DicomDictionary.values():
            setattr(self, tag[-1], Selector(tag[-1]))

    def __getitem__(self, sel):
        if DicomTag.is_tag(sel):
            return Selector(DicomTag(*sel))
        return Selector(sel)

    def __getattr__(self, sel):
        return self[sel]


DICOM = DicomTagSelector()


class DicomElement:
    """pickable DICOM element"""

    def __init__(self, name, tag, VR, VM, value=None, keyword=None):
        """init DICOM element"""
        self.name = name
        self.tag = tag
        self.VR = VR
        self.VM = VM
        self.value = value
        self.keyword = keyword

    @classmethod
    def from_element(cls, element):
        keyword = element.keyword
        if not keyword and element.name.startswith("["):
            # automatic keyword
            keyword = re.sub(r"[^\w\[\]]|_", "", element.name)

        return cls(
            str(element.name),
            DicomTag(element.tag.group, element.tag.element),
            str(element.VR),
            str(element.VM),
            parse_element_value(element.value, element.VR),
            keyword,
        )

    @property
    def sequence(self):
        return self.VR == "SQ"

    def __getitem__(self, item):
        return self.value[item]

    def __repr__(self):
        return self.represent()

    def represent(self, level=0, indent="  "):
        """represent DICOM element"""

        def str_repr(value):
            if value is None:
                return ""
            elif isinstance(value, (list, tuple)):
                return "\\".join(map(str_repr, value))
            elif isinstance(value, str):
                return f"'{value}'"
            else:
                return str(value)

        # value representation
        valuerep = "" if self.sequence else str_repr(self.value)

        if len(valuerep) > 80:
            valuerep = valuerep[:77] + "..."

        # element representation
        repr = f"{indent * level}{self.tag} {self.name:<64.64} {self.VR}: {valuerep:<80.80}"

        if not self.sequence:
            return repr

        # else, if sequence, show sub elements
        for i, dataset in enumerate(self.value):
            # repr += f"\n{indent * (level + 1)}{'-' * 120}\n"
            repr += f"\n{indent * level}[{i}]\n"
            for element in dataset.elements:
                repr += element.represent(level + 1, indent=indent)
                if element is not dataset.elements[-1]:
                    repr += "\n"
            if dataset is self.value[-1]:
                repr += f"\n{indent * level}[-]"

        return repr


def parse_dataset(dataset, index=None, flatten=False):
    """parse dataset to retrieve elements"""
    elements = []
    for element in itertools.chain(getattr(dataset, "file_meta", []), dataset):

        if element.keyword == "PixelData":
            # skip pixel data
            continue

        elif element.keyword == "PerFrameFunctionalGroupsSequence":
            # multi-frame DICOM: flatten sub-dataset
            logging.debug("Parse subdataset PerFrameFunctionalGroupsSequence")
            elements.extend(parse_dataset(element[index], flatten=True))

        elif element.keyword == "SharedFunctionalGroupsSequence":
            # multi-frame DICOM: flatten sub-dataset
            logging.debug("Parse subdataset PerFrameFunctionalGroupsSequence")
            for _dataset in element:
                elements.extend(parse_dataset(_dataset, flatten=True))

        elif flatten and element.VR == "SQ":
            # flatten sequences
            logging.debug(f"Parse subdataset {element}")
            for _dataset in element:
                elements.extend(parse_dataset(_dataset, flatten=True))

        else:
            # normal element: append to elements
            elements.append(DicomElement.from_element(element))

    # make dataset
    return elements


def parse_element_value(value, VR):
    """cast DICOM value type"""

    if value is None or value == "":
        return None  # ""

    elif VR == "SQ":
        return [DicomDataset.from_dataset(dataset) for dataset in value]

    elif VR in ["UI", "SH", "LT", "PN", "UT"]:  # , "OW"]:
        return str(value)

    elif isinstance(value, (pydicom.multival.MultiValue, list, tuple)):
        # if value is an array
        return tuple([parse_element_value(v, VR) for v in value])

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

    elif VR in ["FD", "FL"]:
        return float(value)

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


def parse_keys(string):
    """parse string with optional index suffix
    syntax: "txt" or "txt_num"
    """
    # parse key
    if string[0] == "[":
        # custom private field: don't do anything
        return [string]

    if "_" in string:
        keys, index = string.split("_")
        index = int(index) - 1
    else:
        keys, index = string, None

    def cast_key(value):
        if value.isdigit():
            return int(value) - 1
        # elif value.isalnum():
        elif re.match(r"^[\w\[\]]+$", value):
            return value
        else:
            raise ValueError(f"Invalid field value: {value}")

    # split into sub-fields/index
    keys = [cast_key(key) for key in keys.split(".")]
    if isinstance(keys[-1], int):
        raise ValueError(f"Last item in key must not be an int: {keys}")

    if index is not None:
        keys += [index]

    return keys


def get_zip_path(path):
    """return the zip-file root of path"""
    path = str(path)
    if not ".zip" in path:
        return None
    return path[: path.find(".zip") + 4]


def list_files(pathes):
    """list all files in path and its sub folders"""
    if not isinstance(pathes, (list, tuple)):
        pathes = [pathes]

    files = []
    for path in map(pathlib.Path, pathes):
        if "*" in str(path):
            subs = pathlib.Path().glob(str(path))
        elif path.exists():
            subs = [path]
        else:
            raise FileNotFoundError(path)

        for subpath in subs:
            if subpath.is_file():
                files.append(subpath)
            else:
                files.extend([file for file in subpath.rglob("*") if file.is_file()])

    if not files:  # empty
        return None, []

    # find root directory
    root = pathlib.Path(os.path.commonpath(pathes))
    if root.is_file():
        root = root.parent
    filenames = [str(file.relative_to(root)) for file in files]
    return str(root), filenames
