""" pydicom wrapper class for easy manipulation of DICOM data """

# coding=utf-8

import os
import re
import itertools
from io import StringIO
import logging
import warnings

from . import pixeldata
from .query import Selector, Query
from .files import DicomFrame, Loader, check_dedup, get_zip_path, list_files
from .elements import DicomTag

# moved to .elements and .files: re-exported for backward compatibility
from .elements import (  # noqa: F401
    DICOM,
    DicomDataset,
    DicomElement,
    DicomTagSelector,
    DatasetLookup,
    FileLookup,
    compile_keys,
    element_keyword,
    get_nframe,
    parse_dataset,
    parse_element_value,
    parse_keys,
)
from .files import (  # noqa: F401
    DicomFile,
    InvalidDicomError,
    prefetch,
    read_dataset,
    read_file,
)
from .pixeldata import read_native_pixels  # noqa: F401

LOGGER = logging.getLogger(__name__)


class DuplicatedFramesError(Exception):
    """raise when duplicated frames are found"""


class MultipleValuesError(ValueError):
    """raise when several values are found where a single one is expected"""


class NoValueError(ValueError):
    """raise when no value is found where a single one is expected"""


class DicomStack(object):
    """load, sort and filter DICOM images"""

    def __init__(
        self,
        path=None,
        filenames=None,
        *,
        duplicates="remove",
        dedup="uid",
        workers=None,
        cache=None,
    ):
        """path can be:
        * a directory (or a list of),
        * a file (or a list of)
        * a zip file

        duplicates: 'remove', 'error', 'ignore'
        dedup: how duplicated frames are detected
            'uid': same SOPInstanceUID and frame index (files without UID are never merged)
            'content': same file contents and frame index
        workers: number of threads reading the files ahead (headers are parsed in the
            main thread). Useful on slow (e.g. network) storage only.
        cache: header cache file, to skip parsing unchanged files on later loads
            None/False: no cache; True: default location (`.dicomstack-cache.json.gz`
            in the data folder, or the user cache directory if not writable);
            "refresh": rebuild the cache at the default location; or a path
        """
        self._dedup = check_dedup(dedup)
        self._workers = workers
        self._cache = cache

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
        LOGGER.debug("New DICOM stack (from %s frames)", len(frames))
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
                id = str(studies[uid])
            else:
                numeric = [int(i) for i in tree if i.isdigit()]
                id = str(max(numeric, default=0) + 1)
            if id in tree:
                # several studies with the same StudyID
                id = next(f"{id}-{n}" for n in itertools.count(2) if f"{id}-{n}" not in tree)
            tree[id] = uidtree[uid]

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
            raise MultipleValuesError("Multiple values found for %s" % (fields,))
        elif not values:
            raise NoValueError("No value found for %s" % (fields,))
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

    def remove_duplicates(
        self,
        *,
        keys=["SOPInstanceUID", "ImagePositionPatient"],
        inplace=False,
        warn=False,
    ):
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
            warnings.warn(
                f"Removed {diff} duplicated frames.",
                stacklevel=pixeldata.external_stacklevel(),
            )

        if inplace:
            self.frames = frames
            return self
        else:
            return self.from_frames(frames, root=self.root)

    def sort(self, *fields):
        """reindex database using field values (skip frames with missing values)"""
        LOGGER.debug("Sort by fields: %s" % str(fields))
        frames = self._existing(fields)
        sorted_frames = sorted(frames, key=lambda f: tuple(f[field] for field in fields))
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

    def iter_pixels(self, workers=None):
        """iterate the frames' pixel arrays

        workers: number of threads reading the files ahead (I/O only; useful on slow storage)
        """
        if not workers:
            for frame in self.frames:
                yield frame.pixels
            return

        # threads read the files' bytes (I/O), the main thread decodes the pixels;
        # each file once, in order of first appearance
        # (by object: duplicated files have the same identity)
        files = list({id(frame.dicomfile): frame.dicomfile for frame in self.frames}.values())
        files = [dicomfile for dicomfile in files if dicomfile._pixels is None]
        for dicomfile in files:
            dicomfile.dataset  # headers: read in the main thread
        contents = prefetch(lambda dicomfile: dicomfile._read_pixel_bytes(), files, workers)
        pending = {id(dicomfile) for dicomfile in files}
        for frame in self.frames:
            while id(frame.dicomfile) in pending:
                dicomfile = files[len(files) - len(pending)]
                dicomfile._pixels = dicomfile._decode_pixels(next(contents))
                pending.remove(id(dicomfile))
            yield frame.pixels

    def as_volume(
        self,
        by=None,
        rescale=True,
        reorder=True,
        tol=1e-3,
        strict=False,
        dtype=None,
        workers=None,
    ):
        """as volume

        tol: relative tolerance on the uniformity of the slice spacing
        strict: if True, raise ValueError on non-uniform slice spacing (else warn)
        dtype: data type of the volume (default: float64)
        workers: number of threads reading the pixels (I/O only; useful on slow storage)
        """
        LOGGER.debug("Make volume (use fields: %s)", by)

        if len(self) == 0:
            return None

        def make_volume(stack):
            stack = stack.remove_duplicates(keys=["ImagePositionPatient"], warn=True)
            if reorder:
                # sort by location
                stack = stack.reorder()
            for _ in stack.iter_pixels(workers=workers):
                pass
            return pixeldata.make_volume(
                stack.frames, rescale=rescale, tol=tol, strict=strict, dtype=dtype
            )

        if not by:
            # single non-indexed volume
            return make_volume(self)

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
        volumes = [make_volume(self.filter_by_field(**filter)) for filter in filters]
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
        loader = Loader(self.root, self._dedup, self._workers, self._cache)
        self.frames, self.non_dicom = loader.load(filenames)

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

    # header cache


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
        for seriesnumber in sorted(study, key=_series_key):
            frames = study[seriesnumber]
            len_series = len(frames)
            series_description = frames[0]["SeriesDescription"]
            if regex and not re.search(regex, str(series_description)):
                # skip
                continue
            desc = f"{series_description} (n={len_series})"
            print(f"\t{seriesnumber}: {desc}", file=info)

    return info.getvalue()


def _series_key(seriesnumber):
    """sort key for series numbers: numeric first, then others"""
    try:
        return (0, int(seriesnumber), "")
    except ValueError:
        return (1, 0, seriesnumber)
