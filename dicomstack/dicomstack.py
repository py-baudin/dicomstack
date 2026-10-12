""" pydicom wrapper class for easy manipulation of DICOM data """

# coding=utf-8

import os
import pathlib
import re
import datetime
import itertools
import functools
import hashlib
import collections
import concurrent.futures
from io import BytesIO, StringIO
import zipfile
import logging
import warnings
import numpy as np
import pydicom
from pydicom import config

from . import pixeldata, cache
from .query import Selector, Query

LOGGER = logging.getLogger(__name__)

# exceptions
InvalidDicomError = pydicom.errors.InvalidDicomError


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
        # header cache
        cache_paths = cache.locations(self.root, self._cache) if self._cache else []
        if self._cache and not cache_paths:
            LOGGER.info("No default cache location without a root folder: use cache=<path>")
        reuse = cache_paths and self._cache != "refresh"
        self._cache_old = cache.load(cache_paths, self._dedup) if reuse else {}
        self._cache_new = {} if cache_paths else None
        self._cache_changed = False
        skip = {os.path.abspath(path) for path in cache_paths}

        # group zip members by archive (each archive is opened once)
        items = []  # filename, or zip path
        archives = {}  # zip path: (first filename, [selected members])
        for filename in filenames:
            # skip DICOMDIR and cache files
            if str(filename).endswith("DICOMDIR"):
                continue
            elif os.path.basename(str(filename)) == cache.CACHE_NAME:
                continue
            filepath = self._filepath(filename)
            if os.path.abspath(filepath) in skip:
                continue
            zip_path = get_zip_path(filepath)
            if not zip_path:
                items.append(filename)
                continue
            if not zip_path in archives:
                archives[zip_path] = (filename, [])
                items.append(zip_path)
            member = filepath[len(zip_path) :].replace("\\", "/").strip("/")
            archives[zip_path][1].append(member)

        # regular files: from the cache, or read (optionally ahead, in threads)
        files = [item for item in items if not item in archives]
        cached = {}
        if self._cache_new is not None:
            for filename in files:
                entry = self._cache_entry(self._cache_key(filename), self._file_stat(filename))
                if entry is not None:
                    cached[filename] = entry
        to_read = [filename for filename in files if not filename in cached]
        if self._workers:
            contents = prefetch(read_file, map(self._filepath, to_read), self._workers, errors=True)
        else:
            contents = itertools.repeat(None)

        for item in items:
            if item in archives:
                self._load_zipfile(item, *archives[item])
            elif item in cached:
                self._load_cached(item, self._filepath(item), cached[item])
            else:
                self._load_file(item, next(contents))

        if self._cache_new is not None:
            if self._cache_changed or set(self._cache_new) != set(self._cache_old):
                cache.save(cache_paths, self._dedup, self._cache_new)
        del self._cache_old, self._cache_new, self._cache_changed

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

    def _filepath(self, filename):
        """full path of filename"""
        if self.root:
            return os.path.join(self.root, filename)
        return str(filename)

    # header cache

    def _cache_key(self, filename):
        """cache key of a file: path relative to root (absolute without root)"""
        path = os.path.relpath(self._filepath(filename), self.root) if self.root else filename
        path = path if self.root else os.path.abspath(str(path))
        return str(path).replace("\\", "/")

    def _file_stat(self, filename):
        try:
            return cache.file_stat(self._filepath(filename))
        except OSError:
            return None

    def _cache_entry(self, key, stat):
        """valid cache entry for key (and keep it in the new cache), or None"""
        entry = self._cache_old.get(key)
        if stat is None or entry is None or entry.get("stat") != stat:
            return None
        self._cache_new[key] = entry
        return entry

    def _cache_add(self, key, stat, dicomfile=None):
        """add a new entry (dicomfile: None for non-DICOM files)"""
        if self._cache_new is None or stat is None:
            return
        if dicomfile is None:
            self._cache_new[key] = {"stat": stat, "non_dicom": True}
        else:
            self._cache_new[key] = dicomfile.cache_entry(stat)
        self._cache_changed = True

    def _load_cached(self, filename, filepath, entry, source=None):
        """load a file from its cache entry"""
        if entry.get("non_dicom"):
            self.non_dicom.append(filename)
            return
        dicomfile = DicomFile.from_cache(filepath, entry, source=source, dedup=self._dedup)
        self.frames.extend(dicomfile.frames)

    def _load_file(self, filename, data=None):
        """load single Dicom file

        data: contents of the file if already read (or the exception raised when reading it)
        """
        filepath = self._filepath(filename)

        stat = self._file_stat(filename) if self._cache_new is not None else None
        try:
            if isinstance(data, Exception):
                raise data
            # read dicom object
            dicomfile = DicomFile(filepath, bytes=data, dedup=self._dedup)
            frames = dicomfile.frames
        except (IOError, InvalidDicomError):
            # other files
            self.non_dicom.append(filename)
            self._cache_add(self._cache_key(filename), stat)
        else:
            self.frames.extend(frames)
            self._cache_add(self._cache_key(filename), stat, dicomfile)

    def _load_zipfile(self, zip_path, filename, members):
        """load files in zipfile

        zip_path: path to the zip file
        filename: (first) filename designating the zip file
        members: selected members or folders in the zip file ("": all)
        """
        if not zipfile.is_zipfile(zip_path):
            self.non_dicom.append(filename)
            return

        def selected(member):
            return any(
                not sel or member == sel or member.startswith(sel + "/")
                for sel in members
            )

        if self._cache_new is not None:
            # cache: valid while the archive and the member's CRC are unchanged
            prefix = os.path.relpath(zip_path, self.root) if self.root else os.path.abspath(zip_path)
            prefix = prefix.replace("\\", "/")
            try:
                archive_stat = cache.file_stat(zip_path)
            except OSError:
                archive_stat = None

        with zipfile.ZipFile(zip_path, "r") as zf:
            for zfile in zf.filelist:

                if zfile.filename.endswith("/"):
                    # is a directory: skip
                    continue
                elif "DICOMDIR" in zfile.filename:
                    # skip DICOMDIR for now
                    continue
                elif not selected(zfile.filename.strip("/")):
                    continue
                zfilename = os.path.normpath(zfile.filename)
                full_zfilename = os.path.join(zip_path, zfilename)
                source = (zip_path, zfile.filename)

                key = stat = None
                if self._cache_new is not None and archive_stat is not None:
                    key = f"{prefix}:{zfile.filename}"
                    stat = archive_stat + [zfile.CRC]
                    entry = self._cache_entry(key, stat)
                    if entry is not None:
                        self._load_cached(full_zfilename, full_zfilename, entry, source=source)
                        continue

                try:
                    # read dicom object
                    dicomfile = DicomFile(
                        full_zfilename,
                        bytes=zf.read(zfile),
                        source=source,
                        dedup=self._dedup,
                    )
                    frames = dicomfile.frames
                except (IOError, InvalidDicomError):
                    # other files
                    self.non_dicom.append(full_zfilename)
                    self._cache_add(key, stat)
                else:
                    self.frames.extend(frames)
                    self._cache_add(key, stat, dicomfile)


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


class DicomFile:
    """pickable DICOM file

    Keeps the DICOM header (dataset without pixel data), not the file's bytes.
    Pixels are read on demand.

    filename: displayed path (zip members: zip path / member name)
    source: path to the file, or (zip path, member name)
    dedup: file identity: 'uid' (SOPInstanceUID) or 'content' (hash of the file)
    """

    _dataset = None  # header
    _pixels = None
    _frames = None
    _identity = None
    _lookup = None
    _cache = None  # cache entry (header cache)

    def __init__(self, filename, bytes=None, *, source=None, dedup="uid"):
        """init DicomFile object (the header is read on demand)

        bytes: contents of the file (e.g. zip member), read now but not kept
        """
        self.filename = filename
        self.source = str(filename) if source is None else source
        self.dedup = check_dedup(dedup)
        self.digest = None
        self.pixel_offset = None
        if bytes is not None:
            self._read_header(bytes)

    @classmethod
    def from_cache(cls, filename, entry, *, source=None, dedup="uid"):
        """DicomFile from a header cache entry (the header is read on demand)"""
        dicomfile = cls(filename, source=source, dedup=dedup)
        dicomfile._cache = entry
        dicomfile._identity = entry["identity"]
        dicomfile.pixel_offset = entry["offset"]
        return dicomfile

    def cache_entry(self, stat):
        """header cache entry: converted values of all frames"""
        if self._lookup is None:
            self._lookup = FileLookup(self.dataset)
        nframe = self.nframe
        indices = [None] if nframe is None else range(nframe)
        return {
            "stat": stat,
            "offset": self.pixel_offset,
            "nframe": nframe,
            "identity": self.identity,
            "frames": [cache.frame_values(self._lookup, index) for index in indices],
        }

    def cached_frame(self, index):
        """cached values of frame `index`, or None"""
        if self._cache is None:
            return None
        return self._cache["frames"][index or 0]

    @property
    def identity(self):
        """identity of the file (for duplicate detection)"""
        if self._identity is None:
            element = self.dataset.get(SOP_INSTANCE_UID)
            uid = element.value if element is not None else None
            if self.dedup == "content":
                self._identity = ("content", self.digest)
            elif uid:
                self._identity = ("uid", str(uid))
            else:
                # never merged with other files
                self._identity = ("source", self.source)
        return self._identity

    def __hash__(self):
        return hash(self.identity)

    def __eq__(self, other):
        if not isinstance(other, DicomFile):
            return NotImplemented
        return self.identity == other.identity

    # pickle
    def __getstate__(self):
        state = dict(self.__dict__)
        for name in ["_pixels", "_frames", "_lookup"]:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

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
        """DICOM header (pydicom dataset without pixel data)"""
        if self._dataset is None:
            self._read_header()
        return self._dataset

    @property
    def nframe(self):
        """number of frames in multi-frame DICOM"""
        if self._dataset is None and self._cache is not None:
            return self._cache["nframe"]
        return get_nframe(self.dataset)

    @property
    def frames(self):
        """return list of frames"""
        if self._frames is None:
            nframe = self.nframe
            if nframe is None:
                self._frames = [DicomFrame(self)]
            else:
                self._frames = [DicomFrame(self, index=i) for i in range(nframe)]
        return self._frames

    @property
    def pixels(self):
        """retrieve pixel data"""
        if self._pixels is None:
            self._pixels = self._read_pixels()
        return self._pixels

    def _read_pixels(self):
        """read pixel array, using the stored header"""
        return self._decode_pixels(self._read_pixel_bytes())

    def _read_pixel_bytes(self):
        """bytes needed to decode the pixels: the file from the pixel data element on,
        or the whole file if its position is unknown (I/O only: can run in a thread)
        """
        self.dataset  # make sure the header is read
        if self.pixel_offset is None:
            return self._read_bytes()
        with open(self.source, "rb") as fp:
            fp.seek(self.pixel_offset)
            return fp.read()

    def _decode_pixels(self, data):
        """pixel array from the bytes returned by _read_pixel_bytes"""
        header = self.dataset
        if self.pixel_offset is None:
            # zip member, or unknown offset: parse the whole file
            dataset, _ = read_dataset(data, stop_before_pixels=False)
            return dataset.pixel_array

        syntax = header.file_meta.TransferSyntaxUID
        if syntax in NATIVE_SYNTAXES:
            pixels = read_native_pixels(BytesIO(data), header, implicit=syntax.is_implicit_VR)
            if pixels is not None:
                return pixels
        # other syntaxes: read the pixel data element only, decode with pydicom
        elements = pydicom.filereader.read_dataset(
            BytesIO(data), syntax.is_implicit_VR, syntax.is_little_endian
        )
        dataset = pydicom.Dataset()
        dataset.update(header)
        dataset.update(elements)
        dataset.file_meta = header.file_meta
        return dataset.pixel_array

    def get_element(self, key, index=None):
        """pydicom element of frame `index` for key (keyword, "[Name]" or tag)

        Same rules as DicomDataset.from_dataset(self.dataset, index)[key]
        """
        if self._lookup is None:
            self._lookup = FileLookup(self.dataset)
        return self._lookup.get(key, index)

    def _read_header(self, data=None):
        """read the header, from data (bytes) or from the source

        The file's bytes are read at once (parsing from memory is faster), then dropped.
        """
        if data is None:
            data = self._read_bytes()
        self._dataset, offset = read_dataset(data, stop_before_pixels=True)
        # pixel data can be read directly from regular files only
        self.pixel_offset = None if isinstance(self.source, tuple) else offset
        if self.dedup == "content":
            self.digest = hashlib.blake2b(data, digest_size=16).digest()

    def _read(self, stop_before_pixels=True):
        """read source, return pydicom dataset and offset of the pixel data element"""
        return read_dataset(self._read_bytes(), stop_before_pixels)

    def _read_bytes(self):
        """contents of the source file"""
        if isinstance(self.source, tuple):
            # zip member
            zip_path, member = self.source
            with zipfile.ZipFile(zip_path, "r") as zf:
                return zf.read(member)
        with open(self.source, "rb") as fp:
            return fp.read()


class DicomDataset:
    """collection of DICOM elements"""

    def __init__(self, elements):
        """elements is a list of DicomElement"""
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
        self.dicomfile = dicomfile
        self.index = index
        self._dataset = None
        self._pixels = pixels
        self._values = {}  # cache of converted values

    @property
    def identity(self):
        """identity of the frame (for duplicate detection)"""
        return self.dicomfile.identity + (self.index,)

    def __hash__(self):
        return hash(self.identity)

    def __eq__(self, other):
        if not isinstance(other, DicomFrame):
            return NotImplemented
        return self.identity == other.identity

    # pickle
    def __getstate__(self):
        return {"dicomfile": self.dicomfile, "index": self.index}

    def __setstate__(self, state):
        self.__init__(state["dicomfile"], index=state["index"])

    def __repr__(self):
        """represent DICOM frame"""
        if not self._dataset:
            return "DICOM frame (pending)"
        return "DICOM frame\n" + str(self._dataset)

    @property
    def dataset(self):
        if not self._dataset:
            # convert the whole dataset
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
        keys = compile_keys(field)
        try:
            return self._values[keys]
        except (KeyError, TypeError):  # TypeError: unhashable keys
            pass

        value = cache.NOT_CACHED
        cached = self.dicomfile.cached_frame(self.index)
        if cached is not None:
            value = get_cached_value(cached, keys)
        if value is cache.NOT_CACHED:
            # only the requested element is converted
            node = self.dicomfile.get_element(keys[0], self.index)
            for key in keys[1:]:
                node = get_subitem(node, key)
            value = get_leaf(node)

        try:
            self._values[keys] = value
        except TypeError:
            pass
        return value

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
    """DICOM.Keyword, DICOM["[Private Name]"], DICOM[(group, element)]: query selectors"""

    def __getitem__(self, sel):
        if DicomTag.is_tag(sel):
            return Selector(DicomTag(*sel))
        return Selector(sel)

    def __getattr__(self, sel):
        if sel.startswith("__"):
            raise AttributeError(sel)
        return self[sel]

    def __dir__(self):
        # tab completion of DICOM keywords
        return sorted(set(super().__dir__()) | set(pydicom.datadict.keyword_dict))


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
        keyword = element_keyword(element)
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
            elements.extend(parse_dataset(element[index], flatten=True))

        elif element.keyword == "SharedFunctionalGroupsSequence":
            # multi-frame DICOM: flatten sub-dataset
            for _dataset in element:
                elements.extend(parse_dataset(_dataset, flatten=True))

        elif flatten and element.VR == "SQ":
            # flatten sequences
            for _dataset in element:
                elements.extend(parse_dataset(_dataset, flatten=True))

        else:
            # normal element: append to elements
            elements.append(DicomElement.from_element(element))

    # make dataset
    return elements


#
# thin access to pydicom datasets
# (same results as DicomDataset/DicomElement, converting only the requested elements)

DEDUP_MODES = ("uid", "content")

SHARED_GROUPS = 0x52009229  # SharedFunctionalGroupsSequence
PER_FRAME_GROUPS = 0x52009230  # PerFrameFunctionalGroupsSequence
PIXEL_DATA = 0x7FE00010
SOP_INSTANCE_UID = 0x00080018
ROWS, COLUMNS, BITS_ALLOCATED = 0x00280010, 0x00280011, 0x00280100
NATIVE_SYNTAXES = {pydicom.uid.ImplicitVRLittleEndian, pydicom.uid.ExplicitVRLittleEndian}
SEQUENCE_DELIMITER = b"\xfe\xff\xdd\xe0"


def read_dataset(data, stop_before_pixels=True):
    """read DICOM dataset from bytes (the bytes are not kept)

    returns: dataset, offset of the pixel data element (None if unknown)
    """
    buffer = BytesIO(data)
    with config.strict_reading():
        try:
            dataset = pydicom.dcmread(buffer, stop_before_pixels=stop_before_pixels)
            offset = buffer.tell() if stop_before_pixels else None
        except EOFError as exc:
            warnings.warn(str(exc))
            buffer = BytesIO(data + SEQUENCE_DELIMITER)
            dataset = pydicom.dcmread(buffer, stop_before_pixels=stop_before_pixels)
            offset = None
    # pydicom keeps a reference to the buffer
    dataset.buffer = None
    meta = getattr(dataset, "file_meta", None)
    if meta is None or meta.get("TransferSyntaxUID") == pydicom.uid.DeflatedExplicitVRLittleEndian:
        # the offset is not a position in the file
        offset = None
    return dataset, offset


def check_dedup(dedup):
    if not dedup in DEDUP_MODES:
        raise ValueError(f"Invalid dedup mode: {dedup} (options: {DEDUP_MODES})")
    return dedup


def read_file(path):
    with open(path, "rb") as fp:
        return fp.read()


def prefetch(func, items, workers, window=None, errors=False):
    """yield func(item) for each item, computed ahead in a thread pool

    window: maximum number of items computed ahead (default: 4 x workers)
    errors: yield exceptions raised by func instead of raising them
    """
    window = window or 4 * workers
    items = iter(items)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        pending = collections.deque(pool.submit(func, item) for item in itertools.islice(items, window))
        while pending:
            future = pending.popleft()
            for item in itertools.islice(items, 1):
                pending.append(pool.submit(func, item))
            try:
                yield future.result()
            except Exception as exc:
                if not errors:
                    raise
                yield exc


def read_native_pixels(fp, header, implicit):
    """read native (uncompressed, little endian) pixel data at the current position of fp

    returns the same array as pydicom's pixel_array, or None if not supported
    (several samples per pixel, unusual bit depths, unexpected element)
    """
    rows, columns = header.get(ROWS), header.get(COLUMNS)
    bits = header.get(BITS_ALLOCATED)
    if rows is None or columns is None or bits is None or bits.value not in (8, 16, 32):
        return None
    rows, columns, bits = rows.value, columns.value, bits.value
    samples = header.SamplesPerPixel if "SamplesPerPixel" in header else 1
    stored = header.BitsStored if "BitsStored" in header else bits
    high_bit = header.HighBit if "HighBit" in header else stored - 1
    signed = header.PixelRepresentation == 1 if "PixelRepresentation" in header else False
    nframe = int(header.NumberOfFrames or 1) if "NumberOfFrames" in header else 1
    if samples != 1 or not stored or stored > bits or high_bit != stored - 1:
        return None

    # pixel data element: tag, (explicit VR: VR, reserved), length
    if fp.read(4) != b"\xe0\x7f\x10\x00":
        return None
    if implicit:
        length = int.from_bytes(fp.read(4), "little")
    elif fp.read(4)[:2] in (b"OB", b"OW"):
        length = int.from_bytes(fp.read(4), "little")
    else:
        return None
    dtype = np.dtype(f"<{'i' if signed else 'u'}{bits // 8}")
    size = rows * columns * nframe * dtype.itemsize
    if length == 0xFFFFFFFF or length < size:
        return None
    buffer = bytearray(size)
    if fp.readinto(buffer) != size:
        return None
    pixels = np.frombuffer(buffer, dtype=dtype)

    if stored < bits:
        # as pydicom: mask unused high bits, or extend the sign
        if signed:
            shift = bits - stored
            pixels = (pixels << shift) >> shift
        else:
            pixels &= (1 << stored) - 1
    shape = (nframe, rows, columns) if nframe > 1 else (rows, columns)
    return pixels.reshape(shape)


def element_keyword(element):
    """keyword of a pydicom element, or automatic keyword of a named private element"""
    keyword = element.keyword
    if not keyword and element.name.startswith("["):
        keyword = re.sub(r"[^\w\[\]]|_", "", element.name)
    return keyword


@functools.lru_cache(maxsize=4096)
def _parse_keys(string):
    return tuple(parse_keys(string))


def compile_keys(field):
    """keys (tuple) of a field: string, Selector or tag"""
    if isinstance(field, Selector):
        return tuple(field.keys)
    elif isinstance(field, str):
        return _parse_keys(field)
    return (field,)


class DatasetLookup:
    """element lookup in a pydicom dataset (not flattened), as DicomDataset[key]"""

    def __init__(self, dataset, meta=None, exclude=()):
        self.dataset = dataset
        self.meta = meta  # file meta information
        self.exclude = frozenset(exclude) | {PIXEL_DATA}  # tags not accessible
        self._private = None

    def find(self, key):
        """pydicom element for key (tag, keyword or "[Name]")"""
        if DicomTag.is_tag(key):
            tag = DicomTag(*key)
            element = self.by_tag((tag.group << 16) | tag.element)
            if element is None:
                raise KeyError(tag)
            return element
        elif not isinstance(key, str):
            raise TypeError(f"Invalid key type: {key}")
        element = self.by_keyword(key)
        if element is None:
            raise KeyError(key)
        return element

    def by_tag(self, tag):
        if tag in self.exclude:
            return None
        # the dataset's element takes precedence
        for container in (self.dataset, self.meta):
            if container is not None and tag in container:
                return container[tag]
        return None

    def by_keyword(self, keyword):
        if keyword.startswith("["):
            tag = self._private_index().get(keyword)
            return None if tag is None else self.by_tag(tag)
        # (elements of repeating groups, e.g. OverlayData, have no keyword in pydicom)
        tag = pydicom.datadict.tag_for_keyword(keyword)
        return None if tag is None else self.by_tag(tag)

    def _tags(self):
        """sorted tags (meta, then dataset), last wins"""
        tags = sorted(self.meta.keys()) if self.meta is not None else []
        return tags + sorted(self.dataset.keys())

    def _private_index(self):
        """automatic keyword of named private elements: tag"""
        if self._private is None:
            self._private = {}
            for tag in self._tags():
                if tag.group % 2 == 0 or tag in self.exclude:
                    continue
                keyword = element_keyword(self.by_tag(tag))
                if keyword:
                    self._private[keyword] = tag
        return self._private


class FileLookup:
    """frame-level element lookup in a DICOM header, as DicomDataset.from_dataset(header, index)

    multi-frame files: for frame i, top-level elements after the functional groups
    > per-frame group i > shared groups > other top-level elements (last occurrence wins)
    """

    def __init__(self, header):
        self.header = header
        self.multiframe = SHARED_GROUPS in header or PER_FRAME_GROUPS in header
        exclude = {SHARED_GROUPS, PER_FRAME_GROUPS} if self.multiframe else ()
        meta = getattr(header, "file_meta", None)
        self.top = DatasetLookup(header, meta=meta, exclude=exclude)
        self._shared = None
        self._per_frame = None

    def get(self, key, index=None):
        if not self.multiframe:
            return self.top.find(key)

        if DicomTag.is_tag(key):
            tag = DicomTag(*key)
            group_key = (tag.group << 16) | tag.element
        elif isinstance(key, str):
            group_key = key
        else:
            raise TypeError(f"Invalid key type: {key}")

        try:
            element = self.top.find(key)
        except KeyError:
            element = None
        if element is not None and element.tag > PER_FRAME_GROUPS:
            return element
        for group in self._groups(index):
            if group_key in group:
                return group[group_key]
        if element is None:
            raise KeyError(key)
        return element

    def _flattened_groups(self):
        """flattened functional groups (lists of elements): shared, per-frame"""
        if self._shared is None:
            header = self.header
            shared = header[SHARED_GROUPS].value if SHARED_GROUPS in header else []
            per_frame = header[PER_FRAME_GROUPS].value if PER_FRAME_GROUPS in header else []
            self._shared = list(itertools.chain(*map(flatten_elements, shared)))
            self._per_frame = [list(flatten_elements(item)) for item in per_frame]
            self._shared_index = index_elements(self._shared)
            self._per_frame_index = [index_elements(items) for items in self._per_frame]
        return self._shared, self._per_frame

    def _groups(self, index):
        """indexed functional groups for frame `index`: per-frame, shared"""
        self._flattened_groups()
        if index is None or not self._per_frame:
            return [self._shared_index]
        return [self._per_frame_index[index], self._shared_index]

    def frame_elements(self, index=None):
        """all elements of frame `index`: {tag (int): element}, {keyword: tag}

        Same precedence as `get` (and as DicomDataset.from_dataset(header, index)):
        file meta, then top-level elements in tag order with the functional groups
        flattened in place, the last occurrence winning.
        Top-level public elements are returned unconverted (pydicom RawDataElement)
        when not converted yet.
        """
        by_tag, by_keyword = {}, {}

        def add(element, keyword=None):
            tag = int(element.tag)
            by_tag[tag] = element
            keyword = element_keyword(element) if keyword is None else keyword
            if keyword:
                by_keyword[keyword] = tag

        meta = self.top.meta
        for tag in sorted(meta.keys()) if meta is not None else []:
            add(meta[tag])
        if self.multiframe:
            shared, per_frame = self._flattened_groups()
        for tag in sorted(self.header.keys()):
            if tag == PIXEL_DATA:
                continue
            elif self.multiframe and tag == SHARED_GROUPS:
                for element in shared:
                    add(element)
            elif self.multiframe and tag == PER_FRAME_GROUPS:
                for element in per_frame[index] if index is not None and per_frame else []:
                    add(element)
            elif tag.is_private:
                # the automatic keyword needs the converted element's name
                add(self.header[tag])
            else:
                # as DataElement.keyword: dictionary keyword, "" for repeating groups
                keyword = (
                    pydicom.datadict.dictionary_keyword(tag)
                    if pydicom.datadict.dictionary_has_tag(tag)
                    else ""
                )
                add(self.header.get_item(tag), keyword)
        return by_tag, by_keyword


def flatten_elements(dataset):
    """elements of dataset with nested sequences flattened (cf. parse_dataset(flatten=True))"""
    for element in dataset:
        if element.keyword == "PixelData":
            continue
        elif element.VR == "SQ":
            for item in element.value:
                yield from flatten_elements(item)
        else:
            yield element


def index_elements(elements):
    """dict of elements by tag (int) and keyword, last occurrence wins"""
    index = {}
    for element in elements:
        index[int(element.tag)] = element
        keyword = element_keyword(element)
        if keyword:
            index[keyword] = element
    return index


def get_subitem(node, key):
    """node[key], as DicomElement[key] / DicomDataset[key]"""
    if isinstance(node, pydicom.DataElement):
        if node.VR == "SQ" and isinstance(node.value, pydicom.Sequence) and isinstance(key, int):
            # sequence item: not converted
            return DatasetLookup(node.value[key])
        return parse_element_value(node.value, node.VR)[key]
    elif isinstance(node, DatasetLookup):
        return node.find(key)
    return node[key]


def get_cached_value(cached, keys):
    """value from the cached values of a frame, or cache.NOT_CACHED

    raises KeyError for missing elements (the cache lists all elements)
    """
    key = keys[0]
    if DicomTag.is_tag(key):
        tag = DicomTag(*key)
        tag_key = f"{tag.group:04x},{tag.element:04x}"
    elif isinstance(key, str):
        tag_key = cached["k"].get(key)
        if tag_key is None:
            raise KeyError(key)
    else:
        raise TypeError(f"Invalid key type: {key}")
    if not tag_key in cached["v"]:
        raise KeyError(key)
    value = cached["v"][tag_key]
    if value is cache.NOT_CACHED:
        return value
    for key in keys[1:]:
        # as get_subitem for non-sequence elements
        value = value[key]
    return value


def get_leaf(node):
    """value of node, as DicomFrame.__getitem__ returned it"""
    if isinstance(node, pydicom.DataElement):
        return parse_element_value(node.value, node.VR)
    elif isinstance(node, DatasetLookup):
        return DicomDataset.from_dataset(node.dataset)
    elif isinstance(node, DicomElement):
        return node.value
    return node


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
    if not PER_FRAME_GROUPS in dataset:
        return None
    frames = dataset[PER_FRAME_GROUPS].value
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
    """return the zip-file root of path

    The zip file is the first path component ending with ".zip" (any case)
    which is not an existing directory.
    """
    path = str(path)
    for match in re.finditer(r"\.zip(?=[\\/]|$)", path, flags=re.IGNORECASE):
        zip_path = path[: match.end()]
        if not os.path.isdir(zip_path):
            return zip_path
    return None


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
