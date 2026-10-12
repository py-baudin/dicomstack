""" pydicom wrapper class for easy manipulation of DICOM data """

# coding=utf-8

from __future__ import annotations

import os
import re
import html
import itertools
from io import StringIO
import logging
import warnings
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import pydicom

from . import pixeldata, utils
from .query import Selector, Query
from .files import (
    DicomFrame,
    Loader,
    check_dedup,
    filter_filenames,
    get_zip_path,
    list_files,
)
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

# a field: keyword ("EchoTime", "ImageType_1", "[Private Name]"), selector or tag
Field = Union[str, Selector, DicomTag, Tuple[int, int]]


# default number of threads reading files (I/O only): see DicomStack
WORKERS = 8

# fields ignored by DicomStack.varying: per-instance values
VARYING_EXCLUDE = (
    "SOPInstanceUID",
    "MediaStorageSOPInstanceUID",
    "FileMetaInformationGroupLength",
    "InstanceNumber",
    "ImagePositionPatient",
    "SliceLocation",
    "AcquisitionTime",
    "AcquisitionDateTime",
    "ContentTime",
    "InstanceCreationTime",
    "WindowCenter",
    "WindowWidth",
    "SmallestImagePixelValue",
    "LargestImagePixelValue",
)


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
        workers=WORKERS,
        cache=None,
        include=None,
        exclude=None,
        recursive=True,
        progress="auto",
    ):
        """path can be:
        * a directory (or a list of),
        * a file (or a list of)
        * a zip file

        include, exclude: file name patterns (str or list, fnmatch syntax, e.g. "*.dcm"),
            matched against the file names and the relative paths ("/" separators);
            zip files are filtered by their name, not their members
        recursive: search sub-folders
        progress: progress bar on stderr: "auto" (if stderr is a terminal), True, False;
            or a callable(done, total)
        duplicates: 'remove', 'error', 'ignore'
        dedup: how duplicated frames are detected
            'uid': same SOPInstanceUID and frame index (files without UID are never merged)
            'content': same file contents and frame index
        workers: number of threads reading the files ahead (headers are parsed in the
            main thread); much faster on slow (e.g. network) storage, about the same on a
            local disk. None or 0: sequential
        cache: header cache file, to skip parsing unchanged files on later loads
            None/False: no cache; True: default location (`.dicomstack-cache.json.gz`
            in the data folder, or the user cache directory if not writable);
            "refresh": rebuild the cache at the default location; or a path
        """
        check_dedup(dedup)

        self.frames = []
        self.non_dicom = []
        self.root = None

        if not filenames and not path:
            filenames = []

        elif not filenames:
            # search path
            root, filenames = list_files(path, recursive=recursive)
            self.root = root

        elif not isinstance(filenames, (list, tuple)):
            filenames = [filenames]

        if include is not None or exclude is not None:
            filenames = filter_filenames(filenames, include, exclude)

        if not filenames:
            # empty stack
            LOGGER.info("New DICOM stack (empty)")
            return

        # load dicom files
        LOGGER.info("New DICOM stack (from %s files)", len(filenames))
        loader = Loader(self.root, dedup, workers, cache, progress)
        self.frames, self.non_dicom = loader.load(filenames)
        self._check_duplicates(duplicates)

    @classmethod
    def from_datasets(
        cls, datasets: Iterable[pydicom.Dataset], *, duplicates: str = "remove"
    ) -> DicomStack:
        """create a stack from in-memory pydicom datasets (with their pixel data)

        The datasets are not copied (frames read their values when accessed).
        """
        stack = cls()
        for i, dataset in enumerate(datasets):
            filename = getattr(dataset, "filename", None)
            if not isinstance(filename, str) or not filename:
                filename = f"dataset_{i + 1}.dcm"
            stack.frames.extend(DicomFile.from_dataset(dataset, filename).frames)
        stack._check_duplicates(duplicates)
        return stack

    @classmethod
    def from_frames(cls, frames: Iterable[DicomFrame], root: Optional[str] = None) -> DicomStack:
        """create a new stack from list of frames"""
        LOGGER.debug("New DICOM stack (from %s frames)", len(frames))
        if not all([isinstance(frame, DicomFrame) for frame in frames]):
            raise TypeError("Invalid DicomFrame in %s" % frames)
        stack = cls.__new__(cls)
        stack.frames = list(frames)
        stack.non_dicom = []
        stack.root = root
        return stack

    def __len__(self) -> int:
        """number of DICOM images"""
        return len(self.frames)

    def __bool__(self) -> bool:
        """return True if stack is not empty"""
        return len(self) > 0

    def __iter__(self) -> Iterator[DicomFrame]:
        """iterates dicom frames"""
        return iter(self.frames)

    def __repr__(self) -> str:
        """DICOM(2000 frames | 1 study | 10 series)"""
        if not self.frames:
            return "DICOM(0 frames)"
        nframe = len(self)
        nstudy = len({frame.get("StudyInstanceUID") for frame in self.frames})
        nseries = len({frame.get("SeriesInstanceUID") for frame in self.frames})
        return (
            f"DICOM({nframe} frame{'s' * (nframe != 1)} | "
            f"{nstudy} stud{'ies' if nstudy != 1 else 'y'} | {nseries} series)"
        )

    def _repr_html_(self) -> str:
        """Jupyter: description of the stack"""
        return f"<pre>{html.escape(repr(self))}\n{html.escape(self.describe())}</pre>"

    def __contains__(self, field: Field) -> bool:
        """short for has_field"""
        return self.has_field(field)

    # set operations on frames (order of the left operand kept)

    def __add__(self, other: DicomStack) -> DicomStack:
        """union of the frames"""
        if not isinstance(other, DicomStack):
            return NotImplemented
        seen = set(self.frames)
        frames = list(self.frames)
        for frame in other.frames:
            if not frame in seen:
                seen.add(frame)
                frames.append(frame)
        return self._from_set_operation(frames, other)

    def __sub__(self, other: DicomStack) -> DicomStack:
        """frames not in other"""
        if not isinstance(other, DicomStack):
            return NotImplemented
        others = set(other.frames)
        return self._from_set_operation([f for f in self.frames if not f in others], other)

    def __and__(self, other: DicomStack) -> DicomStack:
        """frames also in other"""
        if not isinstance(other, DicomStack):
            return NotImplemented
        others = set(other.frames)
        return self._from_set_operation([f for f in self.frames if f in others], other)

    def _from_set_operation(self, frames, other):
        # filenames are relative to the common root, absolute otherwise
        root = self.root if self.root == other.root else None
        return self.from_frames(frames, root=root)

    # exploring

    def fields(self, how: str = "all") -> list[str]:
        """keywords of the elements present in all (how="all") or any (how="any") frames"""
        keywords = [set(frame.keywords()) for frame in self.frames]
        if not keywords:
            return []
        elif how == "all":
            return sorted(set.intersection(*keywords))
        elif how == "any":
            return sorted(set.union(*keywords))
        raise ValueError(f"Invalid value for how: {how}")

    def varying(
        self,
        within: Union[Field, Sequence[Field], None] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> list[str]:
        """keywords of the fields whose values differ across frames

        within: field(s): fields varying within the groups of the same value(s)
            (e.g. "SeriesNumber": fields varying inside a series)
        exclude: fields to ignore (default: VARYING_EXCLUDE, per-instance fields)
        Sequences are ignored.
        """
        exclude = set(VARYING_EXCLUDE if exclude is None else exclude)
        if within is None:
            groups = [self]
        else:
            within = [within] if isinstance(within, str) else list(within)
            groups = [substack for _, substack in self.groupby(*within)]
        candidates = [field for field in self.fields("any") if not field in exclude]
        varying = set()
        for substack in groups:
            for field in candidates:
                if field in varying:
                    continue
                values = set()
                try:
                    for frame in substack:
                        values.add(frame.get(field))
                        if len(values) > 1:
                            varying.add(field)
                            break
                except TypeError:
                    # not hashable (sequences)
                    continue
        return sorted(varying)

    def table(self, *fields: Field) -> list[dict]:
        """list of {field: value} for each frame (default fields: fields())"""
        fields = fields or self.fields()
        return [{field: frame.get(field) for field in fields} for frame in self.frames]

    def to_dataframe(self, *fields: Field) -> Any:
        """pandas DataFrame of the fields' values (one row per frame); requires pandas"""
        try:
            import pandas
        except ImportError:
            raise ImportError("pandas is required for to_dataframe (pip install pandas)")
        fields = fields or self.fields()
        return pandas.DataFrame(self.table(*fields), columns=list(fields))

    def export(self, dest: Union[str, os.PathLike], anonymize: bool = True, **kwargs) -> list:
        """export (anonymized) copies of the stack's files in folder dest

        Files keep their path relative to the stack's root (their name without root).
        kwargs: see utils.export_file (mapper, mapkey, remove_private_tags, overwrite)
        Returns the exported files.
        """
        exported = []
        done = set()
        for frame in self.frames:
            dicomfile = frame.dicomfile
            if id(dicomfile) in done:
                continue
            done.add(id(dicomfile))
            if self.root:
                relpath = self._filename(frame)
            else:
                relpath = os.path.basename(str(dicomfile.filename))
            destfile = os.path.join(str(dest), relpath)
            exported.append(
                utils.export_file(
                    dicomfile.full_dataset(), destfile, anonymize=anonymize, **kwargs
                )
            )
        return exported

    @property
    def filenames(self) -> list[str]:
        """return list of dicom files"""
        return [self._filename(frame) for frame in self.frames]

    def _filename(self, frame):
        """frame's filename, relative to root"""
        filename = frame.dicomfile.filename
        return os.path.relpath(filename, self.root) if self.root else filename

    @property
    def zipfiles(self) -> list[str]:
        """return path to zip files"""
        zipfiles = {get_zip_path(file) for file in self.filenames}
        return list(zipfiles)

    @property
    def dicomtree(self) -> dict:
        """return DICOM-tree"""
        LOGGER.info("Make dicom tree")

        def _describe(frame):
            filename = self._filename(frame)
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

    def describe(self, **kwargs) -> str:
        """method for self description"""
        return describe(self.dicomtree, **kwargs)

    def __call__(
        self, query: Optional[Query] = None, *, strict: bool = False, **filters
    ) -> DicomStack:
        """short for filter_by_field or filter_by_query

        strict: (queries) raise the errors of comparisons instead of excluding the frames
        """
        if query is not None:
            return self.filter_by_query(query, strict=strict)
        return self.filter_by_field(**filters)

    def __getitem__(self, items):
        """stack[int]: frame, stack[slice]: substack, else short for get_field_values"""
        if isinstance(items, int) and not isinstance(items, bool):
            return self.frames[items]
        elif isinstance(items, slice):
            return self.from_frames(self.frames[items], root=self.root)
        elif DicomTag.is_tag(items) or not isinstance(items, (tuple, list)):
            items = [items]
        return self.get_field_values(*items)

    def __getattr__(self, name):
        """get stack attribute or init selector"""
        if name[0].isalpha() and name[0].upper() == name[0]:
            return self.select(name)
        return getattr(super(), name)

    def single(self, *fields: Field, default: Any = ..., precision: Optional[int] = None) -> Any:
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

    def unique(self, *fields: Field) -> list:
        """return unique values for field"""
        return sorted(set(self.get_field_values(*fields, ignore_missing=True)))

    def filter_by_field(self, **filters) -> DicomStack:
        """return a sub stack with matching values for the given field"""
        LOGGER.debug("Filter by fields: %s", filters)
        frames = self._filter(filters)
        return self.from_frames(frames, root=self.root)

    def filter_by_query(self, query: Query, strict: bool = False) -> DicomStack:
        """return a sub stack with from frames with validated queries

        strict: raise the errors of comparisons (TypeError, AttributeError, e.g. comparing
            a number with a string) instead of excluding the frames with a warning
        """
        LOGGER.debug("Filter by query: %s", query)
        frames = [frame for frame in self.frames if query.execute(frame.get, strict=strict)]
        return self.from_frames(frames, root=self.root)

    def get_field_values(
        self, field: Field, *fields: Field, ignore_missing: bool = False
    ) -> list:
        """return a list a values for the given fields"""
        _fields = (field,) + fields
        LOGGER.debug("Get fields' values: %s", _fields)

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
        frames = []
        for frame in self.frames:
            index = tuple(frame[key] for key in keys)
            if index in indices:
                continue  # skip
            indices.add(index)
            frames.append(frame)

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

    def sort(self, *fields: Field, missing: str = "drop") -> DicomStack:
        """reindex database using field values

        missing: frames with missing values:
            "drop" (removed), "last" (kept at the end, in their order), "error" (NoValueError)
        """
        LOGGER.debug("Sort by fields: %s", fields)
        if not missing in ("drop", "last", "error"):
            raise ValueError(f"Invalid value for missing: {missing}")
        frames = self._existing(fields)
        sorted_frames = sorted(frames, key=lambda f: tuple(f[field] for field in fields))
        if len(frames) < len(self):
            if missing == "error":
                raise NoValueError(f"Missing values for {fields} in {len(self) - len(frames)} frames")
            elif missing == "last":
                existing = set(map(id, frames))
                sorted_frames += [frame for frame in self.frames if not id(frame) in existing]
        return self.from_frames(sorted_frames, root=self.root)

    def groupby(self, *fields: Field) -> Iterator[Tuple[Any, DicomStack]]:
        """iterate (values, substack) for each value of the field(s), in sorted order

        values: a value (one field) or a tuple of values; frames with missing values are skipped
        """
        groups = {}
        for frame in self._existing(fields):
            values = tuple(frame[field] for field in fields)
            groups.setdefault(values[0] if len(fields) == 1 else values, []).append(frame)
        for values in sorted(groups):
            yield values, self.from_frames(groups[values], root=self.root)

    def series(self) -> Iterator[Tuple[str, DicomStack]]:
        """iterate (SeriesInstanceUID, substack) for each series"""
        return self.groupby("SeriesInstanceUID")

    def getaxis(self) -> int:
        """return the image acquisition axis with respect to the patient"""
        orient = self.single("ImageOrientationPatient", precision=3)
        vec = (
            abs(orient[1] * orient[5] - orient[2] * orient[4]),
            abs(orient[0] * orient[5] - orient[2] * orient[3]),
            abs(orient[0] * orient[4] - orient[1] * orient[3]),
        )
        return vec.index(max(vec))

    def reorder(self) -> DicomStack:
        """reindex stack based on spatial information"""
        if len(self) <= 1:
            return self
        # use ImagePositionPatient
        axis = self.getaxis()
        field = Selector("ImagePositionPatient")[axis]
        if len(self.unique(field)) == len(self):
            return self.sort(field)
        else:
            raise ValueError(
                "Could not sort stack spatially: non-unique slice locations"
            )

    def has_field(self, field: Field, how: str = "all") -> bool:
        """return True if all frame have the given field"""
        func = all if how == "all" else any
        return func(frame.get(field) is not None for frame in self.frames)

    def iter_pixels(self, workers: Optional[int] = WORKERS) -> Iterator[np.ndarray]:
        """iterate the frames' pixel arrays

        workers: number of threads reading the files ahead (I/O only; None: sequential)
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
        workers=WORKERS,
        stack=False,
    ):
        """as volume

        by: field(s): one volume per value, returns (values, volumes)
        tol: relative tolerance on the uniformity of the slice spacing
        strict: if True, raise ValueError on non-uniform slice spacing (else warn)
        dtype: data type of the volume (default: float64)
        workers: number of threads reading the pixels (I/O only; None: sequential)
        stack: (with `by`) return a single volume (x, y, z, n) instead, with the values
            of the `by` fields in vol.metadata[field] (as strings)
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

        # else: indexed volumes, one per (sorted) value of the `by` field(s)
        by = [by] if isinstance(by, str) else list(by)
        groups = list(self.groupby(*by))
        indices = [values for values, _ in groups]
        volumes = [make_volume(substack) for _, substack in groups]
        if not stack:
            return indices, volumes
        elif not volumes:
            return None
        return stack_volumes(volumes, by, indices)

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
            except (KeyError, IndexError):
                # missing field or index: skip (as frame.get)
                continue
            if not all(v in m for v, m in zip(values, matchlist)):
                continue
            filtered.append(frame)
        return filtered

    def select(self, sel: Union[str, Tuple[int, int]]) -> Selector:
        """return query Selector (same as DICOM[sel])"""
        return DICOM[sel]

    def _check_duplicates(self, duplicates):
        """duplicated frames: 'remove', 'error', 'ignore'"""
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


def stack_volumes(volumes, fields, indices, tol=1e-6):
    """stack volumes with the same geometry in a single (x, y, z, n) volume

    the values (indices) of `fields` are stored in the volume's metadata (as strings)
    """
    first = volumes[0]
    for volume in volumes[1:]:
        if volume.shape != first.shape or not all(
            np.allclose(getattr(volume, name), getattr(first, name), atol=tol)
            for name in ("origin", "spacing", "transform")
        ):
            raise ValueError("Cannot stack volumes with different shapes or geometries")
    if len(fields) == 1:
        metadata = {fields[0]: str(list(indices))}
    else:
        metadata = {field: str([values[i] for values in indices]) for i, field in enumerate(fields)}
    return pixeldata.DicomVolume(
        np.stack([np.asarray(volume) for volume in volumes], axis=-1),
        origin=first.origin,
        spacing=first.spacing,
        transform=first.transform,
        metadata=metadata,
    )
