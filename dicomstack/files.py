""" DICOM files and frames: reading headers and pixels, loading folders and zip files """

import os
import gc
import sys
import copy
import fnmatch
import glob
import pathlib
import re
import hashlib
import itertools
import collections
import concurrent.futures
import zipfile
import logging
import warnings
from io import BytesIO

import pydicom
from pydicom import config
from pydicom.dataelem import RawDataElement

from . import pixeldata, cache
from .elements import (
    DicomDataset,
    FileLookup,
    compile_keys,
    get_leaf,
    get_nframe,
    get_subitem,
)

LOGGER = logging.getLogger(__name__)

# exceptions
InvalidDicomError = pydicom.errors.InvalidDicomError

DEDUP_MODES = ("uid", "content")
SOP_INSTANCE_UID = 0x00080018
SEQUENCE_DELIMITER = b"\xfe\xff\xdd\xe0"


class Loader:
    """load DICOM files (regular files, zip members) into frames, using the header cache

    root: folder of the (relative) filenames
    dedup, workers, cache: see DicomStack
    """

    def __init__(self, root=None, dedup="uid", workers=None, cache=None, progress=None):
        self.root = root
        self._dedup = check_dedup(dedup)
        self._workers = workers
        self._cache = cache
        self._progress = progress
        self.frames = []
        self.non_dicom = []

    def load(self, filenames):
        """load filenames: returns frames, non-DICOM files

        The garbage collector is paused while loading: the headers kept in memory create
        many objects, which full collections would scan repeatedly (no reference cycles
        are created here).
        """
        enabled = gc.isenabled()
        gc.disable()
        try:
            return self._load(filenames)
        finally:
            if enabled:
                gc.enable()

    def _load(self, filenames):
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

        progress = make_progress(self._progress, len(items))
        for done, item in enumerate(items, start=1):
            if item in archives:
                self._load_zipfile(item, *archives[item])
            elif item in cached:
                self._load_cached(item, self._filepath(item), cached[item])
            else:
                self._load_file(item, next(contents))
            progress(done)
        progress(None)

        if self._cache_new is not None:
            if self._cache_changed or set(self._cache_new) != set(self._cache_old):
                cache.save(cache_paths, self._dedup, self._cache_new)
        del self._cache_old, self._cache_new, self._cache_changed
        return self.frames, self.non_dicom

    def _filepath(self, filename):
        """full path of filename"""
        if self.root:
            return os.path.join(self.root, filename)
        return str(filename)

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
            # read dicom object (private values deferred after building the cache entry)
            caching = self._cache_new is not None
            dicomfile = DicomFile(
                filepath, bytes=data, dedup=self._dedup, defer_private=not caching
            )
            frames = dicomfile.frames
        except (IOError, InvalidDicomError):
            # other files
            self.non_dicom.append(filename)
            self._cache_add(self._cache_key(filename), stat)
        else:
            self.frames.extend(frames)
            self._cache_add(self._cache_key(filename), stat, dicomfile)
            dicomfile.defer_private_values()

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
    _memory = None  # in-memory pydicom dataset (from_dataset)

    def __init__(self, filename, bytes=None, *, source=None, dedup="uid", defer_private=True):
        """init DicomFile object (the header is read on demand)

        bytes: contents of the file (e.g. zip member), read now but not kept
        defer_private: private values are not kept in memory: read from the file when
            accessed (regular files only; see defer_private_values)
        """
        self.filename = filename
        self.source = str(filename) if source is None else source
        self.dedup = check_dedup(dedup)
        self.digest = None
        self.pixel_offset = None
        self._defer_private = defer_private
        if bytes is not None:
            self._read_header(bytes)

    def defer_private_values(self):
        """drop the values of the private elements of the header (read again when accessed)

        Only for regular files whose header was parsed from the file's own bytes
        (not zip members, deflated files, or files completed with a sequence delimiter).
        """
        if self._dataset is None or self.pixel_offset is None or not isinstance(self.source, str):
            return
        defer_private(self._dataset, self.source)

    @classmethod
    def from_cache(cls, filename, entry, *, source=None, dedup="uid"):
        """DicomFile from a header cache entry (the header is read on demand)"""
        dicomfile = cls(filename, source=source, dedup=dedup)
        dicomfile._cache = entry
        dicomfile._identity = entry["identity"]
        dicomfile.pixel_offset = entry["offset"]
        return dicomfile

    @classmethod
    def from_dataset(cls, dataset, filename, *, dedup="uid"):
        """DicomFile from an in-memory pydicom dataset (with its pixel data)"""
        if check_dedup(dedup) == "content":
            raise ValueError("dedup='content' needs files, not in-memory datasets")
        dicomfile = cls(filename, dedup=dedup)
        dicomfile.source = None
        dicomfile._memory = dataset
        dicomfile._dataset = dataset
        return dicomfile

    def full_dataset(self):
        """the file's complete pydicom dataset (with pixel data), read again"""
        if self._memory is not None:
            return copy.deepcopy(self._memory)
        dataset, _ = read_dataset(self._read_bytes(), stop_before_pixels=False)
        return dataset

    def frame_keywords(self, index=None):
        """keywords of the elements of frame `index`"""
        if self._lookup is None:
            self._lookup = FileLookup(self.dataset)
        return list(self._lookup.frame_elements(index)[1])

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
            elif self._memory is not None:
                self._identity = ("memory", id(self._memory))
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
        if self._memory is not None:
            return None
        self.dataset  # make sure the header is read
        if self.pixel_offset is None:
            return self._read_bytes()
        with open(self.source, "rb") as fp:
            fp.seek(self.pixel_offset)
            return fp.read()

    def _decode_pixels(self, data):
        """pixel array from the bytes returned by _read_pixel_bytes"""
        if self._memory is not None:
            return self._memory.pixel_array
        elif self.pixel_offset is None:
            # zip member, or unknown offset: parse the whole file
            dataset, _ = read_dataset(data, stop_before_pixels=False)
            return dataset.pixel_array
        return pixeldata.decode_pixels(self.dataset, data)

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
        if self._defer_private:
            self.defer_private_values()

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

    # fields shown by repr (all elements: print(frame.dataset))
    REPR_FIELDS = ("Modality", "SeriesNumber", "SeriesDescription", "InstanceNumber")

    def __repr__(self):
        """DicomFrame(path/IM1.dcm, MR, SeriesNumber=5, ...)"""
        source = str(self.dicomfile.filename)
        if self.index is not None:
            source += f" [{self.index}]"
        items = [source]
        for field in self.REPR_FIELDS:
            try:
                value = self.get(field)
            except Exception:  # (e.g. unreadable file)
                value = None
            if value is not None:
                items.append(value if field == "Modality" else f"{field}={value!r}")
        return f"DicomFrame({', '.join(map(str, items))})"

    def keywords(self):
        """keywords of the frame's elements (incl. automatic keywords of private ones)"""
        cached = self.dicomfile.cached_frame(self.index)
        if cached is not None:
            return list(cached["k"])
        return self.dicomfile.frame_keywords(self.index)

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
            value = cache.get_cached_value(cached, keys)
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


def defer_private(dataset, path):
    """drop the values of the top-level private elements of dataset, except private creators

    On access, pydicom reads a deferred value (raw element without value) from the file
    `path`, at the position recorded when parsing (the dataset must have been parsed
    from the whole file's bytes).
    """
    deferred = 0
    for tag in list(dataset.keys()):
        if not tag.is_private or tag.is_private_creator:
            continue
        raw = dataset.get_item(tag, keep_deferred=True)
        if isinstance(raw, RawDataElement) and raw.value is not None and raw.length:
            # (Dataset.__setitem__ would convert private elements: set the raw element)
            dataset._dict[tag] = raw._replace(value=None)
            deferred += 1
    if deferred:
        dataset.filename = path
        dataset.fileobj_type = open
        dataset.timestamp = os.stat(path).st_mtime  # pydicom warns if the file changed


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


def list_files(pathes, recursive=True):
    """list all files in path and (if recursive) its sub folders"""
    if not isinstance(pathes, (list, tuple)):
        pathes = [pathes]

    files = []
    bases = []  # paths found (glob patterns expanded)
    for path in map(pathlib.Path, pathes):
        zip_path = get_zip_path(path)
        if "*" in str(path):
            subs = sorted(map(pathlib.Path, glob.glob(str(path))))
        elif path.exists():
            subs = [path]
        elif zip_path and os.path.isfile(zip_path):
            # member or folder inside a zip file: loaded as a "file", root: the zip's folder
            bases.append(pathlib.Path(zip_path))
            files.append(path)
            continue
        else:
            raise FileNotFoundError(path)

        for subpath in subs:
            bases.append(subpath)
            if subpath.is_file():
                files.append(subpath)
            else:
                found = subpath.rglob("*") if recursive else subpath.glob("*")
                # sorted: same order on all file systems
                files.extend(sorted(file for file in found if file.is_file()))

    if not files:  # empty
        return None, []

    # find root directory
    root = pathlib.Path(os.path.commonpath(bases))
    if root.is_file():
        root = root.parent
    filenames = [str(file.relative_to(root)) for file in files]
    return str(root), filenames


def filter_filenames(filenames, include=None, exclude=None):
    """filenames matching any `include` pattern and no `exclude` pattern

    patterns (str or list of str, fnmatch syntax): matched against the file name and
    against the whole (relative) path with "/" separators, e.g. "*.dcm", "series1/*"
    """

    def matches(filename, patterns):
        if isinstance(patterns, str):
            patterns = [patterns]
        path = str(filename).replace("\\", "/")
        name = path.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(path, p) for p in patterns)

    return [
        filename
        for filename in filenames
        if (include is None or matches(filename, include))
        and (exclude is None or not matches(filename, exclude))
    ]


def make_progress(progress, total):
    """progress function (done, or None when finished)

    progress: a callable(done, total), True (text progress bar on stderr),
        "auto" (progress bar if stderr is a terminal), or None/False (no progress)
    """
    if progress == "auto":
        progress = bool(getattr(sys.stderr, "isatty", None) and sys.stderr.isatty())
    if progress is None or progress is False:
        return lambda done: None
    elif progress is True:
        from .utils import ProgressBar

        bar = ProgressBar(total)

        def update(done):
            if done is None:
                bar.close()
            else:
                bar.update(done)

        return update

    def call(done):
        if done is not None:
            progress(done, total)

    return call
