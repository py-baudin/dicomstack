""" header cache file: converted values of the DICOM elements, to skip parsing on later loads

The cache is a gzip-compressed JSON file (no pickle: a cache in a shared folder must not
be able to execute code). It is valid for a file as long as the file's size and
modification time (and CRC for zip members) are unchanged.

Per frame, the converted values (same rules as frame[...]) are stored by tag ("gggg,eeee"),
with a keyword -> tag map. Sequences, bytes and long values are not stored but marked as
present: accessing them reads the file's header.
"""

import os
import sys
import gzip
import json
import hashlib
import logging
import tempfile
import warnings
import importlib.metadata

import pydicom

from .elements import as_tag, parse_element_value

LOGGER = logging.getLogger(__name__)

FORMAT_VERSION = 1
CACHE_NAME = ".dicomstack-cache.json.gz"
MAX_STRING = 1024  # longer strings are not cached
MAX_ITEMS = 128  # longer multi-values are not cached


class NotCached:
    """marker: the element exists but its value is not in the cache"""

    def __repr__(self):
        return "NOT_CACHED"


NOT_CACHED = NotCached()


def header():
    """validity header of the cache file"""
    try:
        version = importlib.metadata.version("dicomstack")
    except importlib.metadata.PackageNotFoundError:
        version = "0"
    return {
        "format": FORMAT_VERSION,
        "dicomstack": version.split(".")[0],
        "pydicom": pydicom.__version__.split(".")[0],
    }


#
# location


def user_cache_dir():
    """user cache directory for dicomstack"""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "dicomstack")


def locations(root, cache):
    """candidate cache files, in order of preference

    cache: True or "refresh" (default locations), or a path
    """
    if cache not in (True, "refresh"):
        return [os.path.abspath(str(cache))]
    if root is None:
        return []
    root = os.path.abspath(root)
    digest = hashlib.sha1(root.encode("utf-8")).hexdigest()
    return [os.path.join(root, CACHE_NAME), os.path.join(user_cache_dir(), digest + ".json.gz")]


#
# read / write


def load(paths, dedup):
    """load the first existing cache file: returns its entries (or {})"""
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fp:
                data = json.load(fp, object_hook=decode_object)
        except (OSError, ValueError, EOFError) as exc:
            warnings.warn(f"Invalid cache file (rebuilt): {path} ({exc})")
            return {}
        expected = dict(header(), dedup=dedup)
        if not isinstance(data, dict) or data.get("header") != expected:
            LOGGER.info("Outdated cache file (rebuilt): %s", path)
            return {}
        LOGGER.info("Using cache file: %s", path)
        return data.get("files", {})
    return {}


def save(paths, dedup, entries):
    """write the cache to the first writable location"""
    data = {"header": dict(header(), dedup=dedup), "files": entries}
    text = json.dumps(encode(data), default=encode_object, separators=(",", ":"))
    for path in paths:
        try:
            write_atomic(path, gzip.compress(text.encode("utf-8"), compresslevel=6))
        except OSError as exc:
            LOGGER.info("Could not write cache file %s (%s)", path, exc)
            continue
        LOGGER.info("Cache file written: %s", path)
        # remove outdated copies in other locations
        for other in paths:
            if other != path and os.path.isfile(other):
                try:
                    os.remove(other)
                except OSError:
                    pass
        return path
    LOGGER.info("No writable location for the cache file")
    return None


def write_atomic(path, data):
    """write a file atomically (temporary file in the same folder, then rename)"""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".dicomstack-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


#
# entries


def file_stat(path):
    """validity key of a regular file"""
    stat = os.stat(path)
    return [stat.st_size, stat.st_mtime_ns]


def frame_values(lookup, index):
    """cached values of a frame: {"v": {tag: value}, "k": {keyword: tag}}"""
    by_tag, by_keyword = lookup.frame_elements(index)
    values = {}
    for tag, element in by_tag.items():
        if isinstance(element, pydicom.dataelem.RawDataElement):
            if skip_raw(element):
                values[tag] = NOT_CACHED
                continue
            element = lookup.header[tag]  # convert
        values[tag] = cached_value(element, parse_element_value)
    return {
        "v": {tag_key(tag): value for tag, value in values.items()},
        "k": {keyword: tag_key(tag) for keyword, tag in by_keyword.items()},
    }


# value representations of binary values (not cached)
BINARY_VRS = {"OB", "OW", "OD", "OF", "OL", "OV"}


def skip_raw(raw):
    """True if an unconverted element is not cached (saves its conversion)

    (not caching an element is always safe: its value is read from the header on access)
    """
    return raw.VR == "SQ" or raw.VR in BINARY_VRS or raw.length > MAX_STRING


def cached_value(element, convert):
    """converted value of a pydicom element, or NOT_CACHED"""
    if element.VR == "SQ":
        return NOT_CACHED
    value = convert(element.value, element.VR)
    if isinstance(value, bytes):
        return NOT_CACHED
    elif isinstance(value, str) and len(value) > MAX_STRING:
        return NOT_CACHED
    elif isinstance(value, tuple) and (
        len(value) > MAX_ITEMS or not all(is_plain(item) for item in value)
    ):
        return NOT_CACHED
    elif not is_plain(value) and not isinstance(value, tuple):
        return NOT_CACHED
    return value


def get_cached_value(cached, keys):
    """value from the cached values of a frame (frame_values), or NOT_CACHED

    keys: compiled keys (elements.compile_keys)
    raises KeyError for missing elements (the cache lists all elements)
    """
    key = keys[0]
    tag = as_tag(key)
    if tag is not None:
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
    if value is NOT_CACHED:
        return value
    for key in keys[1:]:
        # as get_subitem for non-sequence elements
        value = value[key]
    return value


def is_plain(value):
    return value is None or isinstance(value, (bool, int, float, str))


def tag_key(tag):
    return f"{tag >> 16:04x},{tag & 0xFFFF:04x}"


#
# json encoding: tuples, bytes and NOT_CACHED round-trip


def encode_object(obj):
    """json.dumps default (tuples are handled by `encode`)"""
    if obj is NOT_CACHED:
        return {"__n__": 1}
    elif isinstance(obj, bytes):
        return {"__b__": obj.hex()}
    raise TypeError(f"Cannot encode {type(obj)} in cache")


def encode(value):
    """encode tuples (json would make them lists)"""
    if isinstance(value, tuple):
        return {"__t__": [encode(item) for item in value]}
    elif isinstance(value, list):
        return [encode(item) for item in value]
    elif isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    return value


def decode_object(obj):
    """json.load object_hook"""
    if len(obj) == 1:
        if "__t__" in obj:
            return tuple(obj["__t__"])
        elif "__n__" in obj:
            return NOT_CACHED
        elif "__b__" in obj:
            return bytes.fromhex(obj["__b__"])
    return obj
