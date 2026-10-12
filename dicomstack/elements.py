""" access to DICOM elements: tags, field keys, value conversion, lookup in pydicom datasets

Converted values (see parse_element_value) are what DicomFrame[...] returns.
The lookup (DatasetLookup, FileLookup) converts only the requested elements, with the same
results as the full conversion (DicomDataset.from_dataset).
"""

import re
import datetime
import itertools
import functools

import pydicom

from .query import Selector

# tags
SHARED_GROUPS = 0x52009229  # SharedFunctionalGroupsSequence
PER_FRAME_GROUPS = 0x52009230  # PerFrameFunctionalGroupsSequence
PIXEL_DATA = 0x7FE00010


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

    def __int__(self):
        return (self.group << 16) | self.element

    @classmethod
    def is_tag(cls, obj):
        return as_tag(obj) is not None


def as_tag(key):
    """DicomTag of a tag key (DicomTag or (group, element) tuple), or None"""
    if isinstance(key, DicomTag):
        return key
    elif not isinstance(key, tuple):
        return None
    try:
        return DicomTag(*key)
    except (ValueError, TypeError):
        return None


class DicomTagSelector:
    """DICOM.Keyword, DICOM["[Private Name]"], DICOM[(group, element)]: query selectors"""

    def __getitem__(self, sel):
        tag = as_tag(sel)
        return Selector(sel if tag is None else tag)

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
        tag = as_tag(key)
        if tag is not None:
            # get value by tag
            element = self.elements_by_tag[tag]

        elif isinstance(key, str):
            # get value by keyword
            element = self.elements_by_keyword[key]
        else:
            raise TypeError(f"Invalid key type: {key}")

        # return value
        return element


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


def element_keyword(element):
    """keyword of a pydicom element, or automatic keyword of a named private element"""
    keyword = element.keyword
    if not keyword and element.name.startswith("["):
        keyword = re.sub(r"[^\w\[\]]|_", "", element.name)
    return keyword


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
        tag = as_tag(key)
        if tag is not None:
            element = self.by_tag(int(tag))
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

        tag = as_tag(key)
        if tag is not None:
            group_key = int(tag)
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


def get_leaf(node):
    """value of node, as DicomFrame.__getitem__ returned it"""
    if isinstance(node, pydicom.DataElement):
        return parse_element_value(node.value, node.VR)
    elif isinstance(node, DatasetLookup):
        return DicomDataset.from_dataset(node.dataset)
    elif isinstance(node, DicomElement):
        return node.value
    return node


def get_nframe(dataset):
    """return number of frames if several"""
    if not PER_FRAME_GROUPS in dataset:
        return None
    frames = dataset[PER_FRAME_GROUPS].value
    if frames:
        return len(frames)
    # else
    return None
