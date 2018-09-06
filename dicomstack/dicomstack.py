""" pydicom wrapper class for simple manipulation of DICOM data """
# coding=utf-8

import os
import pydicom
import re
import glob
import functools
import zipfile
from io import BytesIO

import tinydb

try:
    import numpy

    HAS_NUMPY = True
except ImportError:
    # no pixel array support
    HAS_NUMPY = False


class DICOMStack(object):
    """ load, sort and filter DICOM images
    """

    def __init__(self, path=None, filenames=None):
        """ path can be:
            * a directory (or a list of),
            * a file (or a list of)
            * a zip file
        """
        self.db = tinydb.TinyDB(storage=tinydb.storages.MemoryStorage)
        self.non_dicom = []

        if not filenames and not path:
            # empty DICOMStack
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

    def __repr__(self):
        return "DICOMStack(%d)" % len(self.db)

    def __getitem__(self, fields):
        # if len(fields) == 1 and isinstance(fields[0], int):
        # return self.g(fields[0])
        if not isinstance(fields, tuple):
            fields = (fields,)
        return self.get_field_values(*fields)

    def __call__(self, **filters):
        return self.filter_by_field(**filters)

    @classmethod
    def from_elements(cls, elements):
        """ create a new DICOMStack from a db object """
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
        return DICOMStack.from_elements(elements)

    def get_field_values(self, *fields):
        """ return a list a values for the given fields """
        elements = self._existing(*fields)
        return [_get_values(element, fields=fields) for element in elements]

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
            condition = cond if not condition else cond & condition
        return self.db.search(condition)

    def _filter(self, **filters):
        """ return elements filtered by fields """
        query = tinydb.Query()
        condition = None
        for name, value in filters.items():
            condition = None
            for name, value in filters.items():
                field, index = _parse_field(name)
                cond = query[field].value

                if index is not None:
                    cond = cond[index]
                if not isinstance(value, list):
                    value = [value]
                cond = cond.one_of(value)

            condition = cond if not condition else cond & condition
        return self.db.search(condition)

    def _index(self, index):
        """ return element matching index """
        query = tinydb.Query()
        return self.db.get(query.index == index)

    def sort(self, *fields):
        """ reindex database using field values """
        elements = self._existing(*fields)
        sort_key = functools.partial(_get_values, fields=fields)
        sorted_elements = sorted(elements, key=sort_key)
        for i, element in enumerate(sorted_elements):
            element["index"] = i
        return DICOMStack.from_elements(sorted_elements)

    def as_volume(self, by=None, rescale=True):
        """ as volume """
        assert HAS_NUMPY

        # sort by position
        stack = self.sort("SliceLocation")

        vols = []
        if not by:
            for element in stack.elements:
                slope, intercept = 1, 0
                if rescale:
                    slope = element.get("RescaleSlope", {}).get("value", 1)
                    intercept = element.get("RescaleSlope", {}).get("value", 0)
                vols.append(element["array"].T * slope + intercept)
            return numpy.asarray(vols).T

        # unique values
        if isinstance(by, str):
            by = [by]
        unique = sorted(set(stack.get_field_values(*by)))
        if len(by) == 1:
            allfilters = [{by[0]: value} for value in unique]
        else:
            allfilters = [
                dict((field, value) for field, value in zip(by, values))
                for values in unique
            ]
        for filters in allfilters:
            vols_ = []
            for element in stack.filter_by_field(**filters).elements:
                slope, intercept = 1, 0
                if rescale:
                    slope = element.get("RescaleSlope", {}).get("value", 1)
                    intercept = element.get("RescaleSlope", {}).get("value", 0)
                vols_.append(element["array"].T * slope + intercept)
            vols.append(numpy.asarray(vols_).T)

        return unique, vols

    # def geometry(self):
    #     """ return dict of geometrical properties """
    #    TODO

    def load_files(self, filenames):
        """ load filenames """

        filenames = sorted(set(os.path.normpath(f) for f in filenames))
        for filename in filenames:
            zip_path = get_zip_path(filename)
            if zip_path:
                self.load_zipfile(filename)
                continue

            elif not pydicom.misc.is_dicom(filename):
                # if not DICOM
                self.non_dicom.append(filename)
                continue

            dicom_obj = pydicom.dcmread(filename)
            dicom_dict = load_dicom_dataset(dicom_obj)
            dicom_dict["filename"] = filename
            dicom_dict["index"] = len(self.db)
            self.db.insert(dicom_dict)

    def load_zipfile(self, filename):
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
                    dicom_obj = pydicom.read_file(dicom_bytes)
                except IOError:
                    self.non_dicom.append(zfilename)

                dicom_dict = load_dicom_dataset(dicom_obj)
                dicom_dict["filename"] = full_zfilename
                dicom_dict["index"] = len(self.db)
                self.db.insert(dicom_dict)


def get_zip_path(path):
    """ return the zip-file root of path """
    if not ".zip" in path:
        return None
    return path[: path.find(".zip") + 4]


# parse: <FIELD_NAME>_<NUM>
RE_PARSE_FIELD = re.compile(r"([a-zA-Z]+)(?:_(\d+))?")


def _parse_field(name):
    """ parse field name with index"""
    match = RE_PARSE_FIELD.match(name)
    if not match:
        raise ValueError('Cannot parse field: "%s"' % name)
    field, index = match.groups()
    if index:
        return field, int(index)
    return field, None


def _get_values(element, fields):
    """ get element value"""
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


def list_files(dirpath):
    """ list all files in dirpath and its sub folders """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(dirpath):
        all_files.extend([os.path.join(dirpath, name) for name in filenames])
    return all_files


def load_dicom_dataset(dataset):
    """ load all dicom fields in dataset"""
    def _list_dicom_elements(dataset):
        elements, sequences = [], []
        for item in dataset:

            if item.keyword == "PixelData":
                # skip
                continue

            target = sequences if item.VR == "SQ" else elements
            target.append(item)

        for sequence in sequences:
            for subdataset in sequence:
                elements.extend(_list_dicom_elements(subdataset))
        return elements

    elements = _list_dicom_elements(dataset)

    def _get_element_info(element):
        def _cast_element_value(value):
            if isinstance(value, pydicom.multival.MultiValue):
                return [_cast_element_value(v) for v in value]
            elif isinstance(value, pydicom.valuerep.DSfloat):
                if value.is_integer():
                    return int(value)
                return value.real
            return str(value)

        return {
            "name": element.name,
            "tag": element.tag,
            "value": _cast_element_value(element.value),
        }

    dicom_dict = dict(
        (element.keyword, _get_element_info(element)) for element in elements
    )
    if HAS_NUMPY:
        dicom_dict["array"] = dataset.pixel_array

    return dicom_dict
