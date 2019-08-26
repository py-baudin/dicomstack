
""" store/transmit
    - pickable (store)
        - sequence of elements
        - pixels
        - dataset as bytes (load dataset, element and pixels)
    - jsonable (transmit)
        - only the dataset as bytes
"""

import pydicom

class Dataset:
    """ serializable DICOM dataset """
    def __init__(self, filename):
        with open(filename, "rb") as fp:
            self.bytes = fp.read()
        self.filename = filename
        self._dataset = None
        self._pixels = None

    # pickle
    def __getstate__(self):
        return (self.filename, self.bytes)

    def __setstate__(self, state):
        self.filename, self.bytes = state

    @property
    def pixels(self):
        if not self._pixels:
            self._pixels = self.elements.pixel_array
        return self._pixels

    @property
    def dataset(self):
        if not self._dataset:
            self._dataset = pydicom.dcmread(self.bytes, stop_before_pixels=True)
        return self._dataset


class Element:
    """ serializable DICOM element """
    self.name # full name
    self.keyword # search name
    self.value
    self.repr # VR


class Frame:
    """ serializable DICOM frame container """

    def __init__(self, dataset, values=None, pixels=None, index=None):
        """
            index: frame index in multi-frame (enhanced) DICOM
        """
        self.index = index
        self.dataset = dataset
        self._values = values
        self._pixels = pixels


    def serialize(self):
        """ return json-able version """
        ...

    @property
    def values(self):
        """ retrieve DICOM field values """
        if not self._values:
            # retrieve fields from dataset
            ...
        return self._values

    @property
    def pixels(self):
        """ retrieve pixel data """
        if not self._pixels:
            # load dataset
            pixels = self.dataset.pixels
            if self.index if not None:
                pixels = pixels[self.index]
            self._pixels = pixels
        return self._pixels
