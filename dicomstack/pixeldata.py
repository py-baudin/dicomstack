""" Pixel data manipulation """

# coding=utf-8
import os
import sys
import math
import warnings
from io import BytesIO

import numpy as np
import pydicom

# pixel data read directly (other transfer syntaxes: decoded by pydicom)
NATIVE_SYNTAXES = {pydicom.uid.ImplicitVRLittleEndian, pydicom.uid.ExplicitVRLittleEndian}
ROWS, COLUMNS, BITS_ALLOCATED = 0x00280010, 0x00280011, 0x00280100


def decode_pixels(header, data):
    """pixel array of a DICOM file

    header: the file's dataset without pixel data
    data: the file's bytes from the pixel data element on
    """
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


def make_volume(frames, rescale=True, tol=1e-3, strict=False, dtype=None):
    """return volume from a sequence of frames

    tol: relative tolerance on the uniformity of the slice spacing
    strict: if True, raise ValueError on non-uniform slice spacing (else warn)
    dtype: data type of the volume (default: float64)
    """
    dtype = np.dtype(dtype or "float64")

    # find geometry
    nframe = len(frames)
    first = frames[0]
    last = frames[-1]

    origin = first["ImagePositionPatient"]
    end = last["ImagePositionPatient"]
    ax1 = tuple(first["ImageOrientationPatient"][:3])
    ax2 = tuple(first["ImageOrientationPatient"][3:])
    vec3 = [b - a for a, b in zip(origin, end)]
    norm3 = math.sqrt(sum(value**2 for value in vec3))
    if nframe == 1:
        ax3 = (
            ax1[1] * ax2[2] - ax1[2] * ax2[1],
            ax1[2] * ax2[0] - ax1[0] * ax2[2],
            ax1[0] * ax2[1] - ax1[1] * ax2[0],
        )
        spacing3 = 1
    else:
        ax3 = tuple(value / norm3 for value in vec3)
        spacing3 = norm3 / (nframe - 1)
        check_slice_spacing(frames, origin, ax3, spacing3, tol=tol, strict=strict)
    transform = (ax1, ax2, ax3)
    # PixelSpacing: (row spacing, column spacing); axis 0 runs along a row
    row_spacing, col_spacing = first["PixelSpacing"]
    spacing = (col_spacing, row_spacing, spacing3)

    metadata = {
        # (0,0,0) pixel location in mm
        "origin": tuple(origin),
        # pixel dimensions in mm
        "spacing": tuple(spacing),
        # xyz axes orientation
        "transform": tuple(transform),
    }

    # make volume
    slices = []
    for frame in frames:
        slope, intercept = 1, 0
        pixels = frame.pixels.astype(dtype)
        if rescale:
            slope = frame.get("RescaleSlope", default=1)
            intercept = frame.get("RescaleIntercept", default=0)
            pixels = pixels * slope + intercept
        slices.append(pixels)
    return DicomVolume(slices, **metadata).T


def check_slice_spacing(frames, origin, ax3, spacing3, tol=1e-3, strict=False):
    """check slice positions along ax3 are uniformly spaced by spacing3"""
    positions = [
        sum((b - a) * n for a, b, n in zip(origin, frame["ImagePositionPatient"], ax3))
        for frame in frames
    ]
    steps = [b - a for a, b in zip(positions[:-1], positions[1:])]
    error = max(abs(step - spacing3) for step in steps)
    if error <= tol * spacing3:
        return
    msg = (
        f"Non-uniform slice spacing (from {min(steps):g} to {max(steps):g}, "
        f"volume spacing: {spacing3:g}): missing or irregular slices?"
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=external_stacklevel())


def external_stacklevel():
    """stacklevel of the first caller outside the dicomstack package (for warnings.warn)"""
    package = os.path.dirname(os.path.abspath(__file__))
    frame, level = sys._getframe(2), 2
    while frame is not None and os.path.dirname(frame.f_code.co_filename) == package:
        frame, level = frame.f_back, level + 1
    return level


class DicomVolume(np.ndarray):
    """simple layer over np ndarray to add geometry and metadata

    Same attributes as tomio.volume.Volume (`tomio.asvolume(vol)` converts it):
        spacing: voxel size along each axis
        origin: LPS coordinates of voxel (0, 0, 0)
        transform: LPS direction of each axis (transform[i]: axis i)
        metadata: dict of str
    """

    GEOMETRY = ("origin", "spacing", "transform")

    @property
    def tags(self):
        """geometry as a dict (deprecated: use origin, spacing, transform)"""
        warnings.warn(
            "DicomVolume.tags is deprecated: use origin, spacing and transform",
            DeprecationWarning,
            stacklevel=2,
        )
        return {name: getattr(self, name) for name in self.GEOMETRY}

    def __new__(
        cls,
        input_array,
        *,
        spacing=None,
        origin=None,
        transform=None,
        metadata=None,
        tags=None,
    ):
        """create Volume object"""
        obj = np.asarray(input_array).view(cls)
        if tags is not None:
            warnings.warn(
                "DicomVolume(tags=...) is deprecated: "
                "use origin=, spacing= and transform=",
                DeprecationWarning,
                stacklevel=2,
            )
        geometry = dict(tags or {})
        for name, value in zip(cls.GEOMETRY, (origin, spacing, transform)):
            if value is not None:
                geometry[name] = value
        obj.origin = geometry.get("origin")
        obj.spacing = geometry.get("spacing")
        obj.transform = geometry.get("transform")
        obj.metadata = metadata
        return obj

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        metadata = metadata or {}
        self._metadata = {str(key): str(value) for key, value in metadata.items()}

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for name in self.GEOMETRY:
            setattr(self, name, getattr(obj, name, None))
        self.metadata = getattr(obj, "metadata", None)

    def has_geometry(self):
        return get_geometry(self) is not None

    @property
    def affine(self):
        """4x4 affine transform: voxel indices (i, j, k, 1) -> LPS coordinates"""
        if not self.has_geometry():
            raise ValueError("Volume without geometry")
        affine = np.eye(4)
        for axis in range(3):
            affine[:3, axis] = np.asarray(self.transform[axis], dtype=float) * self.spacing[axis]
        affine[:3, 3] = self.origin
        return affine

    def __getitem__(self, key):
        """geometry-aware slicing (as tomio.volume.Volume)

        slices on the spatial axes (first 3): geometry updated
        other indexing on the spatial axes (integer, array, None): plain ndarray
        """
        result = super().__getitem__(key)
        if not isinstance(result, DicomVolume) or self.ndim < 3:
            return result
        slices = get_spatial_slices(key, self.ndim)
        if slices is None:
            return result.view(np.ndarray)
        elif not self.has_geometry():
            return result

        starts, steps = [], []
        for sl, size in zip(slices, self.shape[:3]):
            start, _, step = sl.indices(size)
            starts.append(start)
            steps.append(step)
        result.origin = tuple(
            float(o + sum(i * s * t[d] for i, s, t in zip(starts, self.spacing, self.transform)))
            for d, o in enumerate(self.origin)
        )
        result.spacing = tuple(
            s * abs(step) for s, step in zip(self.spacing, steps)
        ) + tuple(self.spacing[3:])
        result.transform = tuple(
            tuple(v * (1 if step > 0 else -1) for v in t)
            for t, step in zip(self.transform, steps)
        )
        return result

    def __reduce__(self):
        """prevent pickling"""
        raise NotImplementedError("Cannot pickle DicomVolume object")

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        if self.shape != out_arr.shape:
            # if not same shape: drop metadata
            return out_arr
        # else wrap out_array
        return super().__array_wrap__(out_arr, context, return_scalar)


def get_geometry(data):
    """(origin, spacing, transform) of a volume, or None"""
    geometry = [getattr(data, name, None) for name in ("origin", "spacing", "transform")]
    if any(value is None for value in geometry):
        return None
    return geometry


def ds_value(value, digits=10):
    """float fitting in a DICOM DS value (16 characters)"""
    return float(f"{float(value):.{digits}g}")


def get_spatial_slices(key, ndim):
    """slices of the first 3 axes in key, or None if not all simple slices"""
    key = key if isinstance(key, tuple) else (key,)
    if any(k is None for k in key):
        # new axes: axis numbers change
        return None
    ellipsis = [i for i, k in enumerate(key) if k is Ellipsis]
    if len(ellipsis) > 1:
        return None
    elif ellipsis:
        i = ellipsis[0]
        key = key[:i] + (slice(None),) * (ndim - len(key) + 1) + key[i + 1 :]
    key = key + (slice(None),) * (3 - len(key))
    if not all(isinstance(k, slice) for k in key[:3]):
        return None
    return key[:3]


def format_pixels(data, dtype="uint8"):
    """return DICOM tags and pixel bytes of a single image

    data:
        - a volume with geometry (DicomVolume, tomio.Volume) of shape (x, y) or (x, y, 1),
            e.g. `vol[:, :, k:k+1]`: the image is written with rows along y and columns along x,
            and its geometry sets PixelSpacing, ImagePositionPatient and ImageOrientationPatient
        - or a 2d array (rows, columns) without geometry: no geometry tags
    """
    tags = {}
    geometry = get_geometry(data)
    data = np.asarray(data)

    if geometry:
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[..., 0]
        if data.ndim != 2:
            raise ValueError("Invalid data shape: %s (expected (x, y) or (x, y, 1))" % (data.shape,))
        # (x, y) -> (rows, columns)
        data = data.T
        origin, spacing, transform = geometry
        tags["PixelSpacing"] = [ds_value(spacing[1]), ds_value(spacing[0])]
        tags["ImagePositionPatient"] = [ds_value(v) for v in origin]
        tags["ImageOrientationPatient"] = [
            ds_value(v) for v in list(transform[0]) + list(transform[1])
        ]
    elif data.ndim != 2:
        raise ValueError("Invalid data shape: %s" % (data.shape,))

    # shape
    rows, cols = data.shape
    tags["Rows"] = rows
    tags["Columns"] = cols

    # dynamics
    data_max = data.max()
    data_min = data.min()
    reftype = np.dtype(dtype)
    if not np.issubdtype(data.dtype, np.integer) or data.itemsize > reftype.itemsize:

        max_val = np.iinfo(reftype).max
        min_val = np.iinfo(reftype).min

        # stored = (data - intercept) / slope
        slope = (data_max - data_min) / (max_val - min_val)
        intercept = data_min - min_val * slope
        tags["RescaleSlope"] = slope
        tags["RescaleIntercept"] = intercept
        data = ((data - intercept) / slope).astype(reftype)

    # data type
    if np.issubdtype(data.dtype, np.signedinteger):
        tags["PixelRepresentation"] = 1
    elif np.issubdtype(data.dtype, np.unsignedinteger):
        tags["PixelRepresentation"] = 0
    else:
        raise ValueError("Invalid data type: %s", data.dtype)
    tags["BitsAllocated"] = data.itemsize * 8
    tags["BitsStored"] = data.itemsize * 8
    tags["SamplesPerPixel"] = 1

    # return tags
    return tags, data.tobytes()
