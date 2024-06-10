""" Pixel data manipulation """

# coding=utf-8
import math

try:
    import numpy as np
except ImportError:
    # no pixel array support
    AVAILABLE = False
else:
    AVAILABLE = True


def check_available():
    """check functions are available"""
    if not AVAILABLE:
        raise NotImplementedError("numpy is required")


def available(func):
    """decorator"""

    def wrapped(*args, **kwargs):
        check_available()
        return func(*args, **kwargs)

    return wrapped


@available
def make_volume(frames, rescale=True):
    """return volume from a sequence of frames"""

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
    transform = (ax1, ax2, ax3)
    spacing = first["PixelSpacing"] + (spacing3,)

    tags = {
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
        pixels = frame.pixels
        if rescale:
            slope = frame.get("RescaleSlope", default=1)
            intercept = frame.get("RescaleIntercept", default=0)
            pixels = pixels * slope + intercept
        slices.append(pixels)
    return DicomVolume(slices, tags).T


if AVAILABLE:

    class DicomVolume(np.ndarray):
        """simple layer over np ndarray to add attribute: volume.tags"""

        def __new__(cls, input_array, tags=None):
            """create Volume object"""
            # copy the data
            obj = np.asarray(input_array).view(cls)
            obj.tags = tags
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self.tags = getattr(obj, "tags", {})

        def __reduce__(self):
            """prevent pickling"""
            raise NotImplementedError("Cannot pickle DicomVolume object")

        def __array_wrap__(self, out_arr, context=None):
            """propagate metadata if wrap is called"""
            return super().__array_wrap__(self, out_arr, context)

        def __array_wrap__(self, out_arr, context=None):
            if self.shape != out_arr.shape:
                # if not same shape: drop metadata
                return out_arr
            # else wrap out_array
            return np.ndarray.__array_wrap__(self, out_arr, context)


@available
def format_pixels(data, dtype="uint8"):
    """make tags from ndarray"""
    tags = {}
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("Invalid data shape: %s", data.shape)

    # shape
    rows, cols = data.shape
    tags["Rows"] = rows
    tags["Columns"] = cols

    # geometry
    volumetags = getattr(data, "tags", {})
    origin = volumetags.get("origin", (0, 0, 0))
    spacing = volumetags.get("spacing", (1, 1, 1))
    transform = volumetags.get("transform", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    tags["PixelSpacing"] = list(spacing[:2])
    tags["ImagePositionPatient"] = list(origin)
    tags["ImageOrientationPatient"] = list(transform[0]) + list(transform[1])

    # dynamics
    data_max = data.max()
    data_min = data.min()
    reftype = np.dtype(dtype)
    if not np.issubsctype(data, np.integer) or data.itemsize > reftype.itemsize:

        max_val = np.iinfo(reftype).max
        min_val = np.iinfo(reftype).min

        # stored = (data - intercept) / slope
        slope = (data_max - data_min) / (max_val - min_val)
        intercept = data_min - min_val * slope
        tags["RescaleSlope"] = slope
        tags["RescaleIntercept"] = intercept
        data = ((data - intercept) / slope).astype(reftype)

    # data type
    if np.issubsctype(data, np.signedinteger):
        tags["PixelRepresentation"] = 1
    elif np.issubsctype(data, np.unsignedinteger):
        tags["PixelRepresentation"] = 0
    else:
        raise ValueError("Invalid data type: %s", data.dtype)
    tags["BitsAllocated"] = data.itemsize * 8
    tags["BitsStored"] = data.itemsize * 8
    tags["SamplesPerPixel"] = 1

    # return tags
    return tags, data.tobytes()
