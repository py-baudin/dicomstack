# dicomstack

`pydicom` wrapper for easy loading and filtering of DICOM images.

Basic usage:

``` python
    from dicomstack import DICOMStack

    stack = DICOMStack(path)
    # or:
    # stack = DICOMStack(filenames=[list, of, files])

    stack
    > DICOMStack(200) # 200 images in stack

    len(stack)
    > 200

    # access dict of DICOM fields
    frame = stack[0]
    frame["Manufacturer"]
    > {
        "value": "SIEMENS",
        "tag": (0x8, 0x70),
        "name": "Manufacturer",
        "VR": "LO",
      }

```

Access field values:

``` python
    # field values for each image in stack
    echo_times = stack["EchoTime"]
    > [10.0, 10.0, 20.0, ...]

    # multiple values
    stack["StudyNumber", "SeriesNumber"]
    > [(1, 401), (1, 402), ...]

    # multi-valued fields
    stack["ImageType"]
    > [("ORIGINAL", "PRIMARY", "M", ...), ...]

    stack["ImageType_0"]
    > ["ORIGINAL", "ORIGINAL", ...]

```

Make sub-stacks:

``` python
    # substack
    stack(StudyNumber=1, SeriesNumber=401)
    > DICOMStack(50) # 50 images in stack

```

Make ndarray:

``` python
    # make pixel volume (requires numpy)
    volume = stack(SeriesNumber=401).as_volume()

    # volume's type is a subclass of ndarray with added metadata
    volume.shape
    > (220, 440, 64)
    volume.info
    > {
        "spacing": [0.1, 0.1, 0.5],
        "origin": [10, 20, -5],
        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }

    # series of volumes can also be obtained
    series, volumes = stack.as_volume(by="SeriesNumber")
    echo_times, volumes = stack.as_volume(by="EchoTime")

```
