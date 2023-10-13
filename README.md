# dicomstack

A `pydicom` wrapper for simple loading and handling DICOM stacks.

Loading data:

``` python
    from dicomstack import DicomStack, DICOM

    # load a single DICOM file, a full directory or a zipfile
    DATADIR = "path/to/DICOM/data"
    stack = DicomStack(DATADIR)

    len(stack)
    > 200 # 200 frames in stack

    # list DICOM series in stack
    print(stack.describe())

```

Access field values (using simplified field names, private field names or hex tags):

```python
    # field values for each image in stack
    echo_times = stack["EchoTime"]
    > [10.0, 10.0, 20.0, ...]

    # multiple values
    stack["StudyNumber", "SeriesNumber"]
    > [(1, 401), (1, 402), ...]

    # multi-valued fields
    stack["ImageType"]
    > [("ORIGINAL", "PRIMARY", "M", ...), ...]

    # a single item of a multi-valued field with the `DICOM` object
    stack[DICOM.ImageType[0]]
    > ["ORIGINAL", "ORIGINAL", ...]

    # hex tag
    something = stack[(0x01, 0x28)] 

    # unique values 
    series_numbers = stack.unique("SeriesNumber") 

    # single value
    diffusion_direction = stack.single("[DiffusionGradientDirection]") 

```

Make sub-stacks by filtering out frames:

```python

    # filter by keywords
    substack = stack(StudyNumber=1, SeriesNumber=401)
    len(substack)
    > 50 # 50 frames in substack

    # filter by queries with the `DICOM`` object
    substack = stack(DICOM.EchoTime > 0)
    substack = stack(DICOM["[Bmatrix]"] == bmatrix)
    substack = stack(DICOM.SeriesNumber.isin([5, 6]))
    substack = stack(DICOM.SeriesDescription.startswith("prefix_"))
    # chain queries
    substack = stack((DICOM.StudyNumber==1) & (DICOM.SeriesNumber==401)) 

```
    

Make ndarray:

``` python
    # make pixel volume (requires numpy)
    volume = stack(SeriesNumber=401).as_volume()

    # volume's type is a subclass of ndarray with added metadata
    volume.shape
    > (220, 440, 64)
    
    volume.tags
    > {'origin': (-125.000992, -122.842384, 32.496708),
       'spacing': (0.488281, 0.488281, 1),
       'transform': ((1, 0, 0), (0, 0.959915, -0.280292), (0, 0.280292, 0.959915)),
      }
      
    # get series of volumes split by DICOM fields
    series, volumes = stack.as_volume(by="SeriesNumber")
    echo_times, volumes = stack.as_volume(by="EchoTime")

```

