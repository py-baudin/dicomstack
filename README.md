# dicomstack

A thin layer over [pydicom](https://pydicom.github.io/) to load DICOM data (folders, zip files, multi-frame files), explore and query it by field values, sort it, and build 3d volumes with their geometry.

At a glance:

```python
from dicomstack import DicomStack, DICOM

stack = DicomStack("data")              # load a folder (or a file, a zip file...)
print(stack.describe())                 # list its studies and series

stack.unique("EchoTime")                # -> [5, 10, 20]: values of a field
t2 = stack(SeriesNumber=2)              # filter by field values
first_echo = t2(DICOM.EchoTime < 15)    # ... or with queries

volume = first_echo.as_volume()         # numpy array (x, y, z) with its geometry
volume.spacing                          # -> (1.484375, 1.484375, 2.0)
echo_times, volumes = t2.as_volume(by="EchoTime")   # one volume per echo time
```

The examples below use a folder `data/` with two series (a T1 series of 4 slices, and a T2 series with 2 echoes of 4 slices) and a non-DICOM file, the same data zipped in `data.zip`, and a multi-frame file `multiframe.dcm`.

## Installation

```bash
pip install dicomstack
pip install "dicomstack[pandas]"    # optional: stack.to_dataframe()
```

Requires Python ≥ 3.10, pydicom ≥ 3 and numpy.

## Loading

```python
from dicomstack import DicomStack

stack = DicomStack("data")                    # a folder and its sub-folders
stack = DicomStack("data/series1/IM1.dcm")    # a single file
stack = DicomStack(["data/series1", "data/series2/IM1_1.dcm"])  # several paths
stack = DicomStack("data/series*")            # glob pattern
stack = DicomStack("data.zip")                # a zip file
stack = DicomStack("data.zip/series1")        # a folder inside a zip file
stack = DicomStack(filenames=["data/series1/IM1.dcm", "data/series1/IM2.dcm"])

stack = DicomStack("data", include="*.dcm")   # file name patterns (fnmatch)
stack = DicomStack("data", exclude=["*.txt", "series2/*"], recursive=True)

stack = DicomStack("data")
stack.root                                    # folder of the files
stack.filenames[:2]                           # -> ['series1/IM1.dcm', 'series1/IM2.dcm'] (relative to root)
stack.non_dicom                               # -> ['notes.txt']: files which are not DICOM

# multi-frame (enhanced) files: one frame per image
multi = DicomStack("multiframe.dcm")
print(multi)                                  # -> DICOM(90 frames | 1 study | 1 series)
```

Loading reads the files' headers only; field values are converted when accessed, and pixels are read when needed. The values of private fields are not kept in memory: they are read again from the file when accessed (except for zip members). A progress bar is shown while loading when the output is a terminal (`progress=False` to hide it, `True` to always show it, or a function `progress(done, total)`).

**Duplicates.** Frames are identified by their `SOPInstanceUID` (and frame index): the same image loaded twice (two copies of a file, overlapping paths) appears once. `duplicates="error"` raises `DuplicatedFramesError` instead, `duplicates="ignore"` keeps them. `dedup="content"` compares whole file contents instead of UIDs (slower: whole files are read); files without `SOPInstanceUID` are never merged.

```python
stack = DicomStack(["data/series1", "data/series1"])       # same files twice
len(stack)                                    # -> 4
stack = DicomStack("data", duplicates="error", dedup="content")
```

**Header cache.** Re-opening the same data parses all headers again. `cache=True` stores the converted values in a file (`.dicomstack-cache.json.gz` in the data folder, or in the user cache directory if the folder is not writable), used on the next loads for the files that are unchanged (same size and modification time). The first load with `cache=True` is slower (all values are converted); the next ones are much faster. `cache="refresh"` rebuilds it, `cache="path/to/file.json.gz"` sets its location.

```python
stack = DicomStack("data", cache=True)        # writes data/.dicomstack-cache.json.gz
stack = DicomStack("data", cache=True)        # unchanged files: not parsed
```

**Slow storage.** On network file systems, `workers=8` reads the files ahead in threads (headers are still parsed one by one: threads do not speed up parsing on a local disk).

**In-memory datasets.**

```python
import pydicom

datasets = [pydicom.dcmread("data/series1/IM1.dcm"), pydicom.dcmread("data/series1/IM2.dcm")]
stack = DicomStack.from_datasets(datasets)
```

## Exploring

```python
stack = DicomStack("data")
print(stack.describe())
```

```
0: (non-DICOM data)
	0: notes.txt
1: Legs
	1: T1 axial (n=4)
	2: T2 multi-echo (n=8)
```

Studies are listed by `StudyID` (with a suffix if several studies share it), series by `SeriesNumber`. `stack.dicomtree` returns the same structure as a dictionary.

```python
stack.dicomtree.keys()                        # -> dict_keys(['0', '1']): studies ("0": non-DICOM files)
len(stack.fields())                           # -> 116: keywords present in all frames
stack.fields("any")                           # keywords present in any frame
stack.varying()                               # -> ['EchoTime', 'SeriesDescription', 'SeriesInstanceUID', 'SeriesNumber']
stack.varying(within="SeriesNumber")          # -> ['EchoTime']: fields varying inside a series
stack.table("SeriesNumber", "EchoTime")[:2]   # -> [{'SeriesNumber': 1, 'EchoTime': 5}, {'SeriesNumber': 1, 'EchoTime': 5}]
```

`varying()` ignores per-image fields (`SOPInstanceUID`, `InstanceNumber`, positions, window values...: see `dicomstack.dicomstack.VARYING_EXCLUDE`; `exclude=()` to include them).

<!-- skip-test -->
```python
df = stack.to_dataframe("SeriesNumber", "EchoTime", "SliceThickness")  # requires pandas
```

## Field access

```python
from dicomstack import DICOM

stack["EchoTime"]                             # one value per frame
stack["SeriesNumber", "EchoTime"][:2]         # -> [(1, 5), (1, 5)]
frame = stack[0]                              # a frame (stack[1:3]: a sub-stack)
frame["EchoTime"]                             # -> 5
frame.get("StudyComments")                    # -> None (missing field)
```

Field names:

| Syntax | Meaning | Example (value) |
|---|---|---|
| `"Keyword"` | DICOM keyword | `"ImageType"` → `('ORIGINAL', 'PRIMARY', 'M', ...)` |
| `"Keyword_N"` | N-th value of a multi-valued field (**1-based**) | `"ImageType_1"` → `'ORIGINAL'` |
| `"Sequence.N.Keyword"` | field of the N-th item of a sequence (1-based) | `"IconImageSequence.1.Rows"` → `64` |
| `DICOM.Keyword[i]` | selector, i-th value (**0-based**, Python indexing) | `DICOM.ImageType[0]` → `'ORIGINAL'` |
| `"[Private Name]"` | private field, by its name in pydicom's private dictionary | `"[CSAImageHeaderType]"` → `'IMAGE NUM 4'` |
| `(group, element)` | tag | `(0x0018, 0x0081)` (EchoTime) → `5` |

```python
stack["ImageType_1"][0]                       # -> 'ORIGINAL'
stack[DICOM.ImageType[0]][0]                  # -> 'ORIGINAL'
stack["IconImageSequence.1.Rows"][0]          # -> 64
stack["[CSAImageHeaderType]"][0]              # -> 'IMAGE NUM 4'
stack[(0x0018, 0x0081)][0]                    # -> 5
```

Values are converted: dates and times to ISO strings (`'2014-11-24'`, `'14:11:39.046000'`), decimal strings to `int` when integral (`5`) or `float`, multi-valued fields to tuples (`PixelSpacing` → `(1.484375, 1.484375)`), empty values to `None`, sequences to lists of datasets. `frame.dataset` returns all the converted elements of a frame (`frame.dataset.elements`, `elements_by_keyword`, ...).

```python
stack.unique("SeriesNumber")                  # -> [1, 2] (sorted, missing values ignored)
stack.unique("SeriesNumber", "EchoTime")      # -> [(1, 5), (2, 10), (2, 20)]
stack(SeriesNumber=2).single("SeriesDescription")  # -> 'T2 multi-echo'
stack.has_field("EchoTime")                   # -> True (all frames; how="any")
"EchoTime" in stack                           # -> True
```

`single()` raises `MultipleValuesError` if the values differ and `NoValueError` if there is none (both are `ValueError`), unless a `default` is given:

```python
from dicomstack import MultipleValuesError

try:
    stack.single("EchoTime")
except MultipleValuesError as exc:
    print(exc)                                # -> Multiple values found for ('EchoTime',)
stack.single("StudyComments", default=None)   # -> None
```

## Filtering and queries

```python
stack(SeriesNumber=2)                         # frames with SeriesNumber == 2
stack(SeriesNumber=2, EchoTime=[10, 20])      # several fields, list of accepted values
```

Queries are built from selectors (`DICOM.Keyword`, `DICOM["[Private Name]"]`, `DICOM[(group, element)]`, or `stack.Keyword`):

```python
stack(DICOM.EchoTime > 5)
stack(DICOM.ImageType[1] == "PRIMARY")
stack(DICOM.SeriesNumber.isin([1, 2]))
stack(DICOM.SeriesDescription.startswith("T2"))
stack((DICOM.SeriesNumber == 2) & ~(DICOM.EchoTime == 20))   # &, |, ~
stack(stack.EchoTime.between(5, 10))
```

| Selector operator / method | True if the field's value... |
|---|---|
| `==`, `!=`, `<`, `<=`, `>`, `>=` | compares as stated |
| `.is_in(values)`, `.isin(values)`, `.not_in(values)` | is (not) one of the values |
| `.between(low, high)` | is in [low, high] |
| `.isclose(value, tol=1e-6)` | is within `tol` of value (also element-wise for tuples, e.g. `PixelSpacing`) |
| `.exists()`, `.is_null()` | is present and not empty / is missing or empty |
| `.contains(x)` | contains x (strings, tuples) |
| `.intersect(values)`, `.subset(values)`, `.superset(values)` | shares a value with / is included in / includes the values |
| `.startswith(s)`, `.endswith(s)` | starts / ends with s |
| `.regex(pattern)` | matches the regular expression anywhere (`re.search`; `^` to anchor) |
| `.apply(function)` | `function(value)` is true |

A comparison which fails (e.g. a missing value compared with `<`, or a number compared with a string) excludes the frame with a warning; `strict=True` raises the error instead:

<!-- skip-test -->
```python
stack(DICOM.SeriesNumber > "1")               # warning: no frame
stack(DICOM.SeriesNumber > "1", strict=True)  # TypeError
```

## Sorting and grouping

```python
stack.sort("SeriesNumber", "InstanceNumber")  # frames with missing values are dropped
stack.sort("StudyComments", missing="last")   # ... or kept at the end ("error": NoValueError)
stack(SeriesNumber=1).reorder()               # sorted along the slice direction

for (series, echo_time), substack in stack.groupby("SeriesNumber", "EchoTime"):
    print(series, echo_time, len(substack))   # -> 1 5 4, 2 10 4, 2 20 4

for uid, series in stack.series():            # groupby("SeriesInstanceUID")
    print(series.single("SeriesDescription"))

stack[0], stack[-1], stack[2:6]               # frame, frame, sub-stack
t1, t2 = stack(SeriesNumber=1), stack(SeriesNumber=2)
len(t1 + t2), len(stack - t1), len(stack & t2)    # -> (12, 8, 8): union, difference, intersection
```

## Volumes

```python
volume = stack(SeriesNumber=1).as_volume()
volume.shape                                  # -> (256, 192, 4): (x, y, z) = (columns, rows, slices)
volume.origin                                 # -> (-190, -142.5, 2): position of voxel (0, 0, 0)
volume.spacing                                # -> (1.484375, 1.484375, 2.0)
volume.transform                              # -> ((1, 0, 0), (0, 1, 0), (0.0, 0.0, 1.0))
volume.affine                                 # 4x4: voxel indices (i, j, k, 1) -> coordinates
```

Volumes are numpy arrays (`DicomVolume`) with their geometry, in LPS coordinates (as DICOM and ITK): `transform[i]` is the direction of array axis `i` and `spacing[i]` the voxel size along it. Slices are sorted by position (`reorder=False` keeps the stack's order), pixel values are rescaled with `RescaleSlope`/`RescaleIntercept` (`rescale=False` to keep the stored values), the data type is `float64` (`dtype="float32"` halves the memory). A warning is shown if the slices are not evenly spaced (missing slice), `strict=True` raises a `ValueError` instead.

Slicing a volume keeps a consistent geometry (`volume[:, :, 2:3].origin` is the position of the third slice); `tomio.asvolume(volume)` converts it to a [tomio](https://github.com/py-baudin/tomio) `Volume`.

```python
echo_times, volumes = stack(SeriesNumber=2).as_volume(by="EchoTime")
echo_times                                    # -> [10, 20]: one volume per echo time
volume = stack(SeriesNumber=2).as_volume(by="EchoTime", stack=True)
volume.shape                                  # -> (256, 192, 4, 2)
volume.metadata                               # -> {'EchoTime': '[10, 20]'}

for pixels in stack.iter_pixels():            # frames' pixel arrays (rows, columns)
    pass
```

## Export and anonymisation

<!-- skip-test -->
```python
stack(SeriesNumber=2).export("anonymous")     # copies of the stack's files, anonymized
stack.export("copy", anonymize=False, remove_private_tags=True)

from dicomstack import utils
utils.export_stack("data", "anonymous")       # all DICOM files of a folder
utils.export_file("data/series1/IM1.dcm", "anonymous/IM1.dcm")
```

Anonymisation empties the patient's name, ID, birth date and sex, the accession number and the referring physician's name (`utils.FIELDS_TYPE_2`), removes the fields of `utils.FIELDS_TYPE_3` (institution, station, study description, patient's age, size, weight, comments, device serial number, protocol name...), and empties all person names. Curves (groups `50xx`) are removed. Values can be set with a mapping table (`mapper`, `mapkey`: see the command line).

## Command line

```bash
dicomstack list data                          # List DICOM studies, series and non-DICOM files.
dicomstack values data EchoTime --series 2    # Show unique values of one or more DICOM elements.
dicomstack elements data -i 1 -f SeriesNumber=2   # List the DICOM elements of one frame.
dicomstack export data anonymous              # Export a modified/anonymous copy of a DICOM stack.
dicomstack export -t table.csv -k PatientID data anonymous   # with a mapping table
```

`--cache` uses the header cache (see Loading). `dicomstack export -h` describes the mapping tables (CSV or JSON).

## Performance

Indicative numbers (2,000 single-frame MR files, 350 MB, Windows laptop, warm disk cache; `benchmarks/`):

| | time |
|---|---|
| load (headers only) | ~2.3 s |
| first field access (`stack["EchoTime"]`) | ~0.05 s |
| 20 different fields | ~1 s |
| all pixels | ~0.7 s |
| load with the header cache | ~0.3 s |

Loading is dominated by pydicom's header parsing, which threads do not speed up (`workers=` only helps on slow storage). Uncompressed pixel data is read directly from the files; other transfer syntaxes are decoded by pydicom.

## Development

```bash
pip install -e ".[test]"
python -m pytest
```

- `test/make_golden.py` writes a snapshot of the outputs (values, types, volumes) of the reference version in `test/golden/` (not versioned); `test/test_golden.py` compares the current code with it (skipped without snapshot).
- `benchmarks/make_data.py` generates the benchmark data (`benchmarks/data/`, not versioned), `benchmarks/bench.py` runs the benchmark scenarios.
- Every Python example of this README is run by `test/test_readme.py`.
