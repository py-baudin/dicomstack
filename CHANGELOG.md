# Changelog

## 2.0.0 (2026-10-12)

### Added
- Added: `DicomStack(cache=)`: optional header cache file for faster reloads.
- Added: `--cache` option of the `list`, `elements` and `values` commands.
- Added: `workers=` threads reading files in loads and volumes (default: 8).
- Added: `stack.iter_pixels(workers=)` and `as_volume(dtype=)`.
- Added: `DicomStack(dedup=)`: duplicate detection by `"uid"` or `"content"`.
- Added: `fields()`, `varying()`, `table()`, `to_dataframe()` to explore stacks.
- Added: `groupby()`, `series()`, `stack[int]`, `stack[slice]`, `+`, `-`, `&`.
- Added: `sort(missing=)`, `as_volume(stack=True)`, `DicomVolume.affine`.
- Added: `DicomStack(include=, exclude=, recursive=, progress=)`.
- Added: progress bar while loading (default: when stderr is a terminal).
- Added: `DicomStack.from_datasets()`, `stack.export()`, type hints.
- Added: `MultipleValuesError` and `NoValueError` raised by `single()`.
- Added: `isin()` alias of `is_in()` on selectors.
- Added: selector methods `between()`, `isclose()` and `exists()`.
- Added: `stack(query, strict=True)` raises comparison errors.
- Added: `as_volume(tol=, strict=)` to check slice spacing uniformity.

### Changed
- Changed: requires Python >= 3.10 and pydicom >= 3, < 4.
- Changed: duplicates detected by SOPInstanceUID (`dedup="content"`: old way).
- Changed: file bytes are no longer kept in memory; fields converted on access.
- Changed: uncompressed pixel data is read directly from the files.
- Changed: private values are read from the file when accessed (less memory).
- Changed: `regex()` matches anywhere in the value (`re.search`); `^` anchors.
- Changed: queries are evaluated without `eval`.
- Changed: `repr(stack)` shows frames, studies and series.
- Changed: `as_volume` warns on missing or irregular slices.
- Changed: `describe()` keys colliding StudyIDs as "1", "1-2", ...
- Changed: warnings point to the calling code.
- Changed: `DicomVolume.metadata` is a dict of str; geometry in attributes.
- Changed: `DicomVolume.tags` and `DicomVolume(tags=)` are deprecated.
- Changed: `DicomVolume` slicing updates the geometry, as in tomio.
- Changed: integer indexing of a `DicomVolume` spatial axis gives an ndarray.
- Changed: `write_dataset` writes Explicit VR Little Endian with unique UIDs.
- Changed: `format_pixels` writes `(x, y[, 1])` volumes as `(rows, columns)`.
- Changed: `utils.export_file` also accepts a pydicom dataset.
- Changed: `utils.FIELDS_TYPE_2`/`FIELDS_TYPE_3` list DICOM keywords.
- Changed: the CLI exits with code 1 on errors; program name `dicomstack`.

### Removed
- Removed: support of pydicom 2.x.
- Removed: `Query(expr, data)` string expressions (`Query(True/False)` kept).
- Removed: `parse_dataset()` (use `DicomDataset.from_dataset()`).

### Fixed
- Fixed: `sort()` ignored every field after the first.
- Fixed: studies sharing a StudyID were lost in `dicomtree` and `describe()`.
- Fixed: non-numeric StudyID or empty SeriesNumber crashed `describe()`.
- Fixed: zip members selected by substring (`a.zip/IM1` loaded `IM10`).
- Fixed: `.ZIP` files were not read; each archive is now opened once.
- Fixed: a non-zip file named `*.zip` crashed loading.
- Fixed: loading with a glob pattern (`DicomStack("data/series*")`) crashed.
- Fixed: a folder inside a zip file could not be loaded as a path.
- Fixed: importing dicomstack replaced `warnings.showwarning` globally.
- Fixed: in-plane volume spacings were swapped for non-square pixels.
- Fixed: `as_volume(by=)` with a multi-valued field (e.g. orientation).
- Fixed: `write_dataset` files lacked SOP UIDs and had invalid orientation.
- Fixed: `write_dataset` overwrote the geometry tags given by the caller.
- Fixed: `format_pixels` ignored the geometry of volumes.
- Fixed: `make_dataset` replaced given falsy values (e.g. `SeriesNumber=0`).
- Fixed: queries with equal-looking values (`1`, `"1"`) overwrote each other.
- Fixed: `copy`/`pickle` of selectors (special attributes became selectors).
- Fixed: field filters raised `IndexError` on out-of-range indices.
