# Changelog

## Unreleased

### Added
- Added: `DicomStack(cache=)`: optional header cache file for faster reloads.
- Added: `--cache` option of the `list`, `elements` and `values` commands.
- Added: `workers=` threads for reading files (slow storage) in loads and volumes.
- Added: `stack.iter_pixels(workers=)` and `as_volume(dtype=)`.
- Added: `DicomStack(dedup=)`: duplicate detection by `"uid"` or `"content"`.
- Added: `MultipleValuesError` and `NoValueError` raised by `single()`.
- Added: `isin()` alias of `is_in()` on selectors.
- Added: selector methods `between()`, `isclose()` and `exists()`.
- Added: `stack(query, strict=True)` raises comparison errors instead of warning.
- Added: `as_volume(tol=, strict=)` to check slice spacing uniformity.

### Changed
- Changed: requires Python >= 3.10 and pydicom >= 3, < 4.
- Changed: duplicates detected by SOPInstanceUID; dedup="content" for the old behaviour.
- Changed: file bytes are no longer kept in memory; fields are converted on access.
- Changed: uncompressed pixel data is read directly, without parsing the file again.
- Changed: warnings point to the calling code.
- Changed: `regex()` matches anywhere in the value (`re.search`); use `^` to anchor.
- Changed: queries are evaluated without `eval`.
- Changed: `utils.FIELDS_TYPE_2`/`FIELDS_TYPE_3` list DICOM keywords.
- Changed: the CLI exits with code 1 on errors; program name `dicomstack`.

### Removed
- Removed: `Query(expr, data)` string expressions (`Query(True)`/`Query(False)` remain).
- Removed: `parse_dataset()` (use `DicomDataset.from_dataset()`).
- Changed: `as_volume` warns on missing or irregular slices.
- Changed: `describe()` keys colliding StudyIDs as "1", "1-2", ...
- Changed: `write_dataset` writes Explicit VR Little Endian with unique UIDs.
- Changed: `DicomVolume.metadata` is a dict of str; geometry moved to attributes.
- Changed: `DicomVolume.tags` and `DicomVolume(tags=)` are deprecated.
- Changed: `DicomVolume` slicing updates the geometry, as in tomio.
- Changed: integer indexing of a `DicomVolume` spatial axis returns an ndarray.
- Changed: `format_pixels` writes `(x, y[, 1])` volumes as `(rows, columns)`.

### Fixed
- Fixed: `sort()` ignored every field after the first.
- Fixed: studies sharing a StudyID were lost in `dicomtree` and `describe()`.
- Fixed: non-numeric StudyID or empty SeriesNumber crashed `describe()`.
- Fixed: zip members selected by substring (`a.zip/IM1` loaded `IM10`).
- Fixed: `.ZIP` files were not read; each archive is now opened once.
- Fixed: a non-zip file named `*.zip` crashed loading.
- Fixed: importing dicomstack replaced `warnings.showwarning` globally.
- Fixed: in-plane volume spacings were swapped for non-square pixels.
- Fixed: `write_dataset` files lacked SOP UIDs and had invalid orientation.
- Fixed: `format_pixels` ignored the geometry of volumes.
- Fixed: `write_dataset` overwrote the geometry tags given by the caller.
- Fixed: queries mixing equal-looking values (`1` and `"1"`) could overwrite each other.
- Fixed: `copy`/`pickle` of selectors (special attributes became selectors).
- Fixed: field filters raised `IndexError` on out-of-range indices (now skipped).
