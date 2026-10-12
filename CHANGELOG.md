# Changelog

## Unreleased

### Added
- Added: `MultipleValuesError` and `NoValueError` raised by `single()`.
- Added: `isin()` alias of `is_in()` on selectors.
- Added: `as_volume(tol=, strict=)` to check slice spacing uniformity.

### Changed
- Changed: requires Python >= 3.10 and pydicom >= 3, < 4.
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
