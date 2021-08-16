""" DICOM tools """
# coding=utf-8
import os
import pathlib
import datetime
import uuid
import logging
import pydicom
from . import pixeldata

LOGGER = logging.getLogger(__name__)

DEFAULT_MAPPER = {
    "Patient's Name": "Anonymous",
    "Patient ID": "",
    "Patient's Sex": "O",
}

FIELDS_TYPE_2 = [
    "Accession Number",
    "Referring Physician's Name",
    "Patient's Name",
    "Patient ID",
    "Patient's Birth Date",
    "Patient's Sex",
]

FIELDS_TYPE_3 = [
    # "Instance Creator UID",
    "Institution Name",
    "Institution Address",
    "Referring Physician's Address",
    "Referring Physician's Telephone Numbers",
    "Station Name",
    "Study Description",
    "Series Description",
    "Institutional Department Name",
    # "Physician(s) of Record",
    "Performing Physicians' Name",
    # "Name of Physician(s) Reading study",
    # "Operator's Name",
    "Admitting Diagnoses Description",
    "Derivation Description",
    "Patient's Birth Time",
    "Other Patient IDs",
    "Other Patient Names",
    "Patient's Age",
    "Patient's Size",
    "Patient's Weight",
    "Medical Record Locator",
    "Ethnic Group",
    "Occupation",
    "Additional Patient's History",
    "Patient Comments",
    "Device Serial Number",
    "Protocol Name",
    "Image Comments",
    "Request Attributes Sequence",
    "Storage Media File-Set UID",
    # "Related Frame of Reference UID",
]


def tame_field(name):
    """tame DICOM field"""

    for char in ["'s", "s'", "-", "(", ")"]:
        name = name.replace(char, "")
    return name.replace(" ", "")


def anonymize_stack(src, dest, prefix=None, **kwargs):
    """anonymize whole dicom stack"""
    outfile = None
    anonymized = []
    dest = pathlib.Path(dest)

    for root, dirs, files in os.walk(src):
        root = pathlib.Path(root)
        nfile = len(files)
        if not any(pydicom.misc.is_dicom(root / file) for file in files):
            continue
        LOGGER.info(f"Converting folder: '{root.relative_to(src)}'")
        for i, filename in enumerate(files):
            if prefix:
                outfile = f"{prefix}{i+1:{nfile}}"
            else:
                outfile = filename

            infile = root / filename
            outfile = dest / root.relative_to(src) / outfile
            LOGGER.debug(f"Loading file: {infile.relative_to(src)}")
            try:
                filename = anonymize_file(infile, outfile, **kwargs)
            except pydicom.errors.InvalidDicomError:
                continue
            anonymized.append(filename)
    return anonymized


def anonymize_file(
    src,
    dest,
    mapper=None,
    mapkey="PatientID",
    remove_private_tags=True,
    overwrite=False,
):
    """anonymize dicom file"""

    # read dicom
    dataset = pydicom.dcmread(str(src))

    # patient's id for mapping
    mapkey = tame_field(mapkey)
    patient_id = dataset.data_element(mapkey).value
    if not mapper:
        mapper = {}
    custom_mapper = {**DEFAULT_MAPPER, **mapper.get(patient_id, {})}

    # remove private tags
    if remove_private_tags:
        LOGGER.debug("remove private tags")
        dataset.remove_private_tags()

    # Data elements of type 3 (optional) can be easily deleted using ``del``
    for element_name in FIELDS_TYPE_3:
        tag = tame_field(element_name)
        if tag in dataset:
            LOGGER.debug(f"Remove type-3 field: {element_name}")
            delattr(dataset, tag)

    # For data elements of type 2, assign a blank string.
    for element_name in FIELDS_TYPE_2:
        tag = tame_field(element_name)
        LOGGER.debug(f"Erase type-2 field: {element_name}")
        dataset.data_element(tag).value = ""

    #  callback functions to find all tags corresponding to a person name
    def person_names_callback(dataset, data_element):
        if data_element.VR == "PN":
            LOGGER.debug(f"Erasing person's name: {data_element}")
            data_element.value = ""

    # remove deprecated fields
    def curves_callback(dataset, data_element):
        if data_element.tag.group & 0xFF00 == 0x5000:
            del dataset[data_element.tag]

    # run callback functions
    dataset.walk(person_names_callback)
    dataset.walk(curves_callback)

    # map specific fields
    for element_name in custom_mapper:
        tag = tame_field(element_name)
        LOGGER.debug(f"Map field: {element_name} to {custom_mapper[element_name]}")
        dataset.data_element(tag).value = custom_mapper[element_name]

    # save
    dest = pathlib.Path(dest)
    dest.parent.mkdir(exist_ok=True, parents=True)
    if dest.is_file() and not overwrite:
        raise ValueError("Destination file already exists: %s" % dest)
    dataset.save_as(str(dest))
    return dest


def write_dataset(
    data,
    filename,
    ext=".dcm",
    media_storage_class="MRI",  # or storage class UID
    dataset=None,  # reference dataset
    **kwargs,
):
    if dataset is not None:
        dataset = update_dataset(dataset, data=data, **kwargs)
    else:
        # create
        dataset = make_dataset(data, **kwargs)

    """ write valid DICOM file """

    # Create the FileDataset instance
    # (initially no data elements, but file_meta
    # supplied)
    LOGGER.debug("Setting file meta information.")
    file_meta = pydicom.dataset.FileMetaDataset()
    # Populate required values for file meta information
    if media_storage_class == "MRI":
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    elif media_storage_class == "MRS":
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4.2"
    elif not media_storage_class in pydicom.uid.UID_dictionary:
        raise ValueError("Unknown media storage class UID: %s" % media_storage_class)
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"

    dataset = pydicom.FileDataset(
        filename, dataset, file_meta=file_meta, preamble=b"\0" * 128
    )

    # write dataset
    LOGGER.debug("Writing file: %s", filename)
    if ext is not None:
        LOGGER.debug("Setting file extension to: %s", ext)
        basename = os.path.splitext(filename)[0]
        filename = basename + ext
    dataset.save_as(filename, write_like_original=False)


def update_dataset(dataset, data=None, dtype="uint16", **tags):
    """update existing dataset"""
    if data is None:
        data = dataset.PixelData
    newtags = {tag.keyword: dataset[tag.tag] for tag in dataset}
    newtags.update(tags)
    return make_dataset(data, dtype=dtype, **newtags)


def make_dataset(data, dtype="uint16", **tags):
    """make valid DICOM dataset"""
    dataset = pydicom.Dataset()

    # date time
    LOGGER.debug("Setting dataset values.")
    now = datetime.datetime.now()

    def default(name, value):
        # set default value in tags
        if tags.get(name):
            return
        tags[name] = value

    # set required tags
    # patient
    default("PatientName", "Anonymous")
    default("PatientID", "")
    default("PatientSex", "O")
    default("PatientBirthDate", "")
    default("PatientOrientation", "FFS")
    default("ReferringPhysicianName", "")
    # material
    default("Modality", "MR")
    default("Manufacturer", "")
    # reference
    # default("FrameOfReferenceUID", None)
    # default("PositionReferenceIndicator", None)
    # geometry
    default("PixelSpacing", [1, 1])  # overwritten with data's metadata
    default("SliceThickness", 1)
    default("ImageOrientationPatient", [1, 0, 0, 0, 1, 0, 0, 0, 1])
    default("ImagePositionPatient", [0, 0, 0])
    # MR image
    default("PhotometricInterpretation", "MONOCHROME1")
    default("ImageType", ["DERIVED", "PRIMARY"])
    default("ScanningSequence", "RM")
    default("SequenceVariant", "NONE")
    # study
    default("StudyDate", now.strftime("%Y%m%d"))
    default("StudyTime", now.strftime("%H%M%S.%f"))
    default("StudyID", "")
    default("StudyInstanceUID", str(uuid.uuid4()))
    # series
    default("SeriesNumber", 1)
    default("SeriesInstanceUID", str(uuid.uuid4()))
    default("AccessionNumber", "")
    default("InstanceNumber", 1)

    # data
    if not isinstance(data, bytes):
        try:
            _tags, data = pixeldata.format_pixels(data, dtype=dtype)
        except (NotImplementedError, TypeError):
            raise TypeError("data must be a bytes array")
        tags.update(_tags)

    # check some tags
    for tag in [
        "SamplesPerPixel",
        "BitsStored",
        "BitsAllocated",
        "PixelRepresentation",
        "Rows",
        "Columns",
    ]:
        if tags.get(tag) is None:
            raise ValueError("Missing required tag: %s" % tag)

    # set tags
    for name, value in tags.items():
        # check tag
        if isinstance(value, pydicom.DataElement):
            element = value
        else:
            tag = pydicom.dataset.tag_for_keyword(name)
            if not tag:
                raise ValueError("Invalid DICOM keyword: %s" % name)
            vr = pydicom.dataset.dictionary_VR(tag)
            element = pydicom.DataElement(tag, vr, value)
        dataset.add(element)

    # Set the transfer syntax
    dataset.is_little_endian = True
    dataset.is_implicit_VR = True

    # Set creation date/time
    dataset.ContentDate = now.strftime("%Y%m%d")
    dataset.ContentTime = now.strftime("%H%M%S.%f")

    # set pixel's data
    dataset.PixelData = data

    return dataset
