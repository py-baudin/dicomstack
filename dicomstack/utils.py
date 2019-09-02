""" DICOM tools """
# coding=utf-8
import os
import datetime
import uuid
import logging
import pydicom
from . import pixeldata

LOGGER = logging.getLogger(__name__)


def anonymize_stack(src, dest, prefix=None, **kwargs):
    """ anonymize whole dicom stack """
    outfile = None
    anonymized = []

    for root, dirs, files in os.walk(src):
        nfile = len(files)
        for i, filename in enumerate(files):
            if prefix:
                outfile = f"{prefix}{i+1:{nfile}}"

            infile = os.path.join(root, filename)
            outdir = os.path.join(dest, os.path.relpath(root, src))
            try:
                filename = anonymize_file(infile, outdir, filename=outfile, **kwargs)
            except pydicom.errors.InvalidDicomError:
                continue
            anonymized.append(filename)
    return anonymized


def anonymize_file(src, dest, filename=None, remove_private_tags=True, overwrite=False):
    """ anonymize dicom file """

    # read dicom
    dataset = pydicom.dcmread(src)

    #  callback functions to find all tags corresponding to a person name
    def person_names_callback(dataset, data_element):
        if data_element.VR == "PN":
            data_element.value = "Anonymous"

    def curves_callback(dataset, data_element):
        if data_element.tag.group & 0xFF00 == 0x5000:
            del dataset[data_element.tag]

    # run callback functions
    dataset.walk(person_names_callback)
    dataset.walk(curves_callback)

    # remove private tags
    if remove_private_tags:
        dataset.remove_private_tags()

    # Data elements of type 3 (optional) can be easily deleted using ``del`
    # for element_name in data_elements:
    #     delattr(dataset, element_name)

    # For data elements of type 2, assign a blank string.
    # tag = 'PatientBirthDate'
    # if tag in dataset:
    #     dataset.data_element(tag).value = '19000101'

    # save
    if not os.path.exists(dest):
        os.makedirs(dest)
    if not filename:
        filename = os.path.basename(src)
    filepath = os.path.join(dest, filename)
    if os.path.isfile(filepath) and not overwrite:
        raise ValueError("Destination file already exists: %s" % filepath)
    dataset.save_as(filepath)
    return filepath


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
    file_meta = pydicom.Dataset()
    # Populate required values for file meta information
    if media_storage_class == "MRI":
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    elif media_storage_class == "MRS":
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4.2"
    elif not media_storage_class in pydicom.uid.UID_dictionary:
        raise ValueError("Unknown media storage class UID: %s" % media_storage_class)
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"

    ds = pydicom.FileDataset(
        filename, dataset, file_meta=file_meta, preamble=b"\0" * 128
    )

    # write dataset
    LOGGER.debug("Writing file: %s", filename)
    if ext is not None:
        LOGGER.debug("Setting file extension to: %s", ext)
        basename = os.path.splitext(filename)[0]
        filename = basename + ext
    ds.save_as(filename, write_like_original=False)


def update_dataset(dataset, data=None, dtype="uint16", **tags):
    """ update existing dataset """
    if data is None:
        data = dataset.PixelData
    newtags = {tag.keyword: dataset[tag.tag] for tag in dataset}
    newtags.update(tags)
    return make_dataset(data, dtype=dtype, **newtags)


def make_dataset(
    data,
    dtype="uint16",
    PatientName="Anonymous",
    PatientID="",
    PatientSex="O",
    PatientBirthDate="",
    PatientOrientation="FFS",
    ReferringPhysicianName="",
    StudyDate=None,  # now
    StudyTime=None,  # now
    StudyID="",
    StudyUID=None,
    SeriesNumber=1,
    SeriesUID=None,
    InstanceNumber=1,
    **tags,
):
    ds = pydicom.Dataset()

    # date time
    LOGGER.debug("Setting dataset values.")
    now = datetime.datetime.now()

    # default values
    if not StudyDate:
        StudyDate = now.strftime("%Y%m%d")
    if not StudyTime:
        StudyTime = now.strftime("%H%M%S.%f")
    if not SeriesUID:
        SeriesUID = str(uuid.uuid4())
    if not StudyUID:
        StudyUID = str(uuid.uuid4())

    # Add the data elements
    tags.update(
        {
            "PatientName": PatientName,
            "PatientID": PatientID,
            "PatientSex": PatientSex,
            "PatientBirthDate": PatientBirthDate,
            "PatientOrientation": PatientOrientation,
            "ReferringPhysicianName": ReferringPhysicianName,
            "ds.StudyDate": StudyDate,
            "ds.StudyTime": StudyTime,
            "ds.SeriesUID": SeriesUID,
            "ds.SeriesNumber": SeriesNumber,
            "ds.StudyID": StudyID,
            "ds.StudyUID": StudyUID,
            "ds.InstanceNumber": InstanceNumber,
        }
    )

    # set other tags
    for name, value in tags.items():
        # check tag
        if isinstance(value, pydicom.DataElement):
            ds.add(value)
        else:
            setattr(ds, name, value)

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set creation date/time
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S.%f")

    # set data
    if isinstance(data, bytes):
        ds.PixelData = data
    else:
        pixeldata.update_dataset(ds, data, dtype=dtype)

    return ds
