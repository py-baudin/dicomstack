""" DICOM tools """
# coding=utf-8
import os
import pydicom


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
