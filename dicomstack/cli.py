""" DICOM utils cli """
# coding=utf-8
import os
import argparse

from . import utils


def cli():

    parser = argparse.ArgumentParser("dicom", description="Various DICOM utils")

    subparser = parser.add_subparsers()
    parser_anonymize = subparser.add_parser(
        "anonymize", help="Create anonymous copy of DICOM data"
    )
    parser_anonymize.add_argument("src", help="Source file or directory")
    parser_anonymize.add_argument("dest", help="Destination directory")
    parser_anonymize.add_argument(
        "-n",
        "--filename",
        help=(
            "Filename (if src is a single file) or file prefix (src is a directory). "
            "Default: keep original names"
        ),
    )
    parser_anonymize.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action="store_true",
        help="Replace existing files",
    )
    parser_anonymize.set_defaults(func=run_anonymize)

    # parse arguments
    args = parser.parse_args()
    args.func(args)


def run_anonymize(args):
    """ anonymize """
    src = args.src
    dest = args.dest
    name = args.filename
    overwrite = args.overwrite

    if os.path.isfile(src):
        filename = utils.anonymize_file(src, dest, filename=name, overwrite=overwrite)
        files = [filename]
    else:
        files = utils.anonymize_stack(src, dest, prefix=name, overwrite=overwrite)

    print("The following files were anonymized:")
    for filename in files:
        print("\t", filename)
