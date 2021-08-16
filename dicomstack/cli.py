""" DICOM utils cli """
# coding=utf-8
import os
import csv
import json
import logging
import argparse

from . import utils, dicomstack


def cli():
    """entry point into command-line utility"""

    # make parser
    parser = argparse.ArgumentParser("dicom", description="Various DICOM utils")

    # add subprograms
    subparser = parser.add_subparsers()
    cli_list(subparser)
    cli_tags(subparser)
    cli_view(subparser)
    cli_anonymize(subparser)

    # parse arguments
    args = parser.parse_args()

    if "func" in args:
        args.func(args)
    else:
        print(parser.print_help())


def cli_tags(subparser):
    """show DICOM tags for one frame"""

    parser_anonymize = subparser.add_parser(
        "tags", help="Show DICOM tags for one frame."
    )
    parser_anonymize.add_argument("src", help="Source file or directory")
    parser_anonymize.add_argument(
        "-i",
        dest="index",
        type=int,
        default=1,
        help="Show tags for i-th frame (default: 1).",
    )
    parser_anonymize.add_argument("--series", type=int, help="Filter by series number.")
    parser_anonymize.add_argument(
        "-f", "--filters", type=dicom_filters, help="Filter by tag=value,... pairs."
    )

    def _tags(args):
        # load stack
        stack = dicomstack.DicomStack(args.src)

        if args.series:
            # filter by series
            stack = stack(SeriesNumber=args.series)
        if args.filters:
            stack = stack(**args.filters)

        # show dicom tags
        frame = stack.frames[args.index - 1]
        frame.elements
        print(frame)

    parser_anonymize.set_defaults(func=_tags)


def cli_view(subparser):
    """show DICOM values for one or more tags"""

    parser_anonymize = subparser.add_parser(
        "view", help="Show unique DICOM values of one or more tags."
    )
    parser_anonymize.add_argument("src", help="Source file or directory")
    parser_anonymize.add_argument("tags", nargs="+", help="DICOM tags.")
    parser_anonymize.add_argument("--series", type=int, help="Filter by series number.")
    parser_anonymize.add_argument(
        "-f", "--filters", type=dicom_filters, help="Filter by tag=value,... pairs."
    )

    def _view(args):
        # load stack
        stack = dicomstack.DicomStack(args.src)

        if args.series:
            # filter by series
            stack = stack(SeriesNumber=args.series)
        if args.filters:
            stack = stack(**args.filters)

        # show dicom tags
        for tag in args.tags:
            print(f"{tag}:", stack.unique(tag))

    parser_anonymize.set_defaults(func=_view)


def cli_list(subparser):
    """describe DICOM stack"""

    parser_anonymize = subparser.add_parser(
        "list", help="List studies, series and non-DICOM files."
    )
    parser_anonymize.add_argument("src", help="Source file or directory")

    def _list(args):
        # load stack
        stack = dicomstack.DicomStack(args.src)

        # show description
        print(stack.describe())

    parser_anonymize.set_defaults(func=_list)


def cli_anonymize(subparser):
    """anonymize DICOM"""

    # parser
    parser_anonymize = subparser.add_parser(
        "anonymize", help="Create anonymous copy of DICOM data"
    )
    parser_anonymize.add_argument("src", help="Source file or directory.")
    parser_anonymize.add_argument("dest", help="Destination directory.")
    parser_anonymize.add_argument(
        "-n",
        "--filename",
        help=(
            "Filename (if src is a single file) or file prefix (src is a directory). "
            "Default: keep original names."
        ),
    )
    parser_anonymize.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing files.",
    )
    parser_anonymize.add_argument(
        "-t",
        "--table",
        help="Mapping table filename in CSV or JSON format. "
        "CSV: use ;-delimited format, the first column being the mapping key.",
    )
    parser_anonymize.add_argument(
        "-k",
        "--map-key",
        default="Patient ID",
        help="DICOM tag to use as the key in the mapping table (default: 'Patient ID').",
    )

    def _anonymize(args):
        src = args.src
        dest = args.dest
        name = args.filename
        overwrite = args.overwrite
        maptable = args.table
        mapkey = args.map_key

        if maptable:
            if maptable.endswith(".csv"):
                with open(maptable, newline="") as csvfile:
                    csvtable = list(csv.reader(csvfile, dialect="excel", delimiter=";"))
                    # maptable = reader.read()
                # refactor
                index = csvtable[0][1:]
                maptable = {row[0]: dict(zip(index, row[1:])) for row in csvtable[1:]}

            elif maptable.endswith(".json"):
                with open(maptable) as jsonfile:
                    maptable = json.load(jsonfile)
            else:
                print(f"Unknown table format: {maptable}")
                exit()

        logging.basicConfig(level=logging.INFO)
        opts = dict(overwrite=overwrite, mapper=maptable, mapkey=mapkey)

        if os.path.isfile(src):
            filename = utils.anonymize_file(src, dest, **opts)
            files = [filename]
        else:
            files = utils.anonymize_stack(src, dest, prefix=name, **opts)

        print(f"{len(files)} files were created in '{args.dest}'")

    parser_anonymize.set_defaults(func=_anonymize)


def dicom_filters(value):
    """return (tag,value) pairs"""
    filters = {}
    for tag_value in value.split(","):
        tag, value = tag_value.strip().split("=")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except:
                pass
        filters[tag] = value
    return filters
