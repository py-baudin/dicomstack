""" DICOM utils cli """

# coding=utf-8
import os
import csv
import json
import logging
import argparse

from . import utils, dicomstack


def cli():
    """Entry point into command-line utility"""

    # make parser
    parser = argparse.ArgumentParser("dicom", description="Various DICOM utilities.")

    # add subprograms
    subparser = parser.add_subparsers()
    cli_list(subparser)
    cli_view(subparser)
    cli_values(subparser)
    cli_export(subparser)

    # parse arguments
    args = parser.parse_args()

    if "func" in args:
        args.func(args)
    else:
        print(parser.print_help())


def cli_view(subparser):
    """show DICOM elements for one frame"""

    help = "List the DICOM elements of one frame."
    parser = subparser.add_parser("elements", help=help, description=help)
    parser.add_argument("src", help="Source file or directory")
    parser.add_argument(
        "-i",
        dest="index",
        type=int,
        default=1,
        help="Show elements of i-th frame (default: 1).",
    )
    parser.add_argument("--series", type=int, help="Filter by series number.")
    parser.add_argument(
        "-f", "--filters", type=dicom_filters, help="Filter by element=value,... pairs."
    )

    def _elements(args):
        # load stack
        stack = dicomstack.DicomStack(args.src)

        if args.series:
            # filter by series
            stack = stack(SeriesNumber=args.series)
        if args.filters:
            stack = stack(**args.filters)

        # show dicom elements
        frame = stack.frames[args.index - 1]
        # load elements
        frame.dataset.elements
        print(frame)

    parser.set_defaults(func=_elements)


def cli_values(subparser):
    """show DICOM values for one or more elements"""

    help = "Show unique values of one or more DICOM elements."
    parser = subparser.add_parser(
        "values",
        help=help,
        description=help,
    )
    parser.add_argument("src", help="Source file or directory")
    parser.add_argument("elements", nargs="+", help="DICOM elements.")
    parser.add_argument("--series", type=int, help="Filter by series number.")
    parser.add_argument(
        "-f", "--filters", type=dicom_filters, help="Filter by element=value,... pairs."
    )

    def _values(args):
        # load stack
        stack = dicomstack.DicomStack(args.src)

        if args.series:
            # filter by series
            stack = stack(SeriesNumber=args.series)
        if args.filters:
            stack = stack(**args.filters)

        # show dicom elements
        for element in args.elements:
            print(f"{element}:", stack.unique(element))

    parser.set_defaults(func=_values)


def cli_list(subparser):
    """describe DICOM stack"""

    help = "List DICOM studies, series and non-DICOM files."
    parser = subparser.add_parser(
        "list",
        help=help,
        description=help,
    )
    parser.add_argument("src", help="Source file or directory")

    def _list(args):
        # load stack
        stack = dicomstack.DicomStack(args.src)

        # show description
        print(stack.describe())

    parser.set_defaults(func=_list)


export_descr = """
Examples:
    # Simply anonymize DICOM stacks
    dicom export <source> <destination>

    # Using a mapping table to modify values:
    map.json:
        {"Patient ID": "some id", "Series Description": "some description"}
    dicom export -t map.json <source> <destination>

    # Using a mapping table to modify DICOM selectively, based on a DICOM element
    # Example, using the 'Patient ID' field as mapping key:
    table.csv:
            ; Patient ID; Patient's Name
        id1 ; new-id1   ; new-name1
        id2 ; new-id2   ; new-name2
    dicom export -t map.json -k PatientID <source> <destination>

    # use option "--dont-anonymize" in combination with the above to keep
    # identification information

"""


def cli_export(subparser):
    """anonymize DICOM"""

    # parser
    help = "Export a modified/anonymous copy of a DICOM stack."
    parser_export = subparser.add_parser(
        "export",
        help=help,
        description=help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=export_descr,
    )
    parser_export.add_argument("src", help="Source DICOM file or directory.")
    parser_export.add_argument("dest", help="Destination directory.")
    parser_export.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing files.",
    )
    parser_export.add_argument(
        "-n",
        "--dont-anonymize",
        default=False,
        action="store_true",
        help="Dot not remove personal data.",
    )
    parser_export.add_argument(
        "--remove-private",
        default=False,
        action="store_true",
        help="Remove private fields.",
    )
    parser_export.add_argument(
        "-t",
        "--map-table",
        metavar="TABLE",
        help="Mapping table filename in CSV or JSON format. "
        "If CSV: use ';' delimiters, the first column being the mapping key.",
    )
    parser_export.add_argument(
        "-k",
        "--map-key",
        metavar="FIELD",
        help="DICOM field to use as key to mapping table (eg. 'Patient ID').",
    )

    def _export(args):
        src = args.src
        dest = args.dest
        overwrite = args.overwrite
        maptable = args.map_table
        mapkey = args.map_key
        anonymize = not args.dont_anonymize
        remove_private = args.remove_private

        if maptable:
            if maptable.endswith(".csv"):
                if mapkey is None:
                    print("A mapping key must be provided along with the mapping table")
                    exit()
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
        opts = dict(
            overwrite=overwrite,
            mapper=maptable,
            mapkey=mapkey,
            anonymize=anonymize,
            remove_private_tags=remove_private,
        )

        if os.path.exists(dest):
            # destination
            print(f"Warning destination is not emtpy: {dest}")
            while True:
                ans = input("Continue anyway (y/n)?: ")
                if ans == "y":
                    break
                elif ans == "n":
                    exit()

        if os.path.isfile(src):
            # source
            filename = utils.export_file(src, dest, **opts)
            files = [filename]
        elif os.path.isdir(src):
            files = utils.export_stack(src, dest, **opts)
        else:
            print(f"No such file/directory: {src}")
            exit()

        print(f"{len(files)} files were created in '{args.dest}'")

    parser_export.set_defaults(func=_export)


def dicom_filters(string):
    """return (element, value) pairs"""
    filters = {}
    for element_value in string.split(","):
        element, value = element_value.strip().split("=")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except:
                pass
        filters[element] = value
    return filters


if __name__ == "__main__":
    cli()
