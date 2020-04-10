""" DICOM utils cli """
# coding=utf-8
import os
import argparse

from . import utils, dicomstack


def cli():
    """ entry point into command-line utility """

    # make parser
    parser = argparse.ArgumentParser("dicom", description="Various DICOM utils")

    # add subprograms
    subparser = parser.add_subparsers()
    cli_describe(subparser)
    cli_anonymize(subparser)

    # parse arguments
    args = parser.parse_args()

    if "func" in args:
        args.func(args)
    else:
        print(parser.print_help())


def cli_describe(subparser):
    """ describe DICOM stack """

    parser_anonymize = subparser.add_parser(
        "list", help="List studies, series and non-DICOM files."
    )
    parser_anonymize.add_argument("src", help="Source file or directory")
    parser_anonymize.add_argument(
        "--tags", help="Show DICOM tags", default=False, action="store_true"
    )
    parser_anonymize.add_argument(
        "-n", help="Only show n-first frames", type=int, default=None
    )
    parser_anonymize.add_argument(
        "--series", nargs="*", help="Filter by series number", type=int
    )

    def _describe(args):
        # load stack
        stack = dicomstack.DicomStack(args.src)

        if args.series:
            # filter by series
            stack = stack(SeriesNumber=args.series)

        if args.tags:
            # show dicom tags
            frames = stack.frames[: args.n]
            for frame in frames:
                frame.elements
                print(frame)
            print(f"Showing {len(frames)} frame(s).")
        else:
            # show description
            print(stack.describe())

    parser_anonymize.set_defaults(func=_describe)


def cli_anonymize(subparser):
    """ anonymize DICOM """

    # parser
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

    def _anonymize(args):
        src = args.src
        dest = args.dest
        name = args.filename
        overwrite = args.overwrite

        if os.path.isfile(src):
            filename = utils.anonymize_file(
                src, dest, filename=name, overwrite=overwrite
            )
            files = [filename]
        else:
            files = utils.anonymize_stack(src, dest, prefix=name, overwrite=overwrite)

        print("The following files were anonymized:")
        for filename in files:
            print("\t", filename)

    parser_anonymize.set_defaults(func=_anonymize)
