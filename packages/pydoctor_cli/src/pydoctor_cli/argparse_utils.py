from argparse import ArgumentParser


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Update docstrings in Python files. "
        "By default, it only adds new docstrings where they are missing."
        "Use '# pydoctor: ignore' comment to skip a function or class from processing."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Python file or directory containing Python files to process.",
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--replace",
        action="store_const",
        dest="extraction_option",
        const="with_docstring",
        help="Only replace existing docstrings.",
    )

    group.add_argument(
        "--all",
        action="store_const",
        dest="extraction_option",
        const="all",
        help="Process all functions / classes.",
    )

    parser.set_defaults(extraction_option="without_docstring")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without modifying any files and show changes (default: False)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Turn on verbose output (default: False)",
    )

    return parser
