from argparse import ArgumentParser


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Update docstrings in Python files. "
        "By default, it only adds new docstrings in code blocks where they are missing "
        "and also ignores files and directories listed in .gitignore and base directories like .venv/, .git/... "
        "You can also provide custom ignore rules in a .pydoctor_ignore file (same syntax as .gitignore) in the project root "
        "or use '# pydoctor: ignore' comment to skip a class or function / method from processing."
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
        help="Replace only existing docstrings.",
    )

    group.add_argument(
        "--all",
        action="store_const",
        dest="extraction_option",
        const="all",
        help="Process all code blocks.",
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
