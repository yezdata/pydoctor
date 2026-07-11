import logging
import sys
import difflib

SUCCESS_LEVEL_NUM = 25
DIFF_LEVEL_NUM = 26


def add_custom_levels():
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

    logging.Logger.success = success

    logging.addLevelName(DIFF_LEVEL_NUM, "DIFF")

    def diff_method(self, target_id: str, old_str: str, new_str: str, *args, **kws):
        if not self.isEnabledFor(DIFF_LEVEL_NUM):
            return

        old_lines = old_str.strip().splitlines()
        new_lines = new_str.strip().splitlines()

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{target_id} (old)",
            tofile=f"b/{target_id} (new)",
            lineterm="",
        )

        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                self._log(
                    DIFF_LEVEL_NUM, line, args, extra={"diff_type": "added"}, **kws
                )
            elif line.startswith("-") and not line.startswith("---"):
                self._log(
                    DIFF_LEVEL_NUM, line, args, extra={"diff_type": "removed"}, **kws
                )
            elif (
                line.startswith("@@")
                or line.startswith("---")
                or line.startswith("+++")
            ):
                self._log(
                    DIFF_LEVEL_NUM, line, args, extra={"diff_type": "meta"}, **kws
                )
            else:
                self._log(
                    DIFF_LEVEL_NUM, line, args, extra={"diff_type": "plain"}, **kws
                )

    logging.Logger.diff = diff_method

    logging.success = lambda msg, *args, **kws: logging.log(
        SUCCESS_LEVEL_NUM, msg, *args, **kws
    )
    logging.diff = lambda target_id, old_str, new_str, *args, **kws: (
        logging.getLogger().diff(target_id, old_str, new_str, *args, **kws)
    )


class ColorFormatter(logging.Formatter):
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    BRIGHT_GREEN = "\x1b[32;1m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    DIFF_ADDED = "\x1b[42;30m"
    DIFF_REMOVED = "\x1b[41;37m"
    DIFF_META = "\x1b[36m"

    BASE_FORMAT = "%(asctime)s %(message)s"
    INFO_FORMAT = "%(message)s"
    SUCCESS_FORMAT = "[✓] %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + BASE_FORMAT + RESET,
        logging.INFO: GREY + INFO_FORMAT + RESET,
        SUCCESS_LEVEL_NUM: BRIGHT_GREEN + SUCCESS_FORMAT + RESET,
        logging.WARNING: YELLOW + BASE_FORMAT + RESET,
        logging.ERROR: RED + BASE_FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + BASE_FORMAT + RESET,
    }

    def format(self, record):
        diff_type = getattr(record, "diff_type", None)

        if diff_type:
            if diff_type == "added":
                log_fmt = self.DIFF_ADDED + "%(message)s" + self.RESET
            elif diff_type == "removed":
                log_fmt = self.DIFF_REMOVED + "%(message)s" + self.RESET
            elif diff_type == "meta":
                log_fmt = self.DIFF_META + "%(message)s" + self.RESET
            else:
                log_fmt = "%(message)s"
        else:
            log_fmt = self.FORMATS.get(record.levelno, self.BASE_FORMAT)

        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logging(log_level: int) -> None:
    add_custom_levels()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(ColorFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
