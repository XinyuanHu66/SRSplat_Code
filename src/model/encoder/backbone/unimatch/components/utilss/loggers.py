import sys
import logging
from pathlib import Path


class MultiLineFormatter(logging.Formatter):
    """Multi-line formatter."""

    def get_header_length(self, record: logging.LogRecord) -> int:
        """Get the header length of a given record."""
        return len(
            super().format(
                logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg="",
                    args=(),
                    exc_info=None,
                )
            )
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with added indentation."""
        indent = " " * self.get_header_length(record)
        head, *trailing = super().format(record).splitlines(True)
        return head + "".join(indent + line for line in trailing)


class LoggerWriter:
    def __init__(self, logfct: any):
        self.logfct = logfct
        self.buf = []

    def write(self, msg: str) -> None:
        if msg.endswith("\n"):
            self.buf.append(msg.removesuffix("\n"))
            self.logfct("".join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass


def setup_logger(log_file_path: Path) -> None:
    log_formatter = MultiLineFormatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    pl_logger = logging.getLogger("lightning.pytorch")
    pl_lo = logging.getLogger("pytorch_lightning")
    pl_logger.setLevel(logging.INFO)
    root_logger.setLevel(logging.INFO)
    pl_lo.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    pl_logger.addHandler(log_file_handler)
    pl_lo.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)
    pl_lo.addHandler(log_stream_handler)

    sys.stdout = LoggerWriter(root_logger.info)
    sys.stderr = LoggerWriter(root_logger.error)

    logging.info("Logging file is %s" % log_file_path)
