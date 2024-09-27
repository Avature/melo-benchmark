import logging
import os


class CustomFormatter(logging.Formatter):

    def format(self, record):
        date = self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S')
        level = record.levelname[0]

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(
            os.path.join(current_dir, "..", "..", "..")
        )
        rel_path = os.path.relpath(record.pathname, project_root)
        line_no = record.lineno

        message = record.getMessage()
        return f"{date}: {level} {rel_path}:{line_no}] {message}"


def setup_logging(level=logging.INFO):
    if not logging.getLogger().hasHandlers():
        logger = logging.getLogger()
        logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())

        logger.addHandler(console_handler)
