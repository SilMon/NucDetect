import logging
import sched
import os
import threading
import time


class CustomLogger:
    __slots__ = (
        "queue",
        "logging_thread",
        "logger",
        "formatter",
        "file_handler",
        "console_handler"
    )

    def __init__(self, log_path: str,  log_to_console: bool = True):
        """
        :param log_path: The path leading to the logging file
        :param log_to_console: If true, the logging output will also be printed to the console
        """
        self.logger = logging.getLogger("NucDetect Logger")
        self.logger.setLevel(logging.DEBUG)
        self.queue = []
        self.logging_thread = threading.Thread(target=self._process_logging_queue)
        self.logging_thread.daemon = True
        self.check_if_logging_file_exists(log_path)
        self.file_handler = logging.FileHandler(log_path)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(self.formatter)
        self.file_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.file_handler)
        if log_to_console:
            self.console_handler = logging.StreamHandler()
            self.console_handler.setFormatter(self.formatter)
            self.console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(self.console_handler)

    def check_if_logging_file_exists(self, log_path: str) -> None:
        """
        Method to check if the logging file exists and if not to create a new one

        :param log_path: Path leading to the log file
        :return: None
        """
        os.makedirs(os.path.split(log_path)[0], exist_ok=True)
        # Check if the file at the log path exists
        if not os.path.isfile(log_path):
            open(log_path, "w").close()

    def log(self, message: str, level: int) -> None:
        """
        Method to log the specified message

        :param message: The message to log
        :param level: The severity level of the level
        :return: None
        """
        self.queue.append((message, level))
        if not self.logging_thread.is_alive():
            self.logging_thread.start()

    def _process_logging_queue(self) -> None:
        """
        Internal method to execute the logging queue

        :return: None
        """
        while True:
            while self.queue:
                message, level = self.queue.pop()
                self.logger.log(msg=message, level=level)
            time.sleep(0.1)

    def debug(self, message: str) -> None:
        self.log(message, logging.DEBUG)

    def info(self, message: str) -> None:
        self.log(message, logging.INFO)

    def warning(self, message: str) -> None:
        self.log(message, logging.WARNING)

    def error(self, message: str) -> None:
        self.log(message, logging.ERROR)

    def critical(self, message: str) -> None:
        self.log(message, logging.CRITICAL)


