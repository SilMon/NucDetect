"""
Central logging configuration for NucDetect.

This module is the single place where the application's log output is configured. It replaces the
three independent, home-grown logging paths the project used before:

* ``Detector.add_log_message`` / ``Detector.save_log_messages`` -- buffered per-image messages that
  were dumped into the log file with a custom block format
* ``NucDetect.write_to_log``                                   -- a second, unsynchronised writer
  opening the same file
* ``core.custom_logging.CustomLogger``                         -- a thread + queue wrapper around
  ``logging`` that was never instantiated anywhere

All three wrote to ``gui.Paths.log_path`` without a shared handle or lock, and a large number of
diagnostics were emitted with bare ``print()`` and therefore never reached the log file at all.

Usage
-----
Call :func:`configure_logging` exactly once during start-up (see ``NucDetectAppQT.main``). Every
module then obtains its logger at import time::

    from core.logging_config import get_logger

    LOGGER = get_logger(__name__)
    LOGGER.info("something happened")

Multiprocessing
---------------
Handlers cannot be inherited by ``ProcessPoolExecutor`` workers -- on Windows the children are
spawned, not forked, so they start with an unconfigured ``logging`` module, and a handler pickled
from the parent would not work either. :func:`init_worker_logging` is therefore installed as the
executor's ``initializer`` so that every worker builds its *own* handler in its *own* process.

The bulk of the analysis log does not travel through that handler: ``Detector.analyse_image``
returns its accumulated messages with the result, and the parent process replays them (see
``Detector.get_log_messages`` and ``NucDetect._analyze_all``). Only messages a worker emits outside
the analysis log -- errors, mostly -- are written by the worker directly. Concurrent appends from a
handful of processes can in principle interleave; keeping the routine per-image messages out of that
path keeps the exposure to occasional error lines.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Iterable, Optional

import gui.Paths as gpaths

#: Name of the package logger. Every logger handed out by :func:`get_logger` is a child of it, so
#: configuring this one logger configures the whole application.
ROOT_LOGGER_NAME = "nucdetect"

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(processName)-16s %(name)-34s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Guards against installing a second set of handlers, which would duplicate every line
_configured = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Function to get the logger for the given module

    Pass ``__name__``; the module path is used as the logger name so the source of a message can be
    identified in the log file. Names are anchored under :data:`ROOT_LOGGER_NAME`, so obtaining a
    logger never depends on :func:`configure_logging` having run already -- which matters because
    modules call this at import time, before start-up configuration happens.

    :param name: The module name to derive the logger name from, usually ``__name__``
    :return: The logger to use
    """
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def log_messages(lines: Iterable[str], level: int = logging.INFO,
                 logger: Optional[logging.Logger] = None) -> None:
    """
    Function to log a block of already formatted lines

    Used to replay the analysis messages a Detector buffered -- possibly in a worker process -- as
    individual records, so every line carries its own timestamp and is written atomically.

    :param lines: The formatted lines to log
    :param level: The level to log the lines at
    :param logger: The logger to write to. Defaults to the analysis logger
    :return: None
    """
    target = logger if logger is not None else get_logger("analysis")
    for line in lines:
        target.log(level, line)


def _create_file_handler(log_path: str, level: int) -> logging.Handler:
    """
    Function to create the handler writing to the log file

    :param log_path: Path leading to the log file
    :param level: The minimal level a record needs to be written
    :return: The created handler
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Explicit UTF-8: the log contains image file names, so without it Python falls back to the
    # locale codepage (cp1252 on a German Windows) and any name outside that codepage raises
    # UnicodeEncodeError in the middle of an analysis
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.setLevel(level)
    return handler


def _create_console_handler(level: int) -> Optional[logging.Handler]:
    """
    Function to create the handler writing to the console, if a console is available

    A PyQt application started with ``pythonw.exe`` or packaged with ``--noconsole`` has no standard
    streams -- ``sys.stderr`` is then ``None`` and attaching a ``StreamHandler`` to it raises. This
    is exactly the situation in which the replaced ``print()`` calls silently lost their output.

    :param level: The minimal level a record needs to be written
    :return: The created handler or None, if no usable stream is available
    """
    stream = sys.stderr
    if stream is None or not hasattr(stream, "write"):
        return None
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.setLevel(level)
    return handler


def configure_logging(log_path: Optional[str] = None,
                      console: bool = True,
                      level: int = logging.INFO,
                      force: bool = False) -> logging.Logger:
    """
    Function to configure the application wide logging

    Idempotent: repeated calls are ignored unless ``force`` is set, so importing a module twice or
    a stray call from a dialog cannot duplicate log lines.

    :param log_path: Path leading to the log file. Defaults to gui.Paths.log_path
    :param console: If true, log records are additionally written to stderr when one is available
    :param level: The minimal level a record needs to be handled
    :param force: If true, existing handlers are replaced instead of the call being ignored
    :return: The configured package logger
    """
    global _configured
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured and not force:
        return logger
    close_logging()
    logger.setLevel(level)
    # The root logger is left alone on purpose: TensorFlow and matplotlib attach their own handlers
    # to it, and propagating there would duplicate every NucDetect line into their output
    logger.propagate = False
    logger.addHandler(_create_file_handler(log_path or gpaths.log_path, level))
    if console:
        console_handler = _create_console_handler(level)
        if console_handler is not None:
            logger.addHandler(console_handler)
    _configured = True
    return logger


def init_worker_logging(log_path: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Function to configure logging inside a ProcessPoolExecutor worker

    Installed as the executor's ``initializer``. Runs once per worker process and builds a handler
    owned by that process -- handlers are not inheritable across a spawn and are not picklable.
    The console handler is omitted: worker processes have no console of their own on Windows.

    :param log_path: Path leading to the log file. Defaults to gui.Paths.log_path
    :param level: The minimal level a record needs to be handled
    :return: None
    """
    configure_logging(log_path=log_path, console=False, level=level, force=True)


def close_logging() -> None:
    """
    Function to detach and close all handlers of the package logger

    Needed before the log file may be replaced or removed -- on Windows an open handler keeps the
    file locked and truncating it behind the handler's back leaves the write position past the new
    end of file.

    :return: None
    """
    global _configured
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _configured = False


def reset_log_file(log_path: Optional[str] = None) -> None:
    """
    Function to erase the log file and resume logging into the empty file

    :param log_path: Path leading to the log file. Defaults to gui.Paths.log_path
    :return: None
    """
    path = log_path or gpaths.log_path
    close_logging()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass
    configure_logging(log_path=path)
