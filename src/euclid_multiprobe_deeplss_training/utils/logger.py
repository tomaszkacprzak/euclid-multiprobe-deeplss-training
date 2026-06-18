# Copyright (C) 2017 ETH Zurich, Cosmology Research Group

"""
Created on Jul 2021
author: Tomasz Kacprzak
"""

import os, sys, logging, time, datetime

from tqdm import tqdm


logging_levels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

logging_levels_num = {
    "critical": 50,
    "error": 40,
    "warning": 30,
    "info": 20,
    "debug": 10,
}


class CustomFormatter(logging.Formatter):
    RED = "\033[91m"
    VIOLET = "\033[95m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    reset = "\033[0m"
    format = "%(asctime)s %(name)10s %(levelname).3s   %(message)s "

    FORMATS = {
        logging.DEBUG: VIOLET + format + reset,
        logging.INFO: format,
        logging.WARNING: BOLD + YELLOW + format + reset,
        logging.ERROR: BOLD + RED + format + reset,
        logging.CRITICAL: BOLD + RED + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%y-%m-%d %H:%M:%S")
        return formatter.format(record)


class Timer:
    def __init__(self):
        self.time_start = {}
        self.time_start["default"] = time.time()

    def reset(self, name="default"):
        self.time_start[name] = time.time()

    def elapsed(self, name="default"):
        return str(datetime.timedelta(seconds=(time.time() - self.time_start[name])))[:-4]

    def start(self, name="default"):
        self.time_start[name] = time.time()


class Progressbar:
    def __init__(self, logger=None):
        self.logger = logger

    def __call__(self, collection, at_level="info", **kw):
        kw.setdefault("disable", self.logger.level > logging_levels_num[at_level])
        kw.setdefault("bar_format", "{percentage:3.0f}%|{bar:28}|   {r_bar:<40} {desc}")
        kw.setdefault("colour", "blue")
        kw.setdefault("mininterval", 1)
        kw.setdefault("file", sys.stdout)
        return tqdm(collection, **kw)


def get_logger(filepath, logging_level=None, progressbar_color="red"):
    """
    Get logger, if logging_level is unspecified, then try using the environment variable PYTHON_LOGGER_LEVEL.
    Defaults to info.
    :param filepath: name of the file that is calling the logger, used to give it a name.
    :return: logger object
    """

    if logging_level is None:
        if "PYTHON_LOGGER_LEVEL" in os.environ:
            logging_level = os.environ["PYTHON_LOGGER_LEVEL"]
        else:
            logging_level = "info"

    logger_name = "{:>12}".format(os.path.basename(filepath)[:12])
    logger = logging.getLogger(logger_name)

    if len(logger.handlers) == 0:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(CustomFormatter())
        logger.addHandler(stream_handler)
        logger.propagate = False
        set_logger_level(logger, logging_level)

    logger.progressbar = Progressbar(logger)
    logger.timer = Timer()
    logger.islevel = lambda level: logging_levels[level] == logger.level

    return logger


def set_logger_level(logger, level):
    logger.setLevel(logging_levels[level])


def set_all_loggers_level(level):
    os.environ["PYTHON_LOGGER_LEVEL"] = level

    loggerDict = logging.root.manager.loggerDict
    for key in loggerDict:
        try:
            set_logger_level(logger=loggerDict[key], level=level)
        except Exception as err:
            pass


import os
import gc
import time
import psutil
import threading
from contextlib import contextmanager

@contextmanager
def peak_memory(label="block", interval=0.01):
    process = psutil.Process(os.getpid())

    gc.collect()
    start = process.memory_info().rss
    peak = start
    running = True

    def monitor():
        nonlocal peak
        while running:
            rss = process.memory_info().rss
            peak = max(peak, rss)
            time.sleep(interval)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    try:
        yield
    finally:
        running = False
        thread.join()

        gc.collect()
        end = process.memory_info().rss

        print(f"{label}:")
        print(f"  start:      {start / 1024**2:.2f} MiB")
        print(f"  end:        {end / 1024**2:.2f} MiB")
        print(f"  peak:       {peak / 1024**2:.2f} MiB")
        print(f"  peak delta: {(peak - start) / 1024**2:.2f} MiB")
        print(f"  end delta:  {(end - start) / 1024**2:+.2f} MiB")