# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/8
import logging
import sys
import os

def set_logger(stdout_level=logging.INFO, log_filename=None):
    """
    Set logger for current experiment
    :param stdout_level:
    :param log_filename:
    :return:
    """
    # for console
    simple_formatter = logging.Formatter("%(name)s:%(levelname)s: %(message)s")
    # for file
    detailed_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = logging.getLogger('experiment')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # PREVENT IT from propagating messages to the root logger

    # what is handler?
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, stdout_level))  # ?
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    if log_filename is not None:
        log_filename = os.path.abspath(log_filename)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    return logger

if __name__ == '__main__':
    pass

