# -*- coding: utf-8 -*-
import logging
import os
import errno

def setup_custom_logger(name, logging_level, flog=None,
                        log_format='%(asctime)s - %(levelname)s - [%(module)s]\t%(message)s'):

    if flog is None:
        raise TypeError("setup_custom_logger::Argument flog cannot be None")

    if not os.path.exists(os.path.dirname(flog)):
        try:
            os.makedirs(os.path.dirname(flog))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    formatter = logging.Formatter(fmt=(log_format))

    fhandler = logging.FileHandler(flog)
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging_level)

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    logger.addHandler(fhandler)

    return logger