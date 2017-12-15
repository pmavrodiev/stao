# -*- coding: utf-8 -*-
import logging
import os
import errno

# levels shoudl be between INFO (20) and WARNING (30)
custom_levels = {"PARAMETER_SETTING": 21,
                 "COMPUTE_UMSATZ_HAUSHALTE": 22,
                 "COMPUTE_UMSATZ_PENDLER": 23,
                 "COMPUTE_UMSATZ_ARBEITNEHMER": 24,
                 "COMPUTE_UMSATZ_DTB": 25}



logging.addLevelName(custom_levels["PARAMETER_SETTING"],
                     "PARAMETER_SETTING")
logging.addLevelName(custom_levels["COMPUTE_UMSATZ_HAUSHALTE"],
                     "COMPUTE_UMSATZ_HAUSHALTE")
logging.addLevelName(custom_levels["COMPUTE_UMSATZ_PENDLER"],
                     "COMPUTE_UMSATZ_PENDLER")
logging.addLevelName(custom_levels["COMPUTE_UMSATZ_ARBEITNEHMER"],
                     "COMPUTE_UMSATZ_ARBEITNEHMER")
logging.addLevelName(custom_levels["COMPUTE_UMSATZ_DTB"],
                     "COMPUTE_UMSATZ_DTB")



def PARAMETER_SETTING(self, message, *args, **kws):
    self._log(custom_levels["PARAMETER_SETTING"], message, args, **kws)

def COMPUTE_UMSATZ_HAUSHALTE(self, message, *args, **kws):
    self._log(custom_levels["COMPUTE_UMSATZ_HAUSHALTE"], message, args, **kws)


def COMPUTE_UMSATZ_PENDLER(self, message, *args, **kws):
    self._log(custom_levels["COMPUTE_UMSATZ_PENDLER"], message, args, **kws)


def COMPUTE_UMSATZ_ARBEITNEHMER(self, message, *args, **kws):
    self._log(custom_levels["COMPUTE_UMSATZ_ARBEITNEHMER"], message, args, **kws)

def COMPUTE_UMSATZ_DTB(self, message, *args, **kws):
    self._log(custom_levels["COMPUTE_UMSATZ_DTB"], message, args, **kws)


logging.Logger.param_info = PARAMETER_SETTING
logging.Logger.COMPUTE_UMSATZ_PENDLER = COMPUTE_UMSATZ_PENDLER
logging.Logger.COMPUTE_UMSATZ_ARBEITNEHMER = COMPUTE_UMSATZ_ARBEITNEHMER
logging.Logger.COMPUTE_UMSATZ_HAUSHALTE = COMPUTE_UMSATZ_HAUSHALTE
logging.Logger.COMPUTE_UMSATZ_DTB = COMPUTE_UMSATZ_DTB

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