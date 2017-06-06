# -*- coding: utf-8 -*-

import configparser

import logging

import numpy as np
import pandas as pd

from input_reader.input_reader import get_input
from simple_logging.custom_logging import setup_custom_logger

from models.model_MBI import model_MBI


# logger einrichten
LOGGING_LEVEL = logging.INFO
logger = setup_custom_logger('GM_LOGGER', LOGGING_LEVEL, flog="logs/gm.log")

settingsFile = "settings.cfg"

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(settingsFile)

    ##########################
    #### READ-IN AND PREPARE THE DATA ####
    ##########################
    # read-in the data
    (stores_pd, stores_migros_pd, drivetimes_pd, haushalt_pd, referenz_pd) = get_input(settingsFile, logger)

    # get all relevant hektars, i.e. those from which a Migros store is reachable
    # use a 'set' to easily remove duplicates
    logger.info("Subsetting only the hektars from which a Migros store is reachable")
    relevant_hektars = set(drivetimes_pd.loc[stores_migros_pd.ID]['hektar_id'])

    # get all drive times for the relevant hektars
    logger.info("Obtaining all drive times only for hektars from which a Migros store is reachable")
    drivetimes_rel_hektars_pd = drivetimes_pd[drivetimes_pd['hektar_id'].isin(relevant_hektars)]

    # enrich the drive times of the relevant hektars with store information
    # and do an inner join to detect missing stores in stores_pd
    logger.info("Enriching with store information")
    before = len(set(drivetimes_rel_hektars_pd.index))

    drivetimes_rel_hektars_stores_pd = drivetimes_rel_hektars_pd.merge(
        stores_pd[['ID', 'FORMAT', 'vfl', 'RELEVANZ', 'type']],
        left_index=True, right_on='ID', how='inner')

    logger.info("%d stores appear in drivetimes, but have no associated information in stores_sm.csv",
                before - len(set(drivetimes_rel_hektars_stores_pd.index)))

    # enrich the drive times of the relevant hektars with Haushalt information
    logger.info("Enriching with Haushalt information")
    enriched_pd = drivetimes_rel_hektars_stores_pd.merge(haushalt_pd[['Tot_Haushaltausgaben']],
                                                         left_on='hektar_id', right_index=True,
                                                         how='left')
    # try to correct for missing HA info by assuming a default of 1 household / person
    enriched_pd['Tot_Haushaltausgaben_corrected'] = enriched_pd['Tot_Haushaltausgaben'].fillna(7800 / 2.5)

    ##########################
    ##########################
    ##########################

    ##########################
    #### MODEL MBI        ####
    ##########################

    model = model_MBI.model_MBI()
    model.entry(enriched_pd, config, logger, stores_migros_pd, referenz_pd)

