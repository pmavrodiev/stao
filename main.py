# -*- coding: utf-8 -*-

import configparser
import logging
import sys

from input_reader.input_reader import get_input
from simple_logging.custom_logging import setup_custom_logger


from optparse import OptionParser
import importlib


if __name__ == "__main__":

    # Define command-line arguments
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model",
                      help = "The model to use. This should be a package in the models/ directory. The __init__.py"
                             " of the package should create an instance of the model, named 'model'")
    parser.add_option("-c", "--config", dest="config",
                      help="A file containing the settings for the specified model")

    parser.add_option("-l", "--log", dest="logname",
                      help="Location of a log file for the current run. While the log file will be appended to if it"
                           " exists or created if it does not, the base log directory should exist")


    (options, args) = parser.parse_args()

    if not options.model or not options.config or not options.logname:
        parser.print_help()
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(options.config)

    # logger einrichten
    LOGGING_LEVEL = logging.INFO
    logger = setup_custom_logger('GM_LOGGER', LOGGING_LEVEL, flog=options.logname)

    # load the chosen model
    m = importlib.import_module("models." + options.model)

    ##########################
    #### READ-IN AND PREPARE THE DATA ####
    ##########################
    # read-in the data
    (stores_pd, stores_migros_pd, drivetimes_pd, haushalt_pd, referenz_pd) = get_input(options.config, logger)

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

    m.model.entry(enriched_pd, config, logger, stores_migros_pd, referenz_pd)
