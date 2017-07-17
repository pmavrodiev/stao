# -*- coding: utf-8 -*-
import configparser
import logging
import sys

from input_reader.input_reader import get_input
from simple_logging.custom_logging import setup_custom_logger

from optparse import OptionParser
import importlib

if __name__ == "__main__":

    # -------------------------------------
    # Define command-line arguments
	# -------------------------------------
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model",
                      help = "The model to use. This should be a package in the models/ directory. The __init__.py"
                             " of the package should create an instance of the model, named 'model'")
    parser.add_option("-c", "--config", dest="config",
                      help="A file containing the settings for the specified model")

    parser.add_option("-l", "--log", dest="logname",
                      help="Location of a log file for the current run. While the log file will be appended to if it"
                           " exists or created if it does not, the base log directory should exist")
    parser.add_option("-b", "--blubb", dest="bojanvar",
                     help="just testing stuff")

    (options, args) = parser.parse_args()

    print("starting model")
    if not options.model or not options.config or not options.logname:
        parser.print_help()
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(options.config)
    
	# -------------------------------------
    # Set up logger
	# -------------------------------------
    LOGGING_LEVEL = logging.INFO
    logger = setup_custom_logger('GM_LOGGER', LOGGING_LEVEL, flog=options.logname)

	# -------------------------------------
    # Load the chosen model
	# -------------------------------------
    m = importlib.import_module("models." + options.model)

    # -------------------------------------
    # Read-in an prepare the data
    # -------------------------------------
    # read-in the data
    (stores_pd, stores_migros_pd, drivetimes_pd, haushalt_pd, referenz_pd, stations_pd) = get_input(options.config, logger)

    # get all relevant hectars, i.e. those from which a Migros store is reachable
    # use a 'set' to easily remove duplicates
    logger.info("Subsetting only the hectars from which a Migros store is reachable")
    relevant_hectars = set(drivetimes_pd.loc[stores_migros_pd.ID]['hektar_id'])

    # get all drive times for the relevant hectars
    logger.info("Obtaining all drive times only for hectars from which a Migros store is reachable")
    drivetimes_rel_hectars_pd = drivetimes_pd[drivetimes_pd['hektar_id'].isin(relevant_hectars)]

    # enrich the drive times of the relevant hectars with store information
    # and do an inner join to detect missing stores in stores_pd
    logger.info("Enriching with store information")
    before = len(set(drivetimes_rel_hectars_pd.index))

    drivetimes_rel_hectars_stores_pd = drivetimes_rel_hectars_pd.merge(
        stores_pd[['ID', 'FORMAT', 'vfl', 'RELEVANZ', 'type']],
        left_index=True, right_on='ID', how='inner')

    logger.info("%d stores appear in drivetimes, but have no associated information in stores_sm.csv",
                before - len(set(drivetimes_rel_hectars_stores_pd.index)))

    # enrich the drive times of the relevant hectars with Haushalt information
    logger.info("Enriching with Haushalt information")
    enriched_pd = drivetimes_rel_hectars_stores_pd.merge(haushalt_pd[['Tot_Haushaltausgaben']],
                                                         left_on='hektar_id', right_index=True,
                                                         how='left')
   
    # -------------------------------------
    # DONE reading in data
    # -------------------------------------
    
    # -------------------------------------
    # RUN MODEL
    # -------------------------------------
    # pass data frames to model (at model's entry point) --> enriched_pd becomes panads_dt
    m.model.entry(enriched_pd, config, logger, stores_pd, stores_migros_pd, referenz_pd, stations_pd)
