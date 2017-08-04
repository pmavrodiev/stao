# -*- coding: utf-8 -*-
import configparser
import logging
import sys
import numpy as np
import pandas as pd
import os

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

    (options, args) = parser.parse_args()

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
    # Load the chosen model first in order to quit early if misspecified
    # -------------------------------------
    # TODO: this will spit out a python exception. Handle in a more user friendly manner
    m = importlib.import_module("models." + options.model)

    cache_dir = config['cache_config']['cache_dir']
    # -------------------------------------
    # Check if intermediary cache is available. In this case, skip the input reader
    # -------------------------------------
    if config.getboolean('global', 'cache_intermediary') and config.getboolean('global', 'cache_enabled'):
        logger.info("Reading intermediary cache ...")
        enriched_pd = pd.read_pickle(os.path.join(cache_dir, config['cache_config']['intermediary_cache']))
        # TODO very ugly FIX ITTTT
        stations = config['inputdata']['stations']
        stations_pd = pd.read_csv(stations, sep=';', header=0, index_col=False, encoding='latin-1')

        logger.info("Done.")
    else:
        # -------------------------------------
        # Read-in an prepare the data
        # -------------------------------------
        # read-in the data
        (migros_stores_pd, konkurrenten_stores_pd,
        drivetimes_pd, haushalt_pd, stations_pd, regionstypen_pd, arbeitnehmer_pd) = get_input(options.config, logger)

        # --- Get only the relevant hektars from the drivetimes, i.e. those from which a Migros store is reachable
        logger.info("Obtaining all drive times only for hectars from which a Migros store is reachable ... ")
        # first join the migros stores and drivetimes
        migros_merged_pd = pd.merge(left=migros_stores_pd,
                                    right=drivetimes_pd, how='left',
                                    left_on='HARasterID', right_on='ZielHARasterID')
        # second join the competitors' stores and drivetimes
        konkurrenten_merged_pd = pd.merge(left=konkurrenten_stores_pd, right=drivetimes_pd, how='left',
                                          left_on='HARasterID', right_on='ZielHARasterID')
        # now remove all stores without drivetimes info
        migros_merged_pd = migros_merged_pd.loc[~np.isnan(migros_merged_pd.FZ)]
        # sanoty check - all stores must have their HARasterIDs as both start and end HARasterID in drivetimes
        x = migros_merged_pd.loc[
            (migros_merged_pd.HARasterID == migros_merged_pd.StartHARasterID), ['StoreID', 'StoreName', 'HARasterID',
                                                                                     'StartHARasterID',
                                                                                     'ZielHARasterID',
                                                                                 'FZ']]
        if len(x) != len(np.unique(migros_merged_pd.StoreID)):
            logger.warn("Some stores do not have the required 0 entry in drivetimes['FZ']")

        konkurrenten_merged_pd = konkurrenten_merged_pd.loc[~np.isnan(konkurrenten_merged_pd.FZ)]

        konkurrenten_StartHA = np.unique(konkurrenten_merged_pd.StartHARasterID)
        migros_StartHA = np.unique(migros_merged_pd.StartHARasterID)
        # get the StartRasterIDs from which only! the competitors' stores are accessible - np.setdiff1d
        StartHA_to_exclude = np.setdiff1d(konkurrenten_StartHA, migros_StartHA, assume_unique=True)
        # need to re-order the columns, because pandas.concat sorts them
        all_stores_pd = pd.concat([migros_merged_pd, konkurrenten_merged_pd])[migros_merged_pd.columns.tolist()]
        all_stores_pd = all_stores_pd.loc[~all_stores_pd.StartHARasterID.isin(StartHA_to_exclude)]

        # enrich the drivetimes of the relevant hectars with Arbeitnehmer Information
        logger.info("Enriching with Arbeitnehmer information ...")
        enriched_pd = all_stores_pd.merge(arbeitnehmer_pd.set_index('HARasterID'),
                                          left_on='StartHARasterID', right_index=True, how='left')

        logger.info("Number of unique StartHARasterIDs without Arbeitnehmer info: %d out of %d",
                    len(np.unique(enriched_pd.loc[np.isnan(enriched_pd.ANTOT)].StartHARasterID)),
                    len(np.unique(enriched_pd.StartHARasterID)))

        # enrich the drivetimes of the relevant hectars with RegionsTyp Information
        logger.info("Enriching with Regionstyp information ...")
        enriched_pd = enriched_pd.merge(regionstypen_pd.set_index('HARasterID')[['RegionTyp', 'DTB']],
                                          left_on='HARasterID', right_index=True, how='left')

        logger.info("Number of unique ZielHARasterIDs without Regionstyp info: %d out of %d",
                    len(np.unique(enriched_pd.loc[np.isnan(enriched_pd.RegionTyp)].ZielHARasterID)),
                    len(np.unique(enriched_pd.ZielHARasterID)))

        # enrich the drive times of the relevant hectars with Haushalt information
        logger.info("Enriching with Haushalt information ...")
        enriched_pd = enriched_pd.merge(haushalt_pd[['Tot_Haushaltausgaben']],
                                                             left_on='StartHARasterID', right_index=True,
                                                         how='left')
        logger.info("Number of unique StartHektarIDs without Haushaltausgaben info: %d out of %d",
                    len(np.unique(enriched_pd.loc[np.isnan(enriched_pd.Tot_Haushaltausgaben), "StartHARasterID"])),
                    len(np.unique(enriched_pd.StartHARasterID)))


        logger.info('Creating intermediary cache ...')
        enriched_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['intermediary_cache']))
        logger.info("Done.")
        # -------------------------------------
        # DONE reading in data
        # -------------------------------------
    
    # -------------------------------------
    # RUN MODEL
    # -------------------------------------
    # pass data frames to model (at model's entry point) --> enriched_pd becomes pandas_dt
    m.model.entry({"all_stores": enriched_pd, "sbb_stations": stations_pd}, config, logger)
