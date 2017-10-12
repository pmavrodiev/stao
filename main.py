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
from models.model_MBI_v1_2.parallel import apply_parallel, filter
from functools import partial

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
    config.read_file(open(options.config, mode="r", encoding="utf-8"))
    # config.read(options.config)
    
    # -------------------------------------
    # Set up logger
    # -------------------------------------
    LOGGING_LEVEL = logging.DEBUG
    logger = setup_custom_logger('GM_LOGGER', LOGGING_LEVEL, flog=options.logname)

    # -------------------------------------
    # Load the chosen model first in order to quit early if misspecified
    # -------------------------------------
    # TODO: If this fails, it will spit out a python exception. Handle in a more user friendly manner.
    m = importlib.import_module("models." + options.model)

    cache_dir = config['cache_config']['cache_dir']
    # -------------------------------------
    # Check if intermediary cache is available. In this case, skip the input reader
    # -------------------------------------
    if config.getboolean('global', 'cache_intermediary'):
        logger.info("Reading intermediary cache ...")
        enriched_pd = pd.read_pickle(os.path.join(cache_dir, config['cache_config']['intermediary_cache']))
        # TODO very ugly FIX ITTTT
        stations_tuple = eval(config['inputdata']['stations'], {}, {})
        stations = stations_tuple[0]
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
        # migros_merged_pd has index HARasterID
        # drivetimes_pd has index ZielHARasterID
        migros_merged_pd = migros_stores_pd.join(drivetimes_pd)
        logger.info("Done.")
        # now remove all stores without drivetimes info
        logger.info("Removing stores without fahrzeit information")
        migros_merged_pd = migros_merged_pd.loc[~np.isnan(migros_merged_pd.FZ)]
        logger.info("Done.")

        logger.info("Sanity check")
        # sanoty check - all stores must have their HARasterIDs as both start and end HARasterID in drivetimes
        x = migros_merged_pd.loc[
            (migros_merged_pd.index == migros_merged_pd.StartHARasterID), ['StoreID', 'StoreName',
                                                                                'StartHARasterID',
                                                                                'ZielHARasterID',
                                                                                'FZ']]
        if len(x) != len(np.unique(migros_merged_pd.StoreID)):
            logger.warn("Some stores do not have the required 0 entry in drivetimes['FZ']")
            logger.warn("Sanity check failed")
        else:
            logger.info("Sanity check passed.")

        # single-store option
        single_store = config['global']['single_store'].encode("latin-1")
        if len(single_store) > 0:
            # --- SINGLE STORE MODE? ---------
            logger.info('Single store mode chosen - %s', single_store)
            logger.info("Removing Migros stores from which %s cannot be reached", single_store)
            single_store_pd = migros_stores_pd.loc[migros_stores_pd.StoreName == single_store.decode('latin-1')]
            # single_store_pd has index HARasterID
            # drivetimes_pd has index ZielHARasterID
            single_store_merged_pd = single_store_pd.join(drivetimes_pd)
            single_store_startHA = np.unique(single_store_merged_pd.StartHARasterID)
            migros_StartHA = np.unique(migros_merged_pd.StartHARasterID)
            StartHA_to_exclude = np.setdiff1d(migros_StartHA, single_store_startHA, assume_unique=True)
            # logger.info("%d", len(migros_merged_pd))
            # migros_merged_pd = pd.concat([migros_merged_pd, single_store_merged_pd])[migros_merged_pd.columns.tolist()]
            migros_merged_pd = migros_merged_pd.loc[~migros_merged_pd.StartHARasterID.isin(StartHA_to_exclude)]

        # logger.info("%d", len(migros_merged_pd))
        # sys.exit(1)
        logger.info("Removing hectares from which only competitor stores can be reached")
        # konkurrenten_stores_pd has index HARasterID
        # drivetimes_pd has index ZielHARasterID
        konkurrenten_merged_pd = konkurrenten_stores_pd.join(drivetimes_pd)
        konkurrenten_merged_pd = konkurrenten_merged_pd.loc[~np.isnan(konkurrenten_merged_pd.FZ)]

        konkurrenten_StartHA = np.unique(konkurrenten_merged_pd.StartHARasterID)
        migros_StartHA = np.unique(migros_merged_pd.StartHARasterID)
        # get the StartRasterIDs from which only! the competitors' stores are accessible - np.setdiff1d
        StartHA_to_exclude = np.setdiff1d(konkurrenten_StartHA, migros_StartHA, assume_unique=True)
        # need to re-order the columns, because pandas.concat sorts them
        all_stores_pd = pd.concat([migros_merged_pd, konkurrenten_merged_pd])[migros_merged_pd.columns.tolist()]
        all_stores_pd = all_stores_pd.loc[~all_stores_pd.StartHARasterID.isin(StartHA_to_exclude)]
        logger.info("Done.")

        # Pruning
        if "pruning" in config:
            if "number_nearest_stores" in config["pruning"]:
                n_nearest = int(config["pruning"]["number_nearest_stores"])
                logger.info("Pruning mode selected. Number of nearest stores set to %d", n_nearest)
                logger.info("Starting pruning ... ")
                all_stores_grouped = all_stores_pd.groupby('StartHARasterID', as_index=False, sort=False,
                                                          group_keys=False)
                func = partial(filter, n_nearest)
                all_stores_pd = apply_parallel(all_stores_grouped, func, ncpus=100, chunk_size=3)
                logger.info("Done. ")


        # enrich the drive times of the relevant hectars with Haushalt information
        logger.info("Enriching with Haushalt information ...")

        # haushalt_pd has index HARasterID
        # all_stores_pd has index HARasterID
        enriched_pd = all_stores_pd.merge(haushalt_pd[['Tot_Haushaltausgaben', 'AnzahlHH']],
                                          left_on='StartHARasterID', right_index = True,
                                          how='left')

        if LOGGING_LEVEL == logging.DEBUG:
            logger.debug("Number of unique StartHektarIDs without Haushaltausgaben info: %d out of %d",
                     len(np.unique(enriched_pd.loc[np.isnan(enriched_pd.Tot_Haushaltausgaben), "StartHARasterID"])),
                     len(np.unique(enriched_pd.StartHARasterID)))

        logger.info("Done.")

        # enrich the drivetimes of the relevant hectars with RegionsTyp Information
        logger.info("Enriching with Regionstyp information ...")
        # regionstypen_pd has index HARasterID
        # enriched_pd has index HARasterID
        enriched_pd = enriched_pd.join(regionstypen_pd[['RegionTyp', 'DTB']])

        enriched_pd = enriched_pd.merge(regionstypen_pd[['PLZ']], suffixes=('_l', '_r'), left_on = 'StartHARasterID',
                                        right_index = True, how='left')

        if LOGGING_LEVEL == logging.DEBUG:
            logger.debug("Number of unique ZielHARasterIDs without Regionstyp info: %d out of %d",
                        len(np.unique(enriched_pd.loc[np.isnan(enriched_pd.RegionTyp)].RegionTyp)),
                        len(np.unique(enriched_pd.RegionTyp)))

        logger.info("Done.")

        # enrich the drivetimes of the relevant hectars with Arbeitnehmer Information
        # arbeitnehmer_pd has index HARasterID
        logger.info("Enriching with Arbeitnehmer information ...")
        enriched_pd = enriched_pd.merge(arbeitnehmer_pd, how="left", left_on='StartHARasterID', right_index=True)

        if LOGGING_LEVEL == logging.DEBUG:
            logger.debug("Number of unique StartHARasterIDs without Arbeitnehmer info: %d out of %d",
                        len(np.unique(enriched_pd.loc[np.isnan(enriched_pd.ANTOT)].StartHARasterID)),
                        len(np.unique(enriched_pd.StartHARasterID)))

        logger.info("Done.")

        logger.info("Resetting index ... ")
        # give the index back its original name
        enriched_pd.index.name = 'HARasterID'
        enriched_pd.reset_index(inplace=True)
        logger.info("Done.")


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
