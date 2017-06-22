# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import configparser
import numpy as np

def get_input(settingsFile, logger):
    # read the input data
    config = configparser.ConfigParser()
    config.read(settingsFile)
    logger.info("Read settings from settings.cfg")

    # do we read the data from cache? cache loads faster
    use_cache = config.getboolean('global', 'cache_enabled')
    cache_input_data = config.getboolean('global', 'cache_input_data')

    cache_dir = config['cache_config']['cache_dir']
    single_store = config['global']['single_store']

    refenz_resultate = config['inputdata']['referenz_ergebnisse']

    # read the reference results from file always, because it's small
    referenz_pd = pd.read_csv(refenz_resultate, sep=';', header=0, index_col=0, encoding='latin-1')
    # convert all 0s to NaNs, so that the stores can be later ignored
    referenz_pd[referenz_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] == 0] = np.nan

    if use_cache:
        logger.info("Reading input files from cache into Pandas data frames")
        try:
            stores_pd = pd.read_pickle(os.path.join(cache_dir,
                                                    config['cache_config']['stores_cm_cached']))

            stores_migros_pd = pd.read_pickle(os.path.join(cache_dir,
                                                            config['cache_config']['stores_cm_migros_only_cached']))
            drivetimes_pd = pd.read_pickle(os.path.join(cache_dir,
                                                        config['cache_config']['drivetimes_cached']))
            haushalt_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['haushalt_cached']))
            stations_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['stations_cached']))
        except IOError as e:
            logger.error("Cannot find cache for the input data. Generate the cache by running with cache disabled"
                         " in the settings.")
            print(e)
            sys.exit(1)
    else:
        logger.info("Reading input files into Pandas data frames")

        stores = config['inputdata']['stores_cm']
        drivetimes = config['inputdata']['drivetimes']
        haushalt = config['inputdata']['haushalt']
        stations = config['inputdata']['stations']

        stores_pd = pd.read_csv(stores, sep=';', header=0, index_col=0, encoding='latin-1')
        # stores_pd['RELEVANZ'] = 1
        # extract the store type from the its name, e.g. COOP, MIG, DENNER, etc.
        # needed to implement step 4 of the model
        stores_pd['type'] = stores_pd['ID'].str[3:6]
        # M, MM, MMM, DMP, SPEZ, VOI zusammen
        # stores_pd.loc[stores_pd['FORMAT'].isin(['M', 'MM', 'MMM', 'DMP', 'SPEZ', 'VOI']), 'type'] = 'Migros'
        # Migrolino, Alnatura und FM separat
        # stores_pd.loc[stores_pd['FORMAT'] == 'ALNA', 'type'] = 'ALNA'
        # stores_pd.loc[stores_pd['FORMAT'] == 'migrolino', 'type'] = 'migrolino'
        # stores_pd.loc[stores_pd['FORMAT'] == 'FM', 'type'] = 'FM'

        # choose the right Verkaufsflaeche for Frische und Food
        # stores_pd['vfl'] = np.where(pd.isnull(stores_pd['VERKAUFSFLAECHE_SABRINA']), stores_pd['VERKAUFSFLAECHE'],
        #                             stores_pd['VERKAUFSFLAECHE_SABRINA'])

        # stores_pd['vfl'] = np.where(np.isnull(stores_pd['VERKAUFSFLAECHE_SABRINA']), stores_pd['VERKAUFSFLAECHE'] ,
        #                                      stores_pd['VERKAUFSFLAECHE_SABRINA'])

        stores_pd['vfl'] = stores_pd['VERKAUFSFLAECHE_TOTAL']

        drivetimes_pd = pd.read_csv(drivetimes, sep=',', header=None, names=['filiale_id', 'fahrzeit', 'hektar_id'],
                                    index_col=[0, 1, 2], nrows=110299436)

        logger.info("Removing duplicate drive times from drivetimes_pd")
        before = len(drivetimes_pd)
        drivetimes_pd = drivetimes_pd[~drivetimes_pd.index.duplicated(keep='first')]
        # reindex, easier to handle than a multi-index
        drivetimes_pd = drivetimes_pd.reset_index().set_index(keys='filiale_id')
        logger.info("Removed %d duplicates entries", before-len(drivetimes_pd))

        haushalt_pd = pd.read_csv(haushalt, sep=',', header=0, index_col=2)
        # haushalt_pd['Tot_Haushaltausgaben'] = (haushalt_pd['H14P01'] + 2*haushalt_pd['H14P02'] + 3*haushalt_pd['H14P03'] \
        #                                       + 4*haushalt_pd['H14P04'] + 5*haushalt_pd['H14P05'] + 6*haushalt_pd['H14P06'])*(7800 / 2.25)
        haushalt_pd['Tot_Haushaltausgaben'] = haushalt_pd['H14PTOT'] * 7800

        stations_pd = pd.read_csv(stations, sep=';', header=0, index_col=False, encoding='latin-1')

        # Get all Migros stores used by MP Technology OR the single store if in single store mode

        if len(single_store) > 0:
            logger.info('Single store mode chosen - %s', single_store)
            stores_migros_pd = stores_pd[stores_pd['ID'] == single_store]
        else:
            stores_migros_pd = stores_pd[stores_pd['FORMAT'].isin(['M', 'MM', 'MMM', 'FM'])]
            # sanity check: the number of stores must equal 591 - the number of stores
            # in "STAO Vergleich V1 and V2.xlsx". It does.

        if cache_input_data:
            logger.info("Caching input data")
            stores_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stores_cm_cached']))
            stores_migros_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stores_cm_migros_only_cached']))
            drivetimes_pd.to_pickle(os.path.join(cache_dir,
                                                 config['cache_config']['drivetimes_cached']))
            haushalt_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['haushalt_cached']))
            stations_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stations_cached']))
            logger.info("Done caching input data")

        logger.info("Done reading input data")

    return (stores_pd, stores_migros_pd, drivetimes_pd, haushalt_pd, referenz_pd, stations_pd)
