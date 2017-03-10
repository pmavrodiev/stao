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

    stores_pd = None
    drivetimes_pd = None
    drivetimes_migros_pd = None
    haushalt_pd = None

    use_cache = config.getboolean('global', 'cache_enabled')
    cache_dir = config['cache_config']['cache_dir']
    if use_cache:
        logger.info("Reading input files from cache into Pandas data frames")
        try:
            stores_pd = pd.read_pickle(os.path.join(cache_dir,
                                                    config['cache_config']['stores_cm_cached']))

            stores_migros_pd = pd.read_pickle(os.path.join(cache_dir,
                                                            config['cache_config']['stores_cm_migros_only_cached']))
            drivetimes_pd = pd.read_pickle(os.path.join(cache_dir,
                                                        config['cache_config']['drivetimes_cached']))
            drivetimes_migros_pd = pd.read_pickle(os.path.join(cache_dir,
                                                               config['cache_config']['drivetimes_migros_cached']))
            haushalt_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['haushalt_cached']))
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

        stores_pd = pd.read_csv(stores, sep=';', header=0, index_col=1, encoding='latin-1')
        if config.getboolean('global', 'remove_duplicates'):
            before=len(stores_pd)
            logger.info("Removing duplicate store ids from stores_cm")
            stores_pd = stores_pd[~stores_pd.index.duplicated(keep=False)]
            logger.info("Removed %d duplicates", before-len(stores_pd))

        # extract the stores hektar locations from the store names
        stores_pd['own_hektar_id'] = stores_pd.index.str[7:11].str.cat(stores_pd.index.str[13:17]).astype(int)

        drivetimes_pd = pd.read_csv(drivetimes, sep=',', header=None, names=['filiale_id', 'fahrzeit', 'hektar_id'],
                                    index_col=0, nrows=110299436)
        """
        drivetimes_pd.loc['SM_DEN_76515_25781'][ (drivetimes_pd['fahrzeit']==22) &
                                                 (drivetimes_pd['hektar_id']==75792337) ]
        filiale_id           fahrzeit  hektar_id
        SM_DEN_76515_25781        22   75792337
        SM_DEN_76515_25781        22   75792337
        """

        # make sure that for every store, there is an entry in
        # drivetimes from the hektar in which this store resides to
        # the store itself
        """
        own_fahrzeit= pd.merge(stores_pd[['own_hektar_id']].reset_index(),
                               drivetimes_pd.reset_index(),
                               left_on=['ID', 'own_hektar_id'],
                               right_on=['filiale_id', 'hektar_id'],
                               how='left').drop_duplicates()
        """

        if config.getboolean('global', 'remove_duplicates'):
            before = len(drivetimes_pd)
            logger.info("Removing duplicate drive times from drivetimes_pd")
            drivetimes_pd = drivetimes_pd.drop_duplicates(keep='first')
            logger.info("Removed %d duplicates", before-len(drivetimes_pd))

        # only get the first 4 columns - X_COORD, Y_COORD, RELI. H14PTOT
        haushalt_pd = pd.read_csv(haushalt, sep=',', header=0, index_col=2, usecols=[0, 1, 2, 3])

        # Get all Migros stores used by MP Technology
        # stores_migros_pd = stores_pd.loc[['SM_MIG_59483_15585', 'SM_MIG_68921_24352']]
        stores_migros_pd = stores_pd[stores_pd['FORMAT'].isin(['M', 'MM', 'MMM', 'FM'])]
        # sanity check: the number of stores must equal 591 - the number of stores in "STAO Vergleich V1 and V2.xlsx".
        # it does.

        if config['global']['single_store']:
            logger.info("Single store mode")
            stores_migros_pd = stores_migros_pd.loc[config['global']['single_store']]
            # get the drivetimes for hektars from which a Migros store is reachable
            drivetimes_migros_pd = drivetimes_pd.loc[stores_migros_pd.name]
        else:
            drivetimes_migros_pd = drivetimes_pd.loc[stores_migros_pd.index]

        logger.info("Caching input data")
        stores_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stores_cm_cached']))
        stores_migros_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stores_cm_migros_only_cached']))
        drivetimes_pd.to_pickle(os.path.join(cache_dir,
                                             config['cache_config']['drivetimes_cached']))
        drivetimes_migros_pd.to_pickle(os.path.join(cache_dir,
                                                    config['cache_config']['drivetimes_migros_cached']))
        haushalt_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['haushalt_cached']))
        logger.info("Done reading input data")

    return (stores_pd, stores_migros_pd, drivetimes_pd, drivetimes_migros_pd, haushalt_pd)

