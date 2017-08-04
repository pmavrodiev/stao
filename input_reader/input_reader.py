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
    use_cache = config.getboolean('global', 'cache_enabled') # read data from cache
    cache_input_data = config.getboolean('global', 'cache_input_data') # store data to cache

    cache_dir = config['cache_config']['cache_dir'] # cache directory
    single_store = config['global']['single_store'] # single-store option


    if use_cache:
        # --- READ FROM CACHE -----------------------------------------------------------------------------------
        logger.info("Reading input files from cache into Pandas data frames")
        try:
            migros_stores_pd = pd.read_pickle(os.path.join(cache_dir,
                                                    config['cache_config']['migros_stores_cache']))
            konkurrenten_stores_pd = pd.read_pickle(os.path.join(cache_dir,
                                                    config['cache_config']['konkurrenten_stores_cache']))

            drivetimes_pd = pd.read_pickle(os.path.join(cache_dir,
                                                        config['cache_config']['drivetimes_cached']))
            haushalt_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['haushalt_cached']))
            stations_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['stations_cached']))
            regionstypen_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['regionstypen_cached']))

            arbeitnehmer_pd = pd.read_pickle(os.path.join(cache_dir,
                                                          config['cache_config']['arbeitnehmer_cached']))

        except IOError as e:
            logger.error("Cannot find cache for the input data. Generate the cache by running with cache_enabled=False"
                         " in the settings.")
            print(e)
            sys.exit(1)
    else:
        # --- READ FROM CSV --------------------------------------------------------------------------------------
        logger.info("Reading input files into Pandas data frames")
        
        migros_stores = config['inputdata']['migros_stores']
        konkurrenten_stores = config['inputdata']['konkurrenten_stores']
        drivetimes = config['inputdata']['drivetimes']
        haushalt = config['inputdata']['haushalt']
        stations = config['inputdata']['stations']
        regionstypen = config['inputdata']['regionstypen']
        arbeitnehmer = config['inputdata']['arbeitnehmer']

        # --- MIGROS STORES ----------------------------------------------------------------------------------------
        logger.info("Reading Migros stores")
        migros_stores_pd = pd.read_csv(migros_stores, sep=';', header=0, index_col=False, encoding='latin-1')
        # --- Remove stores with VFL 0 or undefined
        kwargs = {"errors": 'coerce'}
        migros_stores_pd.VFL = migros_stores_pd.VFL.apply(pd.to_numeric, **kwargs)
        migros_stores_pd = migros_stores_pd.loc[migros_stores_pd.VFL > 0]
        # --- Remove all stores without Geo coordinates
        migros_stores_pd.lon = migros_stores_pd.lon.apply(pd.to_numeric, **kwargs)
        migros_stores_pd.lat = migros_stores_pd.lat.apply(pd.to_numeric, **kwargs)
        migros_stores_pd = migros_stores_pd.loc[(migros_stores_pd.lon > 0) & (migros_stores_pd.lat > 0)]
        # --- Now make all quantitative columns numeric
        kwargs = {"errors": 'ignore'}
        migros_stores_pd = migros_stores_pd.apply(pd.to_numeric, axis=0, **kwargs)

        # --- SINGLE STORE MODE? -----------------------------------------------------------------------------------
        if len(single_store) > 0:
            logger.info('Single store mode chosen - %s', single_store)
            migros_stores_pd = migros_stores_pd.loc[migros_stores_pd.StoreName == single_store]


        # --- COMPETITORS STORES -------------------------------------4----------------------------------------------
        logger.info("Reading Competitors' stores")
        konkurrenten_stores_pd = pd.read_csv(konkurrenten_stores, sep=';', header=0, index_col=False, encoding='latin-1')
        kwargs = {"errors": 'coerce'}
        konkurrenten_stores_pd.VFL = konkurrenten_stores_pd.VFL.apply(pd.to_numeric, **kwargs)
        konkurrenten_stores_pd = konkurrenten_stores_pd.loc[konkurrenten_stores_pd.VFL > 0]

        konkurrenten_stores_pd.lon = konkurrenten_stores_pd.lon.apply(pd.to_numeric, **kwargs)
        konkurrenten_stores_pd.lat = konkurrenten_stores_pd.lat.apply(pd.to_numeric, **kwargs)
        konkurrenten_stores_pd = konkurrenten_stores_pd.loc[
            (konkurrenten_stores_pd.lon > 0) & (konkurrenten_stores_pd.lat > 0)]

        # make all quantitative columns numeric!!!!
        kwargs = {"errors": 'ignore'}
        konkurrenten_stores_pd = konkurrenten_stores_pd.apply(pd.to_numeric, axis=0, **kwargs)

        # --- DRIVE TIMES ------------------------------------------------------------------------------------------
        kwargs = {"errors": 'coerce'}
        logger.info("Reading Drivetimes, takes a while ...")
        drivetimes_pd = pd.read_csv(drivetimes, sep=',', header=0, index_col=False, encoding='latin-1')
        drivetimes_pd = drivetimes_pd.apply(pd.to_numeric, axis=0, **kwargs)
        drivetimes_pd = drivetimes_pd.set_index(["StartHARasterID", "ZielHARasterID"])
        # remove duplicates from the drive times
        logger.info('Removing duplicates from drivetimes ...')
        before = len(drivetimes_pd)
        drivetimes_pd = drivetimes_pd[~drivetimes_pd.index.duplicated(keep='first')].reset_index()
        after = len(drivetimes_pd)
        logger.info('Done! - %d duplicates removed', before-after)

        # --- generate a unified FZ column
        drivetimes_pd['FZ'] = drivetimes_pd['AutoDistanzMinuten']
        drivetimes_pd = drivetimes_pd.loc[drivetimes_pd.FZ <= 30]

        # --- HAUSHALT ---------------------------------------------------------------------------------------------
        logger.info("Reading information on Haushaltausgaben")
        haushalt_pd = pd.read_csv(haushalt, sep=';', header=0, index_col=0)
        # Durchschnitt über Grossregion, Kanton, Sprachregion, Alter und Einkommen (* 12 für Jahr)
        # haushalt_pd['Tot_Haushaltausgaben'] = haushalt_pd['AnzahlHH'] * haushalt_pd['HHA_AVG_ALL'] * 12
        haushalt_pd['Tot_Haushaltausgaben'] = haushalt_pd['AnzahlHH'] * haushalt_pd['HHA_AVG_EA'] * 12

        # --- SBB --------------------------------------------------------------------------------------------------
        logger.info("Reading oeV info")
        stations_pd = pd.read_csv(stations, sep=';', header=0, index_col=False, encoding='latin-1')

        # --- REGIONSTYPEN -----------------------------------------------------------------------------------------
        # The Regionstypen are coded according to the BfS convention as follows:
        # ==  11	Städtische Gemeinde einer grossen Agglomeration
        # ==  12	Städtische Gemeinde einer mittelgrossen Agglomeration
        # ==  13	Städtische Gemeinde einer kleinen oder ausserhalb einer Agglomeration
        # ==  21	Periurbane Gemeinde hoher Dichte
        # ==  22	Periurbane Gemeinde mittlerer Dichte
        # ==  23	Periurbane Gemeinde geringer Dichte
        # ==  31	Ländliche Zentrumsgemeinde
        # ==  32	Ländliche zentral gelegene Gemeinde
        # ==  33	Ländliche periphere Gemeinde

        logger.info("Reading Regionstyp Info")
        regionstypen_pd = pd.read_csv(regionstypen, sep=';', header=0, index_col=False, encoding='latin-1')

        # --- ARBEITNEHMER ----------------------------------------------------------------------------------------
        logger.info("Reading Arbeitnehmer Info")
        arbeitnehmer_pd = pd.read_csv(arbeitnehmer, sep=';', header=0, index_col=False, encoding='latin-1')

        # --- SAVE TO PICKLE (CACHE) --------------------------------------------------------------------------------
        if cache_input_data:
            logger.info("Caching input data")
            migros_stores_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['migros_stores_cache']))
            konkurrenten_stores_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['konkurrenten_stores_cache']))
            drivetimes_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['drivetimes_cached']))
            haushalt_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['haushalt_cached']))
            stations_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stations_cached']))
            regionstypen_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['regionstypen_cached']))
            arbeitnehmer_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['arbeitnehmer_cached']))
            logger.info("Done caching input data")

        logger.info("Done reading input data")

    return (migros_stores_pd, konkurrenten_stores_pd, drivetimes_pd, haushalt_pd, stations_pd, regionstypen_pd,
            arbeitnehmer_pd)