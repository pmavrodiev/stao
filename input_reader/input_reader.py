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
            single_store_migros_pd = pd.read_pickle(os.path.join(cache_dir,
                                                            config['cache_config']['stores_migros_only_cached']))


            drivetimes_pd = pd.read_pickle(os.path.join(cache_dir,
                                                        config['cache_config']['drivetimes_cached']))
            haushalt_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['haushalt_cached']))
            stations_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['stations_cached']))
        except IOError as e:
            logger.error("Cannot find cache for the input data. Generate the cache by running with cache_enabled=False"
                         " in the settings.")
            print(e)
            sys.exit(1)
    else:
        # --- READ FROM CSV --------------------------------------------------------------------------------------
        logger.info("Reading input files into Pandas data frames")
        
        migros_stores = config['inputdata']['migros_stores']
        denner_stores = config['inputdata']['denner_stores']
        konkurrenten_stores = config['inputdata']['konkurrenten_stores']
        drivetimes = config['inputdata']['drivetimes']
        haushalt = config['inputdata']['haushalt']
        stations = config['inputdata']['stations']

        # --- MIGROS STORES ----------------------------------------------------------------------------------------
        migros_stores_pd = pd.read_csv(migros_stores, sep=';', header=0, index_col=0, encoding='latin-1')
        # Fix the sales area for VOI stores
        migros_stores_pd.loc[migros_stores_pd['VertriebstypInternID'] == 260, 'VFL_TOTAL'] = 220
        # Fix the sales area for DMP stores
        migros_stores_pd.loc[migros_stores_pd['VertriebstypInternID'] == 220, 'VFL_TOTAL'] = 220
        # Fix the sales area for Migrolino stores
        migros_stores_pd.loc[migros_stores_pd['VertriebstypInternID'] == 240, 'VFL_TOTAL'] = 150
        # Now remove stores with VFL 0 or undefined
        # first make the respective column numeric
        valid_areas = pd.to_numeric(migros_stores_pd['VFL_TOTAL'], errors='coerce')
        # now filter
        migros_stores_pd = migros_stores_pd[ valid_areas > 0 ].reset_index()

        # --- DENNER STORES ----------------------------------------------------------------------------------------
        denner_stores_pd = pd.read_csv(denner_stores, sep=';', header=0, index_col=0, encoding='latin-1')
        denner_stores_pd = denner_stores_pd.rename(columns = {'area': 'VFL_TOTAL'})

        # --- COMPETITORS STORES -----------------------------------------------------------------------------------
        konkurrenten_stores_pd = pd.read_csv(konkurrenten_stores, sep=';', header=0, index_col=0, encoding='latin-1')
        # Fix the competitors' sales area according to the heuristics from MP
        # Volg
        konkurrenten_stores_pd.loc[
            konkurrenten_stores_pd['NAME_KONKURRENT'].str.lower() == 'volg', 'VFL_GESCHAETZT'] = 172
        # Spar express
        konkurrenten_stores_pd.loc[
            (konkurrenten_stores_pd['NAME_KONKURRENT'].str.lower() == 'spar') &
            (konkurrenten_stores_pd['FILIAL_BEZ'].str.contains('express', case=False)), 'VFL_GESCHAETZT'] = 150
        # Spar supermarkt
        konkurrenten_stores_pd.loc[
            (konkurrenten_stores_pd['NAME_KONKURRENT'].str.lower() == 'spar') &
            (konkurrenten_stores_pd['FILIAL_BEZ'].str.contains('supermarkt', case=False)), 'VFL_GESCHAETZT'] = 450
        # Manor
        konkurrenten_stores_pd.loc[konkurrenten_stores_pd['NAME_KONKURRENT'].str.lower() == 'manor', 'VFL_GESCHAETZT'] = 1000

        # Take only the Coop Pronto and Coop Supermakrt stores
        konkurrenten_stores_pd = konkurrenten_stores_pd.loc[~((konkurrenten_stores_pd['NAME_KONKURRENT'].str.lower() == 'coop') &
                                     (konkurrenten_stores_pd['FORMAT_BEZ'].str.lower() != 'supermarkt') &
                                     (konkurrenten_stores_pd['FORMAT_BEZ'].str.lower() != 'coop pronto') )]

        # PAM - TODO missing form data
        # Otto's Warenposten - TODO missing from data

        # finally leave only competitors in the Food business
        konkurrenten_stores_pd = konkurrenten_stores_pd.loc[konkurrenten_stores_pd['NAME_KONKURRENT'].str.lower().isin(
                                                            ['aldi', 'coop', 'landi', 'lidl', 'manor', 'spar', 'volg'])]



        # --- DRIVE TIMES ------------------------------------------------------------------------------------------
        drivetimes_pd = pd.read_csv(drivetimes, sep=';', header=0, index_ycol=[0, 1])

        logger.info("Removing duplicate drive times from drivetimes_pd")
        before = len(drivetimes_pd)
        drivetimes_pd = drivetimes_pd[~drivetimes_pd.index.duplicated(keep='first')]
        # reindex, easier to handle than a multi-index
        drivetimes_pd = drivetimes_pd.reset_index().set_index(keys='filiale_id')
        logger.info("Removed %d duplicates entries", before-len(drivetimes_pd))

        # --- HAUSHALT ---------------------------------------------------------------------------------------------
        haushalt_pd = pd.read_csv(haushalt, sep=';', header=0, index_col=0)
        # Durchschnitt über Grossregion, Kanton, Sprachregion, Alter und Einkommen (* 12 für Jahr)
        # haushalt_pd['Tot_Haushaltausgaben'] = haushalt_pd['AnzahlHH'] * haushalt_pd['HHA_AVG_ALL'] * 12
        haushalt_pd['Tot_Haushaltausgaben'] = haushalt_pd['AnzahlHH'] * haushalt_pd['HHA_AVG_EA'] * 12

        # --- SBB --------------------------------------------------------------------------------------------------
        stations_pd = pd.read_csv(stations, sep=';', header=0, index_col=False, encoding='latin-1')

        # --- FILTER MIGROS ----------------------------------------------------------------------------------------
        # Get all Migros stores used by MP Technology OR the single store if in single store mode
        if len(single_store) > 0:
            logger.info('Single store mode chosen - %s', single_store)
            single_store_migros_pd = migros_stores_pd[migros_stores_pd['KostenstelleName'] == single_store]
        
        # --- SAVE TO PICKLE (CACHE) --------------------------------------------------------------------------------
        if cache_input_data:
            logger.info("Caching input data")
            migros_stores_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['migros_stores_cache']))
            single_store_migros_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stores_migros_only_cached']))
            drivetimes_pd.to_pickle(os.path.join(cache_dir,
                                                 config['cache_config']['drivetimes_cached']))
            haushalt_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['haushalt_cached']))
            stations_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stations_cached']))
            logger.info("Done caching input data")

        logger.info("Done reading input data")

    return (migros_stores_pd, single_store_migros_pd, drivetimes_pd, haushalt_pd, stations_pd)