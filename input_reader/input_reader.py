# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
import configparser
import numpy as np


def read_migros_stores(filename, logger=None):
    migros_stores_pd = pd.read_csv(filename, sep=';', header=0, index_col=False, encoding='latin-1')
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
    # finally set the index
    migros_stores_pd.set_index('HARasterID', inplace=True)

    return migros_stores_pd


def read_competitor_stores(filename):

    konkurrenten_stores_pd = pd.read_csv(filename, sep=';', header=0, index_col=False, encoding='latin-1')
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
    # finally set the index
    konkurrenten_stores_pd.set_index('HARasterID', inplace=True)
    return konkurrenten_stores_pd


def read_drivetimes(filename, logger, fz_schwelle = 30):
    kwargs = {"errors": 'coerce'}
    drivetimes_pd = pd.read_csv(filename, sep=',', header=0, index_col=False, encoding='latin-1')
    drivetimes_pd = drivetimes_pd.apply(pd.to_numeric, axis=0, **kwargs)
    drivetimes_pd = drivetimes_pd.set_index(["StartHARasterID", "ZielHARasterID"])
    # remove duplicates from the drive times
    logger.info('Removing duplicates from drivetimes ...')
    before = len(drivetimes_pd)
    drivetimes_pd = drivetimes_pd[~drivetimes_pd.index.duplicated(keep='first')].reset_index()
    after = len(drivetimes_pd)
    logger.info('Done! - %d duplicates removed', before - after)

    # --- generate a unified FZ column
    # drivetimes_pd['FZ'] = drivetimes_pd['AutoDistanzMinuten']
    drivetimes_pd['FZ'] = drivetimes_pd['AutoDistanzKilometer']
    drivetimes_pd = drivetimes_pd.loc[drivetimes_pd.FZ <= fz_schwelle]
    # finally set the index
    drivetimes_pd.set_index('ZielHARasterID', inplace=True)
    return drivetimes_pd


def read_haushalt_info(filename):
    haushalt_pd = pd.read_csv(filename, sep=';', header=0, index_col=False)
    # Durchschnitt über Grossregion, Kanton, Sprachregion, Alter und Einkommen (* 12 für Jahr)
    # haushalt_pd['Tot_Haushaltausgaben'] = haushalt_pd['AnzahlHH'] * haushalt_pd['HHA_AVG_ALL'] * 12
    haushalt_pd['Tot_Haushaltausgaben'] = haushalt_pd['AnzahlHH'] * haushalt_pd['HHA_AVG_EA'] * 12
    haushalt_pd.set_index('HARasterID', inplace=True)
    return haushalt_pd

def read_sbb_info(filename):
    stations_pd = pd.read_csv(filename, sep=';', header=0, index_col=False, encoding='latin-1')
    return stations_pd

def read_regions_info(filename):
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
    regionstypen_pd = pd.read_csv(filename, sep=';', header=0, index_col=False, encoding='latin-1')
    regionstypen_pd.set_index('HARasterID', inplace=True)
    return regionstypen_pd

def read_arbeit_info(filename):
    arbeitnehmer_pd = pd.read_csv(filename, sep=';', header=0, index_col=False, encoding='latin-1')
    arbeitnehmer_pd.set_index('HARasterID', inplace=True)
    return arbeitnehmer_pd


def get_input(settingsFile, logger):
    # read the input data
    config = configparser.ConfigParser()
    config.read_file(open(settingsFile, mode="r", encoding="utf-8"))
    # config.read(settingsFile)
    logger.info("Read settings from settings.cfg")

    cache_dir = config['cache_config']['cache_dir'] # cache directory

    migros_stores_tuple = eval(config['inputdata']['migros_stores'], {}, {})
    konkurrenten_stores_tuple = eval(config['inputdata']['konkurrenten_stores'], {}, {})
    drivetimes_tuple = eval(config['inputdata']['drivetimes'], {}, {})
    haushalt_tuple = eval(config['inputdata']['haushalt'], {}, {})
    stations_tuple = eval(config['inputdata']['stations'], {}, {})
    regionstypen_tuple = eval(config['inputdata']['regionstypen'], {}, {})
    arbeitnehmer_tuple = eval(config['inputdata']['arbeitnehmer'], {}, {})

    # ----------------------------------------------------------------------------------------------------------
    # --- MIGROS STORES ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if migros_stores_tuple[1]:
        logger.info("Reading Migros stores from cache.")
        # read from cache
        try:
            migros_stores_pd = pd.read_pickle(os.path.join(cache_dir,
                                                           config['cache_config']['migros_stores_cache']))
        except IOError as e:
            logger.info("Cache not found. Regenerating.")
            migros_stores_pd = read_migros_stores(migros_stores_tuple[0], logger)
            migros_stores_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['migros_stores_cache']))

    else:
        logger.info("Reading Migros stores from file. Will not be cached")
        migros_stores_pd = read_migros_stores(migros_stores_tuple[0], logger)
    logger.info("Done")

    # ----------------------------------------------------------------------------------------------------------
    # --- COMPETITORS' STORES ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if konkurrenten_stores_tuple[1]:
        logger.info("Reading Competitor stores from cache.")
        # read from cache
        try:
            konkurrenten_stores_pd = pd.read_pickle(os.path.join(cache_dir,
                                                           config['cache_config']['konkurrenten_stores_cache']))
        except IOError as e:
            logger.info("Cache not found. Regenerating.")
            konkurrenten_stores_pd = read_competitor_stores(konkurrenten_stores_tuple[0])
            konkurrenten_stores_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['konkurrenten_stores_cache']))

    else:
        logger.info("Reading Competitor stores from file. Will not be cached")
        konkurrenten_stores_pd = read_competitor_stores(konkurrenten_stores_tuple[0])
    logger.info("Done")

    # ----------------------------------------------------------------------------------------------------------
    # --- DRIVETIMES  ------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if drivetimes_tuple[1]:
        logger.info("Reading Drivetimes from cache.")
        # read from cache
        try:
            drivetimes_pd = pd.read_pickle(os.path.join(cache_dir,
                                                        config['cache_config']['drivetimes_cached']))
        except IOError as e:
            logger.info("Cache not found. Regenerating. Takes a while ...")
            drivetimes_pd = read_drivetimes(drivetimes_tuple[0], logger)
            drivetimes_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['drivetimes_cached']))
    else:
        logger.info("Reading Drivetimes stores from file. Takes a while. Will not be cached")
        drivetimes_pd = read_drivetimes(drivetimes_tuple[0], logger)
    logger.info("Done")


    # ----------------------------------------------------------------------------------------------------------
    # --- HAUSHALT  ------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if haushalt_tuple[1]:
        logger.info("Reading Haushalt info from cache.")
        # read from cache
        try:
            haushalt_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['haushalt_cached']))
        except IOError as e:
            logger.info("Cache not found. Regenerating.")
            haushalt_pd = read_haushalt_info(haushalt_tuple[0])
            haushalt_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['haushalt_cached']))

    else:
        logger.info("Reading Haushalt info from file. Will not be cached")
        haushalt_pd = read_haushalt_info(haushalt_tuple[0])
    logger.info("Done")

    # ----------------------------------------------------------------------------------------------------------
    # --- SBB  -------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if stations_tuple[1]:
        logger.info("Reading SBB info from cache.")
        # read from cache
        try:
            stations_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['stations_cached']))
        except IOError as e:
            logger.info("Cache not found. Regenerating.")
            stations_pd = read_sbb_info(stations_tuple[0])
            stations_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['stations_cached']))

    else:
        logger.info("Reading SBB info from file. Will not be cached")
        stations_pd = read_sbb_info(stations_tuple[0])
    logger.info("Done")

    # ----------------------------------------------------------------------------------------------------------
    # --- REGIONSTYPEN  ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if regionstypen_tuple[1]:
        logger.info("Reading Regionstypen info from cache.")
        # read from cache
        try:
            regionstypen_pd = pd.read_pickle(os.path.join(cache_dir,
                                                      config['cache_config']['regionstypen_cached']))
        except IOError as e:
            logger.info("Cache not found. Regenerating.")
            regionstypen_pd = read_regions_info(regionstypen_tuple[0])
            regionstypen_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['regionstypen_cached']))

    else:
        logger.info("Reading Regionstypen info from file. Will not be cached")
        regionstypen_pd = read_regions_info(regionstypen_tuple[0])
    logger.info("Done")

    # ----------------------------------------------------------------------------------------------------------
    # --- ARBEITNEHMER  ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if arbeitnehmer_tuple[1]:
        logger.info("Reading Arbeitnehmer info from cache.")
        # read from cache
        try:
            arbeitnehmer_pd = pd.read_pickle(os.path.join(cache_dir,
                                                          config['cache_config']['arbeitnehmer_cached']))
        except IOError as e:
            logger.info("Cache not found. Regenerating.")
            arbeitnehmer_pd = read_arbeit_info(arbeitnehmer_tuple[0])
            arbeitnehmer_pd.to_pickle(os.path.join(cache_dir, config['cache_config']['arbeitnehmer_cached']))

    else:
        logger.info("Reading Arbeitnehmer info from file. Will not be cached")
        arbeitnehmer_pd = read_arbeit_info(arbeitnehmer_tuple[0])
    logger.info("Done")



    return (migros_stores_pd, konkurrenten_stores_pd, drivetimes_pd, haushalt_pd, stations_pd, regionstypen_pd,
            arbeitnehmer_pd)