# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
import configparser


from simple_logging.custom_logging import setup_custom_logger
from input_reader.input_reader import get_input

# logger einrichten
LOGGING_LEVEL = logging.INFO
logger = setup_custom_logger('GM_LOGGER', LOGGING_LEVEL, flog="logs/gm.log")

settingsFile = "settings.cfg"

if __name__ == "__main__":

    (stores_pd, stores_migros_pd, drivetimes_pd, drivetimes_migros_pd,
     haushalt_pd) = get_input(settingsFile, logger)

    # get all relevant hektars, i.e. those from which a Migros store is reachable
    relevant_hektars = set(drivetimes_migros_pd['hektar_id'])

    # get all drive times for the relevant hektars
    logger.info("Obtaining all drive times only for hektars from which a Migros store is reachable")
    drivetimes_rel_hektars_pd = drivetimes_pd[drivetimes_pd['hektar_id'].isin(relevant_hektars)]

    # enrich the drive times of the relevant hektars with store information
    # and do an inner join to detect missing stores in stores_pd
    logger.info("Enriching with store information")
    before = len(set(drivetimes_rel_hektars_pd.index))
    drivetimes_rel_hektars_stores_pd = drivetimes_rel_hektars_pd.join(
        stores_pd[['FORMAT', 'VERKAUFSFLAECHE', 'VERKAUFSFLAECHE_TOTAL', 'RELEVANZ']], how='inner')
    logger.info("%d stores appear in drivetimes, but have no associated information in stores_sm.csv",
                before-len(set(drivetimes_rel_hektars_stores_pd.index)))

    # enrich the drive times of the relevant hektars with Haushalt information
    logger.info("Enriching with Haushalt information")
    enriched_pd = drivetimes_rel_hektars_stores_pd.join(haushalt_pd['H14PTOT'], on='hektar_id')
    # try to correct for missing HA info by assuming a default 1
    enriched_pd['H14PTOT_corrected'] = enriched_pd['H14PTOT'].fillna(1)


    # compute LAT and RLAT
    logger.info("Computing LAT and RLAT")
    enriched_pd['LAT'] = 12.0 * enriched_pd['RELEVANZ'] * enriched_pd['VERKAUFSFLAECHE_TOTAL']
    enriched_pd['LAT2'] = 12.0 * enriched_pd['RELEVANZ'] * enriched_pd['VERKAUFSFLAECHE']
    enriched_pd['RLAT'] = enriched_pd['LAT'] * np.power(10, -0.17 * np.fmax(enriched_pd['fahrzeit'] - 5, 0))
    enriched_pd['RLAT2'] = enriched_pd['LAT2'] * np.power(10, -0.17 * np.fmax(enriched_pd['fahrzeit'] - 5, 0))

    # now calculate Marktanteil
    logger.info("Computing Marktanteil. Takes a while")

    def calc_MA(x):
        x['Marktanteil'] = x['RLAT'] / np.nansum(x['RLAT'])
        x['Marktanteil2'] = x['RLAT2'] / np.nansum(x['RLAT2'])
        return x

    enriched_pd = enriched_pd.reset_index().groupby('hektar_id').apply(calc_MA)

    logger.info("Computing local Umsatzpotential")
    enriched_pd['LokalUP'] = enriched_pd['Marktanteil'] * enriched_pd['H14PTOT'] * 7800
    enriched_pd['LokalUP2'] = enriched_pd['Marktanteil2'] * enriched_pd['H14PTOT'] * 7800
    enriched_pd['LokalUP_corrected'] = enriched_pd['Marktanteil'] * enriched_pd['H14PTOT_corrected'] * 7800
    enriched_pd['LokalUP2_corrected'] = enriched_pd['Marktanteil2'] * enriched_pd['H14PTOT_corrected'] * 7800

    logger.info("Serializing final data frame ")

    config = configparser.ConfigParser()
    config.read(settingsFile)
    enriched_pd.to_pickle(config["output"]["output_pickle"])

    logger.info("Done")

    # get only the Migros stores
    if len(list(stores_migros_pd['FORMAT'])) == 1:
        migros_only_pd = enriched_pd[enriched_pd['FORMAT'] == stores_migros_pd['FORMAT']]
    else:
        migros_only_pd = enriched_pd[enriched_pd['FORMAT'].isin(list(set(stores_migros_pd['FORMAT'])))]

    logger.info("Computing total Umsatz potential for relevant Migros stores")
    umsatz_potential_pd = migros_only_pd.groupby('index').agg({'LokalUP': lambda x: np.nansum(x),
                                                               'LokalUP2': lambda x: np.nansum(x),
                                                               'LokalUP_corrected': lambda x: np.nansum(x),
                                                               'LokalUP2_corrected': lambda x: np.nansum(x)})


    umsatz_potential_pd = umsatz_potential_pd.rename(columns={'LokalUP2': 'Umsatzpotential2',
                                                              'LokalUP': 'Umsatzpotential',
                                                              'LokalUP_corrected': 'Umsatzpotential_corrected',
                                                              'LokalUP2_corrected': 'Umsatzpotential2_corrected'})


    logger.info("Done")

    logger.info("Generating output csv")
    umsatz_potential_pd.to_csv(config["output"]["output_csv"])
    logger.info("Done")

