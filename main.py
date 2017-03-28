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

    config = configparser.ConfigParser()
    config.read(settingsFile)

    (stores_pd, stores_migros_pd, drivetimes_pd, haushalt_pd) = get_input(settingsFile, logger)

    logger.info("MAIN ALGO BEGINS")
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
        stores_pd[['ID', 'FORMAT', 'VERKAUFSFLAECHE', 'VERKAUFSFLAECHE_TOTAL', 'RELEVANZ', 'type']],
        left_index=True, right_on='ID', how='inner')

    logger.info("%d stores appear in drivetimes, but have no associated information in stores_sm.csv",
                before-len(set(drivetimes_rel_hektars_stores_pd.index)))

    # enrich the drive times of the relevant hektars with Haushalt information
    logger.info("Enriching with Haushalt information")
    enriched_pd = drivetimes_rel_hektars_stores_pd.merge(haushalt_pd[['H14PTOT']],
                                                         left_on='hektar_id', right_index=True)
    # try to correct for missing HA info by assuming a default 1
    enriched_pd['H14PTOT_corrected'] = enriched_pd['H14PTOT'].fillna(1)

    # compute LAT and RLAT
    logger.info("Computing LAT and RLAT")

    enriched_pd['LAT'] = np.where(enriched_pd.VERKAUFSFLAECHE_TOTAL < 1000,
                                      enriched_pd.RELEVANZ * enriched_pd.VERKAUFSFLAECHE_TOTAL * 0.06,
                                      np.where((enriched_pd.VERKAUFSFLAECHE_TOTAL >= 1000) & (
                                                enriched_pd.VERKAUFSFLAECHE_TOTAL < 2500),
                            enriched_pd.RELEVANZ*(20 * (enriched_pd.VERKAUFSFLAECHE_TOTAL - 1000) / 1500 + 60),
                            enriched_pd.RELEVANZ*(20 * (enriched_pd.VERKAUFSFLAECHE_TOTAL - 2500) / 3500 + 80)))

    enriched_pd['RLAT'] = enriched_pd['LAT'] * np.power(10, -(np.fmin(enriched_pd['LAT']/60.0, 1.0)
                                                             *(0.04-0.1)+0.1)*enriched_pd['fahrzeit'])

    logger.info("Reindexing ...")
    enriched_pd = enriched_pd.reset_index().set_index(keys=['hektar_id', 'type', 'OBJECTID'])
    logger.info("Removing duplicates ...")
    # remove the duplicates introduced after merging drivetimes and store information
    enriched_pd = enriched_pd[~enriched_pd.index.duplicated(keep='first')]
    enriched_pd = enriched_pd.reset_index().set_index(keys=['hektar_id', 'type', 'OBJECTID', 'fahrzeit']).sort_index(
        level=[0, 1, 3])

    # pruning the irrelevant stores as defined in Step 4 of the model
    logger.info("Pruning irrelevant stores. Takes a while")

    def prune(x):
        x['diff1'] = x['RLAT'] - x['RLAT'].iloc[0]
        x['diff2'] = x['RLAT'].iloc[1:len(x)] - x['RLAT'].shift(1)
        x = x[(x['diff1'] >= 0) & ((x['diff2'] >= 0) | np.isnan(x['diff2']))]
        return x

    enriched_pruned_pd = enriched_pd.groupby(level=[0, 1]).apply(prune)
    enriched_pruned_pd.to_pickle(config["output"]["output_pickle"])
    # now calculate Marktanteil
    logger.info("Computing Marktanteil.")

    def calc_MA(x):
        x['Marktanteil'] = x['RLAT'] / np.nansum(x['RLAT'])
        return x

    enriched_pruned_pd = enriched_pruned_pd.reset_index().groupby(by='hektar_id').apply(calc_MA)

    logger.info("Computing local Umsatzpotential")
    enriched_pruned_pd['LokalUP'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd['H14PTOT'] * 7800
    enriched_pruned_pd['LokalUP_corrected'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd['H14PTOT_corrected'] * 7800

    logger.info("Serializing final data frame ")

    enriched_pruned_pd.to_pickle(config["output"]["output_pickle"])

    logger.info("Done")

    # get only the Migros stores
    if len(list(stores_migros_pd['FORMAT'])) == 1:
        migros_only_pd = enriched_pruned_pd[enriched_pruned_pd['FORMAT'] == stores_migros_pd['FORMAT']]
    else:
        migros_only_pd = enriched_pruned_pd[enriched_pruned_pd['FORMAT'].isin(list(set(stores_migros_pd['FORMAT'])))]

    logger.info("Computing total Umsatz potential for relevant Migros stores")
    umsatz_potential_pd = migros_only_pd.groupby('ID').agg({'LokalUP': lambda x: np.nansum(x),
                                                            'LokalUP_corrected': lambda x: np.nansum(x)
                                                            })


    umsatz_potential_pd = umsatz_potential_pd.rename(columns={'LokalUP': 'Umsatzpotential',
                                                              'LokalUP_corrected': 'Umsatzpotential_corrected'})

    logger.info("Done")

    logger.info("Generating output csv")
    umsatz_potential_pd.to_csv(config["output"]["output_csv"])
    logger.info("Done")

