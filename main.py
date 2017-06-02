# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
import configparser
import json


from simple_logging.custom_logging import setup_custom_logger
from input_reader.input_reader import get_input
from utils.brain import prune, gen_umsatz_prognose, calc_gradient

# logger einrichten
LOGGING_LEVEL = logging.INFO
logger = setup_custom_logger('GM_LOGGER', LOGGING_LEVEL, flog="logs/gm.log")

settingsFile = "settings.cfg"

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(settingsFile)

    # first check if we are doing a parameter sweep over a and b
    param_sweep = False # by default False
    if config.has_option('parameter_sweep', 'a_array') and config.has_option('parameter_sweep', 'b_array'):
        a_sweep = json.loads(config.get('parameter_sweep', 'a_array'))
        b_sweep = json.loads(config.get('parameter_sweep', 'b_array'))
        param_sweep = True
    ####

    # read-in the data
    (stores_pd, stores_migros_pd, drivetimes_pd, haushalt_pd, referenz_pd) = get_input(settingsFile, logger)

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
        stores_pd[['ID', 'FORMAT', 'vfl', 'RELEVANZ', 'type']],
        left_index=True, right_on='ID', how='inner')

    logger.info("%d stores appear in drivetimes, but have no associated information in stores_sm.csv",
                before - len(set(drivetimes_rel_hektars_stores_pd.index)))

    # enrich the drive times of the relevant hektars with Haushalt information
    logger.info("Enriching with Haushalt information")
    enriched_pd = drivetimes_rel_hektars_stores_pd.merge(haushalt_pd[['Tot_Haushaltausgaben']],
                                                         left_on='hektar_id', right_index=True,
                                                         how='left')
    # try to correct for missing HA info by assuming a default of 1 household / person
    enriched_pd['Tot_Haushaltausgaben_corrected'] = enriched_pd['Tot_Haushaltausgaben'].fillna(7800 / 2.5)

    # compute LAT
    logger.info("Computing LAT")

    enriched_pd['LAT'] = np.where(enriched_pd.vfl < 1000,
                                  enriched_pd.RELEVANZ * enriched_pd.vfl * 0.06,
                                  np.where((enriched_pd.vfl >= 1000) & (
                                      enriched_pd.vfl < 2500),
                                           enriched_pd.RELEVANZ * (20 * (enriched_pd.vfl - 1000) / 1500 + 60),
                                           enriched_pd.RELEVANZ * (20 * (enriched_pd.vfl - 2500) / 3500 + 80)))

    logger.info("Reindexing ...")
    enriched_pd = enriched_pd.reset_index().set_index(keys=['hektar_id', 'type', 'OBJECTID'])
    logger.info("Removing duplicates ...")
    # remove the duplicates introduced after merging drivetimes and store information
    enriched_pd = enriched_pd[~enriched_pd.index.duplicated(keep='first')]
    enriched_pd = enriched_pd.reset_index().set_index(keys=['hektar_id', 'type'])

    if param_sweep:
        # the parameter sweep always executes the pruning stage for every parameter combination
        logger.info('Will do parameter sweep. This will take a while, better run over night')
        for a in a_sweep:
            for b in b_sweep:
                logger.info("Parameters: a/b = %f / %f", a, b)
                logger.info("Computing RLAT")
                enriched_pd['RLAT'] = enriched_pd['LAT'] * np.power(10,
                                                                    -(a - b * np.fmin(enriched_pd['LAT'], 60)) * enriched_pd[
                                                                    'fahrzeit'])
                if config.getboolean('global', 'prune'):
                    # pruning the irrelevant stores as defined in Step 4 of the model
                    enriched_pruned_pd = prune(enriched_pd, a, b, config, logger)
                else:
                    enriched_pruned_pd = enriched_pd.reset_index()

                # compute the sum of the RLATS of all stores in a given hektar
                logger.info("Computing sum RLATs")
                enriched_pruned_pd['sum_RLATS'] = enriched_pruned_pd.groupby('hektar_id')[["RLAT"]].transform(
                    lambda x: np.sum(x))
                # finally compute the Umsatz predictions for all Migros stores
                umsatz_potential_pd = gen_umsatz_prognose(enriched_pruned_pd, stores_migros_pd, referenz_pd, logger)

                logger.info("Computing prediction errors")
                # LINEAR SQUARE ERROR
                umsatz_potential_pd['E_lsq_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] -
                                                        umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'], 2) / \
                                                     umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
                # RATIO SQUARE ERROR: -1 to make it an optimization problem with a minimum at 0
                umsatz_potential_pd['E_rsq_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] /
                                                            umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] - 1, 2)

                logger.info("TOTAL LINEAR SQUARE ERROR: %f", np.sqrt(umsatz_potential_pd.E_lsq_i.sum()))
                logger.info("TOTAL RATIO SQUARE ERROR: %f", umsatz_potential_pd.E_rsq_i.sum())

                logger.info("Generating output csv")
                columns_to_output = ['OBJECTID', 'ID', 'Umsatzpotential', 'Umsatzpotential_corrected',
                                     'Tatsechlicher Umsatz - FOOD_AND_FRISCHE', 'verhaeltnis_tU', 'verhaeltnis_MP2']
                umsatz_potential_pd.to_csv(config["output"]["output_csv"]+'_pruned_a_'+str(a)+'_b_'+str(b))
    else:
        # No parameter sweep
        # starting values for calibration
        a = float(config["calibration"]["a_start"])
        b = float(config["calibration"]["b_start"])
        if not config.getboolean('calibration', 'use_pruned_cache'):
            logger.info("Computing RLAT")
            enriched_pd['RLAT'] = enriched_pd['LAT'] * np.power(10,
                                                                -(a - b * np.fmin(enriched_pd['LAT'], 60)) *
                                                                enriched_pd['fahrzeit'])
            # TEMPORARY - exclude all stores with VFL > 4000
            # logger.info("TEMPORARY STEP: Filtering stores by VFL")
            # enriched_pd['RLAT'] = np.where(enriched_pd.vfl > 2000, 0, enriched_pd.vfl)

            # logger.info("Saving intermediary results ")
            # enriched_pd.to_pickle(config["output"]["intermediary_pickle"])

            if config.getboolean('global', 'prune'):
                # pruning the irrelevant stores as defined in Step 4 of the model
                enriched_pruned_pd = prune(enriched_pd, a, b, config, logger)
            else:
                enriched_pruned_pd = enriched_pd.reset_index()
            # cache pruned data
            enriched_pruned_pd.to_pickle(config["output"]["output_pickle"])
        else:
            logger.info("Loading pruned data from cache")
            enriched_pruned_pd = pd.read_pickle(config["output"]["output_pickle"])

        a_next = a
        b_next = b

        if config.getboolean('calibration', 'direct_output'):
            # No calibration. Just compute the Umsatz forecast and exit
            logger.info("Computing sum RLATs")
            enriched_pruned_pd['sum_RLATS'] = enriched_pruned_pd.groupby('hektar_id')[["RLAT"]].transform(
                lambda x: np.sum(x))

            umsatz_potential_pd = gen_umsatz_prognose(enriched_pruned_pd, stores_migros_pd,
                                                      referenz_pd, logger)
            logger.info("Generating output csv")
            columns_to_output = ['OBJECTID', 'ID', 'Umsatzpotential', 'Umsatzpotential_corrected',
                                 'Tatsechlicher Umsatz - FOOD_AND_FRISCHE', 'verhaeltnis_tU', 'verhaeltnis_MP2']
            umsatz_potential_pd.to_csv(config["output"]["output_csv"])
        else:
            # MAIN CALIBRATION LOOP BEGINS HERE #####
            error = np.zeros(10)
            stop = False
            logger.info("BEGINNING CALIBRATION")
            for t in range(int(config["calibration"]["T"])):
                logger.info("Parameters: a/b = %f / %f", a_next, b_next)
        
                # calculate the RLAT with the new parameters
                enriched_pruned_pd['RLAT'] = enriched_pruned_pd['LAT'] * np.power(10, -(a_next - b_next * np.fmin(enriched_pruned_pd['LAT'], 60)) *
                                                                            enriched_pruned_pd['fahrzeit'])
                enriched_pruned_pd = calc_gradient(enriched_pruned_pd)

                # compute the Marketshare and generate the Umsatz prediction only for the Migros stores
                umsatz_potential_pd = gen_umsatz_prognose(enriched_pruned_pd, stores_migros_pd,
                                                              referenz_pd, logger)
                logger.info("Generating output csv")
                columns_to_output = ['OBJECTID', 'ID', 'Umsatzpotential', 'Umsatzpotential_corrected',
                                       'Tatsechlicher Umsatz - FOOD_AND_FRISCHE', 'verhaeltnis_tU', 'verhaeltnis_MP2']
                umsatz_potential_pd.to_csv(config["output"]["output_csv"])

                logger.info("Computing prediction errors")
                # LINEAR SQUARE ERROR
                umsatz_potential_pd['E_lsq_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] -
                                                        umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'], 2) / \
                                                     umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
                # RATIO SQUARE ERROR: -1 to make it an optimization problem with a minimum at 0
                umsatz_potential_pd['E_rsq_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] /
                                                            umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] - 1, 2)

                logger.info("TOTAL LINEAR SQUARE ERROR after %d iterations: %f", t, np.sqrt(umsatz_potential_pd.E_lsq_i.sum()))
                logger.info("TOTAL RATIO SQUARE ERROR after %d iterations: %f", t, umsatz_potential_pd.E_rsq_i.sum())

                error[t % len(error)] = umsatz_potential_pd.E_lsq_i.sum()

                logger.debug("Computing gradients")
                # gradient linear square error
                umsatz_potential_pd['dE_lsq_i_da'] = 2.0 * (umsatz_potential_pd['Umsatzpotential'] -
                                                        umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']) * \
                                                    umsatz_potential_pd['sum_terms_a'] / \
                                                         umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
                umsatz_potential_pd['dE_lsq_i_db'] = 2.0 * (umsatz_potential_pd['Umsatzpotential'] -
                                                            umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']) * \
                                                        umsatz_potential_pd['sum_terms_b'] / \
                                                         umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
                # gradient ratio square error
                umsatz_potential_pd['dE_rsq_i_da'] = 2.0 * (
                umsatz_potential_pd['Umsatzpotential'] / umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] - 1) * \
                                                     umsatz_potential_pd['sum_terms_a'] / \
                                                         umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
                umsatz_potential_pd['dE_rsq_i_db'] = 2.0 * (
                umsatz_potential_pd['Umsatzpotential'] / umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] - 1) * \
                                                     umsatz_potential_pd['sum_terms_b'] / \
                                                         umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

                # total gradients
                dE_lsq_da = umsatz_potential_pd.dE_lsq_i_da.sum() / (2.0 * np.sqrt(umsatz_potential_pd.E_lsq_i.sum()))
                dE_lsq_db = umsatz_potential_pd.dE_lsq_i_db.sum() / (2.0 * np.sqrt(umsatz_potential_pd.E_lsq_i.sum()))
                #
                dE_rsq_da = umsatz_potential_pd.dE_rsq_i_da.sum()
                dE_rsq_db = umsatz_potential_pd.dE_rsq_i_db.sum()

                logger.info("\tGRADIENT LINEAR SQUARE ERROR after %d iterations %f (da) , %f (db)", t, dE_lsq_da, dE_lsq_db)
                logger.info("\tGRADIENT RATIO SQUARE ERROR after %d iterations %f (da), %f (db)", t, dE_rsq_da, dE_rsq_db)

                # update the parameters, but limit learning rate
                a_next -= np.sign(dE_lsq_da) * np.fmin(np.abs(dE_lsq_da), 0.01)
                # a_next -= dE_rsq_da*0.01
                # b_next = np.fmax(0, b_next - np.sign(dE_lsq_db) * np.fmin(np.abs(dE_lsq_db), 0.001))

                # stop the gradient descent if the error hasn't changed much in the last 10 time steps
                if t > len(error) and np.diff(error).mean() < float(config["calibration"]["delta_convergence"]):
                    logger.info("Convergence criteria reached")
                    break

            logger.info("DONE CALIBRATION")
