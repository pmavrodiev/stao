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

# noinspection PyUnboundLocalVariable
if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(settingsFile)

    # starting values
    a = float(config["calibration"]["a_start"])
    b = float(config["calibration"]["b_start"])

    (stores_pd, stores_migros_pd,
     drivetimes_pd, haushalt_pd, referenz_pd) = get_input(settingsFile, logger)

    if config.getboolean('global', 'use_pruned_cache'):
        logger.info("Loading pruned data from cache")
        enriched_pruned_pd = pd.read_pickle(config["output"]["output_pickle"])
    else:
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
            stores_pd[['ID', 'FORMAT', 'vfl', 'RELEVANZ', 'type']],
            left_index=True, right_on='ID', how='inner')

        logger.info("%d stores appear in drivetimes, but have no associated information in stores_sm.csv",
                    before-len(set(drivetimes_rel_hektars_stores_pd.index)))

        # enrich the drive times of the relevant hektars with Haushalt information
        logger.info("Enriching with Haushalt information")
        enriched_pd = drivetimes_rel_hektars_stores_pd.merge(haushalt_pd[['H14PTOT']],
                                                             left_on='hektar_id', right_index=True,
                                                             how='left')
        # try to correct for missing HA info by assuming a default 1
        enriched_pd['H14PTOT_corrected'] = enriched_pd['H14PTOT'].fillna(1)

        # compute LAT and RLAT
        logger.info("Computing LAT and RLAT")

        enriched_pd['LAT'] = np.where(enriched_pd.vfl < 1000,
                                          enriched_pd.RELEVANZ * enriched_pd.vfl * 0.06,
                                          np.where((enriched_pd.vfl >= 1000) & (
                                                    enriched_pd.vfl < 2500),
                                enriched_pd.RELEVANZ*(20 * (enriched_pd.vfl - 1000) / 1500 + 60),
                                enriched_pd.RELEVANZ*(20 * (enriched_pd.vfl - 2500) / 3500 + 80)))

        enriched_pd['RLAT'] = enriched_pd['LAT'] * np.power(10, -(a - b*np.fmin(enriched_pd['LAT'], 60)) * enriched_pd['fahrzeit'])

        logger.info("Reindexing ...")
        enriched_pd = enriched_pd.reset_index().set_index(keys=['hektar_id', 'type', 'OBJECTID'])
        logger.info("Removing duplicates ...")
        # remove the duplicates introduced after merging drivetimes and store information
        enriched_pd = enriched_pd[~enriched_pd.index.duplicated(keep='first')]
        enriched_pd = enriched_pd.reset_index().set_index(keys=['hektar_id', 'type'])
        # logger.info("Saving intermediary results ")
        # enriched_pd.to_pickle(config["output"]["intermediary_pickle"])

        # pruning the irrelevant stores as defined in Step 4 of the model
        logger.info("Pruning irrelevant stores. Takes a while ...")

        if config.getboolean('global', 'prune'):
            from parallel.parallel import apply_parallel, group_by_store_type

            # The different hektars are distributed across the threads
            # Each thread locally groups its hektars by store type and prunes the resulting groups.
            groups = enriched_pd.groupby(level=[0])  # group by hektar_id
            logger.info('%d groups after grouping by hektar_id', groups.ngroups)
            ncpus = int(config["parallel"]["cpu_count"])
            chunk_size = int(config["parallel"]["chunk_size"])
            enriched_pruned_pd = apply_parallel(groups, group_by_store_type, ncpus, chunk_size)
            logger.info('DONE')
        else:
            enriched_pruned_pd = enriched_pd.reset_index()

        enriched_pruned_pd.to_pickle(config["output"]["output_pickle"])

        """
            enriched_pruned_pd has the following structure at this point:

        hektar_id	type	OBJECTID	fahrzeit	ID	            FORMAT		vfl   RELEVANZ	H14PTOT	    H14PTOT	        LAT     RLAT
                                                                                        	                _corrected
        ---------
        49971200	MIG	        10	        8	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    NaN	        1.0	       9.44856	1.782190
        49971201	MIG	        10	        8	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    NaN	        1.0	       9.44856	1.782190
        49971204	MIG	        10	        8	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    NaN	        1.0	       9.44856	1.782190
        49971206	MIG	        10	        8	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    2.0	        2.0	       9.44856	1.782190
        49971207	MIG	        10	        9	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    NaN	        1.0	       9.44856	1.446781
        """

    # ##### MAIN CALIBRATION LOOP BEGINS HERE #####
    a_next = a
    b_next = b
    error = np.zeros(10)
    stop = False
    logger.info("BEGINNING CALIBRATION")

    for t in range(int(config["calibration"]["T"])):
        logger.info("Parameters: a/b = %f / %f", a_next, b_next )
        
        # calculate the RLAT with the new parameters
        enriched_pruned_pd['RLAT'] = enriched_pruned_pd['LAT'] * np.power(10, -(a_next - b_next * np.fmin(enriched_pruned_pd['LAT'], 60)) *
                                                                              enriched_pruned_pd['fahrzeit'])

        # compute the total sum of all RLATs for each hektar
        enriched_pruned_pd['sum_RLATS'] = enriched_pruned_pd.groupby('hektar_id')[["RLAT"]].transform(lambda x: np.sum(x))

        # compute the change in RLAT w.r.t. the parameters 'a' and 'b' for each hektar
        enriched_pruned_pd['dRLAT_da'] = -1.0 * enriched_pruned_pd['fahrzeit'] * np.log(10) * enriched_pruned_pd['RLAT']
        enriched_pruned_pd['dRLAT_db'] = enriched_pruned_pd['fahrzeit'] * np.log(10) * enriched_pruned_pd['RLAT'] * \
                                         np.where(enriched_pruned_pd.LAT <= 60, enriched_pruned_pd.LAT, 1.0)

        # compute the derivative of total sum of all RLATs for each hektar
        enriched_pruned_pd['dS_RLATda'] = enriched_pruned_pd.groupby('hektar_id')[['dRLAT_da']].transform(lambda x: np.sum(x))
        enriched_pruned_pd['dS_RLATdb'] = enriched_pruned_pd.groupby('hektar_id')[['dRLAT_db']].transform(lambda x: np.sum(x))

        # compute each term of the inner sum (the sum over the hektars)
        enriched_pruned_pd['inner_sum_terms_a'] = (enriched_pruned_pd['dRLAT_da'] * enriched_pruned_pd['sum_RLATS'] -
                                                   enriched_pruned_pd['RLAT'] * enriched_pruned_pd['dS_RLATda']) * \
                                                   enriched_pruned_pd['H14PTOT'] / np.power(enriched_pruned_pd['sum_RLATS'],
                                                                                       2)

        enriched_pruned_pd['inner_sum_terms_b'] = (enriched_pruned_pd['dRLAT_db'] * enriched_pruned_pd['sum_RLATS'] -
                                                   enriched_pruned_pd['RLAT'] * enriched_pruned_pd['dS_RLATdb']) * \
                                                   enriched_pruned_pd['H14PTOT'] / np.power(enriched_pruned_pd['sum_RLATS'],
                                                                                       2)
        # now sum-up all inner terms over all hektars, i.e. group by Filiale!!!
        enriched_pruned_pd['sum_terms_a'] = enriched_pruned_pd.groupby('OBJECTID')[["inner_sum_terms_a"]].transform(lambda x: np.nansum(x))
        enriched_pruned_pd['sum_terms_b'] = enriched_pruned_pd.groupby('OBJECTID')[["inner_sum_terms_b"]].transform(lambda x: np.nansum(x))

        # now calculate Marktanteil
        logger.debug("Computing Marktanteil.")
        enriched_pruned_pd['Marktanteil'] = enriched_pruned_pd['RLAT'] / enriched_pruned_pd['sum_RLATS']

        logger.debug("Computing local Umsatzpotential")
        enriched_pruned_pd['LokalUP'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd['H14PTOT'] * 7800
        enriched_pruned_pd['LokalUP_corrected'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd['H14PTOT_corrected'] * 7800

        migros_only_pd = enriched_pruned_pd[enriched_pruned_pd['OBJECTID'].isin(stores_migros_pd.index.values)]

        logger.debug("Computing total Umsatz potential for relevant Migros stores")
        umsatz_potential_pd = migros_only_pd.groupby('OBJECTID').agg({'ID': lambda x: x.iloc[0],
                                                                      'sum_terms_a': lambda x: x.iloc[0],
                                                                      'sum_terms_b': lambda x: x.iloc[0],
                                                                      'LokalUP': lambda x: np.nansum(x),
                                                                      'LokalUP_corrected': lambda x: np.nansum(x)
                                                                     })

        umsatz_potential_pd = umsatz_potential_pd.rename(columns={'LokalUP': 'Umsatzpotential',
                                                              'LokalUP_corrected': 'Umsatzpotential_corrected'})

        umsatz_potential_pd = umsatz_potential_pd.merge(referenz_pd, left_index=True, right_index=True, how='inner')

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

        error[t % 10] = umsatz_potential_pd.E_lsq_i.sum()
        # stop the gradient descent if the error hasn't changed much in the last 10 time steps
        if t > 10 and np.diff(error).mean() < float(config["calibration"]["delta_convergence"]):
            stop = True

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

        if config.getboolean('calibration','direct_output'):
            umsatz_potential_pd['verhaeltnis_tU'] = umsatz_potential_pd['Umsatzpotential'] / \
                                                    umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

            umsatz_potential_pd['verhaeltnis_MP2'] = umsatz_potential_pd['Umsatzpotential'] / \
                                                     umsatz_potential_pd['MP - CALCULATED_REVENUE 2']

            logger.info("Generating output csv")
            columns_to_output = ['OBJECTID', 'ID', 'Umsatzpotential', 'Umsatzpotential_corrected',
                                 'Tatsechlicher Umsatz - FOOD_AND_FRISCHE', 'verhaeltnis_tU', 'verhaeltnis_MP2']
            umsatz_potential_pd.to_csv(config["output"]["output_csv"])
            break                    

        # a_next -= np.sign(dE_lsq_da) * np.fmin(np.abs(dE_lsq_da), 0.01)
        # b_next = np.fmax(0, b_next - np.sign(dE_lsq_db) * np.fmin(np.abs(dE_lsq_db), 0.001))
        if stop:
            logger.info("Convergence criteria reached")
            break

    logger.info("DONE CALIBRATION")
