import pandas as pd
import numpy as np


def do_run(enriched_pd, a, b, config, logger):
    logger.info("Computing RLAT")

    enriched_pd['RLAT'] = enriched_pd['LAT'] * np.power(10, -(a - b * np.fmin(enriched_pd['LAT'], 60)) * enriched_pd[
        'fahrzeit'])

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
        from utils.parallel import apply_parallel, group_by_store_type

        # The different hektars are distributed across the threads
        # Each thread locally groups its hektars by store type and prunes the resulting groups.
        groups = enriched_pd.groupby(level=[0])  # group by hektar_id
        logger.info('%d groups after grouping by hektar_id', groups.ngroups)
        ncpus = int(config["utils"]["cpu_count"])
        chunk_size = int(config["utils"]["chunk_size"])
        enriched_pruned_pd = apply_parallel(groups, group_by_store_type, ncpus, chunk_size)
        logger.info('DONE')
    else:
        enriched_pruned_pd = enriched_pd.reset_index()

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

    enriched_pruned_pd.to_pickle(config["output"]["output_pickle"])

    # now calculate Marktanteil
    logger.debug("Computing Marktanteil.")
    enriched_pruned_pd['Marktanteil'] = enriched_pruned_pd['RLAT'] / enriched_pruned_pd['sum_RLATS']

    logger.debug("Computing local Umsatzpotential")
    enriched_pruned_pd['LokalUP'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd['H14PTOT'] * 7800
    enriched_pruned_pd['LokalUP_corrected'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd[
        'H14PTOT_corrected'] * 7800

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

    umsatz_potential_pd['verhaeltnis_tU'] = umsatz_potential_pd['Umsatzpotential'] / \
                                            umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

    umsatz_potential_pd['verhaeltnis_MP2'] = umsatz_potential_pd['Umsatzpotential'] / \
                                             umsatz_potential_pd['MP - CALCULATED_REVENUE 2']

    logger.info("Generating output csv")
    columns_to_output = ['OBJECTID', 'ID', 'Umsatzpotential', 'Umsatzpotential_corrected',
                         'Tatsechlicher Umsatz - FOOD_AND_FRISCHE', 'verhaeltnis_tU', 'verhaeltnis_MP2']
    umsatz_potential_pd.to_csv(config["output"]["output_csv"])
    return enriched_pruned_pd