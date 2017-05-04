import pandas as pd
import numpy as np


def prune(enriched_pd, a, b, config, logger):
    # pruning the irrelevant stores as defined in Step 4 of the model
    logger.info("Pruning irrelevant stores. Takes a while ...")

    from utils.parallel import apply_parallel, group_by_store_type

    # The different hektars are distributed across the threads
    # Each thread locally groups its hektars by store type and prunes the resulting groups.
    groups = enriched_pd.groupby(level=[0])  # group by hektar_id
    logger.info('%d groups after grouping by hektar_id', groups.ngroups)
    ncpus = int(config["parallel"]["cpu_count"])
    chunk_size = int(config["parallel"]["chunk_size"])
    enriched_pruned_pd = apply_parallel(groups, group_by_store_type, ncpus, chunk_size)
    logger.info('DONE')
    
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
    return enriched_pruned_pd


def calc_gradient(enriched_pruned_pd):

    enriched_pruned_pd['sum_RLATS'] = enriched_pruned_pd.groupby('hektar_id')[["RLAT"]].transform(
        lambda x: np.sum(x))

    # compute the change in RLAT w.r.t. the parameters 'a' and 'b' for each hektar
    enriched_pruned_pd['dRLAT_da'] = -1.0 * enriched_pruned_pd['fahrzeit'] * np.log(10) * enriched_pruned_pd['RLAT']
    enriched_pruned_pd['dRLAT_db'] = enriched_pruned_pd['fahrzeit'] * np.log(10) * enriched_pruned_pd['RLAT'] * \
                                     np.where(enriched_pruned_pd.LAT <= 60, enriched_pruned_pd.LAT, 1.0)
    # compute the derivative of total sum of all RLATs for each hektar
    enriched_pruned_pd['dS_RLATda'] = enriched_pruned_pd.groupby('hektar_id')[['dRLAT_da']].transform(
        lambda x: np.sum(x))
    enriched_pruned_pd['dS_RLATdb'] = enriched_pruned_pd.groupby('hektar_id')[['dRLAT_db']].transform(
        lambda x: np.sum(x))
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
    enriched_pruned_pd['sum_terms_a'] = enriched_pruned_pd.groupby('OBJECTID')[["inner_sum_terms_a"]].transform(
        lambda x: np.nansum(x))
    enriched_pruned_pd['sum_terms_b'] = enriched_pruned_pd.groupby('OBJECTID')[["inner_sum_terms_b"]].transform(
        lambda x: np.nansum(x))

    return enriched_pruned_pd


def gen_umsatz_prognose(enriched_pruned_pd, stores_migros_pd, referenz_pd, logger):

    # now calculate Marktanteil
    logger.debug("Computing Marktanteil.")
    # compute the total sum of all RLATs for each hektar
    enriched_pruned_pd['Marktanteil'] = enriched_pruned_pd['RLAT'] / enriched_pruned_pd['sum_RLATS']

    logger.debug("Computing local Umsatzpotential")
    enriched_pruned_pd['LokalUP'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd['H14PTOT'] * 7800
    enriched_pruned_pd['LokalUP_corrected'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd[
        'H14PTOT_corrected'] * 7800

    migros_only_pd = enriched_pruned_pd[enriched_pruned_pd['OBJECTID'].isin(stores_migros_pd.index.values)]

    logger.debug("Computing total Umsatz potential for relevant Migros stores")

    if 'sum_terms_a' in migros_only_pd and 'sum_terms_b' in migros_only_pd:
        umsatz_potential_pd = migros_only_pd.groupby('OBJECTID').agg({'ID': lambda x: x.iloc[0],
                                                                      'sum_terms_a': lambda x: x.iloc[0],
                                                                      'sum_terms_b': lambda x: x.iloc[0],
                                                                      'LokalUP': lambda x: np.nansum(x),
                                                                      'LokalUP_corrected': lambda x: np.nansum(x)
                                                                      })
    else:
        umsatz_potential_pd = migros_only_pd.groupby('OBJECTID').agg({'ID': lambda x: x.iloc[0],
                                                                      'LokalUP': lambda x: np.nansum(x),
                                                                      'LokalUP_corrected': lambda x: np.nansum(x)
                                                                      })
    umsatz_potential_pd = umsatz_potential_pd.rename(columns={'LokalUP': 'Umsatzpotential',
                                                              'LokalUP_corrected': 'Umsatzpotential_corrected'})

    umsatz_potential_pd = umsatz_potential_pd.merge(referenz_pd, left_index=True, right_index=True, how='inner')

    umsatz_potential_pd['verhaeltnis_tU'] = umsatz_potential_pd['Umsatzpotential'] / \
                                            umsatz_potential_pd[
                                                'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

    umsatz_potential_pd['verhaeltnis_MP2'] = umsatz_potential_pd['Umsatzpotential'] / \
                                             umsatz_potential_pd['MP - CALCULATED_REVENUE 2']

    return umsatz_potential_pd
