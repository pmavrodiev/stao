import json

import numpy as np
import pandas as pd

from models.model_base import ModelBase
from models.model_MBI_v_1_1.parallel import apply_parallel, group_by_store_type


@ModelBase.register
class model_MBI_v_1_1(ModelBase):

    logger = None
    umsatz_output_csv = None

    param_sweep = None
    a_sweep = None
    b_sweep = None
    a = None
    b = None
    prune = None
    ncpus = None
    chunk_size = None
    use_pruned_cache = None
    prune = None
    output_pickle = None
    direct_output = None
    calibration_T = None

    # the minimum error after the parameter sweep
    E_min = [(float("inf"), ())]

    def whoami(self):
        return 'Model_MBI_v1.1'

    def process_settings(self, config):
        # first check if we are doing a parameter sweep over a and b
        self.param_sweep = False  # by default False
        if config.has_option('parameter_sweep', 'a_array') and config.has_option('parameter_sweep', 'b_array'):
            self.alpha_sweep = [float(x) for x in json.loads(config.get('parameter_sweep', 'alpha_array'))]
            self.beta_sweep = [float(x) for x in json.loads(config.get('parameter_sweep', 'beta_array'))]
            self.param_sweep = True
        ####
        try:
            self.a = float(config["calibration"]["a_start"])
            self.b = float(config["calibration"]["b_start"])
            self.prune = config.getboolean('global', 'prune')
            self.ncpus = int(config["parallel"]["cpu_count"])
            self.chunk_size = int(config["parallel"]["chunk_size"])
            self.umsatz_output_csv = config["output"]["output_csv"]
            self.use_pruned_cache = config.getboolean('calibration', 'use_pruned_cache')
            self.output_pickle = config["output"]["output_pickle"]
            self.direct_output = config.getboolean('calibration', 'direct_output')
            self.calibration_T = int(config["calibration"]["T"])
            self.calibration_delta = float(config["calibration"]["delta_convergence"])
        except Exception:
            pass # TODO implement error handling

    def preprocess(self, pandas_dt):
        # compute LAT
        self.logger.info("Computing LAT")
        pandas_dt['LAT'] = np.where(pandas_dt.vfl < 1000,
                                    pandas_dt.RELEVANZ * pandas_dt.vfl * 0.06,
                                      np.where((pandas_dt.vfl >= 1000) & (
                                          pandas_dt.vfl < 2500),
                                               pandas_dt.RELEVANZ * (20 * (pandas_dt.vfl - 1000) / 1500 + 60),
                                               pandas_dt.RELEVANZ * (20 * (pandas_dt.vfl - 2500) / 3500 + 80)))

        self.logger.info("Reindexing ...")
        pandas_reindexed_dt = pandas_dt.reset_index().set_index(keys=['hektar_id', 'type', 'OBJECTID'])
        self.logger.info("Removing duplicates ...")
        # remove the duplicates introduced after merging drivetimes and store information
        pandas_reindexed_dt = pandas_reindexed_dt[~pandas_reindexed_dt.index.duplicated(keep='first')]
        pandas_reindexed_dt = pandas_reindexed_dt.reset_index().set_index(keys=['hektar_id', 'type'])
        return pandas_reindexed_dt

    def prune_step(self, pandas_dt):
        # pruning the irrelevant stores as defined in Step 4 of the model
        self.logger.info("Pruning irrelevant stores. Takes a while ...")

        # The different hektars are distributed across the threads
        # Each thread locally groups its hektars by store type and prunes the resulting groups.
        groups = pandas_dt.groupby(level=[0])  # group by hektar_id
        self.logger.info('%d groups after grouping by hektar_id', groups.ngroups)
        pandas_pruned_pd = apply_parallel(groups, group_by_store_type, self.ncpus, self.chunk_size)
        self.logger.info('DONE')

        """
            pandas_pruned_pd has the following structure at this point:

        hektar_id	type	OBJECTID	fahrzeit	ID	            FORMAT		vfl   RELEVANZ	H14PTOT	    H14PTOT	        LAT     RLAT
                                                                                    	                _corrected
        ---------
        49971200	MIG	        10	        8	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    NaN	        1.0	       9.44856	1.782190
        49971201	MIG	        10	        8	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    NaN	        1.0	       9.44856	1.782190
        49971204	MIG	        10	        8	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    NaN	        1.0	       9.44856	1.782190
        49971206	MIG	        10	        8	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    2.0	        2.0	       9.44856	1.782190
        49971207	MIG	        10	        9	SM_MIG_49997_11718	SPEZ	   157.476	  1.0	    NaN	        1.0	       9.44856	1.446781
        """
        return pandas_pruned_pd

    def calc_gradient(self, pandas_dt):

        pandas_dt['sum_RLATS'] = pandas_dt.groupby('hektar_id')[["RLAT"]].transform(
            lambda x: np.sum(x))

        # Compute the change in RLAT w.r.t. the parameters 'a' and 'b' for each hektar
        # This is Eq.8 in Gravitationsmodell.pdf
        pandas_dt['dRLAT_da'] = -1.0 * pandas_dt['fahrzeit'] * np.log(10) * pandas_dt['RLAT']
        pandas_dt['dRLAT_db'] = pandas_dt['fahrzeit'] * np.log(10) * pandas_dt['RLAT'] * \
                                         np.where(pandas_dt.LAT <= 60, pandas_dt.LAT, 60)
        # compute the derivative of total sum of all RLATs for each hektar
        pandas_dt['dS_RLATda'] = pandas_dt.groupby('hektar_id')[['dRLAT_da']].transform(
            lambda x: np.sum(x))
        pandas_dt['dS_RLATdb'] = pandas_dt.groupby('hektar_id')[['dRLAT_db']].transform(
            lambda x: np.sum(x))
        # compute each term of the inner sum (the sum over the hektars)
        pandas_dt['inner_sum_terms_a'] = (pandas_dt['dRLAT_da'] * pandas_dt['sum_RLATS'] -
                                          pandas_dt['RLAT'] * pandas_dt['dS_RLATda']) * \
                                         pandas_dt['Tot_Haushaltausgaben'] / np.power(pandas_dt['sum_RLATS'], 2)

        pandas_dt['inner_sum_terms_b'] = (pandas_dt['dRLAT_db'] * pandas_dt['sum_RLATS'] -
                                          pandas_dt['RLAT'] * pandas_dt['dS_RLATdb']) * \
                                         pandas_dt['Tot_Haushaltausgaben'] / np.power(pandas_dt['sum_RLATS'], 2)
        # now sum-up all inner terms over all hektars, i.e. group by Filiale!!!
        # this is Eq. 7 in Gravitationsmodell.pdf
        pandas_dt['sum_terms_a'] = pandas_dt.groupby('OBJECTID')[["inner_sum_terms_a"]].transform(
            lambda x: np.nansum(x))
        pandas_dt['sum_terms_b'] = pandas_dt.groupby('OBJECTID')[["inner_sum_terms_b"]].transform(
            lambda x: np.nansum(x))


    def analysis_sweep(self, pandas_dt,  stores_migros_pd, referenz_pd):
        # the parameter sweep always executes the pruning stage for every parameter combination
        self.logger.info('Will do parameter sweep. This will take a while, better run over night')
        for a in self.a_sweep:
            for b in self.b_sweep:
                self.logger.info("Parameters: a/b = %f / %f", a, b)
                self.logger.info("Computing RLAT")
                pandas_dt['RLAT'] = pandas_dt['LAT'] * np.power(10,
                                                                    -(a - b * np.fmin(pandas_dt['LAT'], 60)) *
                                                                    pandas_dt['fahrzeit'])
                if self.prune:
                    # pruning the irrelevant stores as defined in Step 4 of the model
                    pandas_pruned_pd = self.prune_step(pandas_dt)
                else:
                    pandas_pruned_pd = pandas_dt.reset_index()

                # compute the sum of the RLATS of all stores in a given hektar
                self.logger.info("Computing sum RLATs")
                pandas_pruned_pd['sum_RLATS'] = pandas_pruned_pd.groupby('hektar_id')[["RLAT"]].transform(
                    lambda x: np.sum(x))
                # finally compute the Umsatz predictions for all Migros stores

                umsatz_potential_pd = self.gen_umsatz_prognose(pandas_pruned_pd, stores_migros_pd, referenz_pd)

                self.logger.info("Computing prediction errors")
                # LINEAR SQUARE ERROR
                umsatz_potential_pd['E_lsq_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] -
                                                          umsatz_potential_pd[
                                                              'Tatsechlicher Umsatz - FOOD_AND_FRISCHE'], 2) / \
                                                 umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
                # RATIO SQUARE ERROR: -1 to make it an optimization problem with a minimum at 0
                # umsatz_potential_pd['E_rsq_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] /
                #                                         umsatz_potential_pd[
                #                                           'Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] - 1, 2)

                total_error = np.sqrt(umsatz_potential_pd.E_i.sum())
                self.logger.info("TOTAL ERROR: %f", total_error)

                if total_error < self.E_min[0][0]:
                    self.E_min = [(total_error, {"a": a, "b": b})]
                    self.logger.info('New minimum found.')
                # self.logger.info("TOTAL RATIO SQUARE ERROR: %f", umsatz_potential_pd.E_rsq_i.sum())

                self.logger.info("Generating output csv")
                umsatz_potential_pd.to_csv(self.umsatz_output_csv + '_pruned_a_' + str(a) + '_b_' + str(b))

        self.logger.info('Found error minimum of %f for a=%f / b=%f ',
                     self.E_min[0][0], self.E_min[0][1]["a"], self.E_min[0][1]["b"])

    def analysis_calibration(self, pandas_dt, stores_migros_pd, referenz_pd):
        self.logger.info("BEGINNING CALIBRATION")
        a_next = self.a
        b_next = self.b

        error = np.zeros(10)

        for t in range(self.calibration_T):
            self.logger.info("Parameters: a/b = %f / %f", a_next, b_next)
            pandas_dt['RLAT'] = pandas_dt['LAT'] * np.power(10, -(a_next - b_next * np.fmin(pandas_dt['LAT'], 60)) *
                                                            pandas_dt['fahrzeit'])

            self.calc_gradient(pandas_dt)
            # compute the Marketshare and generate the Umsatz prediction only for the Migros stores
            umsatz_potential_pd = self.gen_umsatz_prognose(pandas_dt, stores_migros_pd, referenz_pd)

            self.logger.info("Computing prediction errors")
            # LINEAR SQUARE ERROR
            umsatz_potential_pd['E_lsq_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] -
                                                      umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'],
                                                      2) / \
                                             umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
            # RATIO SQUARE ERROR: -1 to make it an optimization problem with a minimum at 0
            # umsatz_potential_pd['E_rsq_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] /
            #                                          umsatz_potential_pd[
            #                                              'Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] - 1, 2)

            self.logger.info("TOTAL LINEAR SQUARE ERROR after %d iterations: %f", t,
                        np.sqrt(umsatz_potential_pd.E_lsq_i.sum()))
            # self.logger.info("TOTAL RATIO SQUARE ERROR after %d iterations: %f", t, umsatz_potential_pd.E_rsq_i.sum())

            error[t % len(error)] = umsatz_potential_pd.E_lsq_i.sum()

            self.logger.debug("Computing gradients")
            # gradient linear square error
            umsatz_potential_pd['dE_lsq_i_da'] = 2.0 * (umsatz_potential_pd['Umsatzpotential'] -
                                                        umsatz_potential_pd[
                                                            'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']) * \
                                                 umsatz_potential_pd['sum_terms_a'] / \
                                                 umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
            umsatz_potential_pd['dE_lsq_i_db'] = 2.0 * (umsatz_potential_pd['Umsatzpotential'] -
                                                        umsatz_potential_pd[
                                                            'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']) * \
                                                 umsatz_potential_pd['sum_terms_b'] / \
                                                 umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
            # gradient ratio square error
            # umsatz_potential_pd['dE_rsq_i_da'] = 2.0 * (
            #     umsatz_potential_pd['Umsatzpotential'] / umsatz_potential_pd[
            #        'Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] - 1) * \
            #                                     umsatz_potential_pd['sum_terms_a'] / \
            #                                     umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

            # umsatz_potential_pd['dE_rsq_i_db'] = 2.0 * (
            #    umsatz_potential_pd['Umsatzpotential'] / umsatz_potential_pd[
            #        'Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] - 1) * \
            #                                     umsatz_potential_pd['sum_terms_b'] / \
            #                                     umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

            # total gradients
            dE_lsq_da = umsatz_potential_pd.dE_lsq_i_da.sum() / (2.0 * np.sqrt(umsatz_potential_pd.E_lsq_i.sum()))
            dE_lsq_db = umsatz_potential_pd.dE_lsq_i_db.sum() / (2.0 * np.sqrt(umsatz_potential_pd.E_lsq_i.sum()))
            #
            # dE_rsq_da = umsatz_potential_pd.dE_rsq_i_da.sum()
            # dE_rsq_db = umsatz_potential_pd.dE_rsq_i_db.sum()

            self.logger.info("\tGRADIENT LINEAR SQUARE ERROR after %d iterations %f (da) , %f (db)", t, dE_lsq_da, dE_lsq_db)
            # self.logger.info("\tGRADIENT RATIO SQUARE ERROR after %d iterations %f (da), %f (db)", t, dE_rsq_da, dE_rsq_db)

            # update the parameters, but limit learning rate
            a_next -= np.sign(dE_lsq_da) * np.fmin(np.abs(dE_lsq_da), 0.01)
            # a_next -= dE_rsq_da*0.01
            # b_next = np.fmax(0, b_next - np.sign(dE_lsq_db) * np.fmin(np.abs(dE_lsq_db), 0.001))

            # stop the gradient descent if the error hasn't changed much in the last 10 time steps
            if t > len(error) and np.diff(error).mean() < self.calibration_delta:
                self.logger.info("Convergence criteria reached")
                break

        self.logger.info("DONE CALIBRATION")
        self.logger.info("Generating output csv")
        umsatz_potential_pd.to_csv(self.umsatz_output_csv)

    def entry(self, pandas_dt, config, logger, stores_migros_pd, referenz_pd):

        self.logger = logger
        self.logger.info("Initialized model %s", self.whoami())

        self.process_settings(config)

        pandas_preprocessed_dt = self.preprocess(pandas_dt)
        """
            pandas_preprocessed_dt looks like this now:

            |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            |              |OBJECTID |fahrzeit|          ID       |FORMAT|  vfl  |RELEVANZ|	Tot_Haushaltausgaben	|Tot_Haushaltausgaben_corrected|LAT     |
            |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            |hektar_id|type|
            |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            | 61341718|	MIG|   6     |21      |	SM_MIG_61607_15939|   M  |878.621| 1      |	7800.0                  |   7800                       |52.71726|
            |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        """


        if self.param_sweep:
            self.analysis_sweep(pandas_preprocessed_dt, stores_migros_pd, referenz_pd)
        else:
            if not self.use_pruned_cache:
                self.logger.info("Computing RLAT")
                pandas_preprocessed_dt['RLAT'] = pandas_preprocessed_dt['LAT'] * np.power(10,
                                                                    -(self.a - self.b * np.fmin(pandas_preprocessed_dt['LAT'], 60)) *
                                                                                          pandas_preprocessed_dt['fahrzeit'])
                if self.prune:
                    # pruning the irrelevant stores as defined in Step 4 of the model
                    pandas_postprocessed_dt = self.prune_step(pandas_preprocessed_dt)
                else:
                    pandas_postprocessed_dt = pandas_preprocessed_dt.reset_index()
                # cache pruned data
                pandas_postprocessed_dt.to_pickle(self.output_pickle)
            else:
                self.logger.info("Loading pruned data from cache")
                pandas_postprocessed_dt = pd.read_pickle(self.output_pickle)

        if self.direct_output:
            # No calibration. Just compute the Umsatz forecast and exit
            self.logger.info("Computing sum RLATs")
            pandas_postprocessed_dt['sum_RLATS'] = pandas_postprocessed_dt.groupby('hektar_id')[["RLAT"]].transform(
                lambda x: np.sum(x))
            umsatz_potential_pd = self.gen_umsatz_prognose(pandas_postprocessed_dt, stores_migros_pd, referenz_pd)
            self.logger.info("Generating output csv")
            umsatz_potential_pd.to_csv(self.umsatz_output_csv)
        else:
            self.analysis_calibration(pandas_postprocessed_dt, stores_migros_pd, referenz_pd)


    def gen_umsatz_prognose(self, pandas_pd, stores_migros_pd, referenz_pd):
        """

        :param enriched_pruned_pd:
        :param stores_migros_pd:
        :param referenz_pd:
        :param logger:
        :return:
        """
        # now calculate Marktanteil
        self.logger.debug("Computing Marktanteil.")
        # compute the total sum of all RLATs for each hektar
        pandas_pd['Marktanteil'] = pandas_pd['RLAT'] / pandas_pd['sum_RLATS']

        self.logger.debug("Computing local Umsatzpotential")
        pandas_pd['LokalUP'] = pandas_pd['Marktanteil'] * pandas_pd['Tot_Haushaltausgaben']
        pandas_pd['LokalUP_corrected'] = pandas_pd['Marktanteil'] * pandas_pd['Tot_Haushaltausgaben_corrected']
        # enriched_pruned_pd['LokalUP'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd['H14PTOT'] * 7800
        # enriched_pruned_pd['LokalUP_corrected'] = enriched_pruned_pd['Marktanteil'] * enriched_pruned_pd[
        #     'H14PTOT_corrected'] * 7800

        migros_only_pd = pandas_pd[pandas_pd['OBJECTID'].isin(stores_migros_pd.index.values)]

        self.logger.debug("Computing total Umsatz potential for relevant Migros stores")

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

        umsatz_potential_pd['verhaeltnis_tU_prozent'] = (umsatz_potential_pd['Umsatzpotential'] -
                                                         umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'] ) / \
                                                        umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

        umsatz_potential_pd['verhaeltnis_MP2'] = umsatz_potential_pd['Umsatzpotential'] / \
                                                 umsatz_potential_pd['MP - CALCULATED_REVENUE 2']

        return umsatz_potential_pd
