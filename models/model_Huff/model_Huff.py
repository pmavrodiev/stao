import json

import numpy as np
import pandas as pd

from models.model_base import ModelBase


@ModelBase.register
class model_Huff(ModelBase):
    logger = None
    umsatz_output_csv = None

    param_sweep = None
    alpha_sweep = None
    beta_M_sweep = None
    beta_REST_sweep = None
    alpha = None
    beta = None

    ncpus = None
    chunk_size = None

    calibrate = None
    calibration_T = None

    # the minimum error after the parameter sweep
    E_min = [(float("inf"), ())]

    def whoami(self):
        return 'Model_Huff'

    def process_settings(self, config):
        # first check if we are doing a parameter sweep over a and b
        self.param_sweep = False  # by default False
        if config.has_option('parameter_sweep', 'alpha_array') and config.has_option('parameter_sweep', 'beta_array_M'):
            self.alpha_sweep = [float(x) for x in json.loads(config.get('parameter_sweep', 'alpha_array'))]
            self.beta_M_sweep = [float(x) for x in json.loads(config.get('parameter_sweep', 'beta_array_M'))]
            self.beta_REST_sweep = [float(x) for x in json.loads(config.get('parameter_sweep', 'beta_array_REST'))]
            self.param_sweep = True
        ####

        try:
            self.alpha = float(config["parameters"]["alpha"])
            self.beta = float(config["parameters"]["beta"])
            self.umsatz_output_csv = config["output"]["output_csv"]
            self.output_pickle = config["output"]["output_pickle"]
            self.calibrate = config.getboolean('parameters', 'calibrate')
            self.calibration_T = int(config["parameters"]["T"])
            self.calibration_delta = float(config["parameters"]["delta_convergence"])
        except Exception:
            pass # TODO implement error handling

    def preprocess(self, pandas_dt):

        self.logger.info("Reindexing ...")
        pandas_reindexed_dt = pandas_dt.reset_index().set_index(keys=['hektar_id', 'type', 'OBJECTID'])
        self.logger.info("Removing duplicates ...")
        # remove the duplicates introduced after merging drivetimes and store information
        pandas_reindexed_dt = pandas_reindexed_dt[~pandas_reindexed_dt.index.duplicated(keep='first')]
        pandas_reindexed_dt = pandas_reindexed_dt.reset_index()  # .set_index(keys=['hektar_id']) # , 'type'])
        return pandas_reindexed_dt

    def compute_huff_market_share2(self, pandas_dt, alpha, beta):
        self.logger.info('Computing local market share. This takes a while ...')
        # this is ln(A_i)
        pandas_dt['ln_A_i'] = np.log(pandas_dt['vfl'])

        pandas_dt['fahrzeit_huff'] = np.where(pandas_dt['FORMAT']=='M', np.power(pandas_dt['fahrzeit'], 2 * beta), np.power(pandas_dt['fahrzeit'], beta))

        # this is A_i^alpha*fahrzeit^beta
        pandas_dt['huff_numerator'] = np.power(np.power(pandas_dt['vfl'], 2.0), alpha) * pandas_pd['fahrzeit_huff']


        pandas_dt['huff_denumerator'] = np.power(np.power(pandas_dt['vfl'], 2.0), alpha) * pandas_pd['fahrzeit_huff']

        # this is A_i^alpha * fahrzeit^beta * log(A_i)
        pandas_dt['huff_denumerator_log'] = np.power(np.power(pandas_dt['vfl'], 2.0), alpha) * \
                                         np.power(pandas_dt['fahrzeit'], beta) * pandas_dt['ln_A_i']

        # this is sum(A_i^alpha * fahrzeit^beta)
        pandas_dt['huff_denumerator'] = pandas_dt.groupby('hektar_id')[["huff_denumerator"]].transform(lambda x: np.sum(x))

        # this is sum(A_i^alpha * fahrzeit^beta * ln(A_i))
        pandas_dt['huff_denumerator_log'] = pandas_dt.groupby('hektar_id')[["huff_denumerator_log"]].transform(
            lambda x: np.sum(x))

        pandas_dt['local_market_share'] = pandas_dt['huff_numerator'] / pandas_dt['huff_denumerator']

        self.logger.info('Done')

    def compute_huff_market_share_parallel(self, pandas_dt, alpha, beta):
        self.logger.info('Computing local market share. This takes a while ...')

        # this is A_i^alpha*fahrzeit^beta
        pandas_dt['huff_numerator'] = np.power(np.power(pandas_dt['vfl'], 2.0), alpha) * np.power(pandas_dt['fahrzeit'], beta)

        # this is sum(A_i^alpha * fahrzeit^beta)
        pandas_dt['huff_denumerator'] = pandas_dt.groupby('hektar_id')[["huff_numerator"]].transform(lambda x: np.sum(x))

        pandas_dt['local_market_share'] = pandas_dt['huff_numerator'] / pandas_dt['huff_denumerator']

        self.logger.info('Done')
        return pandas_dt

    def compute_huff_market_share(self, pandas_dt, alpha, beta_M, beta_REST):

        self.logger.info('Computing local market share. This takes a while ...')
        self.logger.info("Parameters: alpha/beta_M/beta_REST = %f / %f /%f", alpha, beta_M, beta_REST)

        pandas_dt['fahrzeit_huff'] = np.where(pandas_dt['FORMAT'] == 'M', np.power(pandas_dt['fahrzeit'], beta_M), np.power(pandas_dt['fahrzeit'], beta_REST))

        # this is A_i^alpha*fahrzeit^beta
        pandas_dt['huff_numerator'] = np.power(np.power(pandas_dt['vfl'], 2.0), alpha) * pandas_dt['fahrzeit_huff']

        # this is sum(A_i^alpha * fahrzeit^beta)
        pandas_dt['huff_denumerator'] = pandas_dt.groupby('hektar_id')[["huff_numerator"]].transform(lambda x: np.sum(x))

        pandas_dt['local_market_share'] = pandas_dt['huff_numerator'] / pandas_dt['huff_denumerator']

        self.logger.info('Done')
        return pandas_dt

    def entry(self, pandas_dt, config, logger, stores_migros_pd, referenz_pd):
        self.logger = logger
        self.logger.info("Initialized model %s", self.whoami())

        self.process_settings(config)

        pandas_preprocessed_dt = self.preprocess(pandas_dt)
        """
            pandas_preprocessed_dt looks like this now:

            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            |              |OBJECTID |fahrzeit|          ID       |FORMAT|vfl    |RELEVANZ|	Tot_Haushaltausgaben	|Tot_Haushaltausgaben_corrected|
            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            |hektar_id|type|
            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            | 61341718|	MIG|   6     |21      |	SM_MIG_61607_15939|   M  |878.621| 1      |	7800.0                  |   7800                       |
            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        """

        if self.param_sweep:
            self.analysis_sweep(pandas_preprocessed_dt, stores_migros_pd, referenz_pd)
            return 0

        if self.calibrate:
            self.logger.error('Gradient descent is not implemented. Turn off calibration from the settings')
            return 1

        # TODO: Eventually implement gradient descent
        if self.calibrate:
            error = np.zeros(10)
            alpha_next = self.alpha
            beta_next = self.beta
            for t in range(self.calibration_T):
                self.logger.info("Parameters: alpha/beta = %f / %f", alpha_next, beta_next)
                self.compute_huff_market_share(pandas_preprocessed_dt, alpha_next, beta_next)
                # pandas_preprocessed_dt now has a column 'local_market_share' giving the
                # local market share of store i in each hektar
                umsatz_potential_pd = self.gen_umsatz_prognose(pandas_preprocessed_dt, stores_migros_pd, referenz_pd,
                                                               logger)
                # calculate the individual errors
                umsatz_potential_pd['E_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] -
                                                      umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'],
                                                      2) / umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

                self.logger.info("TOTAL ERROR after %d iterations: %f", t, np.sqrt(umsatz_potential_pd.E_i.sum()))
                self.logger.debug("Computing gradients")
                self.calc_gradient(umsatz_potential_pd)
                # umsatz_potential_pd now has two additional columns, 'dE_i_alpha' and 'dE_i_alpha', giving the local
                # gradients for each store. The total gradient is simply the sum
                dE_alpha = umsatz_potential_pd.dE_i_alpha.sum() / np.sqrt(umsatz_potential_pd.E_i.sum())
                dE_beta = umsatz_potential_pd.dE_i_beta.sum() / np.sqrt(umsatz_potential_pd.E_i.sum())

                if t > len(error) and np.diff(error).mean() < self.calibration_delta:
                    self.logger.info("Convergence criteria reached")
                    break

            self.logger.info("DONE CALIBRATION")
            self.logger.info("Generating output csv")
            umsatz_potential_pd.to_csv(self.umsatz_output_csv)

        else:
            pandas_postprocessed_dt = self.compute_huff_market_share(pandas_preprocessed_dt, self.alpha, self.beta)
            # pandas_preprocessed_dt now has a column 'local_market_share' giving the
            # local market share of store i in each hektar
            umsatz_potential_pd = self.gen_umsatz_prognose(pandas_postprocessed_dt, stores_migros_pd, referenz_pd)

            self.logger.info('Exporting Umsatz predictions to csv')
            umsatz_potential_pd.to_csv(self.umsatz_output_csv)

    def analysis_sweep(self, pandas_dt, stores_migros_pd, referenz_pd):
        self.logger.info('Will do parameter sweep. This will take a while, better run over night')
        for a in self.alpha_sweep:
            for b_M in self.beta_M_sweep:
                for b_REST in self.beta_REST_sweep:
                    pandas_sweeped_dt = self.compute_huff_market_share(pandas_dt, a, b_M, b_REST)
                    # pandas_preprocessed_dt now has a column 'local_market_share' giving the
                    # local market share of store i in each hektar
                    umsatz_potential_pd = self.gen_umsatz_prognose(pandas_sweeped_dt, stores_migros_pd, referenz_pd)
                    # calculate the individual errors
                    umsatz_potential_pd['E_i'] = np.power(umsatz_potential_pd['Umsatzpotential'] -
                                                      umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'],
                                                      2) / umsatz_potential_pd[
                                                 'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

                    total_error = np.sqrt(umsatz_potential_pd.E_i.sum())
                    self.logger.info("TOTAL ERROR: %f", total_error)

                    if total_error < self.E_min[0][0]:
                        self.E_min = [(total_error, {"alpha": a, "beta_M": b_M, "beta_REST": b_REST})]
                        self.logger.info('New minimum found.')

                    self.logger.info('Exporting Umsatz predictions to csv')
                    umsatz_potential_pd.to_csv(self.umsatz_output_csv + '_a_' + str(a) + '_bM_' + str(b_M) + "_bREST_" + str(b_REST))

        self.logger.info('Found error minimum of %f for alpha=%f / beta_M=%f / beta_REST=%f',
                         self.E_min[0][0], self.E_min[0][1]["alpha"], self.E_min[0][1]["beta_M"], self.E_min[0][1]["beta_REST"])

    def calc_gradient(self, umsatz_pd):
        # It can always be asszmed that umsatz_pd will have a column 'E_i' with the local errors

        pass

    def gen_umsatz_prognose(self, pandas_pd, stores_migros_pd, referenz_pd):
        self.logger.info('Generating Umsatz predictions ... ')
        pandas_pd['lokal_umsatz_potenzial'] = pandas_pd['Tot_Haushaltausgaben'] * pandas_pd['local_market_share']
        pandas_pd['lokal_umsatz_potenzial_corrected'] = pandas_pd['Tot_Haushaltausgaben_corrected'] * pandas_pd['local_market_share']

        migros_only_pd = pandas_pd[pandas_pd['OBJECTID'].isin(stores_migros_pd.index.values)]

        umsatz_potential_pd = migros_only_pd.groupby('OBJECTID').agg({'ID': lambda x: x.iloc[0],
                                                                      'lokal_umsatz_potenzial': lambda x: np.nansum(x),
                                                                      'lokal_umsatz_potenzial_corrected': lambda x: np.nansum(x)
                                                                      })

        umsatz_potential_pd = umsatz_potential_pd.rename(columns={'lokal_umsatz_potenzial': 'Umsatzpotential',
                                                                  'lokal_umsatz_potenzial_corrected': 'Umsatzpotential_corrected'})

        umsatz_potential_pd = umsatz_potential_pd.merge(referenz_pd, left_index=True, right_index=True, how='inner')
        umsatz_potential_pd['verhaeltnis_tU'] = umsatz_potential_pd['Umsatzpotential'] / \
                                                umsatz_potential_pd[
                                                    'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

        umsatz_potential_pd['verhaeltnis_tU_prozent'] = (umsatz_potential_pd['Umsatzpotential'] -
                                                         umsatz_potential_pd[
                                                             'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']) / \
                                                        umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

        umsatz_potential_pd['verhaeltnis_MP2'] = umsatz_potential_pd['Umsatzpotential'] / \
                                                 umsatz_potential_pd['MP - CALCULATED_REVENUE 2']

        return umsatz_potential_pd



