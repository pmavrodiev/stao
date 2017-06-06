import json

import numpy as np
import pandas as pd

from models.model_base import ModelBase


@ModelBase.register
class model_Huff(ModelBase):

    def whoami(self):
        return 'Model_Huff'

    def process_settings(self, config):
        # first check if we are doing a parameter sweep over a and b
        self.param_sweep = False  # by default False
        if config.has_option('parameter_sweep', 'a_array') and config.has_option('parameter_sweep', 'b_array'):
            self.a_sweep = json.loads(config.get('parameter_sweep', 'a_array'))
            self.b_sweep = json.loads(config.get('parameter_sweep', 'b_array'))
            self.param_sweep = True
        ####

        try:
            self.a = float(config["calibration"]["a_start"])
            self.b = float(config["calibration"]["b_start"])
            self.umsatz_output_csv = config["output"]["output_csv"]
            self.output_pickle = config["output"]["output_pickle"]
            self.direct_output = config.getboolean('calibration', 'direct_output')
            self.calibration_T = int(config["calibration"]["T"])
            self.calibration_delta = float(config["calibration"]["delta_convergence"])
        except Exception:
            pass # TODO implement error handling

    def preprocess(self, pandas_dt):

        self.logger.info("Reindexing ...")
        pandas_reindexed_dt = pandas_dt.reset_index().set_index(keys=['hektar_id', 'type', 'OBJECTID'])
        self.logger.info("Removing duplicates ...")
        # remove the duplicates introduced after merging drivetimes and store information
        pandas_reindexed_dt = pandas_reindexed_dt[~pandas_reindexed_dt.index.duplicated(keep='first')]
        pandas_reindexed_dt = pandas_reindexed_dt.reset_index().set_index(keys=['hektar_id', 'type'])
        return pandas_reindexed_dt

    def compute_huff_market_share(self, pandas_dt, alpha, beta):
        self.logger.info('Computing local market share')
        pandas_dt['huff_numerator'] = np.power(np.power(pandas_dt['vfl'], 2.0), alpha) * np.power(pandas_dt['fahrzeit'], -beta)
        # TODO this will probably fail
        pandas_dt['huff_denumerator'] = pandas_dt.groupby('hektar_id')[["vfl", "fahrzeit"]].transform(
            lambda x: np.sum(np.power(x[0], alpha) * np.power(x[1], -beta)))
        #
        pandas_dt['local_market_share'] = pandas_dt['huff_numerator'] / pandas_dt['huff_denumerator']


    def entry(self, pandas_dt, config, logger, stores_migros_pd, referenz_pd):
        self.logger = logger
        self.logger.info("Initialized model %s", self.whoami())

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

        pandas_postprocessed_dt = self.compute_huff_market_share(pandas_preprocessed_dt, alpha, beta)