import json

import numpy as np
import pandas as pd
import sys

from models.model_base import ModelBase


@ModelBase.register
class model_MBI_v_1_0(ModelBase):
    logger = None
    umsatz_output_csv = None
    # basic parameters
    slope_lat = None
    slope_rlat_M = None
    slope_rlat_REST = None
    fahrzeit_cutoff = None

    # öV parameters
    beta_ov = None
    f_pendler = None

    # bacis parameter sweep
    param_basic_sweep = None
    slope_lat_sweep = None
    slope_rlat_sweep = None
    fahrzeit_cutoff_sweep = None

    # ov parameter sweep
    param_ov_sweep = None
    beta_ov_sweep = None
    f_pendler_sweep = None

    # the minimum error after the parameter sweep
    E_min = [(float("inf"), ())]

    def whoami(self):
        return 'Model_MBI_v1.0'

    def process_settings(self, config):
        # first check if we are doing a parameter sweep
        self.param_basic_sweep = False  # by default False
        self.param_ov_sweep = False  # by default False
        if config.has_option('parameter_basic_sweep', 'slope_lat')       and \
           config.has_option('parameter_basic_sweep', 'slope_rlat_M')    and \
           config.has_option('parameter_basic_sweep', 'slope_rlat_REST') and \
           config.has_option('parameter_basic_sweep', 'fahrzeit_cutoff'):
            self.slope_lat_sweep = [float(x) for x in json.loads(config.get('parameter_basic_sweep', 'slope_lat'))]
            self.slope_rlat_M_sweep = [float(x) for x in json.loads(config.get('parameter_basic_sweep', 'slope_rlat_M'))]
            self.slope_rlat_REST_sweep = [float(x) for x in json.loads(config.get('parameter_basic_sweep', 'slope_rlat_REST'))]
            self.fahrzeit_cutoff_sweep = [float(x) for x in json.loads(config.get('parameter_basic_sweep', 'fahrzeit_cutoff'))]
            self.param_basic_sweep = True
        ####
        if config.has_option('parameters_ov_sweep', 'beta_ov_sweep') and \
           config.has_option('parameters_ov_sweep', 'f_pendler_sweep'):
            self.beta_ov_sweep = [float(x) for x in json.loads(config.get('parameters_ov_sweep', 'beta_ov_sweep'))]
            self.f_pendler_sweep = [float(x) for x in json.loads(config.get('parameters_ov_sweep', 'f_pendler_sweep'))]
            self.param_ov_sweep = True
        #
        if self.param_ov_sweep and self.param_basic_sweep:
            self.logger.error('Parameter sweep over basic and ov parameters not yet implemented')
            sys.exit(1)
        try:
            self.slope_lat = float(config["parameters_basic"]["slope_lat"])
            self.slope_rlat_M = float(config["parameters_basic"]["slope_rlat_M"])
            self.slope_rlat_REST = float(config["parameters_basic"]["slope_rlat_REST"])
            self.fahrzeit_cutoff = float(config["parameters_basic"]["fahrzeit_cutoff"])
            #
            self.beta_ov = float(config["parameters_ov"]["beta_ov"])
            self.f_pendler = float(config["parameters_ov"]["f_pendler"])
            #
            self.umsatz_output_csv = config["output"]["output_csv"]
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

    def compute_market_share(self, pandas_dt, slope_lat, slope_rlat_M, slope_rlat_REST, fahrzeit_cutoff):

        self.logger.info('Computing local market share. This takes a while ...')
        self.logger.info("Parameters: slope_lat/slope_rlat_M/slope_rlat_REST/fahrzeit_cutoff = %f / %f / %f / %f", slope_lat, slope_rlat_M, slope_rlat_REST,
                         fahrzeit_cutoff)

        pandas_dt['LAT'] = slope_lat * pandas_dt['RELEVANZ'] * pandas_dt['vfl']
        pandas_dt['RLAT'] = pandas_dt['LAT'] * np.where(pandas_dt['FORMAT'] == 'M', np.power(10, slope_rlat_M * np.fmax(pandas_dt['fahrzeit'] - fahrzeit_cutoff, 0)), np.power(10, slope_rlat_REST * np.fmax(pandas_dt['fahrzeit'] - fahrzeit_cutoff, 0)))

        self.logger.info('Computing sum RLATs ...')
        pandas_dt['sumRLATs'] = pandas_dt.groupby('hektar_id')[["RLAT"]].transform(lambda x: np.sum(x))
        pandas_dt['local_market_share'] = pandas_dt['RLAT'] / pandas_dt['sumRLATs']

        self.logger.info('Done')
        return pandas_dt

    def entry(self, pandas_dt, config, logger, stores_migros_pd, referenz_pd, stations_pd):
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

        if self.param_basic_sweep:
            self.analysis_sweep(pandas_preprocessed_dt, stores_migros_pd, referenz_pd)
            return 0

        pandas_postprocessed_dt = self.compute_market_share(pandas_preprocessed_dt, self.slope_lat, self.slope_rlat_M,
                                                            self.slope_rlat_REST, self.fahrzeit_cutoff)
        # pandas_preprocessed_dt now has a column 'local_market_share' giving the
        # local market share of store i in each hektar
        umsatz_potential_pd = self.gen_umsatz_prognose(pandas_postprocessed_dt, stores_migros_pd, referenz_pd)

        # pendler einfluss is modelled as a last step of the umsatz prognose
        if self.param_ov_sweep:
            self.analysis_ov_sweep(umsatz_potential_pd, stores_migros_pd, referenz_pd, stations_pd)
            return 0

        pendler_einfluss_pd = self.calc_pendler_einfluss(stores_migros_pd, stations_pd, self.beta_ov, self.f_pendler)
        # left join between the calculated umsatz and the pendler einfluss
        umsatz_potential_pd = pd.merge(umsatz_potential_pd, pendler_einfluss_pd, \
                                       how='left', left_index=True, right_index=True)

        umsatz_potential_pd['umsatz_with_pendler'] = umsatz_potential_pd['Umsatzpotential'] + \
                                                     umsatz_potential_pd['additional_kaeufer'] * 50

        umsatz_potential_pd['verhaeltnis_tU_pendler_prozent'] = (umsatz_potential_pd['umsatz_with_pendler'] -
                                                         umsatz_potential_pd[
                                                             'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']) / \
                                                        umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

        self.logger.info('Exporting Umsatz predictions to csv')
        umsatz_potential_pd.to_csv(self.umsatz_output_csv)

    def calc_pendler_einfluss(self, stores_migros_pd, stations_pd, beta, f):
        def get_reachable_stations(group, radius):
            row = group.iloc[0]
            distanz_squared = np.power(stations_pd['E_LV03'] - row['Y'], 2) + np.power(
                stations_pd['N_LV03'] - row['X'], 2)
            within_circle = distanz_squared <= np.power(radius, 2)
            return_pd = pd.DataFrame(stations_pd[within_circle])
            return_pd['OBJECTID'] = row['OBJECTID']
            return_pd['distanz'] = np.sqrt(distanz_squared[within_circle])
            return return_pd.set_index(['OBJECTID'])
        #
        def calc_pendler_einfluss(group, beta, f):
            row = group.iloc[0]
            einfluss = np.sum(f * group['DWV'] * np.exp(-beta * group['distanz']))
            rv = pd.DataFrame({'OBJECTID': [row['OBJECTID']], 'additional_kaeufer': [einfluss]})
            return rv.set_index(['OBJECTID'])
        #
        self.logger.info('Computing Pendler einfluss ...')
        self.logger.info("Parameters: beta/f_pendler = %f / %f", beta, f)

        x = pd.DataFrame(stores_migros_pd.reset_index().groupby('OBJECTID', group_keys=False).apply(get_reachable_stations, 300))
        x = x.reset_index().groupby('OBJECTID', group_keys=False).apply(calc_pendler_einfluss, beta, f)
        return x

    def analysis_ov_sweep(self, umsatz_dt, stores_migros_pd, referenz_pd, stations_pd):

        self.logger.info('Will do oV parameter sweep. This will take a while, better run over night')
        for beta in self.beta_ov_sweep:
            for f_pendler in self.f_pendler_sweep:
                pendler_einfluss_pd = self.calc_pendler_einfluss(stores_migros_pd, stations_pd, beta, f_pendler)
                # left join between the calculated umsatz and the pendler einfluss
                umsatz_potential_pd = pd.merge(umsatz_dt, pendler_einfluss_pd, \
                                               how='left', left_index=True, right_index=True)
                umsatz_potential_pd['umsatz_with_pendler'] = umsatz_potential_pd['Umsatzpotential'] + \
                                                             umsatz_potential_pd['additional_kaeufer'] * 50

                umsatz_potential_pd['verhaeltnis_tU_pendler_prozent'] = (umsatz_potential_pd['umsatz_with_pendler'] -
                                                                         umsatz_potential_pd[
                                                                             'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']) / \
                                                                        umsatz_potential_pd[
                                                                            'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
                # calculate the individual errors
                umsatz_potential_pd['E_i'] = np.power(umsatz_potential_pd['umsatz_with_pendler'] -
                                                      umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE'],
                                                      2) / umsatz_potential_pd[
                                                 'Tatsechlicher Umsatz - FOOD_AND_FRISCHE']
                total_error = np.sqrt(umsatz_potential_pd.E_i.sum())
                self.logger.info("TOTAL ERROR: %f", total_error)

                if total_error < self.E_min[0][0]:
                    self.E_min = [(total_error, {"beta": beta, "f_pendler": f_pendler})]
                    self.logger.info('New minimum found.')

                self.logger.info('Exporting Umsatz predictions to csv')

                umsatz_potential_pd.to_csv(self.umsatz_output_csv + '_beta_' + str(beta) + '_fPendler_' + str(f_pendler))
        self.logger.info('Found error minimum of %f for beta=%f / f_pendler=%f',
                         self.E_min[0][0], self.E_min[0][1]["beta"], self.E_min[0][1]["f_pendler"])


    def analysis_sweep(self, pandas_dt, stores_migros_pd, referenz_pd):
        self.logger.info('Will do basic parameter sweep. This will take a while, better run over night')

        for lat in self.slope_lat_sweep:
            for rlat_M in self.slope_rlat_M_sweep:
                for fz in self.fahrzeit_cutoff_sweep:
                    rlat_REST = 0.5 * rlat_M
                    pandas_sweeped_dt = self.compute_market_share(pandas_dt, lat, rlat_M, rlat_REST, fz)
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
                        self.E_min = [(total_error, {"lat": lat, "rlat_M": rlat_M, "rlat_REST": rlat_REST, "fz_cutoff": fz})]
                        self.logger.info('New minimum found.')

                    self.logger.info('Exporting Umsatz predictions to csv')

                    umsatz_potential_pd.to_csv(self.umsatz_output_csv + '_lat_' + str(lat) + '_rlatM_' + str(rlat_M) +
                                               '_RLATREST_' + str(rlat_REST) + '_fz_' + str(fz))

        self.logger.info('Found error minimum of %f for lat=%f / rlat_M=%f / rlat_REST=%f / fz_cutoff=%f',
                         self.E_min[0][0], self.E_min[0][1]["lat"], self.E_min[0][1]["rlat_M"],self.E_min[0][1]["rlat_REST"],
                         self.E_min[0][1]["fz_cutoff"])

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



