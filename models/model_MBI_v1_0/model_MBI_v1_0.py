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
    slope_rlat = None
    fahrzeit_cutoff = None

    # öV parameters
    beta_ov = None
    f_pendler = None
    pendler_ausgaben = None

    # bacis parameter sweep
    param_basic_sweep = None
    slope_lat_sweep = None
    slope_rlat_sweep = None
    fahrzeit_cutoff_sweep = None

    # ov parameter sweep
    param_ov_sweep = None
    beta_ov_sweep = None
    f_pendler_sweep = None
    pendler_ausgaben_sweep = None

    # the minimum error after the parameter sweep
    E_min = [(float("inf"), ())]

    def whoami(self):
        return 'Model_MBI_v1.0'

    def process_settings(self, config):
        # first check if we are doing a parameter sweep
        self.param_basic_sweep = False  # by default False
        self.param_ov_sweep = False  # by default False
        if config.has_option('parameter_basic_sweep', 'slope_lat') and \
           config.has_option('parameter_basic_sweep', 'slope_rlat') and \
           config.has_option('parameter_basic_sweep', 'fahrzeit_cutoff'):
            self.slope_lat_sweep = [float(x) for x in json.loads(config.get('parameter_basic_sweep', 'slope_lat'))]
            self.slope_rlat_sweep = [float(x) for x in json.loads(config.get('parameter_basic_sweep', 'slope_rlat'))]
            self.fahrzeit_cutoff_sweep = [float(x) for x in json.loads(config.get('parameter_basic_sweep',
                                                                                  'fahrzeit_cutoff'))]
            self.param_basic_sweep = True
        ####
        if config.has_option('parameters_ov_sweep', 'beta_ov_sweep') and \
           config.has_option('parameters_ov_sweep', 'f_pendler_sweep') and \
           config.has_option('parameters_ov_sweep', 'pendler_ausgaben_sweep'):
            self.beta_ov_sweep = [float(x) for x in json.loads(config.get('parameters_ov_sweep', 'beta_ov_sweep'))]
            self.f_pendler_sweep = [float(x) for x in json.loads(config.get('parameters_ov_sweep', 'f_pendler_sweep'))]
            self.pendler_ausgaben_sweep = [float(x) for x in json.loads(config.get('parameters_ov_sweep',
                                                                                   'pendler_ausgaben_sweep'))]
            self.param_ov_sweep = True
        #
        if self.param_ov_sweep and self.param_basic_sweep:
            self.logger.error('Parameter sweep over both basic and ov parameters not yet implemented')
            sys.exit(1)
        try:
            self.slope_lat = float(config["parameters"]["slope_lat"])
            self.slope_rlat = float(config["parameters"]["slope_rlat"])
            self.fahrzeit_cutoff = float(config["parameters"]["fahrzeit_cutoff"])
            #
            self.beta_ov = float(config["parameters"]["beta_ov"])
            self.f_pendler = float(config["parameters"]["f_pendler"])
            self.pendler_ausgaben = float(config["parameters"]["pendler_ausgaben"])
            #
            self.umsatz_output_csv = config["output"]["output_csv"]
        except Exception:
            self.logger.error('Some of the required parameters for model %s are not supplied in the settings',
                              self.whoami())
            sys.exit(1)


    def compute_market_share(self, pandas_dt, slope_lat, slope_rlat, fahrzeit_cutoff):

        self.logger.info('Computing local market share. This takes a while ...')
        self.logger.info("Parameters: slope_lat/slope_rlat/fahrzeit_cutoff = %f / %f / %f", slope_lat, slope_rlat,
                         fahrzeit_cutoff)

        pandas_dt['LAT'] = slope_lat *  pandas_dt['VFL']
        pandas_dt['RLAT'] = pandas_dt['LAT'] * np.power(10,
                                                        slope_rlat * np.fmax(pandas_dt['FZ'] - fahrzeit_cutoff,
                                                                             0))

        self.logger.info('Computing sum RLATs ...')
        pandas_dt['sumRLATs'] = pandas_dt.groupby('velo_StartHARasterID')[["RLAT"]].transform(lambda x: np.sum(x))
        pandas_dt['LMA'] = pandas_dt['RLAT'] / pandas_dt['sumRLATs']

        self.logger.info('Done')
        return pandas_dt

    def entry(self, tables_dict, config, logger):
        self.logger = logger
        self.logger.info("Initialized model %s", self.whoami())

        self.process_settings(config)

        # check if the model got the right dict with input tables
        # TODO: super basic check for names only. Handle more robustly
        try:
            pandas_dt = tables_dict["all_stores"]
            stations_pd = tables_dict["sbb_stations"]
        except:
            logger.error('Model %s did not get the right input table names. Expected %s got %s',
                        self.whoami(), "['all_stores', 'sbb_stations']", list(tables_dict.keys()))
            sys.exit(1)

        if self.param_basic_sweep:
            self.analysis_sweep(pandas_dt)
            return 0

        pandas_postprocessed_dt = self.compute_market_share(pandas_dt, self.slope_lat, self.slope_rlat,
                                                            self.fahrzeit_cutoff)

        # --- HAUSHALT component -----------------------------------------------------------------------------------
        # pandas_preprocessed_dt now has a column 'LMA'
        # giving the local market share of store i
        # in each hektar from which i can be reached
        umsatz_potential_pd = self.gen_umsatz_prognose(pandas_postprocessed_dt)

        # pendler einfluss is modelled after the Haushalt part
        if self.param_ov_sweep:
            self.analysis_ov_sweep(umsatz_potential_pd, stations_pd)
            return 0

        pendler_einfluss_pd = self.calc_zusaetzliche_kauefer(pandas_postprocessed_dt, stations_pd, self.beta_ov,
                                                             self.f_pendler)
        # left join between the calculated umsatz and the pendler einfluss
        umsatz_potential_pd = pd.merge(umsatz_potential_pd, pendler_einfluss_pd, how='left', left_index=True,

                                       right_index=True)

        umsatz_potential_pd['Umsatz_Pendler'] = umsatz_potential_pd['additional_kaeufer'] * self.pendler_ausgaben
        umsatz_potential_pd['Umsatz_Total'] = np.where(np.isnan(umsatz_potential_pd['Umsatz_Pendler']),
                                                       umsatz_potential_pd['Umsatz_Haushalte'],
                                                       umsatz_potential_pd['Umsatz_Haushalte'] +
                                                       umsatz_potential_pd['Umsatz_Pendler'])

        umsatz_potential_pd['percentage_istU_modelU'] = (umsatz_potential_pd['Umsatz_Total'] -
                                                                 umsatz_potential_pd['istUmsatz']) / \
            umsatz_potential_pd['istUmsatz']

        # re-order only the relevant columns
        column_order = ['StoreName', 'Retailer', 'Format', 'VFL', 'Adresse', 'PLZ', 'Ort',
                        'HARasterID', 'E_LV03', 'N_LV03', 'ProfitKSTID', 'Food', 'Frische', 'Near/Non Food',
                        'Fachmaerkte', 'additional_kaeufer', 'istUmsatz', 'Umsatz_Haushalte', 'Umsatz_Pendler',
                        'Umsatz_Total']

        umsatz_potential_pd = umsatz_potential_pd[column_order]
        self.logger.info('Exporting Umsatz predictions to csv')
        umsatz_potential_pd.to_csv(self.umsatz_output_csv)

    def calc_zusaetzliche_kauefer(self, stores_pd, stations_pd, beta, f):
        def get_reachable_stations(group, radius):
            row = group.iloc[0]
            # Calculate squared distance (Euclidean).
            distanz_squared = np.power(stations_pd['E_LV03'] - row['E_LV03'], 2) + \
                              np.power(stations_pd['N_LV03'] - row['N_LV03'], 2)

            within_circle = distanz_squared <= np.power(radius, 2)
            return_pd = pd.DataFrame(stations_pd[within_circle])
            return_pd['StoreID'] = row['StoreID']
            return_pd['distanz'] = np.sqrt(distanz_squared[within_circle])
            return return_pd.set_index(['StoreID'])

        def calc_pendler_wahrscheinlichkeit(group, beta):
            partition_function = np.sum(np.exp(-beta * group['distanz']))
            probabilities = np.exp(-beta * group['distanz']) / partition_function

            rv = pd.DataFrame({'StoreID': group['StoreID'],
                               'code': group['Code'],
                               'bahnhof': group['Bahnhof_Haltestelle'],
                               'distanz': group['distanz'],
                               'DTV': group['DTV'],
                               'DWV': group['DWV'],
                               'pendler_wahrscheinlichkeit': probabilities})
            return rv.set_index(['StoreID'])
        #
        self.logger.info('Computing Pendler einfluss ...')
        self.logger.info("Parameters: beta/f_pendler = %f / %f", beta, f)

        x = pd.DataFrame(stores_pd.reset_index().groupby('StoreID',
                                                         group_keys=False).apply(get_reachable_stations, 300))

        '''
            'x' looks like this now:

                    Code	Bahnhof_Haltestelle	[...]    lat      lon	    E_LV03	N_LV03	distanz
        StoreID
            8	    MI  	Muri AG	                 47.276660	8.339798	668185	236587	538.396460
            9	    BURI	Burier	                 46.447876	6.877116	556853	144215	1456.600536
        '''

        y = x.reset_index().groupby('Code', group_keys=False).apply(calc_pendler_wahrscheinlichkeit, beta)
        y['additional_kaeufer'] = f * y['DWV'] * y['pendler_wahrscheinlichkeit']

        '''
            'y' looks like this now:

                    DTV 	DWV 	bahnhof	    code	distanz	    pendler_wahrscheinlichkeit	additional_kauefer
        StoreID
            9	    970	    1300	Burier	    BURI	1456.600536	    0.062500	                81.250000
            9	    1900	2200	Clarens	    CL	    240.076282	    0.083333	                183.333333
        '''

        # now aggregate over all stores
        return y.reset_index().groupby('StoreID',
                                       as_index=False)['additional_kaeufer'].aggregate(np.sum).set_index('StoreID')

    def analysis_ov_sweep(self, umsatz_dt, stores_pd, referenz_pd, stations_pd):

        self.logger.info('Will do oV parameter sweep. This will take a while, better run over night')
        for beta in self.beta_ov_sweep:
            for f_pendler in self.f_pendler_sweep:
                for pendler_ausgaben in self.pendler_ausgaben_sweep:
                    pendler_einfluss_pd = self.calc_zusaetzliche_kauefer(stores_pd, stations_pd, beta, f_pendler)

                    # left join between the calculated umsatz and the pendler einfluss
                    umsatz_potential_pd = pd.merge(umsatz_dt, pendler_einfluss_pd,
                                                   how='left', left_index=True, right_index=True)

                    umsatz_potential_pd['umsatz_with_pendler'] = umsatz_potential_pd['Umsatz_Haushalte'] + \
                        umsatz_potential_pd['additional_kaeufer'] * pendler_ausgaben

                    umsatz_potential_pd.loc[np.isnan(umsatz_potential_pd['umsatz_with_pendler']),
                                            'umsatz_with_pendler'] = \
                        umsatz_potential_pd[np.isnan(umsatz_potential_pd['umsatz_with_pendler'])]['Umsatz_Haushalte']

                    umsatz_potential_pd['verhaeltnis_tU_pendler_prozent'] = \
                        (umsatz_potential_pd['umsatz_with_pendler'] -
                         umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']) / \
                        umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

                    # calculate the individual errors
                    umsatz_potential_pd['E_i'] = np.power(umsatz_potential_pd['umsatz_with_pendler'] -
                                                          umsatz_potential_pd['Tatsechlicher Umsatz - '
                                                                              'FOOD_AND_FRISCHE'], 2) / \
                        umsatz_potential_pd['Tatsechlicher Umsatz - FOOD_AND_FRISCHE']

                    total_error = np.sqrt(umsatz_potential_pd.E_i.sum())

                    error_quantile = umsatz_potential_pd[~np.isnan(umsatz_potential_pd['E_i'])]['E_i'].quantile(q=0.99)
                    total_error_0_99 = np.sqrt(umsatz_potential_pd[umsatz_potential_pd['E_i'] <=
                                                                   error_quantile]['E_i'].sum())
                    self.logger.info("TOTAL ERROR: %f / %f", total_error, total_error_0_99)

                    if total_error_0_99 < self.E_min[0][0]:
                        self.E_min = [(total_error, {"beta": beta, "f_pendler": f_pendler,
                                                     "pendler_ausgaben": pendler_ausgaben})]
                        self.logger.info('New minimum found.')

                    self.logger.info('Exporting Umsatz predictions to csv')

                    umsatz_potential_pd.to_csv(self.umsatz_output_csv + '_beta_' + str(beta) + '_fPendler_' +
                                               str(f_pendler) + '_pAusgaben_' + str(pendler_ausgaben))

        self.logger.info('Found error minimum of %f for beta=%f / f_pendler=%f / pendler_ausgaben=%f',
                         self.E_min[0][0], self.E_min[0][1]["beta"], self.E_min[0][1]["f_pendler"],
                         self.E_min[0][1]["pendler_ausgaben"])

    def analysis_sweep(self, pandas_dt):
        self.logger.info('Will do basic parameter sweep. This will take a while, better run over night')

        for lat in self.slope_lat_sweep:
            for rlat in self.slope_rlat_sweep:
                for fz in self.fahrzeit_cutoff_sweep:
                    pandas_sweeped_dt = self.compute_market_share(pandas_dt, lat, rlat, fz)
                    # pandas_preprocessed_dt now has a column 'local_market_share' giving the
                    # local market share of store i in each hektar
                    umsatz_potential_pd = self.gen_umsatz_prognose(pandas_sweeped_dt)
                    # calculate the individual errors
                    umsatz_potential_pd['E_i'] = np.power(umsatz_potential_pd['Umsatz_Haushalte'] -
                                                          umsatz_potential_pd['istUmsatz'], 2) / \
                        umsatz_potential_pd['istUmsatz']

                    total_error = np.sqrt(umsatz_potential_pd.E_i.sum())
                    error_quantile = umsatz_potential_pd[~np.isnan(umsatz_potential_pd['E_i'])]['E_i'].quantile(q=0.99)
                    total_error_0_99 = np.sqrt(umsatz_potential_pd[umsatz_potential_pd['E_i'] <=
                                                                   error_quantile]['E_i'].sum())

                    self.logger.info("TOTAL ERROR: %f / %f", total_error, total_error_0_99)

                    if total_error_0_99 < self.E_min[0][0]:
                        self.E_min = [(total_error, {"lat": lat, "rlat": rlat, "fz_cutoff": fz})]
                        self.logger.info('New minimum found.')

                    self.logger.info('Exporting Umsatz predictions to csv')

                    umsatz_potential_pd.to_csv(self.umsatz_output_csv + '_lat_' + str(lat) + '_rlat_' + str(rlat) +
                                               '_fz_' + str(fz))

        self.logger.info('Found error minimum of %f for lat=%f / rlat=%f / fz_cutoff=%f',
                         self.E_min[0][0], self.E_min[0][1]["lat"], self.E_min[0][1]["rlat"],
                         self.E_min[0][1]["fz_cutoff"])

    def gen_umsatz_prognose(self, pandas_pd):
        self.logger.info('Generating Umsatz predictions ... ')
        pandas_pd['lokal_umsatz_potenzial'] = pandas_pd['Tot_Haushaltausgaben'] * pandas_pd['LMA']

        # get only the Migros stores - StoreID between 0 and 9999
        umsatz_potential_pd = pandas_pd.loc[pandas_pd.StoreID < 10000].groupby('StoreID').agg({
                                                                         'StoreName': lambda x: x.iloc[0],
                                                                         'Retailer': lambda x: x.iloc[0],
                                                                         'Format': lambda x: x.iloc[0],
                                                                         'VFL': lambda x: x.iloc[0],
                                                                         'Adresse': lambda x: x.iloc[0],
                                                                         'PLZ': lambda x: x.iloc[0],
                                                                         'Ort': lambda x: x.iloc[0],
                                                                         'HARasterID': lambda x: x.iloc[0],
                                                                         'E_LV03': lambda x: x.iloc[0],
                                                                         'N_LV03': lambda x: x.iloc[0],
                                                                         'ProfitKSTID': lambda x: x.iloc[0],
                                                                         'Food': lambda x: x.iloc[0],
                                                                         'Frische': lambda x: x.iloc[0],
                                                                         'Near/Non Food': lambda x: x.iloc[0],
                                                                         'Fachmaerkte': lambda x: x.iloc[0],
                                                                    'lokal_umsatz_potenzial': lambda x: np.nansum(x),
                                                                      })
        # stupid Pandas is shuffling the columns for some reason, we need to rename them
        column_order = ['StoreName', 'Retailer', 'Format', 'VFL', 'Adresse', 'PLZ', 'Ort',
                        'HARasterID', 'E_LV03', 'N_LV03', 'ProfitKSTID', 'Food', 'Frische', 'Near/Non Food',
                        'Fachmaerkte', 'lokal_umsatz_potenzial']
        umsatz_potential_pd = umsatz_potential_pd[column_order]
        umsatz_potential_pd = umsatz_potential_pd.rename(columns={'lokal_umsatz_potenzial': 'Umsatz_Haushalte'})

        # sum up the relevant Umsaetze
        umsatz_potential_pd['istUmsatz'] = umsatz_potential_pd['Food']+umsatz_potential_pd['Frische']+\
                                           umsatz_potential_pd['Near/Non Food']

        umsatz_potential_pd['verhaeltnis_tU'] = umsatz_potential_pd['Umsatz_Haushalte'] / \
                                                (umsatz_potential_pd['istUmsatz'])

        umsatz_potential_pd['verhaeltnis_tU_prozent'] = (umsatz_potential_pd['Umsatz_Haushalte'] -
                                                         umsatz_potential_pd['istUmsatz']) / \
                                                        umsatz_potential_pd['istUmsatz']

        return umsatz_potential_pd
