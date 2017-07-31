import json

import numpy as np
import pandas as pd
import sys

from models.model_base import ModelBase


@ModelBase.register
class model_MBI_v1_2(ModelBase):

    """
        This subclass implements the calculation of all the different components of the modell turnover.

        For example calculating the turnover components due to Haushaltausgaben and commuters is implemented
        in the methods 'calc_umsatz_haushalt' and 'calc_umsatz_oev'.

        The idea is that the resulting Pandas dataframe will have one column for each turnover component, which
        can then by either directly summed up or used as a basis for the regression part.

        While each turnover component could be implemented in its own method in the base class,
        a subclass provides better encapsulation and easier code maintenance.
    """
    class _umsatz_components_:
        def __init__(self, parent_logger):
            self.logger = parent_logger

        def calc_umsatz_haushalt(self, pandas_pd, parameters):
            def compute_market_share(pandas_dt, parameters):
                slope_lat = parameters["lat"]
                slope_rlat = parameters["rlat"]
                fahrzeit_cutoff = parameters["fz_cutoff"]

                self.logger.info('Computing local market share. This takes a while ...')
                self.logger.info("Parameters: slope_lat/slope_rlat/fahrzeit_cutoff = %f / %f / %f", slope_lat,
                                 slope_rlat,
                                 fahrzeit_cutoff)

                pandas_dt['LAT'] = slope_lat * pandas_dt['VFL']
                pandas_dt['RLAT'] = pandas_dt['LAT'] * np.power(10,
                                                                slope_rlat * np.fmax(pandas_dt['FZ'] - fahrzeit_cutoff,
                                                                                     0))

                self.logger.info('Computing sum RLATs ...')
                pandas_dt['sumRLATs'] = pandas_dt.groupby('velo_StartHARasterID')[["RLAT"]].transform(
                    lambda x: np.sum(x))
                pandas_dt['LMA'] = pandas_dt['RLAT'] / pandas_dt['sumRLATs']

                self.logger.info('Done')
                return pandas_dt

            def gen_umsatz_prognose(pandas_pd):
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
                umsatz_potential_pd['istUmsatz'] = umsatz_potential_pd['Food'] + umsatz_potential_pd['Frische'] + \
                                                   umsatz_potential_pd['Near/Non Food']

                umsatz_potential_pd['verhaeltnis_tU'] = umsatz_potential_pd['Umsatz_Haushalte'] / \
                                                        (umsatz_potential_pd['istUmsatz'])

                umsatz_potential_pd['verhaeltnis_tU_prozent'] = (umsatz_potential_pd['Umsatz_Haushalte'] -
                                                                 umsatz_potential_pd['istUmsatz']) / \
                                                                umsatz_potential_pd['istUmsatz']

                return umsatz_potential_pd

            pandas_postprocessed_dt = compute_market_share(pandas_pd, parameters)
            umsatz_potential_pd = gen_umsatz_prognose(pandas_postprocessed_dt)
            return umsatz_potential_pd

        def calc_umsatz_oev(self, pandas_pd, parameters):
            def calc_zusaetzliche_kauefer(stores_pd, stations_pd, beta, f):
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
                                               as_index=False)['additional_kaeufer'].aggregate(np.sum).set_index(
                    'StoreID')

            beta_ov = parameters["beta_ov"]
            f_pendler = parameters["f_pendler"]
            stations_pd = parameters["stations_pd"]

            pendler_einfluss_pd = calc_zusaetzliche_kauefer(pandas_pd, stations_pd, beta_ov, f_pendler)
            pendler_einfluss_pd['Umsatz_Pendler'] = pendler_einfluss_pd['additional_kaeufer'] * 50.0

            return pendler_einfluss_pd

    logger = None
    umsatz_output_csv = None

    # all model parameters go into this dictionary
    parameters = {}

   # the minimum error after the parameter sweep
    E_min = [(float("inf"), ())]

    def whoami(self):
        return 'Model_MBI_v1.2'

    def process_settings(self, config):
        try:
            # Parameters Haushaltausgaben
            self.parameters["slope_lat"] = [float(x) for x in json.loads(config["parameters"]["slope_lat"])]
            self.parameters["slope_rlat"] = [float(x) for x in json.loads(config["parameters"]["slope_rlat"])]
            self.parameters["fahrzeit_cutoff"] = [float(x) for x in json.loads(config["parameters"]["fahrzeit_cutoff"])]
            # Parameters oeV
            self.parameters["beta_ov"] = [float(x) for x in json.loads(config["parameters"]["beta_ov"])]
            self.parameters["f_pendler"] = [float(x) for x in json.loads(config["parameters"]["f_pendler"])]
            #
            self.umsatz_output_csv = config["output"]["output_csv"]
        except Exception:
            self.logger.error('Some of the required parameters for model %s are not supplied in the settings',
                              self.whoami())
            sys.exit(1)

    def entry(self, tables_dict, config, logger):
        self.logger = logger
        self.logger.info("Initialized model %s", self.whoami())

        # initialize the
        self.umsatz_components = self._umsatz_components_(logger)

        self.process_settings(config)

        # check if the model got the right dict with input tables
        # TODO: super basic check for names only. Handle more robustly
        try:
            pandas_dt = tables_dict["all_stores"]
            self.parameters["stations_pd"] = tables_dict["sbb_stations"]
        except:
            logger.error('Model %s did not get the right input table names. Expected %s got %s',
                        self.whoami(), "['all_stores', 'sbb_stations']", list(tables_dict.keys()))
            sys.exit(1)


        # Loop over the basic parameters
        for lat in self.parameters["slope_lat"]:
            for rlat in self.parameters["slope_rlat"]:
                for fz_cutoff in self.parameters["fahrzeit_cutoff"]:
                    # ---- 1. UMSATZ_HAUSHALTE
                    umsatz_haushalte_pd = self.umsatz_components.calc_umsatz_haushalt(pandas_dt,
                                                                                        {"lat": lat,
                                                                                         "rlat": rlat,
                                                                                         "fz_cutoff": fz_cutoff})
                    (tot_error, tot_error_quant) = self.calc_error(umsatz_haushalte_pd,
                                                                   col_modelUmsatz="Umsatz_Haushalte",
                                                                   col_istUmsatz="istUmsatz",
                                                                   quant=0.95)
                    if tot_error_quant < self.E_min[len(self.E_min)-1][0]:
                        umsatz_haushalte_optimal_pd = umsatz_haushalte_pd
                        self.logger.info("New minimum found. TOTAL ERROR: %f / %f", tot_error, tot_error_quant)
                        self.E_min[len(self.E_min)-1] = (tot_error_quant,
                                                         {"lat": lat, "rlat": rlat, "fz_cutoff": fz_cutoff})


        self.E_min.append((float("inf"), ()))

        # Proceed with looping over the oeV parameters
        for beta_ov in self.parameters["beta_ov"]:
            for f_pendler in self.parameters["f_pendler"]:
                # ---- 2. UMSATZ_PENDLER
                umsatz_pendler_pd = self.umsatz_components.calc_umsatz_oev(pandas_dt,
                                                                             {"beta_ov": beta_ov,
                                                                              "f_pendler": f_pendler,
                                                                     "stations_pd": self.parameters["stations_pd"]})

                # ----- 3. TOTAL_UMSATZ
                umsatz_total_pd = pd.merge(umsatz_haushalte_optimal_pd, umsatz_pendler_pd, how='left',
                                           left_index=True, right_index=True)

                # ----- FINALIZE
                umsatz_total_pd['Umsatz_Total'] = np.where(
                    np.isnan(umsatz_total_pd['Umsatz_Pendler']),
                    umsatz_total_pd['Umsatz_Haushalte'],
                    umsatz_total_pd['Umsatz_Haushalte'] + umsatz_total_pd['Umsatz_Pendler'])

                umsatz_total_pd['percentage_istU_modelU'] = (umsatz_total_pd['Umsatz_Total'] -
                                                             umsatz_total_pd['istUmsatz']) / \
                                                            umsatz_total_pd['istUmsatz']

                # re-order only the relevant columns
                column_order = ['StoreName', 'Retailer', 'Format', 'VFL', 'Adresse', 'PLZ', 'Ort',
                                'HARasterID', 'E_LV03', 'N_LV03', 'ProfitKSTID', 'Food', 'Frische',
                                'Near/Non Food', 'Fachmaerkte', 'additional_kaeufer', 'istUmsatz',
                                'Umsatz_Haushalte', 'Umsatz_Pendler', 'Umsatz_Total']

                umsatz_total_pd = umsatz_total_pd[column_order]
                (tot_error, tot_error_quant) = self.calc_error(umsatz_total_pd,
                                                               col_modelUmsatz="Umsatz_Total",
                                                               col_istUmsatz="istUmsatz",
                                                               quant=0.95)

                if tot_error_quant < self.E_min[len(self.E_min)-1][0]:
                    umsatz_total_optimal_pd = umsatz_total_pd
                    self.logger.info("New minimum found. TOTAL ERROR: %f / %f", tot_error, tot_error_quant)
                    self.E_min[len(self.E_min)-1] = (tot_error_quant, {"beta_ov": beta_ov, "f_pendler": f_pendler})

        output_fname = self.umsatz_output_csv + \
                       "_lat_" + str(self.E_min[0][1]["lat"]) + \
                       "_rlat_" + str(str(self.E_min[0][1]["rlat"])) + \
                       "_fzcutoff_" + str(self.E_min[0][1]["fz_cutoff"]) + \
                       "_betaov_" + str(self.E_min[1][1]["beta_ov"]) + \
                       "_fpendler_" + str(self.E_min[1][1]["f_pendler"])

        umsatz_total_optimal_pd.to_csv(output_fname)

    def calc_error(self, pandas_pd, col_modelUmsatz, col_istUmsatz, quant):
        # --- Calculate the error ------------------------------------------------
        error_E_i = np.power(pandas_pd[col_modelUmsatz] - pandas_pd[col_istUmsatz], 2) / pandas_pd[col_istUmsatz]
        error_E_i = error_E_i.loc[~np.isnan(error_E_i)]

        total_error = np.sqrt(np.sum(error_E_i))
        error_quantile = error_E_i.quantile(q=quant)
        total_error_quant = np.sqrt(np.sum(error_E_i.loc[error_E_i <= error_quantile]))

        return (total_error, total_error_quant)
        # self.logger.info("TOTAL ERROR: %f / %f", total_error, total_error_quant)
