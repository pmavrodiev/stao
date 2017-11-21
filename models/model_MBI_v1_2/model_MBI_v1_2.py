import json

import numpy as np
import pandas as pd
import sys
import os
import datetime
import dask.dataframe as dd

from scipy.optimize import minimize
from models.model_base import ModelBase
from multiprocessing import Pool
from pyproj import Proj, transform  # coordinate projections and transformations

# from sklearn import linear_model # needed for the regression
# from sklearn.metrics import mean_squared_error, r2_score # needed for the regression

import statsmodels.formula.api as smf

@ModelBase.register
class model_MBI_v1_2(ModelBase):

    def __init__(self):
        self.logger = None
        self.umsatz_components = None

        # all model parameters go into this dictionary
        self.parameters = {}

        # the minimum error after the parameter sweep
        self.E_min = [(float("inf"), ())]

        self.single_store = None

        self.regression_formula = None

    """
    POST-PROCESS DATA FOR VISUALIZATION IN TABLEAU
    Input: Numpy ndarray with RELIs / HARasterIDss
    Output: Coordinates (WGS84) of the 4 corners of this hectare, including a Tableau-compatible plotting order
    Based on original code from Bojan.Skerlak@mgb.ch, August 2017
    Optimized and adapted by Pavlin.Mavrodiev@mgb.ch, August 2017
    """
    class _geo_helpers_:
        def __init__(self):
            # --- Import Coordinate Frames
            self.lv03Proj = Proj(init='epsg:21781')  # LV03 = CH1903 (old Swiss coordinate system, 6 digits)
            self.lv95Proj = Proj(init='epsg:2056')  # LV95 = CH1903+ (new Swiss coordinate system, 7 digits)
            self.wgs84Proj = Proj(init='epsg:4326')  # WGS84 (worldwide coordinate system ('default' lat lon)

        # --- Define functions
        #  Calculates points (corners) of HA, including PlotOrder (needed for Tableau).
        def calcHRpoints(self, reli):
            # Extracts first 4 and last 4 digits (these are NOT coordinates because they've not yet been
            # multiplied by 100)
            x0 = reli // 10000
            y0 = reli - x0 * 10000

            out = pd.DataFrame({'HARasterID': np.repeat(reli, 4),
                                'x_corner': np.multiply(np.array([x0, x0 + 1, x0 + 1, x0]), 100).flatten(order='F'),
                                'y_corner': np.multiply(np.array([y0, y0, y0 + 1, y0 + 1]), 100).flatten(order='F'),
                                'PlotOrder': np.tile(range(1, 5), len(x0))}
                               )
            out.set_index('HARasterID', inplace=True)
            return out

        #  Adds WGS coordinates to data frame
        def addHRpointsWGS(self, reli):
            # create data frame with corners in Swiss coordinates
            df = self.calcHRpoints(reli)
            # calculate WGS84 coordinates
            xyWSG84 = transform(self.lv03Proj, self.wgs84Proj, df['x_corner'].values, df['y_corner'].values)
            df['lon'] = xyWSG84[0]
            df['lat'] = xyWSG84[1]
            return df

    """
    An internal class to output various debugging information
    """
    class _debug_:
        def __init__(self, parent_logger, params, geo_helpers):
            self.logger = parent_logger
            self.params = params
            self.geo_helpers = geo_helpers

        def startRelis2Stores(self, group, pandas_dt):
            row = group.iloc[0]
            all_start_harasters = pandas_dt.loc[pandas_dt.StartHARasterID.isin([row['StartHARasterID']])]
            all_start_harasters['Reference_StoreID'] = row['StoreID']
            all_start_harasters['LUmsatz'] = all_start_harasters['LMA'] * all_start_harasters['Tot_Haushaltausgaben']

            return all_start_harasters[
                ['Reference_StoreID', 'StartHARasterID', 'StoreID', 'StoreName', 'Retailer', 'Format', 'VFL', 'FZ',
                 'PLZ_l', 'PLZ_r', 'Tot_Haushaltausgaben', 'LMA', 'LUmsatz', 'AnzahlHH', 'HARasterID']]

        def export_debug_info_stores(self, pandas_dt):
            store_perspective = pandas_dt.loc[pandas_dt["StoreID"].isin(self.params["store_ids"])]

            if len(store_perspective) == 0:
                self.logger.info( "Debug mode is ON, but specidifed stores are not available. Bailing. "
                                  "Is single store mode also ON?")
            else:
                # ---- Sort the FZ column for each StoreID
                store_perspective = store_perspective.sort_values(['StoreID', 'FZ'])

                store_perspective['LUmsatz'] = store_perspective['LMA'] * store_perspective['Tot_Haushaltausgaben']
                store_perspective['LUmsatz_cumsum'] = store_perspective.groupby('StoreID')['LUmsatz'].transform(np.cumsum)

                store_perspective['LUmsatz_cumsum_percentage'] = store_perspective.groupby('StoreID')['LUmsatz_cumsum']. \
                    transform(lambda x: x / np.max(x))

                writer = pd.ExcelWriter(self.params["umsatz_output_csv"] + ".debugstores.xlsx")

                # ---- Now get the WGS84 coordinates of all StartHARasterID
                self.logger.info("Calculating WGS84 coordinates for the StartHARasterIDs of relevant stores ... ")
                stores_perspective_relis2wgs84 = pd.DataFrame()
                for store in self.params["store_ids"]:
                    startHARasterIDs = np.unique(
                        pandas_dt.loc[pandas_dt['StoreID'] == store, 'StartHARasterID'].astype(int))
                    x = self.geo_helpers.addHRpointsWGS(startHARasterIDs)
                    x['StoreID'] = store
                    stores_perspective_relis2wgs84 = pd.concat([stores_perspective_relis2wgs84, x])

                y = pd.merge(left=stores_perspective_relis2wgs84.reset_index(), right=store_perspective.reset_index(),
                             how='left', left_on='HARasterID', right_on='StartHARasterID',
                             suffixes=('_start', '_target'))

                self.logger.info("Writing to output")
                y.to_excel(writer, "Sheet1")
                writer.close()
                self.logger.info("Done.")

                # --- Compute the top competitors per hectare for each debugged store

                # get just the catchment area per store, i.e. only the hectares making up 80% of the sales
                self.logger.info("Computing top competitors")
                store_perspective = store_perspective.loc[store_perspective.LUmsatz_cumsum_percentage <= 0.8]
                x = store_perspective.groupby('StartHARasterID',
                                              group_keys=False).apply(self.startRelis2Stores, pandas_dt)
                self.logger.info("Done.")

                self.logger.info("Writing to output")
                writer = pd.ExcelWriter(self.params["umsatz_output_csv"] + ".debug_top_competitors.xlsx")
                x.to_excel(writer, "Sheet1")
                self.logger.info("Done.")

        def export_debug_info_hectares(self, pandas_dt):

            haraster_perpective = pandas_dt.loc[pandas_dt["StartHARasterID"].isin(self.params["haraster_ids"])]

            if len(haraster_perpective) == 0:
                self.logger.info(
                    "Debug mode is ON, but specidifed hectare ids are not available. Bailing. "
                    "Is single store mode also ON?")
            else:
                self.logger.info("Exporting debugging info for hectares")
                writer2 = pd.ExcelWriter(self.params["umsatz_output_csv"] + ".debugrelis.xlsx")
                haraster_perpective.to_excel(writer2, "RELI")
                # now get the WSG84 coordinates for all relevant hectares
                relevant_hektarts = haraster_perpective["StartHARasterID"].append(haraster_perpective["HARasterID"])
                x = self.geo_helpers.addHRpointsWGS(np.unique(relevant_hektarts))
                x.to_excel(writer2, "StartRelis2WGS84")
                self.logger.info("Done")

    """
        This subclass implements the calculation of all the different components of the model turnover.

        For example, calculating the turnover components due to Household and Commuter spending is implemented
        in the methods 'calc_umsatz_haushalt' and 'calc_umsatz_oev', respectively.

        The idea is that the resulting Pandas dataframe will have one column for each turnover component, which
        can then by either directly summed up or used as a basis for the regression part.

        While each turnover component could be implemented in its own method in the base class,
        a subclass provides better encapsulation and easier code maintenance.

        Each method that computes a turnover component can expect a pandas dataframe with these columns:
        ['StoreID', 'StoreName', 'Retailer', 'Format', 'VFL', 'Adresse', 'PLZ', 'Ort', 'lon', 'lat',
        'E_LV03', 'N_LV03', 'HARasterID', 'ProfitKSTID', 'KostenstelleID', 'JahrID',
        'Food', 'Frische', 'Near/Non Food', 'Fachmaerkte', 'Oeffnungsdatum', 'StartHARasterID',
        'ZielHARasterID', 'AutoDistanzMinuten', 'AutoDistanzKilometer',
        'FZ', 'RegionTyp', 'DTB', 'ANTOT', 'Tot_Haushaltausgaben']
    """
    class _umsatz_components_:
        def __init__(self, parent_logger, debug):
            self.logger = parent_logger
            self.debug = debug

        """
        class Groupby:
            def __init__(self, keys):
                _, self.keys_as_int = np.unique(keys, return_inverse=True)
                self.n_keys = max(self.keys_as_int)
                self.set_indices()

            def set_indices(self):
                self.indices = [[] for i in range(self.n_keys + 1)]
                for i, k in enumerate(self.keys_as_int):
                    self.indices[k].append(i)
                self.indices = [np.array(elt) for elt in self.indices]

            def apply(self, function, vector):
                result = np.zeros(len(vector))
                for k in range(self.n_keys):
                    result[self.indices[k]] = function(vector.iloc[self.indices[k]])
                return result
        """
        def calc_umsatz_haushalt(self, pandas_pd, parameters):
            def compute_market_share(pandas_dt, parameters):
                factor_LAT = parameters["factor_LAT"]
                hh_halbzeit = parameters["hh_halbzeit"]
                hh_halbzeit_factor_smvm = parameters["hh_halbzeit_factor_smvm"]

                self.logger.COMPUTE_UMSATZ_HAUSHALTE('Computing local market share. This takes a while ...')
                self.logger.COMPUTE_UMSATZ_HAUSHALTE("Parameters: factor_LAT / hh_halbzeit / hh_halbzeit_factor_smvm = %f / %f /%f",
                                 factor_LAT, hh_halbzeit, hh_halbzeit_factor_smvm)

                pandas_dt['LAT'] = np.power(pandas_dt['VFL'], factor_LAT)
                                            # np.where(pandas_dt["RegionTyp"].isin([11, 12]), factor_LAT, 1))

                # hh_halbzeit_vector = np.where(pandas_pd["RegionTyp"].isin([11,12]), 0.8*hh_halbzeit, hh_halbzeit)

                hh_halbzeit_vector = np.where(pandas_pd["Format"].isin(["SM/VM 700", "SM/VM 2000"]),
                                               hh_halbzeit_factor_smvm * hh_halbzeit,
                                               hh_halbzeit)

                pandas_dt['RLAT'] = pandas_dt['LAT'] * np.exp(-1.0*pandas_dt['FZ']*np.log(2) / hh_halbzeit_vector)

                self.logger.COMPUTE_UMSATZ_HAUSHALTE('Computing sum RLATs ...')
                # convert to Dask dataframe and then do a very fast group by
                df = dd.from_pandas(pandas_dt, npartitions=10)
                pd_result = pd.DataFrame(df.groupby(df.StartHARasterID).RLAT.sum().compute())
                pd_result.rename(columns={'RLAT': 'sumRLATs'}, inplace=True)
                pandas_dt = df.join(pd_result, on='StartHARasterID').compute()

                pandas_dt['LMA'] = pandas_dt['RLAT'] / pandas_dt['sumRLATs']

                self.logger.COMPUTE_UMSATZ_HAUSHALTE('Done')

                return pandas_dt

            def gen_umsatz_prognose(pandas_pd):
                pandas_pd['lokal_umsatz_potenzial'] = pandas_pd['Tot_Haushaltausgaben'] * pandas_pd['LMA']
                # pandas_pd['lokal_umsatz_potenzial'] = 9000 * pandas_pd['LMA']
                # get only the Migros stores - StoreID between 0 and 9999
                umsatz_potential_pd = pandas_pd.loc[pandas_pd.StoreID < 10000].groupby('StoreID').agg({
                    'StoreName': lambda x: x.iloc[0],
                    'Retailer': lambda x: x.iloc[0],
                    'Format': lambda x: x.iloc[0],
                    'VFL': lambda x: x.iloc[0],
                    'Adresse': lambda x: x.iloc[0],
                    'PLZ_l': lambda x: x.iloc[0],
                    'Ort': lambda x: x.iloc[0],
                    'RegionTyp': lambda x: x.iloc[0],
                    'DTB': lambda x: x.iloc[0],
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
                column_order = ['StoreName', 'Retailer', 'Format', 'VFL', 'Adresse', 'PLZ_l', 'Ort', 'RegionTyp',
                                'DTB', 'HARasterID', 'E_LV03', 'N_LV03', 'ProfitKSTID', 'Food', 'Frische',
                                'Near/Non Food', 'Fachmaerkte', 'lokal_umsatz_potenzial']
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

            if parameters["debug"]:
                self.logger.COMPUTE_UMSATZ_HAUSHALTE("Exporting debugging info for stores")
                self.debug.export_debug_info_stores(pandas_pd)
                self.logger.COMPUTE_UMSATZ_HAUSHALTE("Exporting debugging info for hectares")
                self.debug.export_debug_info_hectares(pandas_pd)

            umsatz_potential_pd = gen_umsatz_prognose(pandas_postprocessed_dt)
            return umsatz_potential_pd

        def calc_umsatz_oev(self, pandas_pd, parameters):
            def calc_zusaetzliche_kauefer(stores_pd, stations_pd, halbzeit):
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

                def calc_pendler_wahrscheinlichkeit(group, halbzeit):

                    partition_function = np.sum(np.exp(-1.0*np.log(2)* group['distanz'] / halbzeit ))


                    probabilities = np.exp(-1.0*np.log(2)* group['distanz'] / halbzeit ) / partition_function

                    rv = pd.DataFrame({'StoreID': group['StoreID'],
                                       'code': group['Code'],
                                       'bahnhof': group['Bahnhof_Haltestelle'],
                                       'distanz': group['distanz'],
                                       'DTV': group['DTV'],
                                       'DWV': group['DWV'],
                                       'pendler_wahrscheinlichkeit': probabilities})
                    return rv.set_index(['StoreID'])

                #
                self.logger.COMPUTE_UMSATZ_PENDLER('Computing Pendler einfluss ...')
                self.logger.COMPUTE_UMSATZ_PENDLER("Parameters: halbzeit/ausgaben_pendler = %f", halbzeit)

                x = pd.DataFrame(stores_pd.reset_index().groupby('StoreID',
                                                                 group_keys=False).apply(get_reachable_stations, 300))

                '''
                    'x' looks like this now:

                            Code	Bahnhof_Haltestelle	[...]    lat      lon	    E_LV03	N_LV03	distanz
                StoreID
                    8	    MI  	Muri AG	                 47.276660	8.339798	668185	236587	538.396460
                    9	    BURI	Burier	                 46.447876	6.877116	556853	144215	1456.600536
                '''

                y = x.reset_index().groupby('Code', group_keys=False).apply(calc_pendler_wahrscheinlichkeit,
                                                                            halbzeit).reset_index()
                y['additional_kaeufer'] = y['DWV'] * y['pendler_wahrscheinlichkeit']

                '''
                    'y' looks like this now:

                StoreID    DTV 	DWV 	bahnhof	    code	distanz	    pendler_wahrscheinlichkeit	additional_kauefer
                    9	    970	    1300	Burier	    BURI	1456.600536	    0.062500	                81.250000
                    9	    1900	2200	Clarens	    CL	    240.076282	    0.083333	                183.333333
                '''

                # now aggregate over all stores
                return y.loc[y.StoreID < 10000].groupby('StoreID',
                                        as_index=False)['additional_kaeufer'].aggregate(np.sum).set_index('StoreID')

            ov_halbzeit = parameters["ov_halbzeit"]
            ausgaben_pendler = parameters["ausgaben_pendler"]
            stations_pd = parameters["stations_pd"]

            pendler_einfluss_pd = calc_zusaetzliche_kauefer(pandas_pd, stations_pd, ov_halbzeit)
            pendler_einfluss_pd['Umsatz_Pendler'] = pendler_einfluss_pd['additional_kaeufer'] * ausgaben_pendler
            return pendler_einfluss_pd

        def calc_umsatz_statent(self, pandas_pd, parameters):
            statent_halb_zeit = parameters["statent_halb_zeit"]
            statent_halbzeit_factor_smvm = parameters["statent_halbzeit_factor_smvm"]
            ausgaben_arbeitnehmer = parameters["ausgaben_arbeitnehmer"]

            self.logger.COMPUTE_UMSATZ_ARBEITNEHMER("Computing Arbeitnehmer influence ...")
            self.logger.COMPUTE_UMSATZ_ARBEITNEHMER("Parameters: statent_halb_zeit / statent_halbzeit_factor_smvm / ausgaben_arbeitnehmer= %f / %f / %f ",
                             statent_halb_zeit, statent_halbzeit_factor_smvm, ausgaben_arbeitnehmer)

            # beta_zeit_vector = np.where(pandas_pd["Format"].isin(["SM/VM 700"]), 0.5 * statent_halb_zeit,
              #                           statent_halb_zeit)  # best
            beta_zeit_vector = np.where(pandas_pd["Format"].isin(["SM/VM 700", "SM/VM 2000"]),
                                        statent_halbzeit_factor_smvm*statent_halb_zeit,
                                        statent_halb_zeit)
            # beta_zeit_vector = np.where(pandas_pd["RegionTyp"].isin([11, 12]), 0.5*statent_halb_zeit, statent_halb_zeit) - worst

            pandas_pd['statent_numerator'] = np.exp(-1.0*np.log(2)*pandas_pd['AutoDistanzKilometer'] / beta_zeit_vector)
            pandas_pd['statent_denumerator'] = pandas_pd.groupby('StartHARasterID')[
                ['statent_numerator']].transform(sum)

            pandas_pd['statent_probability'] = pandas_pd['statent_numerator'] / pandas_pd['statent_denumerator']
            pandas_pd['statent_additional_kunden'] = pandas_pd['statent_probability'] * pandas_pd['ANTOT']

            aggregated_over_stores = pandas_pd.reset_index()
            aggregated_over_stores = aggregated_over_stores[aggregated_over_stores.StoreID < 10000].groupby('StoreID',
                            as_index=False)['statent_additional_kunden'].aggregate(np.sum).set_index('StoreID')

            aggregated_over_stores['Umsatz_Arbeitnehmer'] = aggregated_over_stores['statent_additional_kunden'] * \
                                                            ausgaben_arbeitnehmer
            return aggregated_over_stores

    """
        Implements the regression part of the model. This takes place after the main turnover components
        have been calculated. 
    """
    class _umsatz_regression_:
        """
            :param parent_logger:
                reference to the parent loggers

            :param data:
                A Pandas DataFrame with all stores as rows and their relevant
                features as columns. At the minimum the following features are required:
                'Umsatz_Pendler', 'Umsatz_Arbeitnehmer' and 'Umsatz_Haushalte'

            :param param_dict:
                A dictionary with the following keys:

                'regr_formula' - R style regression formula read from the settings file.
                The names of the formula components must correspond to the available
                features in the 'data' DataFrame

                'out' - The name of the output file where a summary of the regression
                will be stored

            :return:
                The original pandas DataFrame supplied in 'data' with an additional column
                'Umsatz_Regression'


                """
        def __init__(self, parent_logger, data, param_dict):
            self.logger = parent_logger
            self.data_pd = data
            self.regr_formula = param_dict["regr_formula"]
            self.out = param_dict["output_file"]

        def prepare_data(self):
            self.logger.info("Preparing data for the regression")

            # prepare the data
            self.data_pd.loc[pd.isnull(self.data_pd.Umsatz_Pendler), 'Umsatz_Pendler'] = 0
            self.data_pd.loc[pd.isnull(self.data_pd.Umsatz_Pendler), 'Umsatz_Arbeitnehmer'] = 0
            self.data_pd.loc[pd.isnull(self.data_pd.Umsatz_Pendler), 'Umsatz_Haushalte'] = 0

            # for training take only stores with recorded turnover and add an 'error' column for them
            self.data_pd.loc[~pd.isnull(self.data_pd.istUmsatz), 'error'] = \
                np.abs(self.data_pd.loc[~pd.isnull(self.data_pd.istUmsatz), 'Umsatz_Total'] -
                       self.data_pd.loc[~pd.isnull(self.data_pd.istUmsatz), 'istUmsatz']) / \
                self.data_pd.loc[~pd.isnull(self.data_pd.istUmsatz), 'istUmsatz']

            # finally select the training dataset, after removing irrelevant 'outliers'
            q = self.data_pd['error'].quantile(q=0.99)
            d_training = self.data_pd.loc[self.data_pd['error'] < q]
            del d_training['error']
            return d_training

        def run(self):
            self.logger.info("Running regression: %s", self.regr_formula)

            d_train = self.prepare_data()

            self.logger.info("Training the regression model")
            fitted_model= smf.rlm(formula=self.regr_formula, data=d_train).fit()

            with open(self.out, mode="w") as out_f:
                out_f.write(fitted_model.summary().as_text())

            # now predict on the whole dataset
            self.logger.info("Running regression model on test data ")
            y_train = self.data_pd

            """
                This is to deal with the annoying SettingWithCopyWarning
                When a copy of a data frame is changed, e.g. by adding a new columnn,
                this warning kicks-in and notifies the user that he is working on a copy
                and his changes will not be propagated back to the original data frame.

                We don't care about this here, because it is the modified copy that will be returned to the caller.
                By manually designating this DataFrame as not a copy, this warning won't appear
                For a larger discussion, please refer here:
                https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
            """
            y_train.is_copy = False

            # If 'Format' was in the formula exclude the Migrolino stores from the test data
            # The reason is that the Migrolino format doesn't appear in the training data and so
            # we can't do predictions for it. TODO: Find a smarter way to do this check. This feels hacky.
            if 'Format' in self.regr_formula:
                self.logger.info("Excluding Migrolino stores from test data, because \"Format\" is a regressor")
                y_train = y_train.loc[y_train.Format != 'Migrolino']

            y_train.loc[:, "Umsatz_Regression"] = fitted_model.predict(y_train)

            self.logger.info("DONE with regression")

            # now merge the predicted turnover into the original data set
            return pd.merge(left=self.data_pd.reset_index(),
                            right=y_train.reset_index()[["StoreID", "Umsatz_Regression"]],
                            left_on='StoreID', right_on='StoreID', how='left').set_index('StoreID')

    def whoami(self):
        return 'Model_MBI_v1.2'

    def process_settings(self, config):
        try:
            # single store mode
            self.single_store = config['global']['single_store'].encode("latin-1").decode("latin-1")
            if len(self.single_store) == 0:
                # empty string
                self.single_store = None
                self.logger.param_info("Single store mode is off.")

            # Parameters Haushaltausgaben
            self.parameters["factor_LAT"] = [float(x) for x in json.loads(config["parameters"]["factor_LAT"])]
            self.parameters["hh_halbzeit"] = [float(x) for x in json.loads(config["parameters"]["hh_halbzeit"])]
            self.parameters["hh_halbzeit_factor_smvm"] = [float(x) for x in
                                                  json.loads(config["parameters"]["hh_halbzeit_factor_smvm"])]
            # Parameters oeV
            self.parameters["ov_halbzeit"] = [float(x) for x in json.loads(config["parameters"]["ov_halbzeit"])]
            self.parameters["ausgaben_pendler"] = [float(x) for x in
                                                   json.loads(config["parameters"]["ausgaben_pendler"])]
            # Parameters STATENT
            self.parameters["statent_halb_zeit"] = [float(x) for x in
                                                    json.loads(config["parameters"]["statent_halb_zeit"])]
            self.parameters["statent_halbzeit_factor_smvm"] = [float(x) for x in
                                                    json.loads(config["parameters"]["statent_halbzeit_factor_smvm"])]
            self.parameters["ausgaben_arbeitnehmer"] = [float(x) for x in
                                                    json.loads(config["parameters"]["ausgaben_arbeitnehmer"])]

            #
            self.optimize = config.getboolean('global', 'optimize')
            self.debug = config.getboolean('global', 'debug')

            # Debug options
            self.store_ids = None
            self.ha_rasterids = None
            if self.debug:
                if self.optimize:
                    self.logger.param_info("Debug option chosen, but will be ignored, because optimize is True. ")
                    self.debug = None
                else:
                    try:
                        self.store_ids = [int(x) for x in json.loads(config["debug"]["store_ids"])]
                        self.ha_rasterids = [int(x) for x in json.loads(config["debug"]["ha_rasterids"])]
                        self.logger.param_info("Debug mode chosen. Will output geo statistics for specific stores and hectars.")
                    except Exception as e:
                        print(e)
                        sys.exit(1)

            self.parallelize = config.getboolean('global', 'parallelize')
            self.cpu_count = None
            self.chunk_size = None
            if self.parallelize:
                try:
                    self.cpu_count = [int(x) for x in json.loads(config["parallel"]["cpu_count"])][0]
                    self.chunk_size = [int(x) for x in json.loads(config["parallel"]["chunk_size"])][0]
                    self.logger.param_info("Parallel mode chosen with cpu_count %d / chunk_size %d",
                                     self.cpu_count, self.chunk_size)
                except Exception as e:
                    print(e)
                    sys.exit(1)

            # Regression formula
            if "formula" in config["regression"]:
                self.regression_formula = config["regression"]["formula"]
                if len(self.regression_formula) == 0:
                    self.regression_formula = None

            self.umsatz_output_csv = config["output"]["output_csv"] + "_" + datetime.datetime.now().isoformat()
            # create output directory if it doesn't exist
            try:
                os.makedirs(os.path.dirname(self.umsatz_output_csv))
            except FileExistsError as fe:
                pass
            except Exception as e:
                # also captures PermissionError
                print(e)
                sys.exit(1)
            # now build the cartesian products
            self.habe_params = [(a, b, c) for a in self.parameters["factor_LAT"]
                                for b in self.parameters["hh_halbzeit"]
                                for c in self.parameters["hh_halbzeit_factor_smvm"]]
            self.ov_params = [(a, b) for a in self.parameters["ov_halbzeit"] for b in self.parameters["ausgaben_pendler"]]
            self.statent_params = [(a, b, c) for a in self.parameters["statent_halb_zeit"]
                                   for b in self.parameters["statent_halbzeit_factor_smvm"]
                                   for c in self.parameters["ausgaben_arbeitnehmer"]]

        except Exception as e:
            print(e)
            self.logger.error('Some of the required parameters for model %s are not supplied in the settings',
                              self.whoami())
            self.logger.error("%s", self.parameters.keys())
            sys.exit(1)

    def total_umsatz(self, param_vector, rest):

        # general parameters
        cpu_count = rest["parallel"]["cpu_count"]
        chunk_size = rest["parallel"]["chunk_size"]
        debug = rest["debug"]

        # Parameters STATPOP
        factor_LAT = param_vector[0]
        hh_halbzeit = param_vector[1]
        hh_halbzeit_factor_smvm = param_vector[2]

        # Parameters oeV
        ov_halbzeit = param_vector[3]
        ausgaben_pendler = param_vector[4]

        # Parameters STATENT
        statent_halb_zeit = param_vector[5]
        statent_halbzeit_factor_smvm = param_vector[6]
        ausgaben_arbeitnehmer = param_vector[7]

        umsatz_haushalte = self.umsatz_components.calc_umsatz_haushalt(rest["pandas_dt"],
                                                                       {"factor_LAT": factor_LAT,
                                                                        "hh_halbzeit": hh_halbzeit,
                                                                        "hh_halbzeit_factor_smvm": hh_halbzeit_factor_smvm,
                                                                        "cpu_count": cpu_count,
                                                                        "chunk_size": chunk_size,
                                                                        "debug": debug})

        umsatz_ov = self.umsatz_components.calc_umsatz_oev(rest["pandas_dt"],
                                                                       {"ov_halbzeit": ov_halbzeit,
                                                                        "ausgaben_pendler": ausgaben_pendler,
                                                                        "stations_pd": rest["stations_pd"]})

        umsatz_statent = self.umsatz_components.calc_umsatz_statent(rest["pandas_dt"],
                                                                    {"statent_halb_zeit": statent_halb_zeit,
                                                                     "statent_halbzeit_factor_smvm": statent_halbzeit_factor_smvm,
                                                                     "ausgaben_arbeitnehmer": ausgaben_arbeitnehmer})

        umsatz_merged_pd = pd.merge(umsatz_haushalte, umsatz_ov, how="left", left_index=True, right_index=True)

        umsatz_merged_pd = pd.merge(umsatz_merged_pd, umsatz_statent, how="left", left_index=True, right_index=True)

        (umsatz_total_pd, tot_error, tot_error_quant) = self.calc_error(umsatz_merged_pd,
                                                                    col_modelUmsatz=[
                                                                        "Umsatz_Haushalte",
                                                                        "Umsatz_Arbeitnehmer",
                                                                        "Umsatz_Pendler"],
                                                                    col_istUmsatz="istUmsatz",
                                                                    quant=0.95,
                                                                    single_store = self.single_store)
        return tot_error_quant

    def calc_error(self, pandas_pd, col_modelUmsatz, col_istUmsatz, quant, single_store=None):
        # ----- Calculate the current Total Umsatz
        pandas_pd['Umsatz_Total'] = 0.0
        for column in col_modelUmsatz:
            pandas_pd['Umsatz_Total'] = np.where(np.isnan(pandas_pd[column]), pandas_pd['Umsatz_Total'],
                                                 pandas_pd['Umsatz_Total'] + pandas_pd[column])

        # --- Calculate the error ------------------------------------------------
        if single_store is not None:
            x = pandas_pd.loc[pandas_pd.StoreName == single_store]
            error_E_i = np.power(x['Umsatz_Total'] - x[col_istUmsatz], 2) / x[col_istUmsatz]
        else:
            error_E_i = np.power(pandas_pd['Umsatz_Total'] - pandas_pd[col_istUmsatz], 2) / pandas_pd[col_istUmsatz]

        error_E_i = error_E_i.loc[~np.isnan(error_E_i)]

        total_error = np.sqrt(np.sum(error_E_i))
        error_quantile = error_E_i.quantile(q=quant)
        total_error_quant = np.sqrt(np.sum(error_E_i.loc[error_E_i <= error_quantile]))

        return (pandas_pd, total_error, total_error_quant)

    def entry(self, tables_dict, config, logger):
        self.logger = logger
        self.logger.info("Initialized model %s", self.whoami())

        self.process_settings(config)

        # initialize the internal classes
        # self.umsatz_components = self._umsatz_components_(logger, geo_helpers=self._geo_helpers_())
        self.umsatz_components = self._umsatz_components_(logger,
                                                          debug=self._debug_(logger,
                                                                            params = {
                                                                            'store_ids': self.store_ids or None,
                                                                            'haraster_ids': self.ha_rasterids or None,
                                                                            'umsatz_output_csv': self.umsatz_output_csv
                                                                            },
                                                                            geo_helpers=self._geo_helpers_()))

        # check if the model got the right dict with input tables
        # TODO: super basic check for names only. Handle more robustly
        try:
            pandas_dt = tables_dict["all_stores"]
            self.parameters["stations_pd"] = tables_dict["sbb_stations"]
        except Exception:
            logger.error('Model %s did not get the right input table names. Expected %s got %s',
                        self.whoami(), "['all_stores', 'sbb_stations']", list(tables_dict.keys()))
            sys.exit(1)

        if self.optimize:
            logger.info("Starting parameter optimization ")
            res = minimize(self.total_umsatz,
                           x0=(np.array([self.parameters["factor_LAT"][0],
                                         self.parameters["hh_halbzeit"][0],
                                         self.parameters["hh_halbzeit_factor_smvm"][0],
                                         self.parameters["ov_halbzeit"][0],
                                         self.parameters["ausgaben_pendler"][0],
                                         self.parameters["statent_halb_zeit"][0],
                                         self.parameters["statent_halbzeit_factor_smvm"][0],
                                         self.parameters["ausgaben_arbeitnehmer"][0]])
                              ),
                           args=({"pandas_dt": pandas_dt, "stations_pd": self.parameters["stations_pd"],
                                  "parallel": {"cpu_count": self.cpu_count,
                                               "chunk_size": self.chunk_size},
                                  "debug": None}),
                           method='nelder-mead',
                           options={ 'xtol': 1e-3, 'maxiter': 150},
                           callback=lambda xk: self.logger.info("Iterating. Parameters %s", str(xk)) )

            logger.info("Optimization finished: ")
            logger.info("%s", str(res))
            sys.exit(1)

        # -----------------------
        # ---- HAUSHALT component
        # -----------------------
        cache_haushalte_pd = []
        for habe_p in self.habe_params:
            cache_haushalte_pd.append((habe_p,
                self.umsatz_components.calc_umsatz_haushalt(pandas_dt, {"factor_LAT": habe_p[0],
                                                                        "hh_halbzeit": habe_p[1],
                                                                        "hh_halbzeit_factor_smvm": habe_p[2],
                                                                        "debug": self.debug,
                                                                        "store_ids": self.store_ids or None,
                                                                        "haraster_ids": self.ha_rasterids or None,
                                                                        "umsatz_output_csv": self.umsatz_output_csv,
                                                                        "cpu_count": self.cpu_count,
                                                                        "chunk_size": self.chunk_size})))

        # -----------------------
        # ---- OEV component
        # -----------------------
        cache_pendler_pd = []
        for ov_p in self.ov_params:
            cache_pendler_pd.append((ov_p,
                self.umsatz_components.calc_umsatz_oev(pandas_dt, {"ov_halbzeit": ov_p[0], "ausgaben_pendler": ov_p[1],
                                                                   "stations_pd":self.parameters["stations_pd"]})))

        # -----------------------
        # ---- STATENT component
        # -----------------------
        cache_statent_pd = []
        for statent_p in self.statent_params:
            cache_statent_pd.append((statent_p,
                self.umsatz_components.calc_umsatz_statent(pandas_dt, {"statent_halb_zeit": statent_p[0],
                                                                       "statent_halbzeit_factor_smvm": statent_p[1],
                                                                       "ausgaben_arbeitnehmer": statent_p[2]})))

        # now find the optimal combination
        idx_combinations = [(a, b, c) for a in range(len(cache_haushalte_pd)) for b in range(len(cache_pendler_pd))
                                      for c in range(len(cache_statent_pd))]
        for idx in idx_combinations:
            idx_haushalt = idx[0] # index into the cache_haushalte_pd tuple, i.e. 0 to len(cache_haushalte_pd)
            idx_ov = idx[1] # index into the cache_pendler_pd tuple, i.e. 0 to len(cache_pendler_pd)
            idx_statent = idx[2] # index into the cache_statent_pd tuple, i.e. 0 to len(cache_statent_pd)

            umsatz_merged_pd = pd.merge(cache_haushalte_pd[idx_haushalt][1], cache_pendler_pd[idx_ov][1], how="left",
                                        left_index=True, right_index=True)
            umsatz_merged_pd = pd.merge(umsatz_merged_pd, cache_statent_pd[idx_statent][1], how="left",
                                        left_index=True, right_index=True)
            (umsatz_total_pd, tot_error, tot_error_quant) = self.calc_error(umsatz_merged_pd,
                                                                        col_modelUmsatz=[
                                                                            "Umsatz_Haushalte",
                                                                            "Umsatz_Arbeitnehmer",
                                                                            "Umsatz_Pendler"],
                                                                        col_istUmsatz="istUmsatz",
                                                                        quant=0.95,
                                                                        single_store = self.single_store)
            if tot_error_quant < self.E_min[len(self.E_min) - 1][0]:
                umsatz_total_optimal_pd = umsatz_total_pd
                self.logger.info("New minimum found. TOTAL ERROR: %f / %f", tot_error, tot_error_quant)
                self.logger.info("Parameters: ")
                self.logger.info("factor_LAT: %f ", cache_haushalte_pd[idx_haushalt][0][0])
                self.logger.info("hh_halbzeit: %f ", cache_haushalte_pd[idx_haushalt][0][1])
                self.logger.info("hh_halbzeit_factor_smvm: %f ", cache_haushalte_pd[idx_haushalt][0][2])
                self.logger.info("ov_halbzeit: %f ", cache_pendler_pd[idx_ov][0][0])
                self.logger.info("ausgaben_pendler: %f ", cache_pendler_pd[idx_ov][0][1])
                self.logger.info("statent_halb_zeit: %f ", cache_statent_pd[idx_statent][0][0])
                self.logger.info("statent_halbzeit_factor_smvm: %f ", cache_statent_pd[idx_statent][0][1])
                self.logger.info("ausgaben_arbeitnehmer: %f ", cache_statent_pd[idx_statent][0][2])

                self.E_min[len(self.E_min) - 1] = (tot_error_quant,
                                               {"factor_LAT": cache_haushalte_pd[idx_haushalt][0][0],
                                                "hh_halbzeit": cache_haushalte_pd[idx_haushalt][0][1],
                                                "hh_halbzeit_factor_smvm": cache_haushalte_pd[idx_haushalt][0][2],
                                                "ov_halbzeit": cache_pendler_pd[idx_ov][0][0],
                                                "ausgaben_pendler": cache_pendler_pd[idx_ov][0][1],
                                                "statent_halb_zeit": cache_statent_pd[idx_statent][0][0],
                                                "statent_halbzeit_factor_smvm": cache_statent_pd[idx_statent][0][1],
                                                "ausgaben_arbeitnehmer": cache_statent_pd[idx_statent][0][2]})

        # -----------------------
        # ---- RUN THE REGRESSION
        # -----------------------

        # choose only the relevant columns - StoreID is the index
        column_order = ['StoreName', 'Retailer', 'Format', 'VFL', 'Adresse', 'PLZ_l', 'Ort', 'RegionTyp', 'DTB',
                        'HARasterID', 'E_LV03', 'N_LV03', 'ProfitKSTID', 'Food', 'Frische',
                        'Near/Non Food', 'Fachmaerkte', 'additional_kaeufer', 'statent_additional_kunden',
                        'istUmsatz', 'Umsatz_Haushalte', 'Umsatz_Pendler', 'Umsatz_Arbeitnehmer', 'Umsatz_Total']

        umsatz_total_optimal_pd = umsatz_total_optimal_pd[column_order]

        if self.regression_formula is not None:
            if self.single_store:
                self.logger.warning("Cannot run the regression part in single store mode. Skipping.")
                umsatz_total_optimal_pd = umsatz_total_optimal_pd.loc[umsatz_total_optimal_pd.StoreName == self.single_store]
            else:
                umsatz_total_optimal_pd = self._umsatz_regression_(self.logger, umsatz_total_optimal_pd,
                                                {"regr_formula": self.regression_formula,
                                                 "output_file": self.umsatz_output_csv + '.regression'}).run()

                # calculate the error after the regression
                # (umsatz_total_pd, tot_error, tot_error_quant) = \
                #     self.calc_error(umsatz_total_optimal_pd, col_modelUmsatz=["Umsatz_Regression"],
                #                     col_istUmsatz="istUmsatz", quant=0.95)
                # self.logger.info("Error after regression is %f ", tot_error_quant)

        # --- FINALIZE ------
        output_fname = self.umsatz_output_csv + '.txt'
        output_param_fname = self.umsatz_output_csv + '.params'
        self.logger.info("Exporting results ...")

        with open(output_param_fname, 'w') as f:
            f.write("factor_LAT: " + str(self.E_min[0][1]["factor_LAT"])+"\n")
            f.write("hh_halbzeit: " + str(self.E_min[0][1]["hh_halbzeit"])+"\n")
            f.write("hh_halbzeit_factor_smvm: " + str(self.E_min[0][1]["hh_halbzeit_factor_smvm"])+"\n")
            f.write("betaov: " + str(self.E_min[0][1]["ov_halbzeit"])+"\n")
            f.write("ausgaben_pendler: " + str(self.E_min[0][1]["ausgaben_pendler"])+"\n")
            f.write("statent_halb_zeit: " + str(self.E_min[0][1]["statent_halb_zeit"])+"\n")
            f.write("statent_halbzeit_factor_smvm: " + str(self.E_min[0][1]["statent_halbzeit_factor_smvm"])+"\n")
            f.write("ausgaben_arbeitnehmer: " + str(self.E_min[0][1]["ausgaben_arbeitnehmer"])+"\n")

        umsatz_total_optimal_pd.to_csv(output_fname)
        self.logger.info("Done.")

