import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew


def calc_error(pandas_pd, col_modelUmsatz, col_istUmsatz, quant, inner_range, single_store=None):
    # --- Calculate the error ------------------------------------------------
    if single_store is not None:
        x = pandas_pd.loc[pandas_pd.StoreName == single_store]
        # error_E_i = np.power(x[col_modelUmsatz] - x[col_istUmsatz], 2) / x[col_istUmsatz]
        error_E_i = (x[col_modelUmsatz] - x[col_istUmsatz]) / x[col_istUmsatz]

    else:
        error_E_i = (pandas_pd[col_modelUmsatz] - pandas_pd[col_istUmsatz]) / pandas_pd[col_istUmsatz]
        # error_E_i = np.power(pandas_pd[col_modelUmsatz] - pandas_pd[col_istUmsatz], 2) / pandas_pd[col_istUmsatz]

    # drop NaNs
    error_E_i = error_E_i.loc[~np.isnan(error_E_i)]


    total_error_mean = np.mean(error_E_i)
    total_error_median = np.median(error_E_i)
    error_quantile = error_E_i.quantile(q=quant)

    total_error_quant_median = np.median(error_E_i.loc[error_E_i <= error_quantile])
    total_error_quant_mean = np.mean(error_E_i.loc[error_E_i <= error_quantile])

    skewness = skew(error_E_i.loc[error_E_i <= error_quantile])
    kurt = kurtosis(error_E_i.loc[error_E_i <= error_quantile])

    inner_range_percent = len( error_E_i.loc[(error_E_i >= -inner_range) & (error_E_i <= inner_range)]) / len(error_E_i)

    num_stores = len(error_E_i)
    num_relevant_stores = len(error_E_i.loc[error_E_i <= error_quantile])

    return (total_error_mean, total_error_median,
            total_error_quant_median, total_error_quant_mean,
            skewness+kurt, inner_range_percent,
            (num_relevant_stores, num_stores))


