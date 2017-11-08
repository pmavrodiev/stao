import json

import numpy as np
import pandas as pd
import sys
import os
import datetime

from scipy.optimize import minimize
from models.model_base import ModelBase
from multiprocessing import Pool
from pyproj import Proj, transform  # coordinate projections and transformations


class _model_MBI_v1_2:

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
