#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:41:31 2017

@author: macintosh
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import calendar
from pandas.tseries import offsets
import os
from datetime import date
import holidays
import Classes.regression as regression

blues = [x for x in reversed(sns.color_palette("Blues_d", 11))]


path = r'/Users/macintosh/Documents/OneDrive - Cardiff University/04 - Projects/02 - Warwick/04 - Data/'
file = "compiled_data.csv"

data = pd.read_csv(path+file, index_col=0, parse_dates=True)

data.fillna(method="ffill", inplace=True)
daily_data = data.resample("d").agg({"Heat": np.sum, "Temperature":np.mean})
model = regression.Regression()
model._model(daily_data[["Heat", "Temperature"]].copy(), verbose_eval=False)
model._plot()

print(model.param)