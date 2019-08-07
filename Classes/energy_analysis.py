# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:31:05 2018

@author: sceac10
"""


# Use pyment to create the docstrings of the code: https://github.com/dadadel/pyment


import os
import pandas as pd
import numpy as np


# Change the path to adapt to the path structure of the machine where it is executed

def _get_path_machine(folder):

    computer_path = ""
    for x in os.getcwd().split(os.path.sep):
        computer_path = computer_path+x+os.path.sep
        if x == str(folder):
            break
    return computer_path


#loop through all .csv files in a folder and merge them together in the given axis

def _extract_merge(path=None, axis=0):

    if path is None:
        error = 'Please specify a path'
        raise ValueError(error)
        return None
    else:
        frames = []
        for fn in os.listdir(path):
            if os.path.splitext(fn)[1] == ".csv":
                data = pd.read_csv(path+fn)
                frames.append(data)
        return pd.concat(frames, axis=axis)


# 3 sigma rules and 6 hours window
def _find_fill_outliers(data, serie_name, window=12, rolling=True, fillna=True, verbose=False):

    if isinstance(data, pd.DataFrame):

        data[serie_name + " zscore"] = 0
        if rolling:
            zscore_array = rolling_zscore(data[serie_name], window)
            #data[serie_name +" zscore"].fillna(zscore(data[serie_name]), inplace=True)
        else:
            zscore_array = zscore(data[serie_name])

        data.loc[abs(zscore_array) > 3, serie_name] = np.nan

        if fillna:
            data[serie_name].fillna(data[serie_name].rolling(window=6, center=False).mean(), inplace=True)
            data[serie_name].interpolate(inplace=True)

# return the zscore of eac value of an array
def zscore(x):

    z = (x-x.mean())/x.std()
    return z
    
# return the rolling zscore of eac value of an array
def rolling_zscore(x, window):

    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z
