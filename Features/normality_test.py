import pandas as pd
import numpy as np
from numpy.random import seed
from numpy.random import randn
from scipy.stats import anderson
from scipy.stats import shapiro
from scipy.stats import normaltest

# Shapiro-Wilk Test
# from https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/    
def ShapiroWilk(data):

    # normality test
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
#         print('Sample looks Gaussian (fail to reject H0)')
        return True
    else:
#         print('Sample does not look Gaussian (reject H0)')
        return False

# D'Agostino and Pearson's Test
def Dagostino(data):
    # normality test
    stat, p = normaltest(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
#         print('Sample looks Gaussian (fail to reject H0)')
        return True
    else:
#         print('Sample does not look Gaussian (reject H0)')
        return False

# Anderson-Darling Test
def AndersonDarling(data):

    # normality test
    result = anderson(data)
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            return True
        else:
            print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
            return False


def normality(data):
    #Not sure why it creates an error

    # using a log transform
    normality_dict = {}
    shapiro_list = []
    dagostino_list = []
    anderson_list = []
    for col in data.select_dtypes(exclude=['object']).columns:
        print(col)
        data = data[col].apply(np.log).values
        shapiro_list.append(ShapiroWilk(data))
        dagostino_list.append(Dagostino(data))
        anderson_list.append(AndersonDarling(data))
    normality_dict={'shapiro':shapiro_list, 'dagostino':dagostino_list, 'anderson':anderson_list}
    
    normality_df = pd.DataFrame(normality_dict, index=data.select_dtypes(exclude=['object']).columns)

    #default
    normality_dict = {}
    shapiro_list = []
    dagostino_list = []
    anderson_list = []
    for col in data.select_dtypes(exclude=['object']).columns:
        print(col)
        data = data[col].values
        shapiro_list.append(ShapiroWilk(data))
        dagostino_list.append(Dagostino(data))
        anderson_list.append(AndersonDarling(data))
    normality_dict={'shapiro':shapiro_list, 'dagostino':dagostino_list, 'anderson':anderson_list}
    temp_df = pd.DataFrame(normality_dict, index=data.select_dtypes(exclude=['object']).columns)
    
    normality_df = pd.merge(normality_df, temp_df, left_index=True, right_index=True, how='left')

    # using a square transform
    normality_dict = {}
    shapiro_list = []
    dagostino_list = []
    anderson_list = []
    for col in data.select_dtypes(exclude=['object']).columns:
        print(col)
        data = data[col].apply(np.square).values
        shapiro_list.append(ShapiroWilk(data))
        dagostino_list.append(Dagostino(data))
        anderson_list.append(AndersonDarling(data))
    normality_dict={'shapiro':shapiro_list, 'dagostino':dagostino_list, 'anderson':anderson_list}
    temp_df = pd.DataFrame(normality_dict, index=data.select_dtypes(exclude=['object']).columns)
    normality_df = pd.merge(normality_df, temp_df, left_index=True, right_index=True, how='left')

    return normality_df
