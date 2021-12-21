import pandas as pd
import numpy as np
import csv
import math
from scipy import stats


def shapiro_wilk_test(data, alpha=0.05, verbose=False):
    stat, p = stats.shapiro(data)
    if verbose and p < alpha:
        print("--> WARNING: Data does not look Gaussian!")
    return p > alpha 

def median_confidence_interval_95(data):
    sorted_data = np.sort(data)
    n = sorted_data.size
    low = math.floor((n-1.96*np.sqrt(n))/2)
    high = math.ceil(1+(n+1.96*np.sqrt(n))/2)

    return np.median(data), sorted_data[low], sorted_data[high]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    lower, upper = stats.t.interval(confidence, n-1, loc=m, scale=se)
    return m, lower, upper

def read_csv(path):
    # Find start of header
    with open(path, 'r') as f:
        reader = csv.reader(f)
        idx = next(idx for idx, row in enumerate(reader) if len(row) > 1)
    
    return pd.read_csv(path, skiprows=idx)