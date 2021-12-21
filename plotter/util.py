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

def median_confidence_interval_95(data, confidence=0.95):
    assert confidence > 0 and confidence < 1
    # Sort
    sorted_data = np.sort(data)
    
    # Compute high/low according to J.-Y. L. Boudec. Performance Evaluation of Computer and Communication Systems. EPFL Press, 2011.
    n = sorted_data.size
    alpha = 1 - confidence
    z_a_2 = stats.norm.ppf(1 - alpha / 2)
    low = math.floor((n-z_a_2*np.sqrt(n))/2)
    high = math.ceil(1+(n+z_a_2*np.sqrt(n))/2)

    return np.median(data), sorted_data[low], sorted_data[high]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    lower, upper = stats.t.interval(confidence, n-1, loc=m, scale=se)
    return m, lower, upper

def convert_ns_to_format(time, unit):
    if unit == 'ns':
        return time
    elif unit == 'us':
        return time / 1e3 
    elif unit == 'ms':
        return time / 1e6
    elif unit == 's':
        return time / 1e9
    else:
        raise Exception("Illegal output format")

def read_csv(path):
    # Find start of header
    with open(path, 'r') as f:
        reader = csv.reader(f)
        idx = next(idx for idx, row in enumerate(reader) if len(row) > 1)
    
    return pd.read_csv(path, skiprows=idx)