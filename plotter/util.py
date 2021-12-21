from scipy import stats
import pandas as pd
import csv
from scipy.stats import shapiro


def shapiro_wilk_test(data, alpha=0.05, verbose=False):
    stat, p = shapiro(data)
    if verbose and p < alpha:
        print("--> WARNING: Data does not look Gaussian!")
    return p > alpha 

def read_csv(path):
    # Find start of header
    with open(path, 'r') as f:
        reader = csv.reader(f)
        idx = next(idx for idx, row in enumerate(reader) if len(row) > 1)
    
    return pd.read_csv(path, skiprows=idx)