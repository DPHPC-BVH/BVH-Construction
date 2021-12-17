from scipy import stats
from scipy.stats import shapiro

def shapiro_wilk_test(data, alpha=0.05, verbose=False):
    stat, p = shapiro(data)
    if verbose and p < alpha:
        print("--> WARNING: Data does not look Gaussian!")
    return p > alpha 

