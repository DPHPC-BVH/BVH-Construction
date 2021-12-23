import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.gofplots import qqplot

from util import read_csv

def add_plot_qq_subparser(parser):
    subparser = parser.add_parser('qqplot', help='Plots a Q-Q plot')
    subparser.add_argument('file', type=str, help='The path to the csv file containing the measurements')
    subparser.add_argument('--skip-first-n-iterations', type=int, default=0, help='The number of first iterations to skip (used to skip warm-up iterations)')
    subparser.add_argument('--out', type=str, default='plot.pdf', help='specifies the output file')


def plot_qq(file, skip_first_n_iterations, out):
    df = read_csv(file)
    single_iterations = df[df['iterations'] == 1]
    time = np.array(single_iterations['real_time'][skip_first_n_iterations:])
    fig = qqplot(time, line = 's', fit=True)
    fig.savefig(out)