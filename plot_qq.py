import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.gofplots import qqplot

def add_plot_qq_subparser(parser):
    # subparser = parser.add_parser('qqplot', help='')
    parser.add_argument('files', type=str, help='The path to the csv file containing the measurements')
    parser.add_argument('--out', type=str, default='plot.pdf', help='specifies the output file')

def read_csv(path):
    return pd.read_csv(path, header=9)

def plot_qq(file, out):
    df = read_csv(file)
    single_iterations = df[df['iterations'] == 1]
    time = np.array(single_iterations['real_time'])
    fig = qqplot(time, line = 's')
    fig.savefig(out)

def main():
    parser = argparse.ArgumentParser(description='Plot qq')
    add_plot_qq_subparser(parser)
    args = vars(parser.parse_args())
    plot_qq(*args.values())

if __name__ == '__main__':
    main()