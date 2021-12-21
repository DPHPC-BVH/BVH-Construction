import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from util import shapiro_wilk_test, read_csv


def add_plot_distribution_subparser(parser):
    subparser = parser.add_parser('distribution', help='a help')
    subparser.add_argument('file', type=str, help='The path to the csv file containing the measurements')
    subparser.add_argument('--without-labels', action='store_true', help='Hides the labels')
    subparser.add_argument('--without-additional-info', action='store_true', help='Hides additional information under the plot')
    subparser.add_argument('--unit', choices=['s', 'ms', 'us', 'ns'], default='us', help='The time unit, used for the histogram')
    subparser.add_argument('--skip-first-n-iterations', type=int, default=0, help='The number of first iterations to skip (used to skip warm-up iterations)')
    subparser.add_argument('--out', type=str, default='plot.pdf', help='specifies the output file')


def plot_distribution(file, without_labels, without_additional_info, unit, skip_first_n_iterations, out):
    
    # Read in data and convert to unit
    df = read_csv(file)

    single_iterations = df[df['iterations'] == 1]

    # skip warmup iterations
    single_iterations = single_iterations[skip_first_n_iterations:]

    assert np.all(single_iterations['time_unit'] == 'ns')
    
    # Convert to unit and transform data to fit histogram
    time = np.array([convert_ns_to_format(t, unit) for t in single_iterations['real_time']])
    time, exponent = transform_to_fit(time)
   
    # Normality Test
    _ = shapiro_wilk_test(time, verbose=True)

    # Draw Histogram
    time_max = int(np.ceil(np.max(time)))
    time_min = int(np.floor(np.min(time)))
    bins = np.linspace(time_min, time_max, time_max - time_min + 1)
    n, bins, _ = plt.hist(time, bins=bins, density=True, fc=(0,0,0,0), ec='black', linewidth=0.5)
    
    # Rename ticks
    locs, _ = plt.xticks()
    labels = [str(x / exponent) for x in locs]
    plt.xticks(locs, labels)
    plt.xlim(time_min - 1, time_max + 1)
    
    plt.title("Distribution of completion times \n")
    plt.xlabel(('Time Completion (%s)' % unit))
    plt.ylabel('Density')

    # Draw interpolated line
    x = np.linspace(time_min - 1, time_max + 1, 1000)
    kernel = stats.gaussian_kde(time)
    y = kernel(x)
    plt.plot(x, y, c='blue', linewidth=0.5, alpha=0.5)
    plt.fill_between(x, y, color='blue', alpha=0.25)
    
    # Plot Mean Line
    mean = np.mean(time)
    draw_vertical_line(mean, exponent, label='Mean', linestyle=':', color='purple', without_labels=without_labels)

    # Plot Median Line
    median = np.median(time)
    draw_vertical_line(median,exponent, label='Median', linestyle='-', color='blue', without_labels=without_labels)

    # Plot Min line
    min = np.min(time)
    draw_vertical_line(min, exponent, label='Min', linestyle='-.', color='green', without_labels=without_labels)

    # Plot Max line
    max = np.max(time)
    draw_vertical_line(max, exponent, label='Max', linestyle=(0, (3, 2, 1, 2, 1, 2)), color='orange', without_labels=without_labels)

    # Plot quantile line
    qunatile = np.quantile(time, 0.95)
    draw_vertical_line(qunatile, exponent, label='Quantile 95%', linestyle=(0, (3, 2, 1, 2, 1, 2)), color='red', without_labels=without_labels)

    # Calculate confidence interval (mean)
    _, lower, upper = mean_confidence_interval(time)

    # Additional Info
    if not without_additional_info:
        plt.gcf().text(0.5, 0.075, 'Mean: {:.2f}, Median: {:.2f}, Min: {:.2f}, Max: {:.2f}, Quantile 95%: {:.2f}'.format(mean / exponent, median / exponent, min / exponent, max / exponent, qunatile / exponent), horizontalalignment='center', verticalalignment='center')
        plt.gcf().text(0.5, 0.025, 'CI 95% (Mean): ({:.2f},{:.2f})'.format(lower / exponent, upper / exponent), horizontalalignment='center', verticalalignment='center')
        plt.subplots_adjust(bottom=0.2)    

    # Save Plot
    plt.savefig(out, bbox_inches="tight")


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    lower, upper = stats.t.interval(0.95, n-1, loc=m, scale=se)
    return m, lower, upper


def transform_to_fit(data):
    data_max = int(np.ceil(np.max(data)))
    data_min = int(np.floor(np.min(data)))
    n_bins = data_max - data_min + 1
    exponent = 1
    if n_bins >= 100:
        exponent = 10**(np.floor(np.log10(n_bins)) - 1)
        data /= exponent

    if n_bins < 10:
        exponent = 10**(np.floor(np.log10(np.max(data) - np.min(data))) - 1)
        print(exponent)
        data /= exponent

    return data, exponent

def draw_vertical_line(value, exponent, label, linestyle, color, without_labels=False):
    plt.axvline(value, color=color, linestyle=linestyle, linewidth=1)
    _, max_ylim = plt.ylim()
    if not without_labels:
        plt.text(value, max_ylim * 1.02, label, fontsize='small', horizontalalignment='center', verticalalignment='center', color=color)
        plt.text(value + 0.15, max_ylim*0.85, '{:.2f}'.format(value / exponent), rotation=90, fontsize='small')

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