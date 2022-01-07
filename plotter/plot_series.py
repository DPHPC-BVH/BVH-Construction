from typing import Collection
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import capitalize
from util import read_csv, mean_confidence_interval, median_confidence_interval_95, convert_ns_to_format, median_largest_diff_95ci, mean_largest_diff_95ci, COLORS

def add_plot_series_subparser(parser):
    subparser = parser.add_parser('series', help='')
    subparser.add_argument('files', type=str, nargs='+', help='The path to the csv file containing the measurements')
    subparser.add_argument('--series-labels', type=str, nargs='+', default=['Ours'])
    subparser.add_argument('--baseline', type=str, default=None, help='The path to the optix baseline')
    subparser.add_argument('--strategy', choices=['mean', 'median'], default='median', help='The strategy used to summarize the runs')
    subparser.add_argument('--unit', choices=['s', 'ms', 'us', 'ns'], default='ms', help='The time unit, used for the histogram')
    subparser.add_argument('--with-ci', action='store_true', help='Display CI intervals')
    subparser.add_argument('--out', type=str, default='plot.pdf', help='specifies the output file')
    subparser.add_argument('--skip-first-n-iterations', type=int, default=0, help='The number of first iterations to skip (used to skip warm-up iterations)')


def plot_series(files, series_labels, baseline, strategy, unit, skip_first_n_iterations, with_ci, out):

    # read in data
    data_frames = [read_csv(file) for file in files]

    # pick single iterations and skip first n iterations
    data_frames = [df[df['iterations'] == 1][skip_first_n_iterations:] for df in data_frames]

    # Check unit
    assert np.all(np.array([df['time_unit'] == 'ns' for df in data_frames]))

    # compute medians of each stage for all scenes
    data = np.array([convert_ns_to_format(df['real_time'], unit) for df in data_frames])

    # Arrange data
    barplot_data = {}
    assert len(data) % len(series_labels) == 0
    files_per_series = int(len(data) / len(series_labels))
    scene_names = np.array([df['name'].to_numpy()[0].split('/')[1] for df in data_frames[:files_per_series]])

    assert len(series_labels) <= len(COLORS) 
    colors = COLORS[1:len(series_labels)+1]
    for i, label in enumerate(series_labels):
        barplot_data[label] = data[i*files_per_series:(i+1)*files_per_series]
    
    if baseline is not None:
        baseline_data_dict = read_in_baseline(baseline)
        barplot_data['Baseline'] = []
        for name in scene_names:
            data_in_unit = convert_ns_to_format(baseline_data_dict[name][skip_first_n_iterations:], unit)
            barplot_data['Baseline'].append(data_in_unit)
        colors.append(COLORS[0])
    
    # compute tick locations and bar offsets
    n_scenes = len(scene_names)
    n_keys = len(barplot_data.keys())
    width = 0.4
    tick_locs = np.arange(n_scenes) * (n_keys + 2) * width
    offsets = (np.array(range(n_keys)) * width) - (width / 2) - np.floor((n_keys - 1) / 2) * width

    # Draw barplot
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, key in enumerate(barplot_data.keys()):
        aggregated_data = np.median(barplot_data[key], axis=1) if strategy == 'median' else np.mean(barplot_data[key], axis=1)
        yerr = None
        if with_ci:
            yerr = np.transpose([[mid - lower, upper - mid] for (mid, lower, upper) in (median_confidence_interval_95(x) if strategy == 'median' else mean_confidence_interval(x) for x in barplot_data[key])])
        ax.bar(tick_locs + offsets[i % n_keys], aggregated_data, yerr=yerr, width=width, capsize=2, label=key, color=colors[i])

    ax.set_ylabel('Execution time [{}]'.format(unit), fontsize=16, fontweight='bold', labelpad=10)
    #ax.set_xlabel('Scene')
    #ax.set_title('Execution time for different scenes', fontsize=20, pad=15)
    plt.xticks(tick_locs, scene_names, fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')

    # set border
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    # set grid
    ax.yaxis.grid()
    ax.set_axisbelow(True)

    # add statistical data
    if with_ci:
        text_props = dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=0.5')
        if strategy == 'median':
            diff = median_largest_diff_95ci(data)
            ci_diff = "%.2f" % (diff*100)
            stat_str = r'95% CI: $\pm$ '
        else:
            diff = mean_largest_diff_95ci(data)
            ci_diff = "%.2f" % (diff*100)
            stat_str = r'95% CI(mean): $\pm$ '

        stat_str += ci_diff + r'%' 
        #stat_str += "\n"
        #stat_str += r'measurements: ' + str(len(data[0]))

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.925, stat_str, transform=ax.transAxes, fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left', bbox=text_props) 

    plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left")
    plt.savefig(out, bbox_inches="tight")

def read_in_baseline(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.split(',')
            data[line[0]] = np.array(line[1:], dtype=np.double) * 1000.0 # convert to nanoseconds
    
    return data