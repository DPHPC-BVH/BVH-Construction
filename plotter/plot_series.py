import numpy as np
import matplotlib.pyplot as plt
from util import read_csv, mean_confidence_interval, median_confidence_interval_95, convert_ns_to_format

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
    data = [convert_ns_to_format(df['real_time'], unit) for df in data_frames]

    # Arrange data
    barplot_data = {}
    assert len(data) % len(series_labels) == 0
    files_per_series = int(len(data) / len(series_labels))
    scene_names = np.array([df['name'].to_numpy()[0].split('/')[1] for df in data_frames[:files_per_series]])
    
    for i, label in enumerate(series_labels):
        barplot_data[label] = data[i*files_per_series:(i+1)*files_per_series]
   
    if baseline is not None:
        baseline_data_dict = read_in_baseline(baseline)
        barplot_data['Baseline'] = []
        for name in scene_names:
            data_in_unit = convert_ns_to_format(baseline_data_dict[name][skip_first_n_iterations:], unit)
            barplot_data['Baseline'].append(data_in_unit)
    
    # compute label locations and bar offsets
    x = np.arange(len(scene_names))
    n_keys = len(barplot_data.keys())
    width = 1 / (n_keys + 1)
    offsets = (np.array(range(n_keys)) * width) - (width / 2) - np.floor((n_keys - 1) / 2) * width

    # Draw barplot
    fig, ax = plt.subplots()
    for i, key in enumerate(barplot_data.keys()):
        data = np.median(barplot_data[key], axis=1) if strategy == 'median' else np.mean(barplot_data[key], axis=1)
        yerr = None
        if with_ci:
            yerr = np.transpose([[mid - lower, upper - mid] for (mid, lower, upper) in (median_confidence_interval_95(x) if strategy == 'median' else mean_confidence_interval(x) for x in barplot_data[key])])
        ax.bar(x + offsets[i % n_keys], data, yerr=yerr, width=width, label=key)

    ax.set_ylabel('Execution time ({})'.format(unit))
    ax.set_xlabel('Scene')
    ax.set_title('Execution time for different scenes')
    plt.xticks(x, scene_names)

    plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left")

    plt.savefig(out, bbox_inches="tight")

def read_in_baseline(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.split(',')
            data[line[0]] = np.array(line[1:], dtype=np.double) * 1000.0 # convert to nanoseconds
    
    return data