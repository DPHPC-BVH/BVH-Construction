import numpy as np
import matplotlib.pyplot as plt
from util import read_csv

def add_plot_series_subparser(parser):
    subparser = parser.add_parser('series', help='')
    subparser.add_argument('files', type=str, nargs='+', help='The path to the csv file containing the measurements')
    subparser.add_argument('--series-labels', type=str, nargs='+', default=['Ours'])
    subparser.add_argument('--baseline', type=str, default=None, help='The path to the optix baseline')
    subparser.add_argument('--strategy', choices=['mean', 'median'], default='median', help='The strategy used to summarize the runs')
    subparser.add_argument('--out', type=str, default='plot.pdf', help='specifies the output file')
    subparser.add_argument('--skip-first-n-iterations', type=int, default=0, help='The number of first iterations to skip (used to skip warm-up iterations)')


def plot_series(files, series_labels, baseline, strategy, skip_first_n_iterations, out):

    # read in data
    data_frames = [read_csv(file) for file in files]

    # pick single iterations and skip first n iterations
    data_frames = [df[df['iterations'] == 1][skip_first_n_iterations:] for df in data_frames]

    # compute medians of each stage for all scenes
    data = [np.median(df['real_time']) if strategy == 'median' else np.mean(df['real_time']) for df in data_frames]

    # Arrange data
    barplot_data = {}
    assert len(data) % len(series_labels) == 0
    files_per_series = int(len(data) / len(series_labels))
    scene_names = np.array([df['name'][0].split('/')[1] for df in data_frames[:files_per_series]])
    for i, label in enumerate(series_labels):
        barplot_data[label] = data[i*files_per_series:(i+1)*files_per_series]
   
    if baseline is not None:
        baseline_data_dict = read_in_baseline(baseline)
        barplot_data['Baseline'] = []
        for name in scene_names:
            barplot_data['Baseline'].append(np.median(baseline_data_dict[name][skip_first_n_iterations:]) if strategy == 'median' else np.mean(baseline_data_dict[name][skip_first_n_iterations:]))
    
    # compute label locations and bar offsets
    x = np.arange(len(scene_names))
    n_keys = len(barplot_data.keys())
    width = 1 / (n_keys + 1)
    offsets = (np.array(range(n_keys)) * width) - (width / 2) - np.floor((n_keys - 1) / 2) * width

    # Draw barplot
    fig, ax = plt.subplots()
    for i, key in enumerate(barplot_data.keys()):
        ax.bar(x + offsets[i % n_keys], barplot_data[key], width, label=key)

    ax.set_ylabel('Execution time (ns)')
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