import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.core import base


from util import read_csv, convert_ns_to_format
import seaborn as sns

def add_plot_boxplot_subparser(parser):
    subparser = parser.add_parser('boxplot', help='')
    subparser.add_argument('files', type=str, nargs='+', help='The path to the csv file containing the measurements')
    subparser.add_argument('--series-labels', type=str, nargs='+', default=['Ours'])
    subparser.add_argument('--baseline', type=str, default=None, help='The path to the optix baseline')
    subparser.add_argument('--unit', choices=['s', 'ms', 'us', 'ns'], default='ms', help='The time unit, used for the histogram')
    subparser.add_argument('--out', type=str, default='plot.pdf', help='specifies the output file')
    subparser.add_argument('--skip-first-n-iterations', type=int, default=0, help='The number of first iterations to skip (used to skip warm-up iterations)')

def plot_boxplot(files, series_labels, baseline, unit, skip_first_n_iterations, out):
    
    # read in data
    data_frames = [read_csv(file) for file in files]

    # pick single iterations and skip first n iterations
    data_frames = [df[df['iterations'] == 1][skip_first_n_iterations:] for df in data_frames]

    # Check unit
    assert np.all(np.array([df['time_unit'] == 'ns' for df in data_frames]))

    # compute medians of each stage for all scenes
    data = np.array([convert_ns_to_format(df['real_time'].to_numpy(), unit) for df in data_frames])

    # Arrange data
    boxplot_data = []
    assert len(data) % len(series_labels) == 0
    files_per_series = int(len(data) / len(series_labels))
    scene_names = np.array([df['name'].to_numpy()[0].split('/')[1] for df in data_frames[:files_per_series]])
    
    for i, label in enumerate(series_labels):
        boxplot_data.extend(np.transpose(np.concatenate((data[i*files_per_series:(i+1)*files_per_series], [[label]*data.shape[1]]))))
   
    if baseline is not None:
        baseline_data_dict = read_in_baseline(baseline)
        baseline_data = []
        for name in scene_names:
            data_in_unit = convert_ns_to_format(baseline_data_dict[name][skip_first_n_iterations:], unit)
            baseline_data.append(data_in_unit)
        
        baseline_data = np.array(baseline_data)
        boxplot_data.extend(np.transpose(np.concatenate((baseline_data, [['Baseline']*baseline_data.shape[1]]))))

    # draw boxplot
    df_boxplot_data = pd.DataFrame(boxplot_data, columns=np.concatenate((scene_names, ['Implementation'])))
    df_boxplot_data_long = pd.melt(df_boxplot_data, 'Implementation', var_name="Scenes", value_name="Completion Time")
    df_boxplot_data_long['Completion Time']= df_boxplot_data_long['Completion Time'].astype('float')
    
    sns.boxplot(x="Scenes", hue="Implementation", y="Completion Time", data=df_boxplot_data_long)
    plt.savefig(out)
    

def read_in_baseline(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.split(',')
            data[line[0]] = np.array(line[1:], dtype=np.double) * 1000.0 # convert to nanoseconds
    
    return data
