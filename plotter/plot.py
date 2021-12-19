import argparse
from plot_distribution import plot_distribution, add_plot_distribution_subparser
from plot_stages import plot_stages, add_plot_stages_subparser

def main():

    # Setup parser
    parser = argparse.ArgumentParser(description='Plot distribution')
    subparsers = parser.add_subparsers(dest='plot', help='')
    subparsers.required = True
    add_plot_distribution_subparser(subparsers)
    add_plot_stages_subparser(subparsers)
    
    args = vars(parser.parse_args())
    plot(**args)

def plot(plot, **kwargs):
    if(plot == 'distribution'):
        plot_distribution(**kwargs)
    elif plot == 'stages':
        plot_stages(**kwargs)
    else:
        raise Exception("Invalid plot type")

if __name__ == '__main__':
    main()