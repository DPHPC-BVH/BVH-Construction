import argparse
from plot_distribution import plot_distribution, add_plot_distribution_subparser

def main():

    # Setup parser
    parser = argparse.ArgumentParser(description='Plot distribution')
    subparsers = parser.add_subparsers(dest='plot', help='')
    subparsers.required = True
    add_plot_distribution_subparser(subparsers)
    
    args = vars(parser.parse_args())
    plot(**args)

def plot(plot, **kwargs):
    if(plot == 'distribution'):
        plot_distribution(**kwargs)
    else:
        raise Exception("Invalid plot type")

if __name__ == '__main__':
    main()