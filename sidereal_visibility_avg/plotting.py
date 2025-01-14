from sys import exit
from os import system as run_command
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from casacore.tables import table

from .utils.ms_info import make_ant_pairs, get_station_id


def plot_baseline_track(t_final_name: str = None, t_input_names: list = None, baseline='0-1', UV=True, saveas=None):
    """
    Plot baseline track

    :param:
        - t_final_name: table with final name
        - t_input_names: tables to compare with
        - mappingfiles: baseline mapping files
    """


    if len(t_input_names) > 4:
        exit("ERROR: Can just plot 4 inputs")

    colors = ['red', 'green', 'yellow', 'black']

    if not UV:
        print("MAKE UW PLOT")

    ant1, ant2 = baseline.split('-')
    plt.close()

    for n, t_input_name in enumerate(t_input_names):
        print(t_input_name)

        ref_stats, ref_ids = get_station_id(t_final_name)
        new_stats, new_ids = get_station_id(t_input_name)

        id_map = dict(zip([ref_stats.index(a) for a in new_stats], new_ids))

        print(ref_stats[int(float(ant1))], ref_stats[int(float(ant2))])

        with table(t_final_name, ack=False) as f:
            fsub = f.query(f'ANTENNA1={ant1} AND ANTENNA2={ant2} AND NOT ALL(WEIGHT_SPECTRUM == 0)', columns='UVW')
            uvw1 = fsub.getcol("UVW")

        with table(t_input_name, ack=False) as f:
            fsub = f.query(f'ANTENNA1={id_map[int(ant1)]} AND ANTENNA2={id_map[int(ant2)]} AND NOT ALL(WEIGHT_SPECTRUM == 0)', columns='UVW')
            uvw2 = fsub.getcol("UVW")

        # Scatter plot for uvw1
        if n == 0:
            lbl = 'Final dataset'
        else:
            lbl = None

        if uvw2.ndim>1:

            plt.scatter(uvw1[:, 0], uvw1[:, 2] if UV else uvw1[:, 3], label=lbl, color='blue', edgecolor='black', alpha=0.2, s=130, marker='o')

            # Scatter plot for uvw2
            plt.scatter(uvw2[:, 0], uvw2[:, 2] if UV else uvw2[:, 3], label=f'Dataset {n}', color=colors[n], edgecolor='black', alpha=0.7, s=70, marker='*')


    # Adding labels and title
    plt.xlabel("U (m)", fontsize=14)
    plt.ylabel("V (m)" if UV else "W (m)", fontsize=14)

    # Adding grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adding legend
    plt.legend(fontsize=12)

    plt.tight_layout()

    if saveas is None:
        plt.show()
    else:
        plt.savefig(saveas, dpi=150)
        plt.close()


def make_baseline_uvw_plots(tabl, mslist):
    """
    Make baseline plots
    """

    run_command('mkdir -p baseline_plots')

    ants = table(tabl + "::ANTENNA", ack=False)
    baselines = np.c_[make_ant_pairs(ants.nrows(), 1)]
    ants.close()

    for baseline in baselines:
        bl = '-'.join([str(a) for a in baseline])
        plot_baseline_track(tabl, sorted(mslist), bl, saveas=f'baseline_plots/{bl}.png')


def parse_args():
    """
    Parse input arguments
    """

    parser = ArgumentParser(description='Plot UV tracks for baseline')
    parser.add_argument('msin', nargs='+', help='Measurement sets to combine (up to 4).')
    parser.add_argument('--msout', type=str, default='empty.ms', help='Measurement set output name.')
    parser.add_argument('--baseline', type=str, default='0-1', help='Baseline numbers')
    parser.add_argument('--saveas', type=str, default=None, help='Name of PNG (if not given, it just plots)')

    return parser.parse_args()


def main():
    args = parse_args()
    plot_baseline_track(t_final_name=args.msout, t_input_names=args.msin, baseline=args.baseline, saveas=args.saveas)


if __name__ == '__main__':
    main()
