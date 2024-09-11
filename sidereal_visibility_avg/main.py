"""
LOFAR SIDEREAL VISIBILITY AVERAGER
This script can be used to average visibilities over sidereal time, when using multiple observations of the same FoV.

Example: python ms_merger.py --msout <MS_NAME> *.ms
The wildcard is in this example combining a collection of measurement sets

Strategy:
    1) Make a template using the 'default_ms' option from casacore.tables (Template class).
       The template inclues all baselines, frequency, and smallest time spacing from all input MS.
       Time is converted to Local Sidereal Time (LST).

    2) Map baselines from input MS to template MS.
        This step makes *baseline_mapping folders with the baseline mappings in json files.

    3) Interpolate new UV data with nearest neighbours.

    4) Make exact mapping between input MS and template MS, using only UV data points.

    5) Average measurement sets in the template (Stack class).
        The averaging is done with a weighted average, using the FLAG and WEIGHT_SPECTRUM columns.
"""


import sys
from argparse import ArgumentParser
import time
from .utils.dysco import compress
from .utils.clean import clean_binary_files, clean_mapping_files
from .utils.files import check_folder_exists
from .utils.smearing import time_resolution
from .utils.plot import make_baseline_uvw_plots
from sidereal_visibility_avg.template_ms import Template
from sidereal_visibility_avg.stack_ms import Stack


def parse_args():
    """
    Parse input arguments
    """

    parser = ArgumentParser(description='Sidereal visibility averaging')
    parser.add_argument('msin', nargs='+', help='Measurement sets to combine')
    parser.add_argument('--msout', type=str, default='empty.ms', help='Measurement set output name')
    parser.add_argument('--time_res', type=float, help='Desired time resolution in seconds')
    parser.add_argument('--resolution', type=float, help='Desired spatial resolution (if given, you also have to give --fov_diam)')
    parser.add_argument('--fov_diam', type=float, help='Desired field of view diameter in degrees. This is used to calculate the optimal time resolution.')
    parser.add_argument('--record_time', action='store_true', help='Record wall-time of stacking')
    parser.add_argument('--chunk_mem', type=float, default=1., help='Additional memory chunk parameter (larger for smaller chunks)')
    parser.add_argument('--no_dysco', action='store_true', help='No Dysco compression of data')
    parser.add_argument('--make_only_template', action='store_true', help='Stop after making empty template')
    parser.add_argument('--keep_mapping', action='store_true', help='Do not remove mapping files')
    parser.add_argument('--plot_uv_baseline_coverage', action='store_true', help='make plots with baseline versus UV')

    return parser.parse_args()


def main():
    """
    Main function
    """

    # Make template
    args = parse_args()

    one_lst_day_sec = 86164.1

    # Verify if output exists
    if check_folder_exists(args.msout):
        sys.exit(f"ERROR: {args.msout} already exists! Delete file first if you want to overwrite.")

    avg = 1
    if args.time_res is not None:
        avg = 1
        time_res = args.time_res
        print(f"Use time resolution {time_res} seconds")
    elif args.resolution is not None and args.fov_diam is not None:
        time_res = time_resolution(args.resolution, args.fov_diam)
        print(f"Use time resolution {time_res} seconds")
    elif args.resolution is not None or args.fov_diam is not None:
        sys.exit("ERROR: if --resolution given, you also have to give --fov_diam, and vice versa.")
    else:
        avg = 2
        time_res = None
        print(f"Additional time sampling factor {avg}\n")

    t = Template(args.msin, args.msout)
    t.make_template(overwrite=True, time_res=time_res, avg_factor=avg)
    t.make_uvw()
    print("\n############\nTemplate creation completed\n############")

    # Stack MS
    if not args.make_only_template:
        if args.record_time:
            start_time = time.time()
        s = Stack(args.msin, args.msout, chunkmem=args.chunk_mem)
        s.stack_all()
        if args.record_time:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time for stacking: {elapsed_time//60} minutes")

    if args.plot_uv_baseline_coverage:
        make_baseline_uvw_plots(args.msout, args.msin)

    # Clean up mapping files
    if not args.keep_mapping:
        clean_mapping_files(args.msin)
    clean_binary_files()

    # Apply dysco compression
    if not args.no_dysco:
        compress(args.msout)


if __name__ == '__main__':
    main()
