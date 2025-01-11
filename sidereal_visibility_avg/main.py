"""
LOFAR SIDEREAL VISIBILITY AVERAGER
"""

import sys
from argparse import ArgumentParser
import time
from .utils.dysco import compress
from .utils.clean import clean_binary_files, clean_mapping_files
from .utils.file_handling import check_folder_exists
from .utils.smearing import time_resolution
from .template_ms import Template
from .stack_ms import Stack
from shutil import rmtree


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
    parser.add_argument('--dysco', action='store_true', help='Dysco compression of data')
    parser.add_argument('--safe_memory', action='store_true', help='Use always memmap for DATA and WEIGHT_SPECTRUM (slower but safer if concerned)')
    parser.add_argument('--make_only_template', action='store_true', help='Stop after making empty template')
    parser.add_argument('--interpolate_uvw', action='store_true', help='Interpolate UVW with nearest neighbours')
    parser.add_argument('--keep_mapping', action='store_true', help='Do not remove mapping files (useful for debugging)')

    return parser.parse_args()


def main():
    """
    Main function
    """

    # Make template
    args = parse_args()
    print(args)

    if len(args.msin)<2:
        sys.exit(f"ERROR: Need more than 1 ms, currently given: {' '.join(args.msin)}")

    # Verify if output exists
    if check_folder_exists(args.msout):
        print(f"{args.msout} already exists, will be overwritten")
        rmtree(args.msout)


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
    if args.interpolate_uvw:
        t.interpolate_uvw()
    else:
        t.calculate_uvw()
    print("\n############\nTemplate creation completed\n############")

    # Stack MS
    if not args.make_only_template:
        start_time = time.time()
        s = Stack(args.msin, args.msout)
        s.stack_all(interpolate_uvw=args.interpolate_uvw)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for stacking: {elapsed_time} seconds")

    # Clean up mapping files
    if not args.keep_mapping:
        clean_mapping_files(args.msin)
    clean_binary_files()

    # Apply dysco compression
    if args.dysco:
        compress(args.msout)


if __name__ == '__main__':
    main()
