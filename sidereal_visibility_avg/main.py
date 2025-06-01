"""
LOFAR SIDEREAL VISIBILITY AVERAGER (see https://arxiv.org/pdf/2501.07374)
"""

import sys
from time import time, sleep
from argparse import ArgumentParser
from shutil import rmtree
from multiprocessing import cpu_count
from numba import set_num_threads
from os import environ

# Logging
from .utils.logger import SVALogger
sys.stdout = SVALogger("sva_log.txt")
sys.stderr = sys.stdout

from .utils.clean import clean_binary_files, clean_mapping_files
from .utils.file_handling import check_folder_exists
from .utils.smearing import time_resolution
from .stack_ms import Stack
from .template_ms import Template


def parse_args():
    """
    Parse input arguments
    """

    parser = ArgumentParser(description='Sidereal visibility averaging')
    parser.add_argument('msin', nargs='+', help='Measurement sets to combine.')
    parser.add_argument('--msout', type=str, default='sva_output.ms', help='Measurement set output name.')
    parser.add_argument('--time_res', type=float, help='Desired time resolution in seconds.')
    parser.add_argument('--resolution', type=float, help='Desired spatial resolution (if given, you also have to give --fov_diam).')
    parser.add_argument('--fov_diam', type=float, help='Desired field of view diameter in degrees. This is used to calculate the optimal time resolution.')
    parser.add_argument('--dysco_bitrate', type=int, help='Dysco compression data bitrate.', default=None)
    parser.add_argument('--safe_memory', action='store_true', help='Use always memmap for DATA and WEIGHT_SPECTRUM storage (slower but less RAM cost).')
    parser.add_argument('--chunk_factor', type=float, help='Factor to reduce chunk size if RAM issues', default=1.)
    parser.add_argument('--make_only_template', action='store_true', help='Stop after making empty template.')
    parser.add_argument('--keep_mapping', action='store_true', help='Do not remove mapping files (useful for debugging).')
    parser.add_argument('--extra_cooldowns', action='store_true', help='Add extra 1-minute cooldown moments after intensive parallelisation (seems to magically help with intensive I/O jobs...).')
    parser.add_argument('--tmp', type=str, help='Temporary storage folder.', default='.')
    parser.add_argument('--ncpu', type=int, help='Maximum number of cpus (default is maximum available).', default=None)
    parser.add_argument('--only_lst_mapping', action='store_true', help='Only LST UVW mapping (faster but less accurate).')
    parser.add_argument('--dp3_uvw', action='store_true', help='Make UVW coordinates with DP3.')

    return parser.parse_args()


def main():
    """
    Main function
    """

    # Make template
    args = parse_args()
    print(args)

    # Set number of cores
    if args.ncpu is None:
        cpucount = int(environ.get("SLURM_CPUS_ON_NODE", min(max(cpu_count() - 1, 1), 64)))
    else:
        cpucount = args.ncpu
    set_num_threads(cpucount)

    if len(args.msin)<2:
        sys.exit(f"ERROR: Need more than 1 ms, currently given: {' '.join(args.msin)}")

    # Verify if output exists
    if check_folder_exists(args.msout):
        print(f"{args.msout} already exists, will be overwritten")
        rmtree(args.msout)
        sleep(5) # ensure that file is deleted with extra processing time

    # time averaging (upsampling factor)
    avg = 1
    if args.time_res is not None:
        time_res = args.time_res
        print(f"Use time resolution {time_res} seconds")
    elif args.resolution is not None and args.fov_diam is not None:
        time_res = time_resolution(args.resolution, args.fov_diam)
        print(f"Use time resolution {time_res} seconds")
    elif args.resolution is not None or args.fov_diam is not None:
        sys.exit("ERROR: if --resolution given, you also have to give --fov_diam, and vice versa.")
    else:
        if len(args.msin)>4:
            avg = 2
        time_res = None
        print(f"Additional time sampling factor {avg}\n")

    # Make template
    t = Template(args.msin, args.msout, tmp_folder=args.tmp, ncpu=cpucount)
    t.make_template(overwrite=True, time_res=time_res, avg_factor=avg, dysco_bitrate=args.dysco_bitrate,
                    only_lst_mapping=args.only_lst_mapping, DP3_uvw=args.dp3_uvw)
    print("\n############\nTemplate creation completed\n############")

    # Stack MS
    if not args.make_only_template:
        start_time = time()
        s = Stack(args.msin, args.msout, tmp_folder=args.tmp, chunkmem=args.chunk_factor)
        s.stack_all(keep_DP3_uvw=args.dp3_uvw, safe_mem=args.safe_memory, extra_cooldowns=args.extra_cooldowns)
        elapsed_time = time() - start_time
        print(f"Elapsed time for stacking: {elapsed_time} seconds")

    # Clean up mapping files
    if not args.keep_mapping:
        clean_mapping_files(args.msin, args.tmp)
    clean_binary_files(args.tmp)


if __name__ == '__main__':
    # Run main script
    main()
