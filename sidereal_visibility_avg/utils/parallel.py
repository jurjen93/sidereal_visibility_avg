import numpy as np
from joblib import Parallel, delayed
import tempfile
import json
from os import path, cpu_count
from .helpers import squeeze_to_intlist
from glob import glob
from .helpers import find_closest_index_multi_array
from .ms_info import get_ms_content


def sum_arrays_chunkwise(array1, array2, chunk_size=1000, n_jobs=-1, un_memmap=True):
    """
    Sums two arrays in chunks using joblib for parallel processing.

    :param:
        - array1: np.ndarray or np.memmap
        - array2: np.ndarray or np.memmap
        - chunk_size: int, size of each chunk
        - n_jobs: int, number of jobs for parallel processing (-1 means using all processors)
        - un_memmap: bool, whether to convert memmap arrays to regular arrays if they fit in memory

    :return:
        - np.ndarray or np.memmap: result array which is the sum of array1 and array2
    """

    # Ensure the arrays have the same length
    assert len(array1) == len(array2), "Arrays must have the same length"

    # Check if un-memmap is needed and feasible
    if un_memmap and isinstance(array1, np.memmap):
        try:
            array1 = np.array(array1)
        except MemoryError:
            pass  # If memory error, fall back to using memmap

    if un_memmap and isinstance(array2, np.memmap):
        try:
            array2 = np.array(array2)
        except MemoryError:
            pass  # If memory error, fall back to using memmap

    n = len(array1)

    # Determine the output storage type based on input type
    if isinstance(array1, np.memmap) or isinstance(array2, np.memmap):
        # Create a temporary file to store the result as a memmap
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        result_array = np.memmap(temp_file.name, dtype=array1.dtype, mode='w+', shape=array1.shape)
    else:
        result_array = np.empty_like(array1)

    def sum_chunk_to_result(start, end):
        result_array[start:end] = array1[start:end] + array2[start:end]

    # Create a generator for chunk indices
    chunks = ((i, min(i + chunk_size, n)) for i in range(0, n, chunk_size))

    # Parallel processing with threading preferred for better I/O handling
    Parallel(n_jobs=n_jobs, prefer="threads")(delayed(sum_chunk_to_result)(start, end) for start, end in chunks)

    return result_array

def process_antpair_batch(antpair_batch, antennas, ref_antennas, time_idxs):
    """
    Process a batch of antenna pairs, creating JSON mappings.
    """

    mapping_batch = {}
    for antpair in antpair_batch:
        pair_idx = squeeze_to_intlist(np.argwhere(np.all(antennas == antpair, axis=1)))
        ref_pair_idx = squeeze_to_intlist(np.argwhere(np.all(ref_antennas == antpair, axis=1))[time_idxs])

        # Create the mapping dictionary for each pair
        mapping = {int(pair_idx[i]): int(ref_pair_idx[i]) for i in range(min(len(pair_idx), len(ref_pair_idx)))}
        mapping_batch[tuple(antpair)] = mapping  # Store in batch

    return mapping_batch

def run_parallel_mapping(uniq_ant_pairs, antennas, ref_antennas, time_idxs, mapping_folder):
    """
    Parallel processing of mapping with unique antenna pairs using joblib.
    Writes the mappings directly after each batch is processed.
    """

    # Determine optimal batch size
    batch_size = max(len(uniq_ant_pairs) // (cpu_count() * 2), 1)  # Split tasks across all cores

    # Use joblib's Parallel with delayed for process-based parallelism
    n_jobs = max(cpu_count()-3, 1)  # Use all available CPU cores
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_antpair_batch)(uniq_ant_pairs[i:i + batch_size], antennas, ref_antennas, time_idxs)
        for i in range(0, len(uniq_ant_pairs), batch_size)
    )

    # Write the JSON mappings after processing each batch
    try:
        for mapping_batch in results:
            for antpair, mapping in mapping_batch.items():
                file_path = path.join(mapping_folder, '-'.join(map(str, antpair)) + '.json')
                with open(file_path, 'w') as f:
                    json.dump(mapping, f)

    except Exception as e:
        print(f"An error occurred while writing the mappings: {e}")


def process_ms(ms):
    """Process MS content in parallel (using separate processes)"""
    mscontent = get_ms_content(ms)
    stations, lofar_stations, channels, dfreq, total_time_seconds, dt, min_t, max_t = mscontent.values()
    return stations, lofar_stations, channels, dfreq, dt, min_t, max_t


def process_baseline(baseline, mslist, UVW):
    """Parallel processing for baseline mapping based on UVW coordinates."""
    try:

        folder = '/'.join(mslist[0].split('/')[:-1])
        if not folder:
            folder = '.'

        mapping_files = glob(f'{folder}/*_mapping/' + '-'.join(map(str, baseline)) + '.json')

        # Pre-load and cache reference indices and UVW data to minimize repeated file reads
        idxs_ref = set()  # Using a set for uniqueness and better performance
        for mapping_file in mapping_files:
            with open(mapping_file) as f:
                mapping_data = json.load(f)
                idxs_ref.update(map(int, mapping_data.values()))  # Collect all unique indices

        # Convert to sorted list once for efficient numpy operations
        idxs_ref = sorted(idxs_ref)
        uvw_ref = UVW[idxs_ref]  # Load UVW reference data once

        # Process each mapping file and update based on UVW coordinates
        for mapping_file in mapping_files:
            with open(mapping_file) as f:
                idxs = list(map(int, json.load(f).keys()))  # Original indices from mapping

            ms_file = (glob(f'{folder}/' + '_'.join(mapping_file.split('/')[-1].split('_')[:-2]) + '*')[0]
                       .replace("_baseline_mapping", ""))
            uvw_in = np.memmap(f'{ms_file}_uvw.tmp.dat', dtype=np.float32, mode='r').reshape(-1, 3)[idxs]

            # Efficiently find closest indices using vectorized operations
            idxs_new = np.array(idxs_ref)[
                find_closest_index_multi_array(uvw_in[:, :2], uvw_ref[:, :2])
            ]

            # Update JSON file in one I/O operation
            with open(mapping_file, 'w') as f:
                json.dump(dict(zip(idxs, idxs_new)), f)

    except Exception as exc:
        print(f'Baseline {baseline} generated an exception: {exc}')