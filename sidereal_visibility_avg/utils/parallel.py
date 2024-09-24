import numpy as np
from joblib import Parallel, delayed
import tempfile
import json
from os import path, cpu_count
from .helpers import squeeze_to_intlist
from glob import glob
from .helpers import find_closest_index_multi_array
from .ms_info import get_ms_content
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


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
        # Get indices for the antenna pair
        pair_idx = np.squeeze(np.argwhere(np.all(antennas == antpair, axis=1)))
        ref_pair_idx = np.squeeze(np.argwhere(np.all(ref_antennas == antpair, axis=1)))

        # Ensure indices are valid
        if pair_idx.size == 0 or ref_pair_idx.size == 0:
            print(f"No matching indices found for antenna pair: {antpair}")
            continue  # Skip this antenna pair if no valid indices are found

        # Ensure `time_idxs` are within the bounds of `ref_pair_idx`
        valid_time_idxs = time_idxs[time_idxs < len(ref_pair_idx)]
        if len(valid_time_idxs) == 0:
            print(f"No valid time indices for antenna pair: {antpair}")
            continue

        ref_pair_idx = ref_pair_idx[valid_time_idxs]

        # Create the mapping dictionary for each pair
        mapping = {int(pair_idx[i]): int(ref_pair_idx[i]) for i in range(min(len(pair_idx), len(ref_pair_idx)))}
        mapping_batch[tuple(antpair)] = mapping  # Store in batch

    return mapping_batch


def run_parallel_mapping(uniq_ant_pairs, antennas, ref_antennas, time_idxs, mapping_folder):
    """
    Parallel processing of mapping with unique antenna pairs using ProcessPoolExecutor.
    Writes the mappings directly after each batch is processed.
    """

    # Determine optimal batch size
    batch_size = max(len(uniq_ant_pairs) // (cpu_count() * 2), 1)  # Split tasks across all cores

    def process_antpair_batch_wrapper(ant_pairs_batch):
        """Wrapper function to pass to ProcessPoolExecutor."""
        return process_antpair_batch(ant_pairs_batch, antennas, ref_antennas, time_idxs)

    # Use ProcessPoolExecutor for process-based parallelism
    n_jobs = max(cpu_count() - 3, 1)  # Use all available CPU cores

    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit batches of antenna pairs for parallel processing
            futures = [
                executor.submit(process_antpair_batch_wrapper, uniq_ant_pairs[i:i + batch_size])
                for i in range(0, len(uniq_ant_pairs), batch_size)
            ]

            for future in as_completed(futures):
                mapping_batch = future.result()
                # Write the JSON mappings after processing each batch
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


def process_baseline_uvw(baseline, folder, UVW):
    """Parallel processing baseline"""
    try:
        if not folder:
            folder = '.'
        mapping_folder_baseline = sorted(
            glob(folder + '/*_mapping/' + '-'.join([str(a) for a in baseline]) + '.json'))
        idxs_ref = np.unique(
            [idx for mapp in mapping_folder_baseline for idx in json.load(open(mapp)).values()])
        uvw_ref = UVW[list(idxs_ref)]
        for mapp in mapping_folder_baseline:
            idxs = [int(i) for i in json.load(open(mapp)).keys()]
            ms = glob('/'.join(mapp.split('/')[0:-1]).replace("_baseline_mapping", ""))[0]
            uvw_in = np.memmap(f'{ms}_uvw.tmp.dat', dtype=np.float32).reshape(-1, 3)[idxs]
            idxs_new = [int(i) for i in np.array(idxs_ref)[
                list(find_closest_index_multi_array(uvw_in[:, 0:2], uvw_ref[:, 0:2]))]]
            with open(mapp, 'w+') as f:
                json.dump(dict(zip(idxs, idxs_new)), f)
    except Exception as exc:
        print(f'Baseline {baseline} generated an exception: {exc}')


def process_baseline_int(baseline_indices, baselines, mslist):
    """Process baselines parallel executor"""
    results = []
    for b_idx in baseline_indices:
        baseline = baselines[b_idx]
        c = 0
        uvw = np.zeros((0, 3))
        time = np.array([])
        row_idxs = []
        for ms_idx, ms in enumerate(sorted(mslist)):
            mappingfolder = ms + '_baseline_mapping'
            try:
                mapjson = json.load(open(mappingfolder + '/' + '-'.join([str(a) for a in baseline]) + '.json'))
            except FileNotFoundError:
                c += 1
                continue

            row_idxs += list(mapjson.values())
            uvw = np.append(np.memmap(f'{ms}_uvw.tmp.dat', dtype=np.float32).reshape((-1, 3))[
                [int(i) for i in list(mapjson.keys())]], uvw, axis=0)

            time = np.append(np.memmap(f'{ms}_time.tmp.dat', dtype=np.float64)[[int(i) for i in list(mapjson.keys())]], time)

        results.append((list(np.unique(row_idxs)), uvw, b_idx, time))
    return results