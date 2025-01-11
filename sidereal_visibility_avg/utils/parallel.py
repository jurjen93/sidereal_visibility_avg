import numpy as np
import tempfile
import json
from os import path, cpu_count
from glob import glob
from .arrays_and_lists import find_closest_index_multi_array
from .ms_info import get_ms_content
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from numba import njit, prange


@njit(parallel=True)
def sum_chunks(result, array1, array2, start_indices, end_indices):
    """
    Numba-compiled function to sum chunks of arrays using slicing for efficiency.
    """
    for i in prange(len(start_indices)):
        start, end = start_indices[i], end_indices[i]
        result[start:end] = array1[start:end] + array2[start:end]

def sum_arrays_chunkwise(array1, array2, chunk_size=10_000, un_memmap=True):
    """
    Sums two arrays in chunks using numba for efficient processing.

    :param array1: np.ndarray or np.memmap
    :param array2: np.ndarray or np.memmap
    :param chunk_size: int, size of each chunk
    :param un_memmap: bool, whether to convert memmap arrays to regular arrays if they fit in memory
    :return: np.ndarray or np.memmap
    """

    # Ensure arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError("Arrays must have the same shape")

    original_shape = array1.shape
    n = array1.size  # Flattened length

    # Ensure arrays are contiguous
    array1_flat = np.ascontiguousarray(array1.ravel())
    array2_flat = np.ascontiguousarray(array2.ravel())

    # Optionally convert memmap arrays to regular arrays
    if un_memmap:
        if isinstance(array1_flat, np.memmap):
            array1_flat = np.array(array1_flat, copy=False)
        if isinstance(array2_flat, np.memmap):
            array2_flat = np.array(array2_flat, copy=False)

    # Determine result array type
    if isinstance(array1_flat, np.memmap) or isinstance(array2_flat, np.memmap):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        result_array_flat = np.memmap(temp_file.name, dtype=array1_flat.dtype, mode='w+', shape=array1_flat.shape)
    else:
        result_array_flat = np.empty_like(array1_flat)

    # Create chunk indices
    start_indices = np.arange(0, n, chunk_size)
    end_indices = np.minimum(start_indices + chunk_size, n)

    # Use Numba for summing chunks
    sum_chunks(result_array_flat, array1_flat, array2_flat, start_indices, end_indices)

    return result_array_flat.reshape(original_shape)


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

    n_jobs = max(cpu_count() - 5, 1)

    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit batches of antenna pairs for parallel processing
            futures = [
                executor.submit(
                    process_antpair_batch,
                    uniq_ant_pairs[i:i + batch_size],
                    antennas,
                    ref_antennas,
                    time_idxs
                )
                for i in range(0, len(uniq_ant_pairs), batch_size)
            ]

            for future in as_completed(futures):
                try:
                    mapping_batch = future.result()
                    # Write the JSON mappings after processing each batch
                    for antpair, mapping in mapping_batch.items():
                        file_path = path.join(mapping_folder, '-'.join(map(str, antpair)) + '.json')
                        with open(file_path, 'w') as f:
                            json.dump(mapping, f)
                except Exception as batch_error:
                    print(f"Error processing a batch: {batch_error}")

    except Exception as e:
        print(f"An error occurred while processing or writing mappings: {e}")


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