import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from multiprocessing import cpu_count
from os import path
import gc
from time import sleep

import numpy as np
from numba import jit, njit, prange, set_num_threads

from .arrays_and_lists import find_closest_index_multi_array
from .ms_info import get_ms_content

# Ensure some cores free
cpucount = max(cpu_count() - 1, 1)
set_num_threads(cpucount)


@jit(nopython=True, parallel=True)
def replace_nan(arr):
    for i in prange(arr.shape[0]):
        arr[i][np.isnan(arr[i])] = 0.0
    return arr


@jit(nopython=True, parallel=True)
def inplace_sum_1d(new_data, data, row_idxs):
    """
    Update array at specified indices by adding corresponding values from `data`.
    Uses Numba for efficient parallel computation.

    Parameters:
    - new_data: 1D NumPy array to update.
    - data: 1D NumPy array with values to add.
    - row_idxs: 1D array of row indices in `new_data`.
    """
    for i in prange(len(row_idxs)):
        row = row_idxs[i]
        new_data[row] += data[i]


@jit(nopython=True, parallel=True)
def inplace_sum_2d(new_data, data, row_idxs_new, freq_idxs, row_idxs):
    """
    Update array at specified indices by adding corresponding values from `data`.
    Uses Numba for efficient parallel computation.

    Parameters:
    - new_data: 2D NumPy array to update.
    - data: 2D NumPy array with values to add.
    - row_idxs_new: 1D array of row indices in `new_data`.
    - freq_idxs: 1D array of column indices in `new_data`.
    - row_idxs: 1D array of row indices in `data`.
    """
    for i in prange(len(row_idxs_new)):
        row = row_idxs_new[i]
        row2 = row_idxs[i]
        for j in prange(len(freq_idxs)):
            col = freq_idxs[j]
            new_data[row, col] += data[row2, j]


@jit(nopython=True, parallel=True)
def inplace_multiply(new_data, data, row_idxs_new, freq_idxs, row_idxs):
    """
    Update array at specified indices by multiplying corresponding values from `data`.
    Uses Numba for efficient parallel computation.

    Parameters:
    - new_data: 2D NumPy array to update.
    - data: 2D NumPy array with values to multiply.
    - row_idxs_new: 1D array of row indices in `new_data`.
    - freq_idxs: 1D array of column indices in `new_data`.
    - row_idxs: 1D array of row indices in `data`.
    """
    for i in prange(len(row_idxs_new)):
        row = row_idxs_new[i]
        row2 = row_idxs[i]
        for j in prange(len(freq_idxs)):
            col = freq_idxs[j]
            new_data[row, col] *= data[row2, j]


@njit(parallel=True)
def multiply_flat_arrays_numba(A_flat, B_flat, out_flat):
    """
    Numba kernel that multiplies two flattened arrays into a flattened output.
    A_flat, B_flat, out_flat must all be 1D and of the same size.
    """
    n = A_flat.size
    for i in prange(n):
        out_flat[i] = A_flat[i] * B_flat[i]


def multiply_arrays(A, B):
    """
    Multiplies two NumPy arrays of the same shape elementwise.
    Uses Numba with parallel=True on flattened data for high efficiency.

    Parameters
    ----------
    A, B : np.ndarray
        Arrays of the same shape and compatible dtypes.

    Returns
    -------
    out : np.ndarray
        The elementwise product of A and B.
    """
    # Ensure A and B have the same shape
    assert A.shape == B.shape, "Arrays must have the same shape"

    # Allocate an output array (same shape and dtype as A)
    out = np.empty_like(A)

    # Get flattened (ravel) views of A, B, and out
    if isinstance(A, np.memmap):
        A_flat = np.array(A).ravel()
    else:
        A_flat = A.ravel()
    B_flat = B.ravel()
    out_flat = out.ravel()

    # Call the parallel Numba kernel on the flattened data
    multiply_flat_arrays_numba(A_flat, B_flat, out_flat)

    return out


@jit(nopython=True, parallel=True)
def sum_flat_arrays_numba(A_flat, B_flat, out_flat):
    """
    Numba kernel that sums two flattened arrays into a flattened output.
    """
    n = A_flat.size
    for i in prange(n):
        out_flat[i] = A_flat[i] + B_flat[i]


def sum_arrays(A, B):
    """
    Sums two NumPy arrays of any shape (A and B) elementwise.
    Uses Numba with nopython=True, parallel=True on flattened data.
    """
    # Make sure they have the same shape
    assert A.shape == B.shape, "Arrays must have the same shape"

    # Allocate output array (same shape, dtype as A)
    out = np.empty_like(A)

    # Flatten (ravel) the arrays to 1D
    if isinstance(A, np.memmap):
        A_flat = np.array(A).ravel()
    else:
        A_flat = A.ravel()
    B_flat = B.ravel()
    out_flat = out.ravel()

    # Call the parallel Numba kernel on the flattened data
    sum_flat_arrays_numba(A_flat, B_flat, out_flat)

    return out


@njit(parallel=True)
def sum_chunks(result, array1, array2, start_indices, end_indices):
    """
    Numba-compiled function to sum chunks of arrays.
    """
    for i in prange(len(start_indices)):
        start, end = start_indices[i], end_indices[i]
        for j in range(start, end):
            result[j] = array1[j] + array2[j]  # Avoid slicing for better efficiency


def sum_arrays_chunkwise_old(array1, array2, chunk_size=1000, un_memmap=True):
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

    # Ensure arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    n = len(array1)

    # Adjust chunk size for large arrays
    chunk_size = min(chunk_size, n)

    # Optionally convert memmap arrays to regular arrays
    def try_convert_to_array(arr):
        if un_memmap and isinstance(arr, np.memmap):
            try:
                return np.array(arr)
            except MemoryError:
                return arr  # Fallback to memmap
        return arr

    array1 = try_convert_to_array(array1)
    array2 = try_convert_to_array(array2)

    # Determine result array type
    if isinstance(array1, np.memmap) or isinstance(array2, np.memmap):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        result_array = np.memmap(temp_file.name, dtype=array1.dtype, mode='w+', shape=array1.shape)
    else:
        result_array = np.empty_like(array1)

    # Create chunk indices
    start_indices = np.arange(0, n, chunk_size)
    end_indices = np.minimum(start_indices + chunk_size, n)

    # Use Numba for summing chunks
    sum_chunks(result_array, array1, array2, start_indices, end_indices)

    # If a temporary file was created, return the memmap; otherwise, return the array
    return result_array


def process_antpair_batch(antpair_batch, antennas, ref_antennas, time_idxs):
    """
    Process a batch of antenna pairs, creating JSON mappings.
    """

    mapping_batch = {}

    for antpair in antpair_batch:

        if antpair[0] > antpair[1]:
            inverse=True
            antpair = sorted(antpair)
        else:
            inverse=False

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
        mapping = {int(pair_idx[i]): (-1 if inverse else 1) * int(ref_pair_idx[i]) for i in range(min(len(pair_idx), len(ref_pair_idx)))}
        mapping_batch[tuple(antpair)] = mapping  # Store in batch

    return mapping_batch


def run_parallel_mapping(uniq_ant_pairs, antennas, ref_antennas, time_idxs, mapping_folder):
    """
    Parallel processing of mapping with unique antenna pairs using ProcessPoolExecutor.
    Writes the mappings directly after each batch is processed.
    """

    # Determine optimal batch size
    batch_size = max(len(uniq_ant_pairs) // cpucount, 1)  # Split tasks across all cores

    n_jobs = cpucount

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
                        file_path = path.join(mapping_folder, '-'.join(map(str, sorted(antpair))) + '.json')
                        with open(file_path, 'w') as f:
                            json.dump(mapping, f)
                except Exception as batch_error:
                    print(f"Error processing a batch: {batch_error}")

    except Exception as e:
        print(f"An error occurred while processing or writing mappings: {e}")

    gc.collect()


def process_ms(ms):
    """Process MS content in parallel (using separate processes)"""

    mscontent = get_ms_content(ms)
    stations, lofar_stations, channels, dfreq, total_time_seconds, dt, min_t, max_t = mscontent.values()
    return stations, lofar_stations, channels, dfreq, dt, min_t, max_t


def process_baseline_uvw(baseline, folder, UVW):
    """Parallel processing of one baseline"""
    try:
        folder = folder or '.'
        baseline_str = '-'.join(map(str, baseline))
        mapping_files = sorted(glob(f'{folder}/*_mapping/{baseline_str}.json'))

        if not mapping_files:
            return

        # Load all mappings and collect idxs_ref
        idxs_ref = set()
        mappings = []
        for path in mapping_files:
            with open(path) as f:
                mapping = json.load(f)
                mappings.append((path, mapping))
                idxs_ref.update(mapping.values())

        idxs_ref = np.unique(np.fromiter(idxs_ref, dtype=int))
        uvw_ref = UVW[np.abs(idxs_ref)]

        for path, mapping in mappings:
            idxs = np.fromiter((int(i) for i in mapping.keys()), dtype=int)
            ms_dir = '/'.join(path.split('/')[:-1]).replace("_baseline_mapping", "")
            ms = glob(ms_dir)[0]
            uvw_in = np.memmap(f'{ms}_uvw.tmp.dat', dtype=np.float32).reshape(-1, 3)[idxs]
            idxs_new = np.array(idxs_ref)[find_closest_index_multi_array(uvw_in[:, :2], uvw_ref[:, :2])]
            new_mapping = dict(zip(map(str, idxs), idxs_new.astype(int).tolist()))
            with open(path, 'w') as f:
                json.dump(new_mapping, f)

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

        results.append((list(np.unique(np.abs(row_idxs))), uvw, baseline, time))
    return results
