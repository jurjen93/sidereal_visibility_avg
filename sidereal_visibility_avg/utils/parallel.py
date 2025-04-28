import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from os import path
import gc

import numpy as np
from numba import njit, prange, get_num_threads

from .arrays_and_lists import find_closest_index_multi_array
from .ms_info import get_ms_content

# Settings
n_threads = get_num_threads()
target_chunks_per_thread = 8
min_chunk = 1024
max_chunk = 65536


@njit(parallel=True)
def replace_nan(arr):
    for i in prange(arr.shape[0]):
        arr[i][np.isnan(arr[i])] = 0.0
    return arr


@njit(parallel=True, fastmath=True)
def multiply_flat_arrays_numba(A_flat, B_flat, out_flat):
    """
    Numba kernel that multiplies two flattened arrays into a flattened output.
    """
    for i in prange(A_flat.size):
        out_flat[i] = A_flat[i] * B_flat[i]


def multiply_arrays(A, B):
    """
    Multiplies two NumPy arrays of the same shape elementwise.
    """

    assert A.shape == B.shape, "Arrays must have the same shape"

    out = np.empty_like(A)
    multiply_flat_arrays_numba(A.ravel(), B.ravel(), out.ravel())

    return out


@njit(parallel=True, fastmath=True)
def sum_flat_arrays_numba(A_flat, B_flat, out_flat):
    n = A_flat.size

    chunk_size = max(min_chunk, min(max_chunk, (n + n_threads * target_chunks_per_thread - 1) // (n_threads * target_chunks_per_thread)))

    # Parallel over chunks
    n_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_id in prange(n_chunks):
        chunk_start = chunk_id * chunk_size
        chunk_end = min(chunk_start + chunk_size, n)

        for i in range(chunk_start, chunk_end):
            out_flat[i] = A_flat[i] + B_flat[i]


def sum_arrays(A, B):
    """
    Sums two NumPy arrays elementwise.
    """

    assert A.shape == B.shape, "Arrays must have the same shape"

    out = np.empty_like(A)
    sum_flat_arrays_numba(A.ravel(), B.ravel(), out.ravel())

    return out


@njit(parallel=True)
def inplace_sum_time(A, row_idxs_new, B):
    n_cols = B.size
    for i in prange(row_idxs_new.size):
        row_new = row_idxs_new[i]
        for j in range(n_cols):
            A[row_new, j] += B[j]


@njit(parallel=True)
def inplace_sum_timefreq(A, row_idxs_new, freq_idxs, B, row_idxs):
    for i in prange(row_idxs_new.size):
        row_new = row_idxs_new[i]
        row_old = row_idxs[i]
        for j in range(freq_idxs.size):
            A[row_new, freq_idxs[j]] += B[row_old, j]


@njit
def nanmean_excluding_zeros_flat(a):
    """
    Compute the mean of a 1D array over values that are neither NaN nor 0.
    """
    s = 0.0
    count = 0
    for i in range(a.shape[0]):
        val = a[i]
        # Skip if NaN or exactly 0
        if not np.isnan(val) and val != 0:
            s += val
            count += 1
    if count == 0:
        return np.nan
    else:
        return s / count


@njit
def nanmean_excluding_zeros_axis0(data):
    """
    Compute the mean along axis 0 for a 2D array 'data' where each column’s
    mean is computed over values that are not NaN and not 0.
    """
    n, m = data.shape
    out = np.empty(m, dtype=data.dtype)
    for j in range(m):
        s = 0.0
        count = 0
        for i in range(n):
            val = data[i, j]
            if not np.isnan(val) and val != 0:
                s += val
                count += 1
        if count == 0:
            out[j] = np.nan
        else:
            out[j] = s / count
    return out


def nozeros_nanmean(a, axis=None):
    """
    Compute the mean of an array that are neither NaN nor 0.

    If axis is None, returns a scalar.
    If an axis is specified, the mean is computed along that axis.
    This version uses Numba for very fast execution.
    """
    # Ensure array is contiguous
    a = np.ascontiguousarray(a)

    if axis is None:
        a_flat = a.ravel()
        return nanmean_excluding_zeros_flat(a_flat)
    else:
        # Move the target axis to the front so that we average along axis 0.
        a_moved = np.moveaxis(a, axis, 0)
        n = a_moved.shape[0]
        # Reshape to 2D: shape (n, prod(other_dims))
        m = a_moved.size // n
        data_2d = a_moved.reshape(n, m)
        # Compute mean along axis 0 for each "column"
        out_flat = nanmean_excluding_zeros_axis0(data_2d)
        # Reshape to original shape with the reduced axis removed.
        out_shape = a.shape[:axis] + a.shape[axis + 1:]
        return out_flat.reshape(out_shape)


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


def run_parallel_mapping(uniq_ant_pairs, antennas, ref_antennas, time_idxs, mapping_folder, cpucount):
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


def process_baseline_uvw(baseline, folder, UVW, tmpfolder):
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
        for pathf in mapping_files:
            with open(pathf) as f:
                mapping = json.load(f)
                mappings.append((pathf, mapping))
                idxs_ref.update(mapping.values())

        idxs_ref = np.unique(np.fromiter(idxs_ref, dtype=int))
        uvw_ref = UVW[np.abs(idxs_ref)]

        # Loop over mapping files with nearest neighbouring
        for pathf, mapping in mappings:
            idxs = np.fromiter((int(i) for i in mapping.keys()), dtype=int)
            ms_dir = '/'.join(pathf.split('/')[:-1]).replace("_baseline_mapping", "")
            ms = glob(ms_dir)[0]
            uvw_in = np.memmap(tmpfolder+f'{path.basename(ms)}_uvw.tmp.dat', dtype=np.float32).reshape(-1, 3)[idxs]
            idxs_new = np.array(idxs_ref)[find_closest_index_multi_array(uvw_in[:, :2], uvw_ref[:, :2])]
            new_mapping = dict(zip(map(str, idxs), idxs_new.astype(int).tolist()))
            with open(pathf, 'w') as f:
                json.dump(new_mapping, f)

    except Exception as exc:
        print(f'Baseline {baseline} generated an exception: {exc}')


def process_baseline_int(baseline_indices, baselines, mslist, tmpfolder):
    """Process baselines parallel executor"""

    results = []
    for b_idx in baseline_indices:
        baseline = baselines[b_idx]
        uvw = np.zeros((0, 3))
        time = np.array([])
        row_idxs = []
        for ms_idx, ms in enumerate(sorted(mslist)):
            mappingfolder = tmpfolder+path.basename(ms) + '_baseline_mapping'
            try:
                mapjson = json.load(open(mappingfolder + '/' + '-'.join([str(a) for a in baseline]) + '.json'))
            except FileNotFoundError:
                continue

            row_idxs += list(mapjson.values())
            uvw = np.append(np.memmap(tmpfolder+f'{path.basename(ms)}_uvw.tmp.dat', dtype=np.float32).reshape((-1, 3))[
                [int(i) for i in list(mapjson.keys())]], uvw, axis=0)

            time = np.append(np.memmap(tmpfolder+f'{path.basename(ms)}_time.tmp.dat', dtype=np.float64)[[int(i) for i in list(mapjson.keys())]], time)

        results.append((list(np.unique(np.abs(row_idxs))), uvw, baseline, time))
    return results