import numpy as np
from joblib import Parallel, delayed
import tempfile
import json
from os import path, cpu_count
from glob import glob
from .arrays_and_lists import find_closest_index_multi_array
from .ms_info import get_ms_content
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def sum_arrays_chunkwise(array1, array2, chunk_size):
    """
    Sums two arrays with maximum core utilization.

    Parameters:
    - array1: numpy.ndarray, the first array to sum.
    - array2: numpy.ndarray, the second array to sum.

    Returns:
    - numpy.ndarray, the summed array.
    """
    # Ensure the arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    # Function to sum a chunk
    def sum_chunk(start_idx, end_idx):
        return array1[start_idx:end_idx] + array2[start_idx:end_idx]

    # Calculate the total number of elements
    n_elements = len(array1)

    # Determine the number of available cores
    n_jobs = Parallel(n_jobs=-1)._effective_n_jobs()

    # Dynamically assign chunks to cores
    chunk_size = max(1, n_elements // (n_jobs * 4))  # Small enough for dynamic balancing
    chunk_indices = [(i, min(i + chunk_size, n_elements)) for i in range(0, n_elements, chunk_size)]

    # Parallel processing with dynamic scheduling
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(sum_chunk)(start, end) for start, end in chunk_indices
    )

    # Combine the results
    return np.concatenate(results)

def sum_arrays_chunkwise_old(array1, array2, chunk_size=1000, n_jobs=-1, un_memmap=True):
    """
    Sums two arrays in chunks using parallel processing.

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

    # Convert memmap arrays to regular arrays if `un_memmap` is True
    def try_unmemmap(array):
        if un_memmap and isinstance(array, np.memmap):
            try:
                return np.array(array)
            except MemoryError:
                pass  # Fall back to memmap if memory error occurs
        return array

    array1 = try_unmemmap(array1)
    array2 = try_unmemmap(array2)

    n = len(array1)

    # Determine the output storage type (memmap if input is memmap, else ndarray)
    if isinstance(array1, np.memmap) or isinstance(array2, np.memmap):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        result_array = np.memmap(temp_file.name, dtype=array1.dtype, mode='w+', shape=array1.shape)
    else:
        result_array = np.empty_like(array1)

    # Define the chunk summation task
    def sum_chunk(start, end):
        result_array[start:end] = array1[start:end] + array2[start:end]

    # Create chunk indices
    chunk_indices = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    # Use joblib for parallel processing, favoring threads for memory-bound tasks
    Parallel(n_jobs=n_jobs, prefer="threads")(delayed(sum_chunk)(start, end) for start, end in chunk_indices)

    # Flush changes to disk if using memmap
    if isinstance(result_array, np.memmap):
        result_array.flush()

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

    n_jobs = max(cpu_count() - 3, 1)  # Use all available CPU cores

    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit batches of antenna pairs for parallel processing
            futures = [
                executor.submit(process_antpair_batch, uniq_ant_pairs[i:i + batch_size], antennas, ref_antennas, time_idxs)
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