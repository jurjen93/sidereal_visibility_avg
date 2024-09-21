import numpy as np
from joblib import Parallel, delayed
import tempfile
import json
from os import path, cpu_count
from .helpers import squeeze_to_intlist


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

def run_parallel_mapping(uniq_ant_pairs, antennas, ref_antennas, time_idxs, mapping_folder, io_batch_size=5):
    """
    Parallel processing of mapping with unique antenna pairs using joblib.
    Batch antenna pairs and write results in chunks to minimize I/O overhead.
    """

    # Determine optimal batch size
    batch_size = max(len(uniq_ant_pairs) // 10, 1)

    results = Parallel(n_jobs=max(cpu_count() - 3, 1), backend="loky")(
        delayed(process_antpair_batch)(uniq_ant_pairs[i:i + batch_size], antennas, ref_antennas, time_idxs)
        for i in range(0, len(uniq_ant_pairs), batch_size)
    )

    # Write the JSON mappings in batches to minimize I/O overhead
    for batch_idx in range(0, len(results), io_batch_size):
        try:
            batch_results = results[batch_idx:batch_idx + io_batch_size]
            mappings_to_write = {}

            # Collect all mappings in this I/O batch
            for mapping_batch in batch_results:
                mappings_to_write.update(mapping_batch)

            # Write to disk in a single batch
            for antpair, mapping in mappings_to_write.items():
                file_path = path.join(mapping_folder, '-'.join(map(str, antpair)) + '.json')
                with open(file_path, 'w') as f:
                    json.dump(mapping, f)

        except Exception as e:
            print(f"An error occurred while writing the mappings: {e}")