import numpy as np
from joblib import Parallel, delayed
import tempfile


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