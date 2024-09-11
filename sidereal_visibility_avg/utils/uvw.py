from numpy import argsort
from scipy.interpolate import interp1d


def resample_uwv(uvw_arrays, row_idxs, time, time_ref):
    """
    Resample a uvw array to have N rows.

    :param:
        - uvw_arrays: UVW array with shape (num_points, 3)
        - row_idxs: Indices of rows to resample
        - time: Original time array
        - time_ref: Reference time array to resample to

    :return:
        - Resampled UVW array
    """

    # Get the original shape
    num_points, num_coords = uvw_arrays.shape

    if num_coords != 3:
        raise ValueError("Input array must have shape (num_points, 3)")

    # Sort the time array and corresponding UVW arrays
    sorted_indices = argsort(time)
    time_sorted = time[sorted_indices]
    uvw_sorted = uvw_arrays[sorted_indices, :]

    # Create a single interpolation function for the entire UVW array
    interp_func = interp1d(time_sorted, uvw_sorted, axis=0, kind='nearest', fill_value='extrapolate')

    # Apply the interpolation function to the reference times
    resampled_array = interp_func(time_ref[row_idxs])

    return resampled_array
