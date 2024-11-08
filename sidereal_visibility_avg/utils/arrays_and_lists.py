from sys import exit
import numpy as np
from casacore.tables import table
from scipy.spatial import cKDTree
from math import ceil
from sklearn.neighbors import NearestNeighbors


def make_odd(i):
    """
    Make odd number

    :param:
        - i: digit

    :return: odd digit
    """

    if int(i) % 2 == 0:
        i += 1

    return int(i)


def get_largest_divider(inp, max=1000):
    """
    Get largest divider

    :param inp: input number
    :param max: max divider

    :return: largest divider from inp bound by max
    """

    for r in range(max)[::-1]:
        if inp % r == 0:
            return r
    exit("ERROR: code should not arrive here.")


def repeat_elements(original_list, repeat_count):
    """
    Repeat each element in the original list a specified number of times.

    :param:
        - original_list: The original list to be transformed.
        - repeat_count: The number of times each element should be repeated.

    :return:
        - A new list where each element from the original list is repeated.
    """

    return np.array([element for element in original_list for _ in range(repeat_count)])


def find_closest_index(arr, value):
    """
    Find the index of the closest value in the array to the given value.

    :param:
        - arr: numpy array.
        - value: float value to find the closest value to.

    :return:
        - The index of the closest value in the array.
    """

    # Calculate the absolute difference with the given value
    differences = np.abs(arr - value)
    print(f"Minimal difference: {min(differences)}")

    # Find the index of the minimum difference
    closest_index = np.argmin(differences)

    return closest_index


def find_closest_index_list(a1, a2):
    """
    Find the indices of the closest values between two arrays.
    """

    a1, a2 = np.array(a1), np.array(a2)

    # Create a nearest neighbors model (use 'ball_tree' or 'kd_tree' for small, 'brute' for large datasets)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(a2.reshape(-1, 1))

    # Find the nearest neighbors for each element in a1
    distances, indices = nbrs.kneighbors(a1.reshape(-1, 1))

    return indices.flatten()


def find_closest_index_multi_array(a1, a2):
    """
    Find the indices of the closest values.

    :param:
        - a1: first array (shape NxM)
        - a2: second array (shape PxM)

    :return:
        - A list of indices corresponding to the nearest neighbors in a2 for each point in a1.
    """

    # Ensure inputs are numpy arrays
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    # Build a KDTree for a2 only
    tree = cKDTree(a2)

    # Query the tree for the closest points in a1
    distances, indices = tree.query(a1)

    # Check for negated versions of a1, directly
    neg_distances, neg_indices = tree.query(-a1)

    # Combine the results from both queries
    final_indices = np.where(neg_distances < distances, neg_indices, indices)

    return final_indices.tolist()


def map_array_dict(arr, dct):
    """
    Maps elements of the input_array to new values using a mapping_dict

    :param:
        - arr: numpy array of integers that need to be mapped.
        - dct: dictionary where each key-value pair represents an original value and its corresponding new value.

    :return:
        - An array where each element has been mapped according to the mapping_dict.
        (If an element in the input array does not have a corresponding mapping in the mapping_dict, it will be left unchanged)
    """

    lookup = np.vectorize(lambda x: dct.get(x, x))
    output_array = lookup(arr)

    return output_array


def get_avg_factor(mslist, less_avg=1):
    """
    Calculate optimized averaging factor

    :param:
        - mslist: measurement set list
        - less_avg: factor to reduce averaging
    :return:
        - averaging factor
    """

    uniq_obs = []
    for ms in mslist:
        obs = table(ms + "::OBSERVATION", ack=False)
        uniq_obs.append(obs.getcol("TIME_RANGE")[0][0])
        obs.close()
    obs_count = len(np.unique(uniq_obs))
    avgfactor = ceil(np.sqrt(obs_count / less_avg))
    if avgfactor < 1:
        return avgfactor
    else:
        return int(avgfactor)


def add_axis(arr, ax_size):
    """
    Add ax dimension with a specific size

    :param:
        - arr: numpy array
        - ax_size: axis size

    :return:
        - output with new axis dimension with a particular size
    """

    or_shape = arr.shape
    new_shape = list(or_shape) + [ax_size]
    return np.repeat(arr, ax_size).reshape(new_shape)


def resample_array(data, factor):
    """
    Resamples the input data array such that the number of points increases by a factor.
    The lowest and highest values remain the same, and the spacing between points remains equal.

    :param:
        - data: The input data array to resample.
        - factor: The factor by which to increase the number of data points.

    :return:
        - The resampled data array.
    """

    # Number of points in input
    n_points = len(data)

    # Number of points in the resampled array
    new_n_points = factor * (n_points - 1) + 1

    # New set of equally spaced indices
    original_indices = np.arange(n_points)
    new_indices = np.linspace(0, n_points - 1, new_n_points + 1)

    # Perform linear interpolation
    resampled_data = np.interp(new_indices, original_indices, data)

    return resampled_data


def sort_list(zipped_list):
    """
    Sorts a list of lists (or tuples) based on the first element of each inner list or tuple,
    which is necessary for a zipped list with station names and positions.

    :param:
        - zipped_list (list of lists or tuples): The list to be sorted.

    :return:
        sorted list
    """

    return sorted(zipped_list, key=lambda item: item[0])


def squeeze_to_intlist(arr):
    """
    Squeeze array and make list with integers

    :param:
        - arr: numpy array

    :return:
        squeezed integers
    """

    squeezed = np.squeeze(arr).astype(int)
    if squeezed.ndim == 0:
        return [squeezed.item()]
    elif squeezed.ndim == 1:
        return squeezed.tolist()
    else:
        return squeezed.tolist()
