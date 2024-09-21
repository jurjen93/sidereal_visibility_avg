from casacore.tables import table
import numpy as np
from .lst import mjd_seconds_to_lst_seconds_single
from sys import exit


def same_phasedir(mslist: list = None):
    """
    Have MS same phase center?

    :param:
        - mslist: measurement set list
    """

    for n, ms in enumerate(mslist):
        t = table(ms+'::FIELD', ack=False)
        if n==0:
            phasedir = t.getcol("PHASE_DIR")
        else:
            if not np.all(phasedir == t.getcol("PHASE_DIR")):
                exit("MS do not have the same phase center, check "+ms)



def get_ms_content(ms):
    """
    Get MS content with speed optimizations.

    :param:
        - ms: measurement set

    :return:
        - station names
        - frequency channels
        - total time in seconds
        - delta time
    """

    with table(ms, ack=False) as T, \
         table(ms+"::SPECTRAL_WINDOW", ack=False) as F, \
         table(ms+"::ANTENNA", ack=False) as A, \
         table(ms+"::LOFAR_ANTENNA_FIELD", ack=False) as L, \
         table(ms+"::LOFAR_STATION", ack=False) as S:

        # Get LOFAR antenna info in a single call
        lofar_stations = np.column_stack((S.getcol("NAME"), S.getcol("CLOCK_ID"))).tolist()

        # Retrieve relevant columns from ANTENNA and LOFAR_ANTENNA_FIELD tables
        antenna_cols = [
            A.getcol("NAME"), A.getcol("POSITION"), A.getcol("DISH_DIAMETER"),
            A.getcol("LOFAR_STATION_ID"), A.getcol("LOFAR_PHASE_REFERENCE"),
            L.getcol("NAME"), L.getcol("COORDINATE_AXES"), L.getcol("TILE_ELEMENT_OFFSET")
        ]
        stations = list(zip(*antenna_cols))

        # Read channel frequencies and calculate delta frequency
        channels = F.getcol("CHAN_FREQ")[0]
        dfreq = np.diff(channels)[0]  # Assuming uniform spacing

        # Get unique and sorted times to calculate total time and delta time
        time = np.unique(T.getcol("TIME"))
        if len(time) > 1:
            total_time_seconds = time[-1] - time[0]
            dt = np.min(np.diff(time))
        else:
            total_time_seconds = 0
            dt = 0

        # Convert time to LST in one call
        time_lst = mjd_seconds_to_lst_seconds_single(time)
        time_min_lst, time_max_lst = time_lst.min(), time_lst.max()

    # Print output (could be skipped for speed)
    print(f'\nCONTENT from {ms}\n'
          '----------\n'
          f'Stations: {", ".join([s[0] for s in lofar_stations])}\n'
          f'Number of channels: {len(channels)}\n'
          f'Channel width: {dfreq} Hz\n'
          f'Total time: {round(total_time_seconds / 3600, 2)} hrs\n'
          f'Delta time: {dt} seconds\n'
          f'----------')

    return {
        'stations': stations,
        'lofar_stations': lofar_stations,
        'channels': channels,
        'dfreq': dfreq,
        'total_time_seconds': total_time_seconds,
        'dt': dt,
        'time_min_lst': time_min_lst,
        'time_max_lst': time_max_lst
    }


def get_station_id(ms):
    """
    Get station with corresponding id number

    :param:
        - ms: measurement set

    :return:
        - antenna names, IDs
    """

    t = table(ms+'::ANTENNA', ack=False)
    ants = t.getcol("NAME")
    t.close()

    t = table(ms+'::FEED', ack=False)
    ids = t.getcol("ANTENNA_ID")
    t.close()

    return ants, ids


def unique_station_list(station_list):
    """
    Filters a list of stations only based on first element

    :param:
        - station_list: Stations to be filtered.

    :return:
        - filtered list of stations
    """
    unique_dict = {}
    for item in station_list:
        if item[0] not in unique_dict:
            unique_dict[item[0]] = item
    return list(unique_dict.values())


def n_baselines(n_antennas: int = None):
    """
    Return number of baselines

    :param:
        - n_antennas: number of antennas

    :return: number of baselines
    """

    return n_antennas * (n_antennas - 1) // 2


def make_ant_pairs(n_ant, n_time):
    """
    Generate ANTENNA1 and ANTENNA2 arrays for an array with M antennas over N time slots.

    :param:
        - n_ant: Number of antennas in the array.
        - n_int: Number of time slots.

    :return:
        - ANTENNA1
        - ANTENNA2
    """

    # Generate all unique pairs of antennas for one time slot
    antenna_pairs = [(i, j) for i in range(n_ant) for j in range(i + 1, n_ant)]

    # Expand the pairs across n_time time slots
    antenna1 = np.array([pair[0] for pair in antenna_pairs] * n_time)
    antenna2 = np.array([pair[1] for pair in antenna_pairs] * n_time)

    return antenna1, antenna2


def get_data_arrays(column: str = 'DATA', nrows: int = None, freq_len: int = None):
    """
    Get data arrays (new data and weights)

    :param:
        - column: column name (DATA, WEIGHT_SPECTRUM, WEIGHT, OR UVW)
        - nrows: number of rows
        - freq_len: frequency axis length

    :return:
        - new_data: new data array (empty array with correct shape)
        - weights: weights corresponding to new data array (empty array with correct shape)
    """

    tmpfilename = column.lower()+'.tmp.dat'
    tmpfilename_weights = column.lower()+'_weights.tmp.dat'

    if column in ['UVW']:
        weights = np.memmap(tmpfilename_weights, dtype=np.float16, mode='w+', shape=(nrows, 3))
        weights[:] = 0
    else:
        weights = None

    if column in ['DATA', 'WEIGHT_SPECTRUM']:
        if column == 'DATA':
            dtp = np.complex128
        elif column == 'WEIGHT_SPECTRUM':
            dtp = np.float32
        else:
            dtp = np.float32
        shape = (nrows, freq_len, 4)

    elif column == 'WEIGHT':
        shape, dtp = (nrows, freq_len), np.float32

    elif column == 'UVW':
        shape, dtp = (nrows, 3), np.float32

    else:
        exit("ERROR: Use only DATA, WEIGHT_SPECTRUM, WEIGHT, or UVW")

    new_data = np.memmap(tmpfilename, dtype=dtp, mode='w+', shape=shape)

    return new_data, weights
