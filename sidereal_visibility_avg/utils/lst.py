import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def mjd_seconds_to_lst_seconds_single(mjd_seconds, longitude_deg=6.869837):
    """
    Convert time in modified Julian Date time to LST for a single value or small chunk.
    """

    # Convert seconds to days for MJD
    mjd_days = mjd_seconds / 86400.0
    time_utc = Time(mjd_days, format='mjd', scale='utc')

    # Observer's location
    location = EarthLocation(lon=longitude_deg * u.deg, lat=52.915122 * u.deg)

    # LST in hours
    lst_hours = time_utc.sidereal_time('apparent', longitude=location.lon).hour

    # Convert LST from hours to seconds
    lst_seconds = lst_hours * 3600.0

    return lst_seconds


def mjd_seconds_to_lst_seconds(mjd_seconds, longitude_deg=6.869837):
    """
    Convert time in modified Julian Date time to LST with parallel processing.

    :param:
        - mjd_seconds: Array of modified Julian date time in seconds
        - longitude_deg: Longitude telescope in degrees (default: 6.869837 for LOFAR core)

    :return:
        - Array of time in LST (seconds)
    """

    # Split the mjd_seconds array into chunks for parallel processing
    n_jobs = max(cpu_count() - 1, 1)  # Use all available cores minus 1
    chunk_size = len(mjd_seconds) // n_jobs

    # Perform parallel processing
    lst_seconds = Parallel(n_jobs=n_jobs)(
        delayed(mjd_seconds_to_lst_seconds_single)(mjd_seconds[i:i + chunk_size], longitude_deg)
        for i in range(0, len(mjd_seconds), chunk_size)
    )

    # Concatenate results from different workers
    return np.concatenate(lst_seconds)
