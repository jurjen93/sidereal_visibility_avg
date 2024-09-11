from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u


def mjd_seconds_to_lst_seconds(mjd_seconds, longitude_deg=6.869837):
    """
    Convert time in modified Julian Date time to LST

    :param:
        - mjd_seconds: modified Julian date time in seconds
        - longitde_deg: longitude telescope in degrees (6.869837 for LOFAR )

    :return:
        time in LST
    """

    # Convert seconds to days for MJD
    mjd_days = mjd_seconds / 86400.0

    # Create an astropy Time object
    time_utc = Time(mjd_days, format='mjd', scale='utc')

    # Define the observer's location using longitude (latitude doesn't affect LST)
    location = EarthLocation(lon=longitude_deg * u.deg, lat=52.915122 * u.deg)

    # Calculate LST in hours
    lst_hours = time_utc.sidereal_time('apparent', longitude=location.lon).hour

    # Convert LST from hours to seconds
    lst_seconds = lst_hours * 3600.0

    return lst_seconds
