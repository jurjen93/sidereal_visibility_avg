import numpy as np


def time_resolution(resolution, fov_diam, time_smearing=0.95):
    """
    Calculate the best time resolution, given a time_smearing allowance

    Using formulas from Bridle & Schwab (1999)

    :params:
        - resolution: resolution in arcseconds
        - fov_diam: longest FoV diameter in degrees
        - time_smearing: allowable time smearing
    :return: integration time in seconds
    """

    # Convert distance from degrees to radians
    distance_from_phase_center_rad = np.deg2rad(fov_diam/2)

    # Calculate angular resolution (radians)
    angular_resolution_rad = resolution*4.8481*1e-6

    int_time = 2.9*10**4*(angular_resolution_rad*np.sqrt(1-time_smearing)/
                          distance_from_phase_center_rad)

    return int_time