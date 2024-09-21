from casacore.tables import table, default_ms, taql
import numpy as np
from os import path, makedirs, cpu_count
from os import system as run_command
from shutil import rmtree
from pprint import pprint
from concurrent.futures import ProcessPoolExecutor, as_completed

from .utils.parallel import run_parallel_mapping, process_ms, process_baseline_uvw, process_baseline_int
from .utils.dysco import decompress
from .utils.helpers import print_progress_bar, repeat_elements, map_array_dict, find_closest_index_list
from .utils.files import check_folder_exists
from .utils.ms_info import get_station_id, same_phasedir, unique_station_list, n_baselines, make_ant_pairs
from .utils.uvw import resample_uwv
from .utils.lst import mjd_seconds_to_lst_seconds, mjd_seconds_to_lst_seconds_single


class Template:
    """Make template measurement set based on input measurement sets"""
    def __init__(self, msin: list = None, outname: str = 'empty.ms'):
        self.mslist = msin
        self.outname = outname

        # Time offset to sidereal day from output MS
        self._time_lst_offset = None

    @property
    def time_lst_offset(self):
        """Get time LST offset to average to same day"""

        if self._time_lst_offset is None:
            times = []
            for ms in self.mslist:
                with table(f"{ms}::OBSERVATION", ack=False) as t:
                    tr = t.getcol("TIME_RANGE")[0][0]
                    lst_tr = mjd_seconds_to_lst_seconds_single(t.getcol("TIME_RANGE")[0][0])
                    lst_offset = tr - lst_tr
                    times.append(lst_offset)
            self._time_lst_offset = np.median(times)
        return self._time_lst_offset

    def add_spectral_window(self):
        """
        Add SPECTRAL_WINDOW as sub table
        """

        print("Add table ==> " + self.outname + "::SPECTRAL_WINDOW")

        tnew_spw_tmp = table(self.ref_table.getkeyword('SPECTRAL_WINDOW'), ack=False)
        newdesc = tnew_spw_tmp.getdesc()
        for col in ['CHAN_WIDTH', 'CHAN_FREQ', 'RESOLUTION', 'EFFECTIVE_BW']:
            newdesc[col]['shape'] = np.array([self.channels.shape[-1]])

        tnew_spw = table(self.outname + '::SPECTRAL_WINDOW', newdesc, readonly=False, ack=False)
        tnew_spw.addrows(1)
        chanwidth = np.expand_dims([np.squeeze(np.diff(self.channels))[0]]*self.chan_num, 0)
        tnew_spw.putcol("NUM_CHAN", np.array([self.chan_num]))
        tnew_spw.putcol("CHAN_FREQ", self.channels)
        tnew_spw.putcol("CHAN_WIDTH", chanwidth)
        tnew_spw.putcol("RESOLUTION", chanwidth)
        tnew_spw.putcol("EFFECTIVE_BW", chanwidth)
        tnew_spw.putcol("REF_FREQUENCY", np.nanmean(self.channels))
        tnew_spw.putcol("MEAS_FREQ_REF", np.array([5]))  # Why always 5?
        tnew_spw.putcol("TOTAL_BANDWIDTH", [np.max(self.channels)-np.min(self.channels)-chanwidth[0][0]])
        tnew_spw.putcol("NAME", 'Stacked_MS_'+str(int(np.nanmean(self.channels)//1000000))+"MHz")
        tnew_spw.flush(True)
        tnew_spw.close()
        tnew_spw_tmp.close()

    def add_stations(self):
        """
        Add ANTENNA and FEED tables
        """

        print("Add table ==> " + self.outname + "::ANTENNA")

        stations = [sp[0] for sp in self.station_info]
        st_id = dict(zip(set(
            [stat[0:8] for stat in stations]),
            range(len(set([stat[0:8] for stat in stations])))
        ))
        ids = [st_id[s[0:8]] for s in stations]
        positions = np.array([sp[1] for sp in self.station_info])
        diameters = np.array([sp[2] for sp in self.station_info])
        phase_ref = np.array([sp[4] for sp in self.station_info])
        names = np.array([sp[5] for sp in self.station_info])
        coor_axes = np.array([sp[6] for sp in self.station_info])
        tile_element = np.array([sp[7] for sp in self.station_info])
        lofar_names = np.array([sp[0] for sp in self.lofar_stations_info])
        clock = np.array([sp[1] for sp in self.lofar_stations_info])

        tnew_ant_tmp = table(self.ref_table.getkeyword('ANTENNA'), ack=False)
        newdesc = tnew_ant_tmp.getdesc()
        tnew_ant_tmp.close()

        tnew_ant = table(self.outname + '::ANTENNA', newdesc, readonly=False, ack=False)
        tnew_ant.addrows(len(stations))
        print('Total number of output stations: ' + str(tnew_ant.nrows()))
        tnew_ant.putcol("NAME", stations)
        tnew_ant.putcol("TYPE", ['GROUND-BASED']*len(stations))
        tnew_ant.putcol("POSITION", positions)
        tnew_ant.putcol("DISH_DIAMETER", diameters)
        tnew_ant.putcol("OFFSET", np.array([[0., 0., 0.]] * len(stations)))
        tnew_ant.putcol("FLAG_ROW", np.array([False] * len(stations)))
        tnew_ant.putcol("MOUNT", ['X-Y'] * len(stations))
        tnew_ant.putcol("STATION", ['LOFAR'] * len(stations))
        tnew_ant.putcol("LOFAR_STATION_ID", ids)
        tnew_ant.putcol("LOFAR_PHASE_REFERENCE", phase_ref)
        tnew_ant.flush(True)
        tnew_ant.close()

        print("Add table ==> " + self.outname + "::FEED")

        tnew_ant_tmp = table(self.ref_table.getkeyword('FEED'), ack=False)
        newdesc = tnew_ant_tmp.getdesc()
        tnew_ant_tmp.close()

        tnew_feed = table(self.outname + '::FEED', newdesc, readonly=False, ack=False)
        tnew_feed.addrows(len(stations))
        tnew_feed.putcol("POSITION", np.array([[0., 0., 0.]] * len(stations)))
        tnew_feed.putcol("BEAM_OFFSET", np.array([[[0, 0], [0, 0]]] * len(stations)))
        tnew_feed.putcol("POL_RESPONSE", np.array([[[1. + 0.j, 0. + 0.j], [0. + 0.j, 1. + 0.j]]] * len(stations)).astype(np.complex64))
        tnew_feed.putcol("POLARIZATION_TYPE", {'shape': [len(stations), 2], 'array': ['X', 'Y'] * len(stations)})
        tnew_feed.putcol("RECEPTOR_ANGLE", np.array([[-0.78539816, -0.78539816]] * len(stations)))
        tnew_feed.putcol("ANTENNA_ID", np.array(range(len(stations))))
        tnew_feed.putcol("BEAM_ID", np.array([-1] * len(stations)))
        tnew_feed.putcol("INTERVAL", np.array([28799.9787008] * len(stations)))
        tnew_feed.putcol("NUM_RECEPTORS", np.array([2] * len(stations)))
        tnew_feed.putcol("SPECTRAL_WINDOW_ID", np.array([-1] * len(stations)))
        tnew_feed.putcol("TIME", np.array([5.e9] * len(stations)))
        tnew_feed.flush(True)
        tnew_feed.close()

        print("Add table ==> " + self.outname + "::LOFAR_ANTENNA_FIELD")

        tnew_ant_tmp = table(self.ref_table.getkeyword('LOFAR_ANTENNA_FIELD'), ack=False)
        newdesc = tnew_ant_tmp.getdesc()

        tnew_ant_tmp.close()

        tnew_field = table(self.outname + '::LOFAR_ANTENNA_FIELD', newdesc, readonly=False, ack=False)
        tnew_field.addrows(len(stations))
        tnew_field.putcol("ANTENNA_ID", np.array(range(len(stations))))
        tnew_field.putcol("NAME", names)
        tnew_field.putcol("COORDINATE_AXES", np.array(coor_axes))
        tnew_field.putcol("TILE_ELEMENT_OFFSET", np.array(tile_element))
        tnew_field.putcol("TILE_ROTATION", np.array([0]*len(stations)))
        # tnew_field.putcol("ELEMENT_OFFSET", ???) TODO: fix for primary beam construction
        # tnew_field.putcol("ELEMENT_RCU", ???) TODO: fix for primary beam construction
        # tnew_field.putcol("ELEMENT_FLAG", ???) TODO: fix for primary beam construction
        tnew_field.flush(True)
        tnew_field.close()

        print("Add table ==> " + self.outname + "::LOFAR_STATION")

        tnew_ant_tmp = table(self.ref_table.getkeyword('LOFAR_STATION'), ack=False)
        newdesc = tnew_ant_tmp.getdesc()
        tnew_ant_tmp.close()

        tnew_station = table(self.outname + '::LOFAR_STATION', newdesc, readonly=False, ack=False)
        tnew_station.addrows(len(lofar_names))
        tnew_station.putcol("NAME", lofar_names)
        tnew_station.putcol("FLAG_ROW", np.array([False] * len(lofar_names)))
        tnew_station.putcol("CLOCK_ID", np.array(clock, dtype=int))
        tnew_station.flush(True)
        tnew_station.close()

    def make_mapping_lst(self):
        """
        Make mapping json files essential for efficient stacking.
        These map LST times from input MS to template MS.
        """

        outname = self.outname  # Cache instance variables locally
        time_lst_offset = self.time_lst_offset

        with taql(f"SELECT TIME,ANTENNA1,ANTENNA2 FROM {path.abspath(outname)} ORDER BY TIME") as T:
            ref_time = T.getcol("TIME")
            ref_antennas = np.c_[T.getcol("ANTENNA1"), T.getcol("ANTENNA2")]

        ref_uniq_time = np.unique(ref_time)

        ref_stats, ref_ids = get_station_id(outname)

        # Process each MS file in parallel
        for ms in self.mslist:
            print(f'\nMapping: {ms}')

            # Open the MS table and read columns
            with taql(f"SELECT TIME,ANTENNA1,ANTENNA2 FROM {path.abspath(ms)} ORDER BY TIME") as t:

                # Mapping folder for the current MS
                mapping_folder = ms + '_baseline_mapping'

                if not check_folder_exists(mapping_folder):
                    makedirs(mapping_folder, exist_ok=False)

                    # Fetch MS info and map antenna IDs
                    new_stats, new_ids = get_station_id(ms)
                    id_map = {new_id: ref_stats.index(stat) for new_id, stat in zip(new_ids, new_stats)}

                    # Convert TIME to LST
                    time = mjd_seconds_to_lst_seconds(t.getcol("TIME")) + time_lst_offset
                    uniq_time = np.unique(time)
                    time_idxs = find_closest_index_list(uniq_time, ref_uniq_time)

                    # Map antennas and compute unique pairs
                    antennas = np.c_[
                        map_array_dict(t.getcol("ANTENNA1"), id_map), map_array_dict(t.getcol("ANTENNA2"), id_map)]
                    uniq_ant_pairs = np.unique(antennas, axis=0)

                    # Run parallel mapping
                    run_parallel_mapping(uniq_ant_pairs, antennas, ref_antennas, time_idxs, mapping_folder)
                else:
                    print(f'{mapping_folder} already exists')

    def calculate_uvw(self):
        """
        Calculate UVW with DP3
        """

        # Make baseline/time mapping
        self.make_mapping_lst()

        # Use DP3 to upsample and downsample, recalculating the UVW coordinates
        run_command(f"DP3 msin={self.outname} msout={self.outname}.tmp steps=[up,avg] "
                    f"up.type=upsample up.timestep=2 up.updateuvw=True avg.timestep=2 avg.type=averager "
                    f"&& rm -rf {self.outname} && mv {self.outname}.tmp {self.outname}")

        # Update baseline mapping
        self.make_mapping_uvw()

    def interpolate_uvw(self):
        """
        Fill UVW data points
        """

        # Make baseline/time mapping
        self.make_mapping_lst()

        # Get baselines
        ants = table(self.outname + "::ANTENNA", ack=False)
        baselines = np.c_[make_ant_pairs(ants.nrows(), 1)]
        ants.close()

        T = table(self.outname, readonly=False, ack=False)
        UVW = np.memmap('UVW.tmp.dat', dtype=np.float32, mode='w+', shape=(T.nrows(), 3))
        TIME = np.memmap('TIME.tmp.dat', dtype=np.float64, mode='w+', shape=(T.nrows()))
        TIME[:] = T.getcol("TIME")

        for ms_idx, ms in enumerate(sorted(self.mslist)):
            with table(ms, ack=False) as f:
                uvw = np.memmap(f'{ms}_uvw.tmp.dat', dtype=np.float32, mode='w+', shape=(f.nrows(), 3))
                time = np.memmap(f'{ms}_time.tmp.dat', dtype=np.float64, mode='w+', shape=(f.nrows()))

                uvw[:] = f.getcol("UVW")
                time[:] = mjd_seconds_to_lst_seconds(f.getcol("TIME")) + self.time_lst_offset

        # Determine number of workers
        num_workers = max(cpu_count()-3, 1)  # I/O-bound heuristic

        print(f"Using {num_workers} workers for making UVW column and accurate baseline mapping."
              f"\nThis is an expensive operation. So, be patient..")

        batch_size = max(1, len(baselines) // num_workers)  # Ensure at least one baseline per batch

        print("Multithreading...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_baseline = {
                executor.submit(process_baseline_int, range(i, min(i + batch_size, len(baselines))), baselines,
                                self.mslist): i
                for i in range(0, len(baselines), batch_size)
            }

            for future in as_completed(future_to_baseline):
                batch_start_idx = future_to_baseline[future]
                try:
                    results = future.result()
                    for row_idxs, uvws, b_idx, time in results:
                        UVW[row_idxs] = resample_uwv(uvws, row_idxs, time, TIME)
                except Exception as exc:
                    print(f'Batch starting at index {batch_start_idx} generated an exception: {exc}')

        UVW.flush()
        T.putcol("UVW", UVW)
        T.close()

        # Make final mapping
        self.make_mapping_uvw()

    def make_mapping_uvw(self):
        """
        Make mapping json files essential for efficient stacking based on UVW points
        """

        # Get baselines
        with table(self.outname + "::ANTENNA", ack=False) as ants:
            baselines = np.c_[make_ant_pairs(ants.nrows(), 1)]

        if not path.exists('UVW.tmp.dat'):
            with table(self.outname, readonly=False, ack=False) as T:
                UVW = np.memmap('UVW.tmp.dat', dtype=np.float32, mode='w+', shape=(T.nrows(), 3))
                with table(self.outname, ack=False) as T:
                    UVW[:] = T.getcol("UVW")

                for ms_idx, ms in enumerate(sorted(self.mslist)):
                    with table(ms, ack=False) as f:
                        np.memmap(f'{ms}_uvw.tmp.dat', dtype=np.float32, mode='w+', shape=(f.nrows(), 3))

        UVW = np.memmap('UVW.tmp.dat', dtype=np.float32).reshape(-1, 3)

        num_workers = min(cpu_count()-3, len(baselines))

        print('\nMake new mapping based on UVW points')
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_baseline = {executor.submit(process_baseline_uvw, baseline,
                                                  '/'.join(self.mslist[0].split('/')[0:-1]), UVW): baseline for baseline
                                  in baselines}

            for n, future in enumerate(as_completed(future_to_baseline)):
                baseline = future_to_baseline[future]
                try:
                    future.result()  # Get the result
                except Exception as exc:
                    print(f'Baseline {baseline} generated an exception: {exc}')

                print_progress_bar(n + 1, len(baselines))

    def make_template(self, overwrite: bool = True, time_res: int = None, avg_factor: float = 1):
        """
        Make template MS based on existing MS

        :param:
            - overwrite: overwrite output file
            - time_res: time resolution in seconds
            - avg_factor: averaging factor
        """

        if overwrite:
            if path.exists(self.outname):
                rmtree(self.outname)

        same_phasedir(self.mslist)

        # Get data columns
        unique_stations, unique_channels, unique_lofar_stations = [], [], []
        min_t_lst, min_dt, dfreq_min, max_t_lst = None, None, None, None

        with ProcessPoolExecutor() as executor:
            future_to_ms = {executor.submit(process_ms, ms): ms for ms in self.mslist}
            for future in as_completed(future_to_ms):
                stations, lofar_stations, channels, dfreq, dt, min_t, max_t = future.result()

                if min_t_lst is None:
                    min_t_lst, min_dt, dfreq_min, max_t_lst = min_t, dt, dfreq, max_t
                else:
                    min_t_lst = min(min_t_lst, min_t)
                    min_dt = min(min_dt, dt)
                    dfreq_min = min(dfreq_min, dfreq)
                    max_t_lst = max(max_t_lst, max_t)

                unique_stations.extend(stations)
                unique_channels.extend(channels)
                unique_lofar_stations.extend(lofar_stations)

        self.station_info = unique_station_list(unique_stations)
        self.lofar_stations_info = unique_station_list(unique_lofar_stations)

        chan_range = np.arange(min(unique_channels), max(unique_channels) + dfreq_min, dfreq_min)
        self.channels = np.sort(np.expand_dims(np.unique(chan_range), 0))
        self.chan_num = self.channels.shape[-1]

        if time_res is not None:
            time_range = np.arange(min_t_lst + self.time_lst_offset,
                                   max_t_lst + min_dt + self.time_lst_offset, time_res)

        else:
            time_range = np.arange(min_t_lst + self.time_lst_offset,
                                   max_t_lst + min_dt + self.time_lst_offset, min_dt/avg_factor)

        baseline_count = n_baselines(len(self.station_info))
        nrows = baseline_count*len(time_range)

        # Take one ms for temp usage
        tmp_ms = self.mslist[0]

        # Remove dysco compression
        self.tmpfile = decompress(tmp_ms)
        self.ref_table = table(self.tmpfile, ack=False)

        # Data description
        newdesc_data = self.ref_table.getdesc()

        # Reshape
        for col in ['DATA', 'FLAG', 'WEIGHT_SPECTRUM']:
            newdesc_data[col]['shape'] = np.array([self.chan_num, 4])

        newdesc_data.pop('_keywords_')

        pprint(newdesc_data)
        print()

        # Make main table
        default_ms(self.outname, newdesc_data)
        tnew = table(self.outname, readonly=False, ack=False)
        tnew.addrows(nrows)
        ant1, ant2 = make_ant_pairs(len(self.station_info), len(time_range))
        t = repeat_elements(time_range, baseline_count)
        tnew.putcol("TIME", t)
        tnew.putcol("TIME_CENTROID", t)
        tnew.putcol("ANTENNA1", ant1)
        tnew.putcol("ANTENNA2", ant2)
        tnew.putcol("EXPOSURE", np.array([np.diff(time_range)[0]] * nrows))
        tnew.putcol("FLAG_ROW", np.array([False] * nrows))
        tnew.putcol("INTERVAL", np.array([np.diff(time_range)[0]] * nrows))
        tnew.flush(True)
        tnew.close()

        # Set SPECTRAL_WINDOW info
        self.add_spectral_window()

        # Set ANTENNA/STATION info
        self.add_stations()

        # Set other tables (annoying table locks prevent parallel processing)
        for subtbl in ['FIELD', 'HISTORY', 'FLAG_CMD', 'DATA_DESCRIPTION',
                       'LOFAR_ELEMENT_FAILURE', 'OBSERVATION', 'POINTING',
                       'POLARIZATION', 'PROCESSOR', 'STATE']:
            try:
                print("Add table ==> " + self.outname + "::" + subtbl)

                tsub = table(self.tmpfile+"::"+subtbl, ack=False, readonly=False)
                tsub.copy(self.outname + '/' + subtbl, deep=True)
                tsub.flush(True)
                tsub.close()
            except:
                print(subtbl+" unknown")

        self.ref_table.close()

        # Cleanup
        if 'tmp' in self.tmpfile:
            rmtree(self.tmpfile)