from casacore.tables import table, default_ms, taql
import numpy as np
from os import path, makedirs, cpu_count
from os import system as run_command
import sys
from shutil import rmtree
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from functools import partial

from .utils.parallel import run_parallel_mapping, process_ms, process_baseline_uvw, process_baseline_int
from .utils.dysco import decompress
from .utils.arrays_and_lists import repeat_elements, map_array_dict, find_closest_index_list
from .utils.file_handling import check_folder_exists
from .utils.ms_info import get_station_id, same_phasedir, unique_station_list, n_baselines, make_ant_pairs
from .utils.uvw import resample_uwv
from .utils.lst import mjd_seconds_to_lst_seconds, mjd_seconds_to_lst_seconds_single
from .utils.printing import print_progress_bar


class Template:
    """
    Make template measurement set based on input measurement sets
    :param:
        - msin: List of input MS
        - outname: Output name for MS
        - tmp_folder: Directory to store temporary files (mapping, .dat files etc.), such as a local scratch or RAM disk
    """

    def __init__(self, msin: list = None, outname: str = 'sva_output.ms', tmp_folder: str = '.'):
        if type(msin)!=list:
            sys.exit("ERROR: input needs to be a list of MS.")
        self.mslist = msin
        self.outname = outname

        # Time offset to sidereal day from output MS
        self._time_lst_offset = None
        self.tmp_folder = tmp_folder
        if self.tmp_folder[-1]!='/':
            self.tmp_folder+='/'

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

    def get_element_offset(self, station):
        """
        Get element offsets from mslist
        (hacky way to work around shape difference between Dutch and international stations)
        """

        for ms in self.mslist:
            with taql(f'SELECT ROWID() as row_id FROM {ms}::ANTENNA WHERE NAME="{station}"') as rows:
                if len(rows) == 1:
                    id = rows[0]['row_id']
                    with taql(f'SELECT ELEMENT_OFFSET FROM {ms}::LOFAR_ANTENNA_FIELD WHERE ANTENNA_ID={id}') as el:
                        return el.getcol("ELEMENT_OFFSET"), id, ms
        sys.exit(f"ERROR: No {station} in {self.mslist}?")

    def add_spectral_window_table(self):
        """
        Add SPECTRAL_WINDOW as sub table
        """

        with table(self.ref_table.getkeyword('SPECTRAL_WINDOW'), ack=False) as tnew_spw_tmp:
            newdesc = tnew_spw_tmp.getdesc()
        for col in ['CHAN_WIDTH', 'CHAN_FREQ', 'RESOLUTION', 'EFFECTIVE_BW']:
            newdesc[col]['shape'] = np.array([self.channels.shape[-1]])

        with table(self.outname + '::SPECTRAL_WINDOW', newdesc, readonly=False, ack=False) as tnew_spw:
            tnew_spw.addrows(1)
            chanw = np.squeeze(np.diff(self.channels))
            while chanw.size != 1: chanw = chanw[0]
            chanwidth = np.expand_dims([chanw]*self.chan_num, 0)
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

    def add_stations_tables(self):
        """
        Add ANTENNA and FEED tables
        """

        # Extract information from collected subtables
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

        with table(self.ref_table.getkeyword('FEED'), ack=False) as tnew_ant_tmp:
            newdesc = tnew_ant_tmp.getdesc()

        with table(self.outname + '::FEED', newdesc, readonly=False, ack=False) as tnew_feed:
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

        with table(self.ref_table.getkeyword('ANTENNA'), ack=False) as tnew_ant_tmp:
            newdesc = tnew_ant_tmp.getdesc()

        with table(self.outname + '::ANTENNA', newdesc, readonly=False, ack=False) as tnew_ant:
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

        with table(self.ref_table.getkeyword('LOFAR_ANTENNA_FIELD'), ack=False) as tnew_ant_tmp:
            newdesc = tnew_ant_tmp.getdesc()

        with table(self.outname + '::LOFAR_ANTENNA_FIELD', newdesc, readonly=False, ack=False) as tnew_field:
            for n, station in enumerate(stations):
                _, ind, ms = self.get_element_offset(station)

                # Using taql because the shapes for Dutch and International stations are not similar (cannot be opened in Python)
                taql(f"INSERT INTO {self.outname}::LOFAR_ANTENNA_FIELD SELECT FROM {ms}::LOFAR_ANTENNA_FIELD b WHERE b.ANTENNA_ID={ind}")
            tnew_field.putcol("ANTENNA_ID", np.array(range(len(stations))))

        with table(self.ref_table.getkeyword('LOFAR_STATION'), ack=False) as tnew_ant_tmp:
            newdesc = tnew_ant_tmp.getdesc()

        with table(self.outname + '::LOFAR_STATION', newdesc, readonly=False, ack=False) as tnew_station:
            tnew_station.addrows(len(lofar_names))
            tnew_station.putcol("NAME", lofar_names)
            tnew_station.putcol("FLAG_ROW", np.array([False] * len(lofar_names)))
            tnew_station.putcol("CLOCK_ID", np.array(clock, dtype=int))

    def make_mapping_lst(self):
        """
        Make mapping json files essential for efficient stacking.
        These map LST times from input MS to template MS.
        """

        outname = self.outname  # Cache instance variables locally
        time_lst_offset = self.time_lst_offset

        with taql(f"SELECT TIME,ANTENNA1,ANTENNA2 FROM {path.abspath(outname)}") as T:
            ref_time = T.getcol("TIME")
            ref_antennas = np.c_[T.getcol("ANTENNA1"), T.getcol("ANTENNA2")]

        ref_uniq_time = np.unique(ref_time)

        ref_stats, ref_ids = get_station_id(outname)

        # Process each MS file in parallel
        for ms in self.mslist:
            print(f'\nMapping: {ms}')

            # Open the MS table and read columns
            with taql(f"SELECT TIME,ANTENNA1,ANTENNA2 FROM {path.abspath(ms)}") as t:

                # Mapping folder for the current MS
                mapping_folder = self.tmp_folder + ms + '_baseline_mapping'

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
                    antennas = np.c_[map_array_dict(t.getcol("ANTENNA1"), id_map), map_array_dict(t.getcol("ANTENNA2"), id_map)]
                    uniq_ant_pairs = np.unique(antennas, axis=0)
                    antennas = np.sort(antennas, axis=1)

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
                    f"up.type=upsample up.timestep=2 up.updateuvw=True avg.type=averager avg.timestep=2 "
                    f"&& rm -rf {self.outname} && mv {self.outname}.tmp {self.outname}")

        # Update baseline mapping
        self.make_mapping_uvw()

    def interpolate_uvw(self):
        """
        Fill UVW data points through interpolation
        """

        # Make baseline/time mapping
        self.make_mapping_lst()

        # Get baselines
        with table(self.outname + "::ANTENNA", ack=False) as ants:
            baselines = np.c_[make_ant_pairs(ants.nrows(), 1)]

        print("Resample UVW through interpolation")
        with table(self.outname, readonly=False, ack=False) as T:
            UVW = np.memmap(self.tmp_folder+'UVW.tmp.dat', dtype=np.float32, mode='w+', shape=(T.nrows(), 3))
            TIME = np.memmap(self.tmp_folder+'TIME.tmp.dat', dtype=np.float64, mode='w+', shape=(T.nrows()))
            TIME[:] = T.getcol("TIME")

            for ms_idx, ms in enumerate(sorted(self.mslist)):
                with table(ms, ack=False) as f:
                    uvw = np.memmap(self.tmp_folder+f'{ms}_uvw.tmp.dat', dtype=np.float32, mode='w+', shape=(f.nrows(), 3))
                    time = np.memmap(self.tmp_folder+f'{ms}_time.tmp.dat', dtype=np.float64, mode='w+', shape=(f.nrows()))

                    uvw[:] = f.getcol("UVW")
                    time[:] = mjd_seconds_to_lst_seconds(f.getcol("TIME")) + self.time_lst_offset

            # Determine number of workers
            num_workers = max(cpu_count()-1, 1)

            batch_size = max(1, len(baselines) // num_workers)  # Ensure at least one baseline per batch

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_baseline = {
                    executor.submit(process_baseline_int, range(i, min(i + batch_size, len(baselines))), baselines,
                                    self.mslist): i
                    for i in range(0, len(baselines), batch_size)
                }

                for future in as_completed(future_to_baseline):
                    try:
                        results = future.result()
                        for row_idxs, uvws, baseline, time in results:
                            UVW[row_idxs] = resample_uwv(uvws, row_idxs, time, TIME)
                    except Exception:
                        print(f"No data for baseline {baseline}")

            UVW.flush()
            T.putcol("UVW", UVW)

        gc.collect()

        # Make final mapping
        self.make_mapping_uvw()

    def make_mapping_uvw(self):
        """
        Update mapping json files essential for efficient and accurate UVW averaging
        """

        # Get baselines
        with table(self.outname + "::ANTENNA", ack=False) as ants:
            baselines = np.c_[make_ant_pairs(ants.nrows(), 1)]

        if not path.exists(self.tmp_folder+'UVW.tmp.dat'):
            with table(self.outname, readonly=False, ack=False) as T:
                UVW = np.memmap(self.tmp_folder+'UVW.tmp.dat', dtype=np.float32, mode='w+', shape=(T.nrows(), 3))
                with table(self.outname, ack=False) as T:
                    UVW[:] = T.getcol("UVW")

                for ms_idx, ms in enumerate(sorted(self.mslist)):
                    with table(ms, ack=False) as f:
                        uvw = np.memmap(f'{ms}_uvw.tmp.dat', dtype=np.float32, mode='w+', shape=(f.nrows(), 3))
                        uvw[:] = f.getcol("UVW")

        else:
            UVW = np.memmap(self.tmp_folder+'UVW.tmp.dat', dtype=np.float32).reshape(-1, 3)

        num_workers = min(cpu_count()-1, len(baselines))

        print('\nMake final UVW mapping to output dataset')
        msdir = '/'.join(self.mslist[0].split('/')[0:-1])
        process_func = partial(process_baseline_uvw, folder=msdir, UVW=UVW)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_baseline = {executor.submit(process_func, baseline): baseline for baseline in baselines}

            for n, future in enumerate(as_completed(future_to_baseline)):
                baseline = future_to_baseline[future]
                try:
                    future.result()  # Get the result
                except Exception as e:
                    sys.exit(f'ERROR: Baseline {baseline} generated an exception: {e}')

                print_progress_bar(n + 1, len(baselines))

        gc.collect()

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

        # Get station information
        self.station_info = unique_station_list(unique_stations)
        self.lofar_stations_info = unique_station_list(unique_lofar_stations)

        # Make frequency channels for output MS
        chan_range = np.arange(min(unique_channels), max(unique_channels) + dfreq_min, dfreq_min)
        self.channels = np.sort(np.expand_dims(np.unique(chan_range), 0))
        self.chan_num = self.channels.shape[-1]

        # Make time axis for output MS
        if time_res is not None:
            time_range = np.arange(min_t_lst + self.time_lst_offset,
                                   max_t_lst + self.time_lst_offset, time_res)

        else:
            time_range = np.arange(min_t_lst + self.time_lst_offset,
                                   max_t_lst + self.time_lst_offset, min_dt/avg_factor)

        baseline_count = n_baselines(len(self.station_info))
        nrows = baseline_count*len(time_range)

        # Take one ms for temp usage (to remove dysco, and modify)
        tmp_ms = self.mslist[0]

        # Remove dysco compression (otherwise running into tricky errors when modifying MS)
        self.tmpfile = decompress(tmp_ms)

        with table(self.tmpfile, ack=False) as self.ref_table:

            # Data description
            newdesc_data = self.ref_table.getdesc()

            # Reshape
            for col in ['DATA', 'FLAG', 'WEIGHT_SPECTRUM']:
                newdesc_data[col]['shape'] = np.array([self.chan_num, 4])

            newdesc_data.pop('_keywords_')

            # Make main table
            default_ms(self.outname, newdesc_data)
            with table(self.outname, readonly=False, ack=False) as tnew:
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

            # Add SPECTRAL_WINDOW info
            self.add_spectral_window_table()

            # Add ANTENNA/STATION info
            self.add_stations_tables()

            # Get other tables (annoying table locks prevent parallel processing)
            for subtbl in ['FIELD', 'HISTORY', 'FLAG_CMD', 'DATA_DESCRIPTION',
                           'LOFAR_ELEMENT_FAILURE', 'OBSERVATION', 'POINTING',
                           'POLARIZATION', 'PROCESSOR', 'STATE']:
                try:
                    with table(self.tmpfile+"::"+subtbl, ack=False, readonly=False) as tsub:
                        tsub.copy(self.outname + '/' + subtbl, deep=True)
                        tsub.flush(True)
                except Exception as e:
                    print(f"Error processing '{subtbl}': {e}")

        # Cleanup
        if 'tmp' in self.tmpfile:
            rmtree(self.tmpfile)
