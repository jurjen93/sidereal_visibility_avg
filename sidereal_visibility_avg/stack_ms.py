from casacore.tables import table, taql
import numpy as np
from os import path
import sys
import psutil
from glob import glob
from scipy.ndimage import gaussian_filter1d
import gc
from time import sleep

from .utils.arrays_and_lists import find_closest_index_list, add_axis, is_range
from .utils.file_handling import load_json, read_mapping
from .utils.ms_info import make_ant_pairs, get_data_arrays
from .utils.printing import print_progress_bar
from .utils.clean import clean_binary_file
from .utils.parallel import (multiply_arrays, sum_arrays, _set_nan_to_zero_complex, _conjugate_masked, _nanmean_axis1,
                             _scatter_add_2d, _scatter_add_uvw)


class Stack:
    """
    Stack measurement sets in template sva_output.ms
    """
    def __init__(self, msin: list = None, outname: str = 'sva_output.ms',
                 chunkmem: float = 1., tmp_folder: str = '.'):
        if not path.exists(outname):
            sys.exit(f"ERROR: Template {outname} has not been created or is deleted")
        print("\n\n==== Start stacking ====\n")
        self.template = table(outname, readonly=False, ack=False)
        self.mslist = msin
        self.outname = outname
        self.flag = False

        F = table(self.outname + '::SPECTRAL_WINDOW', ack=False)
        self.ref_freqs = F.getcol("CHAN_FREQ")[0]
        self.freq_len = len(self.ref_freqs)
        F.close()

        self.total_memory = psutil.virtual_memory().total / (1024 ** 3) / chunkmem
        self.chunk_size = min(
            int(self.total_memory * (1024 ** 3) / np.dtype(np.float64).itemsize / 8 / self.freq_len),
            30_000_000 // self.freq_len
        )
        print(f"\n---------------\nChunk size ==> {self.chunk_size}")

        self.tmp_folder = tmp_folder.rstrip('/') + '/'

    def smooth_uvw(self):
        """Smooth UVW values (EXPERIMENTAL, CURRENTLY NOT USED)"""
        uvw, _ = get_data_arrays('UVW', self.T.nrows())
        uvw[:] = self.T.getcol("UVW")
        time = self.T.getcol("TIME")

        ants = table(self.outname + "::ANTENNA", ack=False)
        baselines = np.c_[make_ant_pairs(ants.nrows(), 1)]
        ants.close()

        print('Smooth UVW')
        for idx_b, baseline in enumerate(baselines):
            print_progress_bar(idx_b, len(baselines))
            idxs = []
            for baseline_json in glob(self.tmp_folder + f"*baseline_mapping/{baseline[0]}-{baseline[1]}.json"):
                idxs += list(load_json(baseline_json).values())
            sorted_indices = np.argsort(time[idxs])
            idxs_arr = np.array(idxs)[sorted_indices]
            for i in range(3):
                uvw[idxs_arr, i] = gaussian_filter1d(uvw[idxs_arr, i], sigma=2)

        self.T.putcol('UVW', uvw)

    def stack_all(self, column: str = 'DATA', keep_DP3_uvw: bool = False,
                  safe_mem: bool = False, extra_cooldowns: bool = False):
        """
        Stack all MS.

        :param column:        column name (currently only DATA)
        :param keep_DP3_uvw:  keep DP3 UVW, no weighted average
        :param safe_mem:      limit RAM usage
        """
        if column != 'DATA':
            sys.exit("ERROR: Only column 'DATA' allowed (for now)")

        columns = [column, 'WEIGHT_SPECTRUM'] if keep_DP3_uvw else ['UVW', column, 'WEIGHT_SPECTRUM']

        with table(path.abspath(self.outname), readonly=False, ack=False) as self.T:

            for col in columns:

                # Allocate output arrays
                if col == 'UVW':
                    new_data, uvw_weights = get_data_arrays(
                        col, self.T.nrows(), self.freq_len,
                        always_memmap=safe_mem, tmp_folder=self.tmp_folder
                    )
                    if keep_DP3_uvw:
                        new_data = self.T.getcol("UVW")
                else:
                    new_data, _ = get_data_arrays(
                        col, self.T.nrows(), self.freq_len,
                        always_memmap=safe_mem, tmp_folder=self.tmp_folder
                    )

                is_memmap = isinstance(new_data, np.memmap)

                # Loop over input MS
                for ms in self.mslist:
                    print(f'\n{col} :: {ms}')

                    t = table(path.abspath(ms), ack=False, readonly=True)

                    # Frequency mapping (not needed for UVW)
                    if col != 'UVW':
                        f = table(ms + '::SPECTRAL_WINDOW', ack=False)
                        freqs = f.getcol("CHAN_FREQ")[0]
                        freq_idxs = np.array(find_closest_index_list(freqs, self.ref_freqs))
                        f.close()

                    mapping_folder = self.tmp_folder + path.basename(ms) + '_baseline_mapping'
                    print('Read baseline mapping')
                    indices, ref_indices = read_mapping(mapping_folder)

                    if len(indices) == 0:
                        sys.exit('ERROR: cannot find *_baseline_mapping folders')

                    # Complex conjugate mask (DATA only)
                    if "DATA" in col:
                        comp_conj = np.array(ref_indices) < 0
                        print(f"{col} needs to complex conjugate {np.sum(comp_conj)} values.")
                    else:
                        comp_conj = None

                    ref_indices = np.abs(ref_indices)

                    # Chunked stacking
                    n_total = len(indices)
                    chunks = n_total // self.chunk_size + 1
                    print(f'Stacking in {chunks} chunks')

                    for chunk_idx in range(chunks):
                        print_progress_bar(chunk_idx, chunks + 1)

                        start = chunk_idx * self.chunk_size
                        end = min(start + self.chunk_size, n_total)
                        if start >= end:
                            break

                        row_idxs_new = ref_indices[start:end]
                        row_idxs = np.array([int(i - start) for i in indices[start:end]])
                        nrow = end - start

                        data = t.getcol(col, startrow=start, nrow=nrow)

                        # Apply row remapping if not a contiguous range
                        if not is_range(row_idxs):
                            data = data[row_idxs]
                            norange = True
                        else:
                            norange = False

                        # Complex conjugate inverted baselines in-place
                        if comp_conj is not None:
                            mask = comp_conj[start:end]
                            if norange:
                                mask = mask[row_idxs]
                            flat_data = data.ravel()
                            flat_mask = np.repeat(mask, data.shape[1] if data.ndim > 1 else 1)
                            _conjugate_masked(flat_data, flat_mask)

                        if col == 'DATA':
                            # Zero NaNs in-place (no temp boolean array)
                            _set_nan_to_zero_complex(data.ravel())

                            weights = t.getcol('WEIGHT_SPECTRUM', startrow=start, nrow=nrow)
                            if norange:
                                weights = weights[row_idxs]
                            multiply_arrays(data, weights, out=data)   # reuse data buffer
                            del weights

                        elif col == 'WEIGHT_SPECTRUM':
                            data = data[..., 0]  # reduce to one polarisation

                        if col == 'UVW':
                            weights_raw = t.getcol("WEIGHT_SPECTRUM", startrow=start, nrow=nrow)[..., 0]
                            if norange:
                                weights_raw = weights_raw[row_idxs]
                            # nanmean over freqs → shape (n,), then reshape to (n,1)
                            w = _nanmean_axis1(np.ascontiguousarray(weights_raw))[:, np.newaxis]

                            # Single-pass scatter-add for data*w and uvw_weights
                            _scatter_add_uvw(new_data, uvw_weights, row_idxs_new, data, w)

                            try:
                                uvw_weights.flush()
                            except AttributeError:
                                pass

                        else:
                            # Scatter-add into output (parallelised, no temp fancy-index read)
                            if is_memmap:
                                buf = new_data[row_idxs_new[:, None], freq_idxs].copy()
                                sum_arrays(buf, data, out=buf)
                                new_data[row_idxs_new[:, None], freq_idxs] = buf
                            else:
                                _scatter_add_2d(new_data, row_idxs_new, freq_idxs, data)

                        data = None

                    try:
                        gc.collect()
                        new_data.flush()
                        if extra_cooldowns:
                            sleep(60)
                    except AttributeError:
                        pass

                    print_progress_bar(chunks, chunks + 1)
                    t.close()

                # Write column back
                print(f'\nPut column {col}')
                if col == 'UVW':
                    uvw_weights[uvw_weights == 0] = 1
                    new_data /= uvw_weights
                    new_data[np.isnan(new_data)] = 0.

                chunks_range = range(self.T.nrows() // self.chunk_size + 1)
                for chunk_idx in chunks_range:
                    print_progress_bar(chunk_idx, len(chunks_range))
                    startp = chunk_idx * self.chunk_size
                    endp = min(startp + self.chunk_size, self.T.nrows())

                    subdat = new_data[startp:endp]
                    if col == 'WEIGHT_SPECTRUM':
                        subdat = add_axis(subdat, 4)
                    elif col == 'DATA':
                        subdat = subdat.copy()
                        subdat[subdat == 0] = np.nan

                    self.T.putcol(col, subdat, startrow=startp, nrow=endp - startp)

                del new_data
                clean_binary_file(self.tmp_folder + col.lower() + '.tmp.dat')

        # Final FLAG and normalise via TaQL (fast server-side)
        print('Put column FLAG')
        taql(f'UPDATE {self.outname} SET FLAG = (WEIGHT_SPECTRUM == 0)')
        print('Normalise column DATA')
        taql(f'UPDATE {self.outname} SET DATA = (DATA / WEIGHT_SPECTRUM) WHERE ANY(WEIGHT_SPECTRUM > 0)')
        print("----------\n")