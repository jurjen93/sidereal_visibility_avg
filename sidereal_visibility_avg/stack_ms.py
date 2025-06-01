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
from .utils.parallel import multiply_arrays, sum_arrays


class Stack:
    """
    Stack measurement sets in template sva_output.ms
    """
    def __init__(self, msin: list = None, outname: str = 'sva_output.ms', chunkmem: float = 1., tmp_folder: str = '.'):
        if not path.exists(outname):
            sys.exit(f"ERROR: Template {outname} has not been created or is deleted")
        print("\n\n==== Start stacking ====\n")
        self.template = table(outname, readonly=False, ack=False)
        self.mslist = msin
        self.outname = outname
        self.flag = False

        # Freq
        F = table(self.outname+'::SPECTRAL_WINDOW', ack=False)
        self.ref_freqs = F.getcol("CHAN_FREQ")[0]
        self.freq_len = self.ref_freqs.__len__()
        F.close()

        # Memory and chunk size
        # Set number of cores
        self.total_memory = psutil.virtual_memory().total / (1024 ** 3)  # in GB
        self.total_memory /= chunkmem
        self.chunk_size = min(int(self.total_memory * (1024 ** 3) / np.dtype(np.float128).itemsize / 8 / self.freq_len), 30_000_000 // self.freq_len)
        print(f"\n---------------\nChunk size ==> {self.chunk_size}")

        self.tmp_folder = tmp_folder
        if self.tmp_folder[-1]!='/':
            self.tmp_folder+='/'

    def smooth_uvw(self):
        """
        Smooth UVW values (EXPERIMENTAL, CURRENTLY NOT USED)
        """

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
            for baseline_json in glob(self.tmp_folder+f"*baseline_mapping/{baseline[0]}-{baseline[1]}.json"):
                idxs += list(load_json(baseline_json).values())
            sorted_indices = np.argsort(time[idxs])
            for i in range(3):
                uvw[np.array(idxs)[sorted_indices], i] = gaussian_filter1d(uvw[np.array(idxs)[sorted_indices], i], sigma=2)

        self.T.putcol('UVW', uvw)


    def stack_all(self, column: str = 'DATA', keep_DP3_uvw: bool = False, safe_mem: bool = False, extra_cooldowns: bool = False):
        """
        Stack all MS

        :param:
            - column: column name (currently only DATA)
            - keep_DP3_uvw: keep DP3 UVW, no weighted average
            - safe_mem: limit RAM usage
        """

        if column == 'DATA':
            if keep_DP3_uvw:
                columns = [column, 'WEIGHT_SPECTRUM']
            else:
                columns = ['UVW', column, 'WEIGHT_SPECTRUM']
        else:
            sys.exit("ERROR: Only column 'DATA' allowed (for now)")

        # Get output data
        with table(path.abspath(self.outname), readonly=False, ack=False) as self.T:

            # Loop over columns
            for col in columns:

                # Arrays to fill
                if col == 'UVW':
                    new_data, uvw_weights = get_data_arrays(col, self.T.nrows(), self.freq_len, always_memmap=safe_mem, tmp_folder=self.tmp_folder)
                    if keep_DP3_uvw:
                        new_data = self.T.getcol("UVW")
                elif col=='WEIGHT_SPECTRUM':
                    new_data, _ = get_data_arrays(col, self.T.nrows(), self.freq_len, always_memmap=safe_mem, tmp_folder=self.tmp_folder)
                else:
                    new_data, _ = get_data_arrays(col, self.T.nrows(), self.freq_len, always_memmap=safe_mem, tmp_folder=self.tmp_folder)

                # Loop over measurement sets
                for ms in self.mslist:

                    print(f'\n{col} :: {ms}')

                    # Open MS table to stack on output data
                    t = table(f'{path.abspath(ms)}', ack=False, readonly=True)

                    # Get freqs offset
                    if col != 'UVW':
                        f = table(ms+'::SPECTRAL_WINDOW', ack=False)
                        freqs = f.getcol("CHAN_FREQ")[0]
                        freq_idxs = find_closest_index_list(freqs, self.ref_freqs)
                        f.close()

                    print('Collect relevant frequencies')

                    # Make antenna mapping in parallel
                    mapping_folder = self.tmp_folder + path.basename(ms) + '_baseline_mapping'

                    print('Read baseline mapping')
                    indices, ref_indices = read_mapping(mapping_folder)

                    # Only complex conjugate check for DATA columns
                    if "DATA" in col:
                        comp_conj = np.array(ref_indices) < 0
                        print(f"{col} needs to complex conjugate {np.sum(comp_conj)} values.")
                    else:
                        comp_conj = None

                    ref_indices = np.abs(ref_indices)

                    if len(indices)==0:
                        sys.exit('ERROR: cannot find *_baseline_mapping folders')

                    # Chunked stacking!
                    chunks = len(indices)//self.chunk_size + 1
                    print(f'Stacking in {chunks} chunks')
                    for chunk_idx in range(chunks):
                        print_progress_bar(chunk_idx, chunks+1)

                        start = chunk_idx * self.chunk_size
                        end = start + self.chunk_size

                        # Get indices
                        row_idxs_new = ref_indices[start:end]
                        row_idxs = np.array([int(i - start) for i in indices[start:end]])

                        data = t.getcol(col, startrow=start, nrow=self.chunk_size)

                        # Check if the row_idxs is a proper range
                        if not is_range(row_idxs):
                            data = data[row_idxs, :]
                            norange = True
                        else:
                            norange = False

                        # Take complex conjugate for inverted baselines
                        if comp_conj is not None:
                            comp_conj_mask = comp_conj[start:end]
                            if norange:
                                comp_conj_mask = comp_conj_mask[row_idxs]
                            if np.any(comp_conj_mask):
                                np.conjugate(data[comp_conj_mask], out=data[comp_conj_mask])

                        if col=='DATA':
                            # convert NaN to 0
                            data[np.isnan(data)] = 0.

                            # Multiply with weight_spectrum for weighted average
                            weights = t.getcol('WEIGHT_SPECTRUM', startrow=start, nrow=self.chunk_size)
                            if norange:
                                weights = weights[row_idxs, :]
                            data = multiply_arrays(data, weights)
                            del weights

                        # Reduce to one polarisation, since weights have same values for other polarisations
                        elif col=='WEIGHT_SPECTRUM':
                            data = data[..., 0]

                        if col == 'UVW':

                            weights = t.getcol("WEIGHT_SPECTRUM", startrow=start, nrow=self.chunk_size)[..., 0]
                            if norange:
                                weights = weights[row_idxs, :]
                            weights = add_axis(np.nanmean(weights, axis=1), 3)

                            # Stacking
                            uvw_weights[row_idxs_new, :] = sum_arrays(uvw_weights[row_idxs_new, :], weights)
                            subdata = multiply_arrays(data, weights)
                            if isinstance(new_data, np.memmap):
                                buffer = new_data[row_idxs_new, :].copy()
                                new_data[row_idxs_new, :] = sum_arrays(buffer, subdata)
                            else:
                                new_data[row_idxs_new, :] = sum_arrays(new_data[row_idxs_new, :], subdata)

                            # cleanup
                            subdata = None
                            weights = None

                            try:
                                uvw_weights.flush()
                            except AttributeError:
                                pass

                        else:
                            # Stacking
                            if isinstance(new_data, np.memmap):
                                buffer = new_data[row_idxs_new[:, None], freq_idxs].copy()
                                new_data[row_idxs_new[:, None], freq_idxs] = sum_arrays(buffer, data)
                            else:
                                new_data[row_idxs_new[:, None], freq_idxs] = sum_arrays(new_data[row_idxs_new[:, None],
                                                                                        freq_idxs], data)

                        # cleanup
                        data = None
                        buffer = None

                    try:
                        gc.collect()
                        new_data.flush()
                        if extra_cooldowns:
                            sleep(60)
                    except AttributeError:
                        pass

                    print_progress_bar(chunk_idx, chunks)
                    t.close()

                print(f'\nPut column {col}')
                if col == 'UVW':
                    uvw_weights[uvw_weights == 0] = 1
                    new_data /= uvw_weights
                    new_data[new_data != new_data] = 0.

                chunks = range(self.T.nrows() // self.chunk_size + 1)
                for chunk_idx in chunks:
                    print_progress_bar(chunk_idx, len(chunks))
                    startp = chunk_idx * self.chunk_size
                    endp = min(startp + self.chunk_size, self.T.nrows())

                    # Get chunk
                    subdat = new_data[startp:endp]

                    if col == 'WEIGHT_SPECTRUM':
                        subdat = add_axis(subdat, 4)
                    elif col == 'DATA':
                        subdat[subdat == 0] = np.nan

                    self.T.putcol(col, subdat, startrow=startp, nrow=endp - startp)

                # clean up
                del new_data
                clean_binary_file(self.tmp_folder + col.lower() + '.tmp.dat')

        # ADD FLAG
        print(f'Put column FLAG')
        taql(f'UPDATE {self.outname} SET FLAG = (WEIGHT_SPECTRUM == 0)')

        # NORM DATA
        print(f'Normalise column DATA')
        taql(f'UPDATE {self.outname} SET DATA = (DATA / WEIGHT_SPECTRUM) WHERE ANY(WEIGHT_SPECTRUM > 0)')

        print("----------\n")
