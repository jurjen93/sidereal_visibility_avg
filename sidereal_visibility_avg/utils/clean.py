from shutil import rmtree, move
from glob import glob
from os import system as run_command
from casacore.tables import table
from os import path, remove


def clean_mapping_files(msin):
    """
    Clean-up mapping files
    """

    for ms in msin:
        rmtree(ms + '_baseline_mapping')

def clean_binary_files(folder):
    """
    Clean-up binary files
    """

    if folder[-1] != '/':
        folder += '/'

    for b in glob(folder+'*.tmp.dat'):
        run_command(f'rm {b}')


def clean_binary_file(file):
    """
    Clean-up binary file
    """

    if path.exists(file):
        remove(file)


def remove_flagged_entries(input_table):
    """
    Remove flagged entries.
    Note that this corrupts the time axis.
    """

    # Define the output table temporary name
    output_table = input_table + '.copy.tmp'

    # Open the input table
    with table(input_table, ack=False) as tb:
        # Select rows that do not match the deletion criteria
        selected_rows = tb.query('NOT all(WEIGHT_SPECTRUM == 0)')

        # Create a new table with the selected rows
        selected_rows.copy(output_table, deep=True)

    # Overwrite the input table with the new table
    rmtree(input_table)
    move(output_table, input_table)

    # Make new time axis TODO: issues with new time axis for BDA datasets
    # t = table(input_table, ack=False, readonly=False)
    # time_old = np.unique(t.getcol("TIME"))
    # tm = np.linspace(time_old.min(), time_old.max(), t.nrows())
    #
    # ants = table(input_table + "::ANTENNA", ack=False)
    # baselines = np.c_[make_ant_pairs(ants.nrows(), 1)]
    # ants.close()
    #
    # t = repeat_elements(time_range, baseline_count)
