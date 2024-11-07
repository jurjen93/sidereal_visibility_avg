from casacore.tables import table
from os import path, system as run_command
from shutil import rmtree, move
from sys import exit


def is_dysco_compressed(ms):
    """
    Check if MS is dysco compressed

    :param:
        - ms: measurement set
    """

    with table(ms, readonly=True, ack=False) as t:
        return t.getdesc()["DATA"]['dataManagerGroup'] == 'DyscoData'


def decompress(ms):
    """
    running DP3 to remove dysco compression

    :param:
        - ms: measurement set
    """

    if is_dysco_compressed(ms):

        print('\n----------\nREMOVE DYSCO COMPRESSION\n----------\n')

        if path.exists(f'{ms}.tmp'):
            rmtree(f'{ms}.tmp')
        run_command(f"DP3 msin={ms} msout={ms}.tmp steps=[]")
        print('----------')
        return ms + '.tmp'

    else:
        return ms


def compress(ms):
    """
    running DP3 to apply dysco compression

    :param:
        - ms: measurement set
    """

    if not is_dysco_compressed(ms):

        print('DYSCO COMPRESSION')

        cmd = f"DP3 msin={ms} msout={ms}.tmp msout.overwrite=true msout.storagemanager=dysco"

        steps = []

        steps = str(steps).replace("'", "").replace(' ','')
        cmd += f' steps={steps}'

        run_command(cmd)

        try:
            t = table(f"{ms}.tmp", ack=False) # test if exists
            t.close()
        except RuntimeError:
            exit(f"ERROR: dysco compression failed (please check {ms})")

        rmtree(ms)
        move(f"{ms}.tmp", ms)

        print('----------')
        return ms

    else:
        return ms
