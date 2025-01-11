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

        print('Remove Dysco compression')

        if path.exists(f'{ms}.tmp'):
            rmtree(f'{ms}.tmp')
        run_command(f"DP3 msin={ms} msout={ms}.tmp steps=[] > /dev/null 2>&1")
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

        print('Apply Dysco compression')

        cmd = (f"DP3 msin={ms} msout={ms}.tmp msout.overwrite=true msout.storagemanager=dysco "
               f"msout.storagemanager.databitrate=12 msout.storagemanager.weightbitrate=12")

        steps = []

        steps = str(steps).replace("'", "").replace(' ','')
        cmd += f' steps={steps}'

        run_command(cmd+' > /dev/null 2>&1')

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
