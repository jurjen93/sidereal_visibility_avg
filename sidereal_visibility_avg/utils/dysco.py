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


def decompress(ms, msout='tmp.ms'):
    """
    running DP3 to remove dysco compression

    :param:
        - ms: measurement set
        - msout: measurement set output name
    """

    if is_dysco_compressed(ms):

        print('Remove Dysco compression')

        if path.exists(f'{msout}'):
            rmtree(f'{msout}')
        run_command(f"DP3 msin={ms} msout={msout} steps=[]")
        if msout=='tmp.ms':
            run_command(f"rm -rf {ms} && mv {msout} {ms}")
            return ms
        else:
            return msout

    else:
        return ms


def compress(ms, bitrate):
    """
    running DP3 to apply dysco compression

    :param:
        - ms: measurement set
        - bitrate: dysco bitrate
    """

    if not is_dysco_compressed(ms):

        print('Apply Dysco compression')

        cmd = (f"DP3 msin={ms} msout={ms}.tmp msout.overwrite=true msout.storagemanager=dysco "
               f"msout.storagemanager.databitrate={bitrate} msout.storagemanager.weightbitrate=12")

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
