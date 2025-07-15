from casacore.tables import table
import time

def parse_input_args(namespace):
    return [f"{k}={repr(v)}" for k, v in vars(namespace).items()]

def insert_history(ms_path, parameters=[], message='parameters', app='SVA', appver='2.0.0', origin='SVA (https://github.com/jurjen93/sidereal_visibility_avg)'):
    """
    Insert history in MeasurementSet

    Args:
        ms_path: Path to MeasurementSet
        parameters: Parameters
        message: Message
        app: Application
        appver: Application ersion
        origin: Software origin

    """
    history_table_path = ms_path.rstrip('/') + '/HISTORY'
    with table(history_table_path, readonly=False, ack=False) as t:

        nrows = len(t)
        t.addrows(1)

        now = time.time()
        columns = t.colnames()

        if 'TIME' in columns:
            t.putcell('TIME', nrows, now)
        if 'OBSERVATION_ID' in columns:
            t.putcell('OBSERVATION_ID', nrows, 0)
        if 'MESSAGE' in columns:
            t.putcell('MESSAGE', nrows, message)
        if 'PRIORITY' in columns:
            t.putcell('PRIORITY', nrows, 'NORMAL')
        if 'ORIGIN' in columns:
            t.putcell('ORIGIN', nrows, origin+" "+appver)
        if 'APPLICATION' in columns:
            t.putcell('APPLICATION', nrows, app)
        if 'CLI_COMMAND' in columns:
            t.putcell('CLI_COMMAND', nrows, [])
        if 'APP_PARAMS' in columns:
            t.putcell('APP_PARAMS', nrows, parameters)
