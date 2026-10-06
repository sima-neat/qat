
import os
import datetime
from pathlib import Path



def find_latest_file_string(root_path: str, tag_str: str='.onnx') -> str:
    ''' Recursively find the most recent 'tag_str' file in the specified path and
        parse it.
        This function is used to find the most recent output files of a subcommand
        in order to post-process the outputs.
    '''
    most_recent_dt = None
    most_recent_file = None
    for f_dir, f_subdirs, f_names in os.walk(root_path):
        for f in [fx for fx in f_names if tag_str in fx]:
            visit_file = os.path.join(f_dir, f)
            m_time = os.path.getmtime(visit_file)
            # convert timestamp into DateTime object
            dt_m = datetime.datetime.fromtimestamp(m_time)
            if not most_recent_dt:
                most_recent_dt = dt_m
                most_recent_file = visit_file
            else:
                if dt_m > most_recent_dt:
                    most_recent_dt = dt_m
                    most_recent_file = visit_file
    
    return most_recent_file

