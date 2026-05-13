import numpy as np
import os
import pandas as pd
from pyevtk.hl import imageToVTK

def store_data(data, names, dir_path=os.getcwd()):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for obj, name in zip(data, names):
        if isinstance(obj, np.ndarray):
            if name.endswith('.npy'):
                np.save(os.path.join(dir_path, name[:-4]), obj)
            elif name.endswith('.vti'):
                get_vti(obj, name=os.path.join(dir_path, name[:-4]))
        if isinstance(obj, dict):
            pd.DataFrame(obj).to_csv(os.path.join(dir_path, f"{name}"), index=False)

def get_vti(array, name='output'):
    """
    Converts a 3D numpy array to a VTI file format using pyevtk.

    Parameters
    ----------
    array : np.ndarray
        A 3D numpy array to be converted.
    name : str, optional
        The name of the output VTI file. If not provided, the file will be named 'output.vti'. Default is 'output'.
    """
    imageToVTK(name, cellData={"array": np.ascontiguousarray(array)})

def format_seconds(seconds):
    """
    Formats seconds into a string of the form 'hh:mm:ss'.

    Parameters:
    -----------
    seconds : int
        the number of seconds to format.

    Returns:
    --------
    a string representing the time in hours, minutes, and seconds.
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
