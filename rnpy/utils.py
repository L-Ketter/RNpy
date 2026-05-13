import os
import numpy as np
import pandas as pd
import .network as nw
from pyevtk.hl import imageToVTK

def store_data(data, names, dir_path=os.getcwd()):
    """
    Stores data in the specified directory. The data can be NumPy arrays or dictionaries.

    Parameters
    ----------
    data : list
        A list of data objects to be stored. Each object can be a NumPy array or a dictionary.
    names : list
        A list of names for the data objects to be stored. The suffix of the name will determine the file format (e.g., '.npy' for numpy arrays, '.csv' for dictionaries, '.vti' for VTI files).
    dir_path : str, optional
        The path to the directory where the data will be stored. If not provided, the current working directory will be used.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for obj, name in zip(data, names):
        obj = nw._shuttle_to_cpu(obj)
        if isinstance(obj, np.ndarray):
            if name.endswith('.npy'):
                np.save(os.path.join(dir_path, name[:-4]), obj)
            elif name.endswith('.vti'):
                get_vti(obj, name=os.path.join(dir_path, name[:-4]))
        elif isinstance(obj, dict):
            pd.DataFrame(obj).to_csv(
                os.path.join(dir_path, f"{name}"), index=False
            )

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
    Format seconds into a string of the form 'hh:mm:ss'.

    Parameters
    ----------
    seconds : int
        The number of seconds to format.

    Returns
    -------
    str
        A string representing the time in hours, minutes, and seconds.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
