import numpy as np
import os
import pandas as pd
from pyevtk.hl import imageToVTK

def store_data(data, names, path=os.getcwd()):
    if not os.path.exists(path):
        os.makedirs(path)

    for obj, name in zip(data, names):
        if isinstance(obj, np.ndarray):
            if name.endswith('.npy'):
                np.save(os.path.join(path, name), obj)
            elif name.endswith('.vti'):
                get_vti(obj, name=os.path.join(path, name[:-4]))
        if isinstance(obj, dict):
            pd.DataFrame(obj).to_csv(os.path.join(path, f"{name}.csv"), index=False)

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