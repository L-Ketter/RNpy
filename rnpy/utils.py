import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .network import Network as nw
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
        if isinstance(obj, np.ndarray):
            obj = nw._shuttle_to_cpu(obj)
            if name.endswith('.npy'):
                np.save(os.path.join(dir_path, name[:-4]), obj)
            elif name.endswith('.vti'):
                get_vti(obj, name=os.path.join(dir_path, name[:-4]))
        elif isinstance(obj, dict):
            pd.DataFrame(obj).to_csv(
                os.path.join(dir_path, f"{name}"), index=False
            )

def compositefigure(array, show=True, save=False, name='', colors=['darkorchid', 'khaki']):
    """
    Generates a 3D composite figure from a 3D numpy array and saves or displays it.
    The array should contain integer values representing different phases, where each unique value corresponds to a different phase.
    The phases are represented by different colors in the figure.

    Parameters
    ----------
    array : np.ndarray
        A 3D numpy array containing integer values representing different phases.
    show : bool, optional
        If True, the figure will be displayed. Default is True.
    save : bool, optional
        If True, the figure will be saved as a PDF file. Default is False.
    name : str, optional
        The output file name (without extension). Default is an empty string.
    colors : list of str, optional
        A list of colors to represent the different phases in the figure. Default is ['darkorchid', 'khaki'].
        Length of the list should match the number of unique phases in the array.
    """
    filled = np.ones(array.shape, dtype=bool)
    color_arr = np.zeros(array.shape, dtype=object)

    for i in range(array.max()+1):
        color_arr[array == i] = colors[i] if i < len(colors) else 'gray'
    # plot and save the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=color_arr)
    ax.set(xlim=(0,array.shape[0]), ylim=(0,array.shape[1]), zlim=(0,array.shape[2]))
    ax.set_aspect('equal')
    ax.set_axis_off()
    if save is True:
        figname = name+'_compositefig.pdf'
        plt.savefig(figname) # save the image
    if show is True:
        plt.show() # show the image
    plt.close()

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
