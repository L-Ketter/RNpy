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

def compositefigure(
        arr,
        show=True,
        save=False,
        name='',
        color_lst=['darkorchid', 'khaki'],
        color_map=None,
        vmin=None,
        vmax=None):
    """
    Generates a 3D composite figure from a 3D numpy array and saves or displays it.
    The array should contain integer values representing different phases, where each unique value corresponds to a different phase.
    The phases are represented by different colors in the figure.

    Parameters
    ----------
    arr : np.ndarray
        A 3D numpy array containing integer values representing different phases.
    show : bool, optional
        If True, the figure will be displayed. Default is True.
    save : bool, optional
        If True, the figure will be saved as a PDF file. Default is False.
    name : str, optional
        The output file name (without extension). Default is an empty string.
    color_lst: list of str, optional
        A list of colors to represent the different phases in the figure.
        This is needed to represent arrays with discrete values.
        The length of the list should be >= the number of unique phases in the array.
        Default is ['darkorchid', 'khaki'].
    color_map: str, optional
        A colormap to represent the different phases in the figure.
        This is needed to represent arrays with continuous values.
        Note: Either color_lst or color_map should be provided, but not both.
        Default is None.
    vmin: float, optional
        If colormap is provided, the minimum value for the colormap can be specified by vmin.
        If colormap is provided and vmin is not provided, the minimum value of arr will be used.
    vmax: float, optional
        If colormap is provided, the maximum value for the colormap can be specified by vmax.
        If colormap is provided and vmax is not provided, the maximum value of arr will be used.
    """
    filled = np.ones(arr.shape, dtype=bool)
    color_arr = np.zeros(arr.shape, dtype=object)
    if color_lst and color_map:
        raise ValueError(
            "Either color_lst or color_map should be provided, but not both. " \
            f"You chose color_lst={color_lst} and color_map={color_map}. " \
            "One of them should be None."
        )
    if color_lst:
        for i in range(arr.max()+1):
            color_arr[arr == i] = color_lst[i] if i < len(color_lst) else 'gray'
    elif color_map:
        vmin = vmin if vmin is not None else arr.min()
        vmax = vmax if vmax is not None else arr.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(color_map)
        color_arr = cmap(norm(arr))
    # plot and save the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=color_arr)
    ax.set(xlim=(0,arr.shape[0]), ylim=(0,arr.shape[1]), zlim=(0,arr.shape[2]))
    ax.set_aspect('equal')
    ax.set_axis_off()
    if save is True:
        figname = name+'_compositefig.pdf'
        plt.savefig(figname)
    if show is True:
        plt.show()
    plt.close()

def get_vti(arr, name='output'):
    """
    Converts a 3D numpy array to a VTI file format using pyevtk.

    Parameters
    ----------
    arr : np.ndarray
        A 3D numpy array to be converted.
    name : str, optional
        The name of the output VTI file. If not provided, the file will be named 'output.vti'. Default is 'output'.
    """
    imageToVTK(name, cellData={"array": np.ascontiguousarray(arr)})

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
