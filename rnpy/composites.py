import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import imageToVTK   
  
def blobs3D(size, disp_volfrac, sclust):
    """
    Generates a 3D cubic array representing a binary composite with dispersed phase and inclusions of size sclust.
    The dispersed phase is represented by 1s and the continuous phase by 0s.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    disp_volfrac : float
        Volume fraction of the dispersed phase in percent (0-100).
    sclust : int
        Number of voxels per cluster of the dispersed phase inclusions.

    Returns
    -------
    np.ndarray
        A 3D cubic array with the dispersed phase inclusions represented by 1s and the continuous phase by 0s.
    """
          
    if disp_volfrac == 100 or disp_volfrac == 0 or sclust == 1:
        array = sc_random(size, disp_volfrac)
    else:
        array = np.zeros((size, size, size), dtype=int) 
        disp_voxel_num = size**3 * disp_volfrac / 100
        clust_num = int(disp_voxel_num/sclust)         
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]     
        # generate and insert clusters 
        for num in range(clust_num): 
            cluster = [] 
            start_pos = np.random.choice(size, 3)
            cluster.append(tuple(start_pos))                          
            while len(cluster) < sclust: # iteratively generate a clusters of connected positions           
                chosen_direc = directions[np.random.choice(len(directions))] 
                chosen_pos = cluster[np.random.choice(len(cluster))] 
                newpos = list(x + y for x, y in zip(chosen_pos, chosen_direc)) 
                newpos = tuple(coord % size for coord in newpos) # ensure periodic boundaries                    
                if newpos not in cluster:
                    cluster.append(newpos)                    
            for x,y,z in cluster: # insert a cluster into the array
                array[x,y,z] = 1            
        # add voxels to reach desired volume fraction       
        missing = disp_voxel_num - (array==1).sum() 
        count = 0   
        while count < missing:       
            x,y,z = np.random.choice(size, 3)       
            if array[x,y,z] == 0:
                array[x,y,z] = 1
                count+=1    
        print('# total voxels:', (array==0).sum() + (array==1).sum())
        print('# disp phase voxels:', (array==1).sum()) 
    return array

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
    colors = np.zeros(array.shape, dtype=object) 

    for i in range(array.max()+1):
        colors[array == i] = colors[i] if i < len(colors) else 'gray' 
    # plot and save the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=colors)  
    ax.set(xlim=(0,array.shape[0]), ylim=(0,array.shape[1]), zlim=(0,array.shape[2]))
    ax.set_aspect('equal')        
    ax.set_axis_off()        
    if save is True:       
        figname = name+'_compositefig.pdf'             
        plt.savefig(figname) # save the image 
    if show is True:
        plt.show() # show the image
    plt.close()           

def parallelconnection3D(size, volfracs):
    """
    Generates a 3D cubic array representing a parallel connected composite. (phases stacked along the z-axis))
    The number of voxels in each phase is determined by the volume fractions provided.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    volfracs : list of float
        List of volume fractions for each phase in percent (0-100). The sum of the
        volume fractions should equal 100.

    Returns
    -------
    np.ndarray
        A 3D cubic array with phases stacked along the z-axis.
    """

    array = seriesconnection3D(size, volfracs)  
    array = np.rot90(array, k=1, axes=(0,2))
    return array
        
def sc_random(size, volfracs):
    """
    Generates a 3D cubic array representing a random composite with specified volume fractions.
    The number of voxels in each phase is determined by the volume fractions provided.
    
    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    volfracs : list of float
        List of volume fractions for each phase in percent (0-100). The sum of the
        volume fractions should equal 100.
    
    Returns
    -------
    np.ndarray
        A 3D cubic array with phases randomly distributed.
    """    

    ordered = seriesconnection3D(size, volfracs).flatten()
    rng = np.random.default_rng()
    shuffled = rng.permutation(ordered)
    array = shuffled.reshape((size, size, size))
    return array 

def seriesconnection3D(size, volfracs):
    """
    Generates a 3D cubic array representing a series connected composite. (phases stacked along the x-axis))
    The number of voxels in each phase is determined by the volume fractions provided.

    Parameters
    ----------
    size : int
        Edge length of the cubic array in voxels.
    volfracs : list of float
        List of volume fractions for each phase in percent (0-100). The sum of the
        volume fractions should equal 100.

    Returns
    -------
    np.ndarray
        A 3D cubic array with phases stacked along the x-axis.
    """

    vox_nums = size**3 * np.array(volfracs)/100 # convert to fraction
    array_1D = [[i]*vox_nums[i] for i in range(len(vox_nums))][0]   
    array = array_1D.reshape((size, size, size))  
    return array  

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

    if len(array.shape) == 2:
        array = array[:, :, np.newaxis]    
    imageToVTK(name, cellData={"array": np.ascontiguousarray(array)})



    
    




    