import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import imageToVTK   
  
def blobs3D(size, disp_volfrac, sclust):
    ''' size = edge length in voxels,    
    disp_volfrac = volume fraction in percent,
    sclust = number of voxels per cluster \n
    returns a 3D cubic array representing a composite with dispersed phase inclusions of size sclust.'''        
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

def compositefigure(array, show=True, save=False, name='', cont_color='darkorchid', disp_color='khaki', filled=None):
    ''' array = 3D array,
    show = bool,    
    save = bool,
    name = str,\n
    plots and saves the array.'''
    filled = np.ones(array.shape, dtype=bool) if filled is None else filled # define to show all voxels
    colors = np.zeros(array.shape, dtype=object) # define the colors of continuous and dispersed phase
    colors[array == 0] = cont_color 
    colors[array == 1] = disp_color  
    colors[array == 2] = 'k'
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

def parallelconnection3D(size, disp_volfrac):
    ''' size = edge length in voxels,    
    disp_volfrac = volume fraction in percent \n
    returns a 3D cubic array representing a parallel connected composite.'''
    array = seriesconnection3D(size, disp_volfrac)  
    array = np.rot90(array, k=1, axes=(0,2))
    return array
        
def sc_random(size, disp_volfrac):
    ''' size = edge length in voxels,    
    disp_volfrac = volume fraction in percent \n
    returns a 3D cubic array representing a randomly mixed composite.'''
    disp_voxel_num = int(size**3 * disp_volfrac / 100)
    cont_voxel_num = int(size**3 - disp_voxel_num)
    array_1D = np.array([1]*disp_voxel_num + [0]*cont_voxel_num)
    np.random.shuffle(array_1D)
    array = array_1D.reshape((size,size,size))
    print('# total voxels', len(array_1D))
    print('# disp phase voxels:', (array == 1).sum())
    return array 

def seriesconnection3D(size, disp_volfrac):
    ''' size = edge length in voxels,    
    disp_volfrac = volume fraction in percent \n
    returns a 3D cubic array representing a series connected composite.'''
    disp_voxel_num = int(size**3 * disp_volfrac / 100)
    cont_voxel_num = int(size**3 - disp_voxel_num)
    array_1D = np.array([1]*disp_voxel_num + [0]*cont_voxel_num)
    array = array_1D.reshape((size,size,size))  
    print('# total voxels', len(array_1D))
    print('# disp phase voxels:', (array == 1).sum())
    return array  

def get_vti(array, name=''):  
    if len(array.shape) == 2:
        array = array[:, :, np.newaxis]    
    imageToVTK(name, cellData={"array": np.ascontiguousarray(array)})



    
    




    