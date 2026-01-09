import time 
import os
import numpy as np
from . import hotplate_utils as utils
import warnings
from numba import njit

@njit
def _get_T_new_SOR(T_arr, omega, ks):
    """
    Generates a temperature array with updated temperatures. The updated temperatures
    are calculated based on the conductivities and current temperatures using the 
    successive over relaxation (SOR) method. 
    numbas njit is utilized and speeds up the calculation significantly.

    Parameters
    ----------
    T_arr : np.ndarray
        an array with temperatures of all voxel_struc entries and boundaries.
    omega: float
        over-relaxation parameter. Typical values lie between 1 < omega < 2.
        omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
        rule of thumb: small values -> more stability,
                       large values -> faster convergence
    ks: list
        list of the conductivities in all directions. The conductivities
        entries need to be specified in the following order ['r','l','ba','f','t','b'] 
         
    Returns
    -------
    T_new : np.ndarray
        3D array of temperatures.
        - the temperature entries are updated according to the SOR method
    """     
    kr_rel, kl_rel, kba_rel, kf_rel, kt_rel, kb_rel = ks
    nx, ny, nz = T_arr.shape
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1): 
                T_new = (
                    T_arr[i+1, j, k] * kr_rel[i-1, j-1, k-1]
                    + T_arr[i-1, j, k] * kl_rel[i-1, j-1, k-1] 
                    + T_arr[i, j+1, k] * kba_rel[i-1, j-1, k-1]
                    + T_arr[i, j-1, k] * kf_rel[i-1, j-1, k-1] 
                    + T_arr[i, j, k+1] * kt_rel[i-1, j-1, k-1] 
                    + T_arr[i, j, k-1] * kb_rel[i-1, j-1, k-1]
                ) 
                T_arr[i, j, k] += omega * (T_new - T_arr[i, j, k])
    return T_arr

class HotPlate:    
    """
    Class for simulating conduction in a 3D voxel structure.
    
    Parameters
    ----------
    voxel_struc : np.ndarray
        3D array representing the voxel structure of the material. Must contain integer values
        representing different phases.
    phase_conds : list
        List of conductivities for each phase in the voxel structure. Indexes in the list correspond to
        the integer values in `voxel_struc`. 
    name : str, optional
        Nametag for the simulation. Default is an empty string.
    dtype : data-type, optional
        Data type for the arrays used in the simulation. Default is None, which uses the default NumPy data type.
        Choose complex128 for simulations using complex numbers.
    set_dir : str, optional 
        Directory where the simulation results will be stored. Default is the current directory ('.').
    use_gpu : bool, optional
        If True, uses CuPy for GPU acceleration; if False, uses NumPy. Default is False.
    mat_bounds : list of list, optional 
        2D list (shape: [n_phases][n_phases]) of boundary resistances between materials. Default is None.
        If provided, it must be accompanied by `int_bounds`.   
    int_bounds : list, optional
        List of boundary resistances between materials and hot plates. Default is None.
        If provided, it must be accompanied by `mat_bounds`.
    T_start : str, optional
        Path to a file containing the initial temperature array. If None, a linear temperature array is
        generated between the hotplates. Default is None.

    Other Parameters
    ----------------
    T_l : float, optional
        Temperature of the left hotplate. Default is 1.
    T_r : float, optional
        Temperature of the right hotplate. Default is 0.
    Lx : float, optional
        Length of the hotplate in the x-direction. Default is 1.  
    
    Raises
    ------
    ValueError
        If only one of `mat_bounds` or `int_bounds` is provided without the other.
    ImportError
        If `use_gpu=True` but CuPy is not installed.
    RuntimeError
        If `use_gpu=True` but CUDA is not available.
    """
    def __init__(self, voxel_struc, phase_conds, name='', dtype=None, set_dir='.', use_gpu=False, mat_bounds=None, int_bounds=None, T_start=None, **kwargs):
        self.voxel_struc = voxel_struc
        self.phase_conds = phase_conds
        self.name = name
        self.dtype = dtype
        self.set_dir = set_dir        
        self._build_working_directory(self.set_dir)

        self.use_gpu = use_gpu             
        self._choose_array_backend()   
        if self.use_gpu:
            self._get_SOR_redblack_mask() 

        self.mat_bounds = mat_bounds
        self.int_bounds = int_bounds
        if (self.mat_bounds is not None) != (self.int_bounds is not None):
            raise ValueError("To consider interfaces, both 'mat_bounds' and 'int_bounds' must be provided together. Only one was provided.")
        elif self.mat_bounds is not None and self.int_bounds is not None:
            self.mat_bounds = np.array(self.mat_bounds, dtype=self.dtype)
            self.int_bounds = np.array(self.int_bounds, dtype=self.dtype)
        
        self.T_l = kwargs.get('T_l', 1)
        self.T_r = kwargs.get('T_r', 0)    
        self.Lx = kwargs.get('Lx', 1)
        self.dx = self.Lx/voxel_struc.shape[0]
        
        self.directions = ['r','l','ba','f','t','b']
        self.arr_slices = {
            'center': (slice(1, -1), slice(1, -1), slice(1, -1)),
            'r': (slice(2, None), slice(1, -1), slice(1, -1)),
            'l': (slice(None, -2), slice(1, -1), slice(1, -1)),
            'ba': (slice(1, -1), slice(2, None), slice(1, -1)),
            'f': (slice(1, -1), slice(None, -2), slice(1, -1)),
            't': (slice(1, -1), slice(1, -1), slice(2, None)),
            'b': (slice(1, -1), slice(1, -1), slice(None, -2))
        }

        self._get_calc_arr() # build conductivity arrays in all directions and the summed conductivities
        self.T_start = self._get_T_start(T_start)
    
    def _build_working_directory(self, directory):
        """
        Creates the directory for storing the calculation results if it does not exist.
    
        Parameters
        ----------
        directory : str
            Path to the directory to be created.
        """        
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def _choose_array_backend(self):
        """
        Selects NumPy (CPU) or CuPy (GPU) as the array backend.
            
        Raises
        ------
        ImportError
            If `use_gpu=True` but CuPy is not installed.
        RuntimeError
            If `use_gpu=True` but CUDA is unavailable.
        """        
        if self.use_gpu:
            try:
                import cupy as cp 
            except ImportError: 
                raise ImportError("CuPy is not installed.")     
            if not cp.cuda.is_available():
                raise RuntimeError("CuPy is installed, but CUDA is not available.")
            self.xp = cp 
        else:
            self.xp = np 
                        
    def _get_kappa_arr(self):        
        """
        Builds a 3D conductivity array with voxel conductivities of the pure phases and pads boundaries. 
    
        Returns
        -------
        kappa_arr : xp.ndarray
            3D array of conductivities with padded boundary values.
            - The corresponding conductivities from `phase_conds` are assigned
              for entries in `voxel_struc` matching the index in `phase_conds`
            - boundary planes are padded:
              * +x and -x faces set to `inf` (hotplates)
              * +y, -y, +z, -z faces set to 0 (adiabatic boundaries)
        """        
        kappa_arr = self.xp.zeros(self.voxel_struc.shape, dtype=self.dtype)
        for i, cond in enumerate(self.phase_conds):
            kappa_arr[self.voxel_struc == i] = cond
        kappa_arr = self.xp.pad(
                                kappa_arr,
                                pad_width = ((1,1), (1,1), (1,1)),
                                mode = 'constant',
                                constant_values = ((float('inf'), float('inf')), (0,0), (0,0)))
        return kappa_arr
    
    def _get_R_interface_arr(self):
        """
        Builds the interface resistance arrays (on the cpu, and later shuttles it on the gpu if needed)
        First pad integers for the plates and adiabatic boundaries to the voxel_struc.
        Then combine `mat_bounds` and `int_bounds` into a boundary matrix as in the following example: 
        int_bounds = [0,1]
        mat_bounds = [[2,3],
                      [4,5]]
        boundary_matrix = [[ 2.  3.  0. inf]
                           [ 4.  5.  1. inf]
                           [ 0.  1. inf inf]
                           [inf inf inf inf]]      
        Then generate the interface resistance arrays for each direction, based on the padded voxel structure and the boundary matrix.

        Returns
        -------
        R_ints : dict
            Dictionary with interface resistance arrays for each direction.
            The keys are the directions ('r', 'l', 'ba', 'f', 't', 'b') and the values are the corresponding resistance arrays.
        """
        voxel_struc_np = self._to_cpu(self.voxel_struc)
        plate_integ = voxel_struc_np.max() + 1
        adiabatic_integ = voxel_struc_np.max() + 2        
        voxel_struc_w_bound_integ = np.pad(voxel_struc_np,
                                pad_width = ((1,1), (1,1), (1,1)),
                                mode = 'constant',
                                constant_values = ((plate_integ,plate_integ),
                                                   (adiabatic_integ,adiabatic_integ),
                                                   (adiabatic_integ,adiabatic_integ)))          
        
        boundary_matrix = np.full((plate_integ+2, plate_integ+2), np.inf, dtype=self.dtype)
        boundary_matrix[:plate_integ, :plate_integ] = self.mat_bounds
        boundary_matrix[:plate_integ, plate_integ] = self.int_bounds
        boundary_matrix[plate_integ, :plate_integ] = self.int_bounds

        R_ints = {}        
        integ_center = voxel_struc_w_bound_integ[self.arr_slices['center']]
        for direction in self.directions:
            integ_neighbor = voxel_struc_w_bound_integ[self.arr_slices[direction]]
            interface_arr = boundary_matrix[integ_center, integ_neighbor] 
            R_ints[direction] = self.xp.asarray(interface_arr)       
        return R_ints
            
    def _get_T_start(self, T_start_center): 
        """
        Initializes the temperature array with a linear temperature distribution between the hotplates.
        If `T_start_center` is provided, it loads the initial temperature array from the specified file.
        It sets all temperatures for nodes with 0 conductivity in all directions to 0.

        Parameters
        ----------
        T_start_center : str, optional
               Path to a file containing the initial temperature array. If None, a linear temperature array is generated.
        
        Returns
        -------
        T_start : xp.ndarray
            3D array of initial temperatures.
        """       
        T_start = self._get_T_arr_linguess()
        if T_start_center:
            T_start[self.arr_slices['center']] = self.xp.load(os.path.join(self.set_dir, T_start_center))  
        T_zero_mask = (self._get_kappa_arr() == 0)
        T_zero_mask[self.arr_slices['center']] = (self.ksum_arr==0)
        T_start[T_zero_mask] = 0
        return T_start 
        
    def _get_calc_arr(self):
        """
        Builds the conductivity arrays for the hotplate simulation.
        If `mat_bounds` and `int_bounds` are provided, the interface resistances are calculated 
        using the _get_R_interface_arr method and used in the bond conductivity calculation. 
        For each spacial direction, the bond conductivities are calculated using the internal _calc_bond_cond method. 
        Then the sum of all bond conductivities is calculated and stored in `ksum_arr`.
        The conductivities in each direction, relative to `ksum_arr`, are stored in `k{direction}_rel_arr` attributes.
        """
        if self.mat_bounds is not None and self.int_bounds is not None:
            R_ints = self._get_R_interface_arr()
        
        kappa_arr = self._get_kappa_arr()
        def _calc_bond_cond(k_spot, k_neighbor, R_int=None):
            """
            Calculates the bond conductivities between two neighboring voxels.

            Parameters
            ----------
            k_spot : xp.ndarray
                Conductivities of the central voxels.
            k_neighbor : xp.ndarray
                Conductivities of the neighboring voxels.
            R_int : xp.ndarray, optional
                Interface resistances between the voxels. 
            
            Returns
            -------
            k_link : xp.ndarray
                Conductivities of the bonds between the voxels.
            """
            zero_mask = (k_spot == 0) | (k_neighbor == 0) | (R_int == np.inf)           
            other_mask = ~(zero_mask)            
            k_link = self.xp.zeros(k_spot.shape, dtype=self.dtype)
            if R_int is not None:
                k_link[other_mask] = (0.5*(1/k_spot[other_mask] + 1/k_neighbor[other_mask]) + R_int[other_mask]/self.dx)**-1
            else:
                k_link[other_mask] = (0.5*(1/k_spot[other_mask] + 1/k_neighbor[other_mask]))**-1
            return k_link      
        
        k_spot = kappa_arr[self.arr_slices['center']]  
        k_arrs = {}              
        for direc in self.directions:
            k_neighbor = kappa_arr[self.arr_slices[direc]]
            if self.mat_bounds is not None and self.int_bounds is not None:
                k_arrs[direc] = _calc_bond_cond(k_spot,k_neighbor,R_ints[direc])
            else:                
                k_arrs[direc] = _calc_bond_cond(k_spot,k_neighbor)
        
        self.ksum_arr = self.xp.sum(list(k_arrs.values()), axis=0)         
        k_rel_arrs = {
            direc: self.xp.divide( 
                k_arrs[direc],
                self.ksum_arr,
                out=self.xp.zeros_like(k_arrs[direc], dtype=self.dtype),
                where=self.ksum_arr!=0
                ) # use xp.divide to avoid division by zero
            for direc in self.directions            
        }        
        for direc in self.directions:
            setattr(self, f'k{direc}_rel_arr', k_rel_arrs[direc])
        
    def _get_T_arr_linguess(self):
        """
        Builds an initial temperature array with linearly de- or increasing
        temperatures between the hotplates
            
        Returns
        -------
        kappa_arr : xp.ndarray
            3D array of temperatures.
            - the temperature entries are linearly de- or increasing
              from `T_l`, the temperature of the left hotplate
              to `T_r`, the temperature of the right hotplate. 
        """
        nx, ny, nz = [dim+2 for dim in self.voxel_struc.shape] # size of voxelstruc + boundaries
        T_arr = self.xp.zeros((nx, ny, nz), dtype=self.dtype)
        T_arr[0] = self.T_l
        T_arr[-1] = self.T_r        
        positions = self.xp.arange(1,nx-1) - 1/2
        T_arr[1:-1] = (self.T_l+((self.T_r-self.T_l)/(nx-2))*positions)[:, None, None]
        return T_arr
    
    def _get_SOR_redblack_mask(self):
        """
        Generates checkerboard views needed to update the Temperature array 
        via the _update_SOR_redblack method.
        """              
        nx, ny, nz = self.voxel_struc.shape
        i_even, i_odd = slice(0,nx,2), slice(1,nx,2)
        j_even, j_odd = slice(0,ny,2), slice(1,ny,2)
        k_even, k_odd = slice(0,nz,2), slice(1,nz,2)        
        self.red_slices = [
            (i_even, j_even, k_even),
            (i_even, j_odd,  k_odd),
            (i_odd,  j_even, k_odd),
            (i_odd,  j_odd,  k_even)
        ]
        self.black_slices = [
            (i_even, j_even, k_odd),
            (i_even, j_odd,  k_even),
            (i_odd,  j_even, k_even),
            (i_odd,  j_odd,  k_odd)
        ]

    def _get_T_new_Jacobi(self):  
        """
        Generates a temperature array with updated temperatures. The updated temperatures
        are calculated based on the conductivities and current temperatures using the Jacobi method. 
            
        Returns
        -------
        T_new : xp.ndarray
            3D array of temperatures, where temperature entries are updated
            temperatures calculated via the Jacobi method.
        """     
        T_new = self.xp.zeros((self.voxel_struc.shape), dtype=self.dtype)
        for direc in self.directions:
            T_new += self.T_arr[self.arr_slices[direc]] * getattr(self, f'k{direc}_rel_arr')
        return T_new          
            
    def _update_SOR_redblack(self, omega):
        """
        Updates temperatures in the current temperature array using the SOR redblack algorithm.
        The algorithm is utilizing a two-step procedure: In the first step, the temperatures
        in the red-mask, and in the second step, temperatures in the black mask are updated.
        In each step, new temperatures are updated using the Jacobi method with over-relaxation
        
        Parameters
        ----------
        omega: float
            over-relaxation parameter. Typical values lie between 1 < omega < 2.
            omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
            rule of thumb: small values -> more stability,
                           large values -> faster convergence
        """
        if not hasattr(self, "red_slices"):
            self._get_SOR_redblack_mask() 
            
        for color_slices in [self.red_slices, self.black_slices]:
            for slice_i, slice_j, slice_k in color_slices:
                self.T_arr[self.arr_slices['center']][slice_i, slice_j, slice_k] += (                    
                    omega * (
                        self.T_arr[self.arr_slices['r']][slice_i, slice_j, slice_k] * self.kr_rel_arr[slice_i, slice_j, slice_k] 
                        + self.T_arr[self.arr_slices['l']][slice_i, slice_j, slice_k] * self.kl_rel_arr[slice_i, slice_j, slice_k] 
                        + self.T_arr[self.arr_slices['ba']][slice_i, slice_j, slice_k] * self.kba_rel_arr[slice_i, slice_j, slice_k]
                        + self.T_arr[self.arr_slices['f']][slice_i, slice_j, slice_k] * self.kf_rel_arr[slice_i, slice_j, slice_k] 
                        + self.T_arr[self.arr_slices['t']][slice_i, slice_j, slice_k] * self.kt_rel_arr[slice_i, slice_j, slice_k] 
                        + self.T_arr[self.arr_slices['b']][slice_i, slice_j, slice_k] * self.kb_rel_arr[slice_i, slice_j, slice_k]
                        - self.T_arr[self.arr_slices['center']][slice_i, slice_j, slice_k]
                    )
                )
                
    def _update_SOR(self, omega):
        """
        Updates temperatures in the current temperature array using the SOR algorithm.

        Parameters
        ----------
        omega: float
            over-relaxation parameter. Typical values lie between 1 < omega < 2.
            omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
            rule of thumb: small values -> more stability,
                           large values -> faster convergence
        """
        T_arr = self.T_arr
        ks = tuple(getattr(self, f'k{direc}_rel_arr') for direc in self.directions)
        self.T_arr = _get_T_new_SOR(T_arr, omega, ks)

    def _get_residuals(self):
        """
        Calculates the mean residual. The mean residual is defined as the absolute 
        temperature difference between the actual temperatures and updated temperatures 
        calculated using the Jacobi method.
        
        Returns
        -------
        mean_resid: float
            The mean residual of the current temperature distribution.
        """  
        T_new = self._get_T_new_Jacobi()
        Resid = self.xp.abs(self.T_arr[self.arr_slices['center']] - T_new) 
        mean_resid = self.xp.mean(Resid)
        return mean_resid
    
    def _get_kappa(self):
        """
        Calculated the effective conductivity. First the average flux density going through the network
        along the field direction is calculated. Next, considering the mean flux density, 
        the temperature difference specified by T_l and T_r, 
        as well as the spacial distance between the hotplates, the effective conductivity is calculated using Fouriers law.
               
        Returns
        -------
        kappa: float
            The effective conductivity calculated using Fouriers law.
        """
        nx = self.voxel_struc.shape[0]
        x_len = nx*self.dx
        Q_lr = (- self.kl_rel_arr*self.ksum_arr
                * (self.T_arr[self.arr_slices['center']] - self.T_arr[self.arr_slices['l']])) / self.dx
        Q_lr_last = (- self.kr_rel_arr[-1,:,:]*self.ksum_arr[-1,:,:]
                     * (self.T_r - self.T_arr[-2, 1:-1, 1:-1])) / self.dx 
        mean_Q_lr = np.mean(np.concatenate((Q_lr, Q_lr_last[np.newaxis,:,:]), axis=0))
        kappa = - mean_Q_lr*x_len / (self.T_r-self.T_l)
        return kappa

    def _get_jloc(self):
        """
        Note: When jloc is calculated with default values (dT=-1, Lx=1) it might need to be rescaled. 
        To get the real jloc for your system, multiply it with dT_real/Lx_real. Often it is sufficient 
        to just show normalized jloc. In such cases, simply divide jloc by its maximum entry.
        ------------------------------------------------------------------------------
        Builds an array storing the local flux densities going through each node. 
        The calculation assumes steady state conditions (flux into a node equals flux out of a node).
        1) Add the sums of all flux terms (note: relative conductivities are used) in each direction.
        2) Multiply with the sum of all conductivities in each direction (to get real fluxes) 
        and divide by the distance between neighboring nodes to get real current densities.
        3) Due to steady state conditions, half of the calculated sum of absolute fluxes
        is going into the node and half is going out of the node. As we are interested in the local current density,
        going through the node, we divide by 2. 
        """
        jloc_term_sum = self.xp.zeros(self.voxel_struc.shape, dtype=self.dtype)
        for direc in self.directions:
            jloc_term_sum += np.abs(
                getattr(self, f'k{direc}_rel_arr') 
                * (self.T_arr[self.arr_slices[direc]] - self.T_arr[self.arr_slices['center']])
                )
        self.jloc = jloc_term_sum * self.ksum_arr/(self.dx * 2) 
        
    def _store_data(self, tag, save_residual_plot=False, remove_act_files=True):  
        """
        Stores the simulation data (conductivities, residuals, Temperature array) in the specified directory
        and removes files with the 'act' tag if specified.

        Parameters
        ----------
        name : str
            Name of the simulation. Used to create the filename for storing the data.
        tag : str
            Tag should be "act" or "final" and is used to differentiate between different states of the simulation.
        """           
        if remove_act_files:
            self._remove_files(tag='act') 
        if self.kappas_sim is not None:
            utils.store_x_and_y_data(np.array(self.iterations), np.array(self.kappas_sim), 'iteration','kappa_sim',
                                     name=os.path.join(self.set_dir, f'{self.name}_iter_{self.iterations[-1]}_{tag}kappa_{self.kappas_sim[-1]:.5f}'))
        if self.residuals is not None:
            utils.store_x_and_y_data(np.array(self.iterations), np.array(self.residuals), 'iteration', 'residual',
                                     name=os.path.join(self.set_dir, f'{self.name}_iter_{self.iterations[-1]}_{tag}resid_{self.residuals[-1]:.2e}')) 
            if save_residual_plot:
                utils.makelogplot(np.array(self.iterations), np.array(self.residuals), 'iteration', 'residual / K',
                                xscale='lin', yscale='log', save=True, show=False,
                                name=os.path.join(self.set_dir, f'{self.name}_residuals_{tag}')) 
                
        if self.T_arr is not None:
            self.xp.save(os.path.join(self.set_dir, f'{self.name}_iter_{self.iterations[-1]}_{tag}Tarr.npy'), self.T_arr[self.arr_slices['center']])
        
        self._get_jloc()
        if self.jloc is not None:
            self.xp.save(os.path.join(self.set_dir, f'{self.name}_iter_{self.iterations[-1]}_{tag}jloc.npy'), self.jloc)

    def _remove_files(self, tag='act'):
        """
        Removes all files in the set directory that contain the specified tag.

        Parameters
        ----------
        tag : str
            Tag to search for in the filenames. Files containing this tag will be removed.
        """
        for filename in os.listdir(self.set_dir):
            if tag in filename and self.name in filename:
                os.remove(os.path.join(self.set_dir, filename))
    
    def _to_cpu(self, val):
        """ 
        Converts a value to CPU if it is a CuPy array, otherwise returns the value as is.

        Parameters
        ----------
        val : any
            The value to be converted. It can be a CuPy array, NumPy array, or any other type.

        Returns
        -------
        any
            The value converted to CPU if it was a CuPy array, otherwise the original value.
        """
        return val.get() if hasattr(val, "get") else val
                
    def _update_lists(self, print_output=True):
        """
        Updates the lists of iterations, kappas, and residuals with the current values. 
        It also prints the current iteration, effective conductivity, and residual if `print_output` is True.

        Parameters
        ----------
        print_output : bool, optional
            If True, prints the current iteration, effective conductivity, and residual.
        """                
        self.iterations.append(self._to_cpu(self.iteration))
        self.kappas_sim.append(self._to_cpu(self._get_kappa()))
        self.residuals.append(self._to_cpu(self._get_residuals()))           
        if print_output:
            print(f"\riteration: {self.iterations[-1]} | conductivity: {self.kappas_sim[-1]:.4f} | "
                  f"residual: {self.residuals[-1]:.2e} ==> {self.cutoff_resid:.2e}{10*' '}", end="", flush=True)
        
    def _tune_omega(self):
        """
        Tunes the omega parameter based on the convergence behavior of the residuals.
        If last four residuals are increasing, it reduces omega by 0.1 and restarts the simulation.
        If omega is at 1.0, it is not changed any further.
        """
        if len(self.residuals) >= 4 and self.omega > 1 and self.residuals[-4] < self.residuals[-3] < self.residuals[-2] < self.residuals[-1]:
            self.omega = max(self.omega - 0.1, 1.0)            
            print(f"\nrestarting with new omega ({self.omega:.3f}) ...")
            self.T_arr = self.T_start.copy()
            self.iteration, self.iterations, self.residuals, self.kappas_sim = 0, [], [], []
            self._update_lists(print_output=True)

    def run(self, omega=1.979, max_iter=1e7, cutoff_resid=1e-9, log_iter=50, save_iter=None, tune_omega=False, get_voxelstruc=True):
        """        
        Runs the hotplate simulation until convergence or until the maximum number of iterations is reached.
        The simulation iteratively updates the temperature array using the SOR method (or in gpu-mode SOR red-black) until the residuals
        are below the specified cutoff or the maximum number of iterations is reached.

        Parameters
        ----------
        omega: float, optional
            Overt-relaxation parameter for the SOR method. Default is 1.979
            Typical values lie between 1 < omega < 2.
            omega > 2 leads to unstable convergence. omega < 1 leads to slow convergence.
            rule of thumb: small values -> more stability,
                        large values -> faster convergence   
        max_iter : int, optional
            Maximum number of iterations to run the simulation. Default is 1e7.
        cutoff_resid : float, optional
            Cutoff value for the residuals. The simulation stops when the mean residual is below this value.
            Default is 1e-9.    
        log_iter : int, optional
            Interval for logging the current iteration, effective conductivity, and residual. 
            If None, no logging is done. Default is 50. 
        save_iter : int, optional
            Interval for saving the current state of the simulation. If None, no saving is done until the end of the calculation.
            Default is None.
        tune_omega : bool, optional    
            If True, tunes the omega parameter based on the convergence behavior of the residuals.
            If False, uses the specified omega value throughout the simulation. Default is False.
        get_voxelstruc: bool, optional
            If True, saves the voxelstruc as .npy in the specified directory
        """
        if get_voxelstruc == True:
            self.xp.save(os.path.join(self.set_dir, f'{self.name}_voxelstruc.npy'), self.voxel_struc)
        self.omega = omega
        self.max_iter = max_iter
        self.cutoff_resid = cutoff_resid
        print("-"*70,
              f"\n{self.name = }"
              "\nsetting up the RN simulation ...")
        t_start = time.time() 
        self.T_arr = self.T_start.copy()
        self.iteration = 0
        self.iterations, self.residuals, self.kappas_sim = [], [], []
        print('approximating the steady state ...')    
        while self.iteration <= self.max_iter:          
            if log_iter is not None and self.iteration % log_iter == 0:
                self._update_lists(print_output=True)             
                if tune_omega:
                    self._tune_omega() 
            if save_iter is not None and self.residuals[-1] > self.cutoff_resid and self.iteration % save_iter == 0:
                self._store_data(tag='act')
            if self.residuals[-1] < self.cutoff_resid or self.iteration == self.max_iter:
                if self.iteration == self.max_iter:
                    warnings.warn(
                        f"maximum number of iterations ({self.max_iter}) reached without convergence "
                        f"(final residual: {self.residuals[-1]:.2e} ==> {self.cutoff_resid:.2e}).") 
                self._store_data(tag='final', save_residual_plot=True)                
                print(f'\nFinished calculation in {utils.format_seconds(time.time()-t_start)} (hh:mm:ss)')
                break 

            self.iteration += 1         
            if not self.use_gpu:
                self._update_SOR(self.omega)
            elif self.use_gpu:
                self._update_SOR_redblack(self.omega) 
                


                
                
            
                    
                    
        
        