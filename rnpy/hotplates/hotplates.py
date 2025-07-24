import time 
import os
import numpy as np
import hotplate_utils as utils
import warnings
import sys 
from numba import njit, prange

@njit
def SOR_update_numba(T_arr, omega, ks):
    kr_rel, kl_rel, kba_rel, kf_rel, kt_rel, kb_rel = ks
    nx, ny, nz = T_arr.shape
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                
                T_new = (
                    T_arr[i+1, j, k] * kr_rel[i-1, j-1, k-1] +
                    T_arr[i-1, j, k] * kl_rel[i-1, j-1, k-1] +
                    T_arr[i, j+1, k] * kba_rel[i-1, j-1, k-1] +
                    T_arr[i, j-1, k] * kf_rel[i-1, j-1, k-1] +
                    T_arr[i, j, k+1] * kt_rel[i-1, j-1, k-1] +
                    T_arr[i, j, k-1] * kb_rel[i-1, j-1, k-1]
                ) 
                # Divide by sum of conductivities
                T_arr[i, j, k] += omega * (T_new - T_arr[i, j, k])
    return T_arr


class HotPlate:    
    
    def __init__(self, voxel_struc, phase_conds=None, **kwargs):
        self.voxel_struc = voxel_struc
        self.phase_conds = phase_conds
        self.T_l = kwargs.get('T_l', 1)
        self.T_r = kwargs.get('T_r', 0)    
        self.Lx = kwargs.get('L', 1)
        self.dx = self.Lx/voxel_struc.shape[0]
        self.dtype = kwargs.get('dtype', None)

        self.set_dir = kwargs.get('set_dir', '.')
        self._build_working_directory(self.set_dir)
        self._choose_array_backend(kwargs.get('use_gpu', False)) 
        
        self.mat_bounds = kwargs.get('mat_bounds', None)
        self.int_bounds = kwargs.get('int_bounds', None)
        if (self.mat_bounds is not None) != (self.int_bounds is not None):
            raise ValueError("To consider interfaces, both 'mat_bounds' and 'int_bounds' must be provided together: Only one was given")
        elif self.mat_bounds is not None and self.int_bounds is not None:
            self.mat_bounds = self.xp.array(self.mat_bounds, dtype=self.dtype)
            self.int_bounds = self.xp.array(self.int_bounds, dtype=self.dtype)
        
        self.arr_slices = {
            'center': (slice(1, -1), slice(1, -1), slice(1, -1)),
            'r': (slice(2, None), slice(1, -1), slice(1, -1)),
            'l': (slice(None, -2), slice(1, -1), slice(1, -1)),
            'ba': (slice(1, -1), slice(2, None), slice(1, -1)),
            'f': (slice(1, -1), slice(None, -2), slice(1, -1)),
            't': (slice(1, -1), slice(1, -1), slice(2, None)),
            'b': (slice(1, -1), slice(1, -1), slice(None, -2))
        }
    
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
    
    def _choose_array_backend(self, use_gpu):
        """
        Selects NumPy (CPU) or CuPy (GPU) as the array backend.
    
        Parameters
        ----------
        use_gpu : bool
            True to use CuPy with GPU accelaration, False to use numpy.
            
        Raises
        ------
        ImportError
            If `use_gpu=True` but CuPy is not installed.
        RuntimeError
            If `use_gpu=True` but CUDA is unavailable.
        """        
        
        if use_gpu:
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
            - The corresponding conductivity from `phase_conds` is assigned
              for entries in `voxel_struc` matching the index in `phase_conds`
            - Boundary planes are padded:
              * +x and -x faces set to `inf`
              * +y, -y, +z, -z faces set to 0
        """
        
        kappa_arr = self.xp.zeros(self.voxel_struc.shape, dtype=self.dtype)
        # assign conductivities of the pure phases
        for i, cond in enumerate(self.phase_conds):
            kappa_arr[self.voxel_struc == i] = cond
        # add boundaries and assign noundary conducuctivities    
        kappa_arr = self.xp.pad(kappa_arr,
                                pad_width = ((1,1), (1,1), (1,1)),
                                mode = 'constant',
                                constant_values = ((float('inf'), float('inf')), (0,0), (0,0)))
        return kappa_arr
    
    def _get_R_interface_arr(self):
        plate_integ = self.mat_bounds.shape[0]
        adiabatic_integ = plate_integ + 1
        
        boundary_matrix = np.full((plate_integ+2, plate_integ+2), np.inf, dtype=self.dtype)
        boundary_matrix[:plate_integ,:plate_integ] = self.mat_bounds
        boundary_matrix[:plate_integ, plate_integ] = self.int_bounds
        boundary_matrix[plate_integ, :plate_integ] = self.int_bounds
        
        voxel_struc_w_bound_integ = self.xp.pad(self.voxel_struc,
                                pad_width = ((1,1), (1,1), (1,1)),
                                mode = 'constant',
                                constant_values = ((plate_integ,plate_integ),
                                                   (adiabatic_integ,adiabatic_integ),
                                                   (adiabatic_integ,adiabatic_integ)))        
        
        integ_center = voxel_struc_w_bound_integ[self.arr_slices['center']]
        R_ints = {}
        for direction in ['r','l','ba','f','t','b']:
            integ_neighbor = voxel_struc_w_bound_integ[self.arr_slices[direction]]
            interface_arr = boundary_matrix[integ_center, integ_neighbor] 
            R_ints[direction] = interface_arr
        return R_ints
            
    def _get_calc_arr(self):
        if self.mat_bounds is not None and self.int_bounds is not None:
            R_ints = self._get_R_interface_arr()
        
        kappa_arr = self._get_kappa_arr()
        def calc_bond_cond(k_spot, k_neighbor, R_int=None):
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
        for direction in ['r','l','ba','f','t','b']:
            k_neighbor = kappa_arr[self.arr_slices[direction]]
            if self.mat_bounds is not None and self.int_bounds is not None:
                k_arrs[direction] = calc_bond_cond(k_spot,k_neighbor,R_ints[direction])
            else:                
                k_arrs[direction] = calc_bond_cond(k_spot,k_neighbor)
        
        self.ksum_arr = np.zeros(self.voxel_struc.shape)
        for direc in ['r','l','ba','f','t','b']:
            self.ksum_arr += k_arrs[direc]
        self.ksum_arr[self.ksum_arr == 0] = 1e-100 # replace zero with a small number to avoid division with zero
                
        for direc in ['r','l','ba','f','t','b']:
            setattr(self, f'k{direc}_rel_arr', k_arrs[direc]/self.ksum_arr)
        
    def _get_T_arr_linguess(self):
        length, width, height = [dim+2 for dim in self.voxel_struc.shape] # size of voxelstruc + boundaries
        T_arr = self.xp.zeros((length, width, height), dtype=self.dtype)
        T_arr[0] = self.T_l
        T_arr[-1] = self.T_r        
        positions = self.xp.arange(1,length-1) - 1/2
        T_arr[1:-1] = (self.T_l+((self.T_r-self.T_l)/(length-2))*positions)[:, None, None]
        return T_arr
    
    def _get_SOR_redblack_mask(self):
        self.red_mask = (self.xp.indices(self.voxel_struc.shape).sum(axis=0) % 2 == 0)
        self.black_mask = ~(self.red_mask)

    def _get_T_new_Jacobi(self):        
        neighbor_terms = self.xp.zeros((self.voxel_struc.shape))
        for direc in ['r', 'l', 'ba', 'f', 't', 'b']:
            neighbor_terms += self.T_arr[self.arr_slices[direc]] * getattr(self, f'k{direc}_rel_arr')
        T_new = neighbor_terms 
        return T_new          
            
    def _update_SOR_redblack(self, omega): 
        for color_mask in [self.red_mask, self.black_mask]:
            T_old = self.T_arr[self.arr_slices['center']]
            T_new = self._get_T_new_Jacobi()                
            T_change = omega*(T_new - T_old)             
            T_old[color_mask] += T_change[color_mask]      
    
    def _update_SOR(self, omega):
        T_arr = self.T_arr
        ks = []
        for direc in ['r', 'l', 'ba', 'f', 't', 'b']:
            ks.append(getattr(self, f'k{direc}_rel_arr'))
                    
        self.T_arr = SOR_update_numba(T_arr, omega, tuple(ks))
    
    def _get_residuals_new(self):
        ''' calculate the mean residual. '''  
        neighbor_terms = self.xp.zeros((self.voxel_struc.shape))
        for direc in ['r', 'l', 'ba', 'f', 't', 'b']:
            neighbor_terms += self.T_arr[self.arr_slices[direc]] * getattr(self, f'k{direc}_rel_arr')        
        Resid = self.xp.abs(self.T_arr[self.arr_slices['center']] - neighbor_terms) 
        mean_Resid = self.xp.mean(Resid)
        return mean_Resid
    
    def get_kappa_left(self):
        length = self.T_arr.shape[0]
        x_len = (length-2)*self.dx
        Q_arr = -(self.kl_rel_arr[0]*self.ksum_arr[0])*(self.T_arr[1, 1:-1, 1:-1]-self.T_l)/self.dx 
        meanQ = self.xp.mean(Q_arr)
        kappa_sim = -meanQ*(x_len)/(self.T_r-self.T_l)
        return kappa_sim

    def _store_data(self, name, iterations, residuals=None, kappas=None, T_arr=None, save_residual_plot=False, remove_act_files = True, tag='act'):        
        if remove_act_files:
            self._remove_act_files(name) 
        if kappas is not None:
            utils.store_x_and_y_data(np.array(iterations), np.array(kappas), 'iteration','kappa_sim',
                                     name=os.path.join(self.set_dir, f'{name}_iter_{iterations[-1]}_{tag}kappa_{kappas[-1]:.5f}'))
        if residuals is not None:
            utils.store_x_and_y_data(np.array(iterations), np.array(residuals), 'iteration', 'residual',
                                     name=os.path.join(self.set_dir, f'{name}_iter_{iterations[-1]}_{tag}resid_{residuals[-1]:.2e}')) 
            if save_residual_plot:
                utils.makelogplot(np.array(iterations), np.array(residuals), 'iteration', 'residual / K',
                                xscale='lin', yscale='log', save=True, show=False,
                                name=os.path.join(self.set_dir, f'{name}_residuals_{tag}')) 
        if T_arr is not None:
            self.xp.save(os.path.join(self.set_dir, f'{name}_iter_{iterations[-1]}_{tag}Tarr.npy'), self.T_arr)

    def _remove_act_files(self, name):
        for filename in os.listdir(self.set_dir):
            if '_act' in filename and name in filename:
                os.remove(os.path.join(self.set_dir, filename))

        
    def _update_lists(self, iterations, residuals, kappas_sim, iteration, cutoff_resid, print_output=True):
        def to_cpu(val):
            return val.get() if hasattr(val, "get") else val

        iterations.append(to_cpu(iteration))
        kappas_sim.append(to_cpu(self.get_kappa_left()))
        residuals.append(to_cpu(self._get_residuals_new()))           
        if print_output:
            print(f"\riteration: {iterations[-1]} | conductivity: {kappas_sim[-1]:.4f} | "
                  f"residual: {residuals[-1]:.2e} ==> {cutoff_resid:.2e}{10*' '}", end="", flush=True)
        
    def _tune_omega(self, omega, iteration, iterations, residuals, kappas_sim, cutoff_resid, T_start):
        if len(residuals) >= 3 and omega > 1 and residuals[-3] < residuals[-2] < residuals[-1]:
            omega = max(omega - 0.1, 1.0)            
            print(f"\nrestarting with new omega ({omega:.3f}) ...")
            self.T_arr = self.xp.load(os.path.join(self.set_dir, T_start)) if T_start else self._get_T_arr_linguess()
            iteration, iterations, residuals, kappas_sim = 0, [], [], []
            self._update_lists(iterations, residuals, kappas_sim, iteration, cutoff_resid, print_output=True)
        return omega, iteration, iterations, residuals, kappas_sim

    def run(self, name, omega=1.979, max_iter=1000000, cutoff_resid = 1*10**-9, T_start=None, log_iter=50, save_iter=None, tune_omega=False):
        
        print("-"*70,
              f"\n{name = }"
              "\nsetting up the RN simulation ...")
        t_start = time.time() 
        self._get_calc_arr()
        self.T_arr = self.xp.load(os.path.join(self.set_dir, T_start)) if T_start else self._get_T_arr_linguess()
        self._get_SOR_redblack_mask()
        
        iteration = 0
        iterations, residuals, kappas_sim = [], [], []
        print('approximating the steady state ...')         
        # iterating using the SOR method until cutoff criterion or max number of iterations is reached
        while iteration < (max_iter+1):
          
            if log_iter is not None and iteration % log_iter == 0:
                self._update_lists(iterations, residuals, kappas_sim, iteration, cutoff_resid, print_output=True)             
                if tune_omega:
                    omega, iteration, iterations, residuals, kappas_sim = self._tune_omega(omega, iteration, iterations, residuals, kappas_sim, cutoff_resid, T_start) 

            if save_iter is not None and residuals[-1] > cutoff_resid and iteration % save_iter == 0:
                self._store_data(name, kappas=kappas_sim, residuals=residuals, T_arr=self.T_arr, iterations=iterations, tag='act')

            if residuals[-1] < cutoff_resid or iteration == max_iter:
                self._store_data(name, kappas=kappas_sim, residuals=residuals, T_arr=self.T_arr, iterations=iterations, save_residual_plot=True, tag='final')                
                print(f'\nFinished calculation in {utils.format_seconds(time.time()-t_start)} (hh:mm:ss)')
                break 

            iteration += 1 
            #self._update_SOR(omega)
            self._update_SOR_redblack(omega) # updating all temperatures using the SOR method              \n
                    
                    
        
        