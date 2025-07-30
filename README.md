# `RNpy`

A python tool for running resistor network simulations based on voxel structures
  
## System requirements

- NumPy 
- Matplotlib 
- Numba
- pyevtk 

## Installation

You can install `RNpy` using conda with the following command:

```bash
conda install rnpy
```

## How to cite?

If you use `RNpy` in your published work, please cite the publication below:

Ketter, L., Greb, N., Bernges, T., Zeier, W. G., Using resistor network models to predict the transport properties of solid-state battery composites. *Nat. Commun.* **16**, 1411 (2025). https://doi.org/10.1038/s41467-025-56514-5

## Example usage

In the first step, we will create a 3D microstructure using the `blobs` function from the `composites` module. The function generates a cubic microstructure, whereby the overall edge length in voxels is controlled by the `size` parameter. Voxel clusters, each containing a number of `sclust`voxels, are inserted into a continuous medium until a volume fraction of `disp_volfrac` (volume fraction in %) is reached. The resulting NumPy array consists of 0s and 1s as entries, where 0 represents the continuous phase and 1 represents the dispersed phase.

**Note:**
1. Inserted clusters overlap.  
2. The function sets the total number of clusters at the start. After all clusters are placed, any remaining difference to the target volume fraction (`disp_volfrac`) is filled with unclustered voxels.
```python
import rnpy.hotplates as hot
import rnpy.composites as comp 

arr = comp.blobs3D(size=20, 
                   disp_volfrac=50, 
                   sclust=10)
comp.compositefigure(arr)
```
Basic figures of small voxel structures can be made using the `compositefigure` function. However, for larger structures, the function becomes significantly slower. For such cases and for advanced visualizations it is recommended to convert the numpy array into a .vti and do visualizations using ParaView.
```python
comp.get_vti(arr, name="my_first_try")
```
Next we will set up and run a resistor network. At first we build a `HotPlate` object using the `hotplates` module. We pass in the 3D microstructure generated earlier and specify phase_conds, a list where each entry corresponds to the conductivity value of the respective phase (i.e., the conductivity at index i applies to phase i in arr). The simulation outputs a few files having the name-tag specified through `name` that are stored in the `save_dir` directory. Finally we run our simulation using the `run` method. The simulation stops, when the specified `cutoff_resid` is reached. 
```python
RN = hot.HotPlate(voxel_struc=arr,
                  phase_conds=[3,5],
                  name="arr1",
                  save_dir="my_first_try") 
RN.run(cutoff_resid=10**-9)
```


