r'''
SNOW: Sub-Network of an Oversegmented Watershed
Copyright (C) 2017 Jeff Gostick

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
# this file is a modified version of the original SNOW implementation
# by Jeff Gostick
#
# Modified by Lukas Ketter in 2026 to allow periodic boundary conditions in the snow labeling algorithm.

import numpy as np
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import footprint_rectangle
import warnings
from .._label_cc import _relabel

def _get_snow_labels(arr, phase_id, periodic=True, pad_width=50, sigma=0.35, r_max=5):
    def _get_inner_arr(arr, pad_width):
        return arr[
            pad_width:-pad_width,
            pad_width:-pad_width,
            pad_width:-pad_width
        ]
    warnings.warn("" \
        "You are using a modified version of the original SNOW implementation. " \
        "If you use this function in your work, cite the original publication (https://doi.org/10.1103/PhysRevE.96.023307)" \
    )
    im = (arr==phase_id).astype(bool)
    if periodic:
        im = np.pad(im, pad_width=pad_width, mode='wrap')
    dt = spim.distance_transform_edt(input=im)
    dt = spim.gaussian_filter(input=dt, sigma=sigma)
    coords = peak_local_max(image=dt, min_distance=r_max-1, exclude_border=0)
    peaks = np.zeros_like(dt, dtype=bool)
    peaks[tuple(coords.T)] = True
    peaks = _trim_saddle_points(peaks=peaks, dt=dt)
    peaks = _trim_nearby_peaks(peaks=peaks, dt=dt)
    markers = spim.label(peaks)[0]
    if periodic:
        markers_center = _get_inner_arr(markers, pad_width=pad_width)
        markers = np.pad(markers_center, pad_width=pad_width, mode='wrap')
    regions = watershed(image=-dt, markers=markers, mask=im)
    if periodic:
        regions = _relabel(_get_inner_arr(regions, pad_width=pad_width))
    return regions

def _trim_nearby_peaks(peaks, dt):
    peaks, N = spim.label(peaks, structure=footprint_rectangle((3,)*dt.ndim))
    crds = spim.center_of_mass(
        peaks,
        labels=peaks,
        index=np.arange(1, N+1)
    )
    crds = np.vstack(crds).astype(int)  # Convert to numpy array of ints
    # Get distance between each peak as a distance map
    tree = sptl.cKDTree(data=crds)
    temp = tree.query(x=crds, k=2)
    nearest_neighbor = temp[1][:, 1]
    dist_to_neighbor = temp[0][:, 1]
    del temp, tree  # Free-up memory
    dist_to_solid = dt[tuple(crds.T)]  # Get distance to solid for each peak
    hits = np.where(dist_to_neighbor < dist_to_solid)[0]
    # Drop peak that is closer to the solid than its neighbor
    drop_peaks = []
    for peak in hits:
        if dist_to_solid[peak] < dist_to_solid[nearest_neighbor[peak]]:
            drop_peaks.append(peak)
        else:
            drop_peaks.append(nearest_neighbor[peak])
    drop_peaks = np.unique(drop_peaks)
    # Remove peaks from image
    slices = spim.find_objects(input=peaks)
    for s in drop_peaks:
        peaks[slices[s]] = 0
    return (peaks > 0)

def _extend_slice(s, shape, pad=1):
    a = []
    for i, dim in zip(s, shape):
        start = 0
        stop = dim
        if i.start - pad >= 0:
            start = i.start - pad
        if i.stop + pad < dim:
            stop = i.stop + pad
        a.append(slice(start, stop, None))
    return tuple(a)

def _trim_saddle_points(peaks, dt, max_iters=10):
    labels, N = spim.label(peaks)
    slices = spim.find_objects(labels)
    for i in range(N):
        s = _extend_slice(s=slices[i], shape=peaks.shape, pad=10)
        peaks_i = labels[s] == i + 1
        dt_i = dt[s]
        im_i = dt_i > 0
        iters = 0
        peaks_dil = peaks_i.copy()
        while iters < max_iters:
            iters += 1
            peaks_dil = spim.binary_dilation(
                input=peaks_dil,
                structure=footprint_rectangle((3,)*dt.ndim)
            )
            peaks_max = peaks_dil*np.max(dt_i*peaks_dil)
            peaks_extended = (peaks_max == dt_i)*im_i
            if np.all(peaks_extended == peaks_i):
                break  # Found a true peak
            elif np.sum(peaks_extended*peaks_i) == 0:
                peaks_i = False
                break  # Found a saddle point
            peaks[s] = peaks_i
    return peaks

