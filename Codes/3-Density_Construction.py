# Numerical and Scientific Computing
import numpy as np
import pandas as pd
import scipy as sp
import math
from numba import jit
from numba import njit
import multiprocessing
import mkl  # Only works if Intel MKL is used



from scipy import special
from scipy.integrate import quad
from scipy import integrate
from scipy import stats
from scipy.interpolate import interp1d

# Plotting and Data Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib as mpl
from matplotlib import ticker, cm
from matplotlib.colors import LinearSegmentedColormap
from tabulate import tabulate

import pyregion

# File Handling
import os
import glob
import csv
import sys

# Astronomical

from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astroML.stats import binned_statistic_2d
import matplotlib.path as mpath


# Other
from tqdm.notebook import tqdm
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon



############################

c0 = 3e5
H_0 = 70
Omega_l = 0.7
Omega_m = 0.3

lim_deltaz = 2

cosmo = FlatLambdaCDM(H0 = H_0, Om0 = Omega_m)

###########################################################################
############################ LSS functions ################################
###########################################################################

def M_lim(z):
    """Fitting function for the mass completeness limit Weaver et. al 2022"""
    return np.log10(-1.51e6 * (1+z) + 6.8e7 * (1+z)**2)

def M_lim_ks(z):
    """Fitting function for mass completeness limit (on K_s) Weaver et al 2022"""
    return np.log10(-3.55e8 * (1+z) + 2.7e8 * (1+z)**2)

def slice_width(z):
    """Calculate the width of each redshift slice of size _physical_width_ (Mpc h^-1)"""
    return physical_width * 100 / c0 * np.sqrt(Omega_m * (1+z)**3 + Omega_l)


def redshift_bins(zmin, zmax):
    """returns the slice centers and widths, given a physical length in (Mpc h^-1) """
    centers = []
    centers.append(zmin + 0.5 * slice_width(zmin))

    i = 0
    while (centers[i] + slice_width(centers[i]) < zmax ):
        centers.append(centers[i] + slice_width(centers[i]))
        i += 1

    centers = np.array(centers)

    "redshift edges"
    edges = np.zeros((len(centers), 2))

    for i in range(0, len(centers)):
        edges[i, 0] = centers[i] - slice_width(centers[i]) / 2
        edges[i, 1] = centers[i] + slice_width(centers[i]) / 2

    return (centers, edges)


def cartesian_from_polar(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def cos_dist(alpha, delta, alpha0, delta0):
    """ gets all angles in [deg]"""
    phi = alpha * np.pi / 180
    theta = np.pi / 2 - delta * np.pi / 180
    phi0 = alpha0 * np.pi / 180
    theta0 = np.pi / 2 - delta0 * np.pi / 180
    
    x = cartesian_from_polar(phi, theta)
    x0 = cartesian_from_polar(phi0, theta0)
    cosdist = np.tensordot(x, x0, axes=[[0], [0]])
    return np.clip(cosdist, 0, 1)

def logsinh(x):
    if np.any(x < 0):
        raise ValueError("logsinh only valid for positive arguments")
    return x + np.log(1-np.exp(-2*x)) - np.log(2)

def Log_K(alpha, delta, alpha0, delta0, kappa):
    norm = -np.log(4 * np.pi / kappa) - logsinh(kappa)
    return norm + cos_dist(alpha, delta, alpha0, delta0) * kappa

@jit(nopython=True)
def σ_k(X0, b, points):
    kappa = 1 / (b * np.pi / 180)**2
    X0_x = points[X0, 0]
    X0_y = points[X0, 1]
    rem = np.delete(points, X0, axis = 0)
    arr = rem[:, 2] * np.exp(Log_K(rem[:, 0], rem[:, 1], X0_x, X0_y, kappa))
    return np.sum(arr)
    
@jit(nopython=True)
def LCV(b, points):
    N = len(points)
    arr1 = [np.log(σ_k(i, b, points)) for i in range(0, len(points))]
    return (1 / N) * np.sum(arr1)

def σ_k_gaussian(X0, b, points):
    X0_x = points[X0, 0]
    X0_y = points[X0, 1]
    rem = np.delete(points, X0, axis = 0)

    Cosdists = cos_dist(rem[:, 0], rem[:, 1], X0_x, X0_y)
    arr = rem[:, 2] * norm.pdf(np.arccos(Cosdists[:]), loc = 0, scale = b * np.pi / 180)
    return np.sum(arr)

def σ(alpha, delta, b_i, points):
    kappa = 1 / (b_i * np.pi / 180)**2
    arr2 = points[:, 2] * np.exp(Log_K(points[:, 0], points[:, 1], alpha, delta, kappa))
    return np.sum(arr2)


def Adaptive_b(b, points):
    g_i = np.array([np.log(points[i, 4] * σ(points[i, 0], points[i, 1], b, points)) for i in range(0, len(points))])
    log_g = 1 / len(points) * np.sum(g_i)
    b_i = np.array([(b * (points[i, 4] * σ(points[i, 0], points[i, 1], b, points) / np.exp(log_g))** -0.5) for i in tqdm(range(0, len(points)))])
    return b_i

@jit(nopython=True)
def divider_NUV(rj):
    return (3*rj+1)

###############################################################################


def setup(work_path='.'):
    '''
    Set up all of the necessary directories
    '''
    for subdir in ('inputs', 'outputs', 'bin', 
                   'outputs/plots', 'outputs/weights', 'outputs/density'):
        path = os.path.join(work_path, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Built directory: {os.path.abspath(path)}')
    
    outputs_dir = os.path.join(work_path, 'outputs')
    plots_dir = os.path.join(work_path, 'outputs', 'plots')
    inputs_dir = os.path.join(work_path, 'inputs')
    weight_dir = os.path.join(work_path, 'outputs', 'weights')
    density_dir = os.path.join(work_path, 'outputs', 'density')
    return outputs_dir, plots_dir, inputs_dir, weight_dir, density_dir


cat_dir = "where you want to set up the catalog directories"

outputs_dir, plots_dir, inputs_dir, weights_dir, density_dir = setup(work_path=cat_dir)



############################################################################



z_min, z_max = 0.4, 9.5

physical_width = 35 # h^-1 Mpc

slice_centers, z_edges = redshift_bins(z_min, z_max)

z_width = z_edges[:, 1] - z_edges[:, 0]

Data = "path to your data file"



threshold = 0.05

# Load the .npy files
weights = np.load(os.path.join(weights_dir, f'weights_unthresholded_normalized_thresh{threshold}_lengh{physical_width}.npy'))
weights_block = np.load(os.path.join(weights_dir, f'weightsBlock_unthresholded_normalized_thresh{threshold}_lengh{physical_width}.npy'))
W = np.load(os.path.join(weights_dir, f'weightsBlock_thresh{threshold}_normalized_lengh{physical_width}.npy'))
normalized_delta_z_median = np.load(os.path.join(weights_dir, f'normalized_delta_z_median_thresh{threshold}_lengh{physical_width}.npy'))
delta_z_median = np.load(os.path.join(weights_dir, f'delta_z_median_thresh{threshold}_lengh{physical_width}.npy'))
count_in_zslice = np.load(os.path.join(weights_dir, f'count_in_zslice_thresh{threshold}_lengh{physical_width}.npy'))

inds_th = [np.where(weights_block[:, i] >= threshold)[0] for i in range(len(slice_centers))]

# Extract RA and Dec values
ra_detect = Data['ra_detec']
dec_detect = Data['dec_detec']

# Find required values
min_RA = np.min(ra_detect)
max_RA = np.max(ra_detect)

# Find Dec values at min_RA and max_RA
dec_at_min_RA = dec_detect[np.argmin(ra_detect)]
dec_at_max_RA = dec_detect[np.argmax(ra_detect)]

# Find RA values at min_Dec and max_Dec
min_Dec = np.min(dec_detect)
max_Dec = np.max(dec_detect)

ra_at_min_Dec = ra_detect[np.argmin(dec_detect)]
ra_at_max_Dec = ra_detect[np.argmax(dec_detect)]

# Define the four key points
quad_corners = np.array([
    [min_RA, dec_at_min_RA],
    [max_RA, dec_at_max_RA],
    [ra_at_min_Dec, min_Dec],
    [ra_at_max_Dec, max_Dec]
])

# Compute centroid of the four points
centroid = np.mean(quad_corners, axis=0)

# Compute angles relative to the centroid
angles = np.arctan2(quad_corners[:,1] - centroid[1], quad_corners[:,0] - centroid[0])

# Sort points in counterclockwise order based on angles
sorted_indices = np.argsort(angles)
sorted_corners = quad_corners[sorted_indices]

# Create the correctly ordered polygon
region_polygon = Polygon(sorted_corners)


# Compute the convex hull (ensures proper ordering)
hull = ConvexHull(quad_corners)

# Get the correctly ordered polygon
polygon_coords = quad_corners[hull.vertices]
region_polygon = Polygon(polygon_coords)

# Compute the area of the rotated region (RA, Dec in degrees)
field_area = region_polygon.area  

# Compute the masked holes area
holes_area = 0.1274 * field_area  #the coefficient is from the HSC masked area in the COSMOS-Web field

# Compute the corrected area
corrected_area = field_area - holes_area

# Compute the bounding box (min area rectangle enclosing the rotated shape)
min_x, min_y, max_x, max_y = region_polygon.bounds
bounding_width = max_x - min_x
bounding_height = max_y - min_y

# Compute aspect ratio
wh_ratio = bounding_width / bounding_height

# circles = np.load('inputs/circles.npy')
circles = "path to your circles file"

bg_density = np.zeros(len(slice_centers))
COL_pts = 10
edge = 0.11

density_Table_col = 135

den_Table = np.zeros((len(Data), density_Table_col))



################################################################################

# Vectorized function for computing the integral
@jit(nopython=True)
def gaussian_kernel(delta, alpha, i, b, pts):
    return np.exp(-((delta - pts[i, 1])**2 + (alpha - pts[i, 0])**2) / (2 * b**2))

def compute_integral(i, x_min, x_max, y_min, y_max, b, pts):
    return integrate.dblquad(
        gaussian_kernel, x_min, x_max, 
        lambda alpha: y_min, lambda alpha: y_max,
        args=(i, b, pts)
    )[0]

# Boundary detection function
@jit(nopython=True)
def identify_boundaries(pts, x_min, x_max, y_min, y_max, edge):
    return np.nonzero(
        (pts[:, 0] < (x_min + edge)) | ((x_max - edge) < pts[:, 0]) |
        (pts[:, 1] < (y_min + edge)) | ((y_max - edge) < pts[:, 1])
    )[0]

# Density field computation
def density_field(grid_x, grid_y, bi, pts):
    density = np.zeros_like(grid_x)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            density[i, j] = σ(grid_x[i, j], grid_y[i, j], bi, pts)
    return density


# Compute sigma values for points
def compute_sigma_values(pts, bg_density_slice, d):
    for i in range(len(d)):
        pts[i, 7] = σ(pts[i, 0], pts[i, 1], pts[i, 3], pts) * pts[i, 4]
    pts[0:len(d), 8] = (pts[0:len(d), 7] / bg_density_slice) - 1
    pts[0:len(d), 9] = pts[0:len(d), 2] * pts[0:len(d), 7]

# Function to mark grid points with circles
@jit(nopython=True)
def mark_grid_with_circles(grid, circles_array):
    # circles_array shape: (N_circles, 3) where columns are center_x, center_y, and radius.
    for i in range(circles_array.shape[0]):
        cx = circles_array[i, 0]
        cy = circles_array[i, 1]
        r = circles_array[i, 2]
        r2 = r * r
        for j in range(grid.shape[0]):
            dx = grid[j, 0] - cx
            dy = grid[j, 1] - cy
            if dx * dx + dy * dy <= r2:
                grid[j, 2] = 1

def rotate_coordinates(ra, dec, theta):
    x = np.cos(theta) * ra + np.sin(theta) * dec
    y = -np.sin(theta) * ra + np.cos(theta) * dec
    return x, y


################################################################################

# Rotation matrix for the field
tan = (1.7259 - 1.9644) / (149.66 - 150.31) 
theta = -np.arctan(tan)
x = np.cos(theta) * Data['ra_detec'] + np.sin(theta) * Data['dec_detec']
y = -np.sin(theta) * Data['ra_detec'] + np.cos(theta) * Data['dec_detec']

xx = np.cos(theta) * Data['ra_detec'] + np.sin(theta) * Data['dec_detec']
yy = -np.sin(theta) * Data['ra_detec'] + np.cos(theta) * Data['dec_detec']
x_min, x_max, y_min, y_max = np.min(xx), np.max(xx), np.min(yy), np.max(yy)

x = np.cos(theta) * Data['ra_detec'] + np.sin(theta) * Data['dec_detec']
y = -np.sin(theta) * Data['ra_detec'] + np.cos(theta) * Data['dec_detec']
circles_array = circles[['RA (deg)', 'Dec (deg)', 'Radius (deg)']].to_numpy()
circles_array[:, 0], circles_array[:, 1] = rotate_coordinates(circles_array[:, 0], circles_array[:, 1], theta)

B = "load your bandwidth array here"



def process_slice(s):
    global Data, W, slice_centers, wh_ratio, corrected_area, field_area, circles_array  # Ensure necessary global variables are accessible

    sliceNo = s
    ind_slice = inds_th[sliceNo]
    d = Data.iloc[ind_slice]
    w = W[ind_slice, sliceNo]

    x = np.cos(theta) * d['ra_detec'] + np.sin(theta) * d['dec_detec']
    y = -np.sin(theta) * d['ra_detec'] + np.cos(theta) * d['dec_detec']


    print(f"Slice {s+1}/{len(slice_centers)} processed ({(s+1)/len(slice_centers)*100:.1f}%).")

    n_dens = len(d) / corrected_area

    # Determine grid size based on density
    N_y = int(np.sqrt(n_dens * field_area / wh_ratio))
    N_x = int(wh_ratio * N_y)

    # Create the grid with extra columns:
    x_fill = np.linspace(x_min, x_max, N_x)
    y_fill = np.linspace(y_min, y_max, N_y)
    x_fill, y_fill = np.meshgrid(x_fill, y_fill)
    num_of_points = N_x * N_y

    # Initialize the grid with an extra column for marking
    grid = np.zeros((num_of_points, COL_pts))
    grid[:, 0] = x_fill.ravel()
    grid[:, 1] = y_fill.ravel()

    # Use the jitted function to mark grid points inside any circle
    mark_grid_with_circles(grid, circles_array)

    # Fill holes with average weight
    filled_holes = grid[grid[:, 2] == 1]
    avg_w = np.full(len(filled_holes), (np.sum(w) / len(d)))

    # Combine data points and filled holes
    p = np.zeros((len(d), COL_pts))
    p[:, 0] = x    
    p[:, 1] = y           
    p[:, 2] = w
    p[:, 5] = ind_slice

    fp = np.zeros((len(filled_holes), COL_pts))
    fp[:, 0] = filled_holes[:, 0]   
    fp[:, 1] = filled_holes[:, 1]   
    fp[:, 2] = avg_w[:]             

    pts = np.concatenate((p, fp))
    pts[:, 4] = 1

    b = B[sliceNo]

    # Identify boundary points
    Boundaries_ind = identify_boundaries(pts, x_min, x_max, y_min, y_max, edge)


    if len(Boundaries_ind) > 0:
        integrals = np.array([compute_integral(i, x_min, x_max, y_min, y_max, b, pts) for i in Boundaries_ind])
        pts[Boundaries_ind, 4] = (2 * np.pi * b**2) / integrals

    bi = Adaptive_b(b, pts)
    pts[:, 3] = bi

    # Create mesh grid for density computation
    mesh_y = 120
    mesh_x = int(wh_ratio * mesh_y)
    gr_x, gr_y = np.linspace(x_min, x_max, mesh_x), np.linspace(y_min, y_max, mesh_y)
    grid_x, grid_y = np.meshgrid(gr_x, gr_y)

    # Compute density field
    density = density_field(grid_x, grid_y, bi, pts)


    bg_density[sliceNo] = np.median(density)

    # Compute density contrast and density excess
    density_contrast = density / bg_density[sliceNo] - 1
    density_excess = density / bg_density[sliceNo]

    pts[0:len(d), 6] = Data.iloc[ind_slice]['id'].to_numpy()


    compute_sigma_values(pts, bg_density[sliceNo], d)

    # Define the output file name for the slice
    output_filename = f"output_slice_{sliceNo}_z={slice_centers[sliceNo]:.3f}_b={b}.npz"
    output_filename = os.path.join(density_dir, output_filename)

    # Save the results to a .npz file
    np.savez(
        output_filename,
        pts=pts,
        x = x,
        y = y,
        w = w,
        density=density,
        density_contrast=density_contrast,
        density_excess=density_excess,
        bg_density=bg_density[sliceNo],
        adaptive_b=bi,
        boundary_correction=pts[:, 4],
    )

    print(f"Saved: {output_filename}")
    
    return s  # Return the slice number for tracking progress



# Multiprocessing to process slices in parallel
# this multiprocessing part may not work on some systems like Apple silicon Macs. You can set num_workers = 1 to run it in single-core mode.
if __name__ == "__main__":
    num_workers = min(multiprocessing.cpu_count(), len(slice_centers)) 
    num_workers = min(8, num_workers)

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_slice, range(len(slice_centers)))

    print(f"Completed processing for slices: {results}")