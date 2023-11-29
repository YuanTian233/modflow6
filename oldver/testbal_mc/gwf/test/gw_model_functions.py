"""
Module contains classes and functions needed to create files for/ read files from MATLAB groundwater model.
"""

import os
import numpy as np
import scipy.stats as stats
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import math
from copy import deepcopy

matplotlib.use("Qt5Agg")


class GRID:
    """
    Class saves the grid properties for a model setup:

    Parameters:
        grid_size: list[2,1], with number of cells in the [x, y] directions
        grid_distance: list[2, 1], with length of each cell in the [x, y] direction
    Attributes:
        self.grid_size = list[2,1], with number of cells in the [x, y] directions
        self.grid_distance = list[2, 1], with length of each cell in the [x, y] direction
        self.n_cells = int, number of cells in the grid
        self.domain_size = list[2, 1], total length of domain in the [x, y] distance
        self.x_grid = np.array [n_x, n_y] with x coordinates of each cell, where n_x and n_y are the number of cells in
         each direction
        self.y_grid = np.array[n_x, n_y] with y coordinates of each cell, where n_x and n_y are the number of cells in
         each direction
        self.x_vec = np.array[n_cells, 1], with x coordinates of each cell, arranged column-wise (columns from
        self.x_grid were stacked on top of each other, in order)
        self.y_vec = np.array[n_cells, 1], with y coordinates of each cell, arranged column-wise (columns from
        self.x_grid were stacked on top of each other, in order)
    """
    def __init__(self, grid_size, grid_distance):
        self.grid_size = grid_size
        self.grid_distance = grid_distance

        self.grid_points = None
        self.n_cells = None
        self.n_pts = None
        self.domain_size = None
        self.x_grid = None
        self.y_grid = None
        self.x_vec = None
        self.y_vec = None

        self.calculate_attributes()
        self.get_grid()

    def calculate_attributes(self):
        """ Function calculates basic attributes"""
        self.n_cells = int(self.grid_size[0] * self.grid_size[1])
        self.domain_size = np.array(self.grid_size) * np.array(self.grid_distance)

        self.grid_points = np.array([[int(self.grid_size[0]+1), int(self.grid_size[0]+1)]])
        self.n_pts = int((self.grid_size[0]+1) * (self.grid_size[1]+1))

    def get_grid(self):
        """Function generates 2 2D grids with the x and y coordinates of each cell in the grid"""
        # Create grids with x and y locations
        x_loc = np.arange(0, self.domain_size[0], self.grid_distance[0])
        y_loc = np.arange(0, self.domain_size[1], self.grid_distance[1])
        self.x_grid, self.y_grid = np.meshgrid(x_loc, y_loc)

        # Create vector with grid locations (stacks elements column-wise):
        self.x_vec = np.reshape(self.x_grid, (self.n_cells, 1), order="F")
        self.y_vec = np.reshape(self.y_grid, (self.n_cells, 1), order="F")

    def update(self, new_grid_size=None, new_grid_distance=None):
        """
        Function allows to change the grid_size and/or grid_distance to generate a new grid, so it recalculates all
        attributes.
        """
        if new_grid_size is not None:
            self.grid_size = new_grid_size
        if new_grid_distance is not None:
            self.grid_distance = new_grid_distance
        self.calculate_attributes()
        self.get_grid()

    def vec_to_mat(self, vec):
        """
        Transforms a vector to a matrix, using the grid data
        :param vec: <np.array[n_cells, 1]> vector with data to convert to matrix
        :return: <np.array[x_grid, y_grid]> reshaped vector as a matrix
        """
        grid = np.reshape(vec, (int(self.grid_size[0]), int(self.grid_size[1])), order='F')
        return grid


def reshape_vector(vec, size_array):
    """
    Function reshapes a vector into an array with size [y, x]
    Args:
        vec: array [n, 1], with values to turn to a matrix form
        size_array: array [1,2] with number of [rows, columns] to reshape vec into

    Returns: array [rows, columns], with reshaped data

    """
    return np.reshape(vec, (size_array[0, 0], size_array[0, 1]), "F")


def get_loc_vector(grid, meas_loc):
    """
    Function extracts the vector location of a set of (x,y) coordinates (locations) from a matrix

    Args:
        grid: <GRID class instance>
        meas_loc: array [n_obs, 2], with [y(row), x(column)] location of each measurement point

    Returns: arrays [n_obs, 1] with vector location of each observation .
    """
    n_obs = meas_loc.shape[0]  # number of observations = number of location values

    vec = np.arange(0, grid.n_pts).astype(int)
    vec_rs = reshape_vector(vec, grid.grid_points)

    data_at_loc = np.zeros((n_obs, 1))
    # Extract observation at measurement locations
    for i in range(0, n_obs):
        data_at_loc[i, 0] = vec_rs[meas_loc[i, 0], meas_loc[i, 1]]

    data_at_loc = data_at_loc.astype(int)
    return data_at_loc


def extract_modeled_at_loc(vec_loc, vec):
    """
    Function extracts the values from "vec" at the given locations in vec_loc.
    Args:
        vec_loc: array[n_observations, 1], with location of the observation in the data array, in vector form
        vec: array[n_grid_points, n_runs], with values on grid vertices, in vector form

    Returns: array[n_runs, n_obs], with extracted data
    """
    if vec.ndim == 1:
        mc_size = 1
    else:
        mc_size = vec.shape[1]
    output_vals = np.full((mc_size, vec_loc.shape[0]), 0.0)
    for i in range(0, mc_size):
        output_vals[i, :] = np.take(vec[:, i], vec_loc)[:, 0]

    return output_vals


def extract_obs_at_loc(data_path, ind_loc):
    """
    Function opens .mat file with synthetic run data and reads the vectors with hydraulic head (h) and concentration (c)
    data and extracts the values at the observation locations. The vectors with h and c data have a size of
    [n_cells, 1].
    Args:
        data_path: string, with location of .mat file with forward model run data
        ind_loc: np.array[n_obs, 2], array with [y=row, x=column] location of each observation

    Returns: 2 np.arrays [1, n_obs] with hydraulic head and concentration values at observation locations.
    """
    mat_true_run = loadmat(data_path)

    if "h_true" in mat_true_run:
        h_obs = np.take(mat_true_run["h_true"], ind_loc)
        c_obs = np.take(mat_true_run["m0_true"], ind_loc)
    else:
        h_obs = np.take(mat_true_run["h_grid"], ind_loc)
        c_obs = np.take(mat_true_run["c_grid"], ind_loc)
    return h_obs.T, c_obs.T


def add_noise(true_meas, error_vals, ret_error=True):
    """
    Function adds noise to synthetically-generated observation values.
    Args:
        true_meas: np.array[1, n_obs], with observed values
        error_vals: list[2, 1] with [absolute error, relative error]
        ret_error: bool, True to return error value and observations with noise, False to only return observation values
        with noise
    Returns: np.array[1, n_obs] with observations with noise and (opt) np.array[n_obs,] with measurement error
    associated to each observation
    """
    n_obs = true_meas.shape[1]

    if ret_error:
        total_error = true_meas * error_vals[1] + error_vals[0]
    else:
        total_error = error_vals

    rnd_noise = np.random.normal(size=(1, n_obs)) * total_error
    data_with_noise = true_meas + rnd_noise

    for i in range(0, n_obs):
        if data_with_noise[0, i] > 1:
            data_with_noise[0, i] = 1
        elif data_with_noise[0, i] < 0:
            data_with_noise[0, i] = 0
    if ret_error:
        total_error = total_error.flatten()
        return total_error, data_with_noise
    else:
        return data_with_noise


def getAdjacent(arr, i, j):
    # Size of given 2d array
    n = len(arr)-1
    m = len(arr[0])-1

    # Initialising a vector array where
    # adjacent elements will be stored
    v = []

    # Checking for adjacent elements
    # and adding them to array

    # Deviation of row that gets adjusted
    # according to the provided position
    for dx in range(-1 if (i > 0) else 0, 2 if (i < n) else 1):

        # Deviation of the column that
        # gets adjusted according to
        # the provided position
        for dy in range(-1 if (j > 0) else 0, 2 if (j < m) else 1):
            if dx is not 0 or dy is not 0:
                v.append(arr[i + dx][j + dy])

    # Returning the vector array
    return v


def smoothen_zones(zone_mat):
    """
    Function smoothens out the original zoned map so there are no single-valued zones
    :param zone_mat: <np.array[n_cells_x, n_cells_y]> with original zone distribution
    :return: <np.array[n_cells_x, n_cells_y]> with smoothened-out zone distribution
    """
    mod_zones = deepcopy(zone_mat)
    for i in range(0, zone_mat.shape[0]):
        for j in range(0, zone_mat.shape[1]):
            # Get adjacent indexes
            v = getAdjacent(mod_zones, i, j)
            if np.count_nonzero(v == mod_zones[i, j]) < np.count_nonzero(v != mod_zones[i, j]):
                # mod_zones[i, j] = stats.mode(v)[0][0]
                mod_zones[i, j] = stats.mode(v)[0]
    return mod_zones


def generate_zoned_model(true_k_vector, n_zones, grid):
    """
    Function generates the zone distribution based on a true log(K) distribution.

    :param true_k_vector: <np.array[np.cells, 1]>
        with synthetic, true log(K) values
    :param n_zones: <int>
        number of zones
    :param grid: <GRID class instance>

    :return: <np.array[n_cells_x, n_cells_y]>
        with zone distribution (ints from 1 to n_zones)
    """
    # Get max/min values to generate ranges
    y_max = math.ceil(np.max(true_k_vector))
    y_min = math.floor(np.min(true_k_vector))
    # Zone distribution
    step = (y_max-y_min)/n_zones
    zone_range = np.arange(start=y_min, stop=y_max+step, step=step)

    # Get zones based on true log(K) field
    index_vector = np.searchsorted(zone_range, true_k_vector)
    # Convert to grid
    original_zones = grid.vec_to_mat(index_vector)
    # Smoothen-out the original zone distribution
    smoothened_zones = smoothen_zones(original_zones)

    return smoothened_zones


def generate_zoned_k(zones, parameter_values, n_cells):
    """
    Assigns the corresponding log(K) to each zone.

    :param zones: array [n_cells, n_cells]
        array with zone associated to each cell
    :param parameter_values: array [n_mc, n_parameters]
        with parameter sets that need to be transformed into log(K) fields
    :param n_cells: int
        number of cells in the grid
    :return:
    """
    zones_vec = np.reshape(zones, (n_cells, 1), order="F")
    log_k_fields = np.zeros((n_cells, parameter_values.shape[0]))

    for i in range(0, parameter_values.shape[0]):  # for each parameter set
        for j in range(0, parameter_values.shape[1]):
            log_k_fields[np.where(zones_vec == j+1)[0], i] = parameter_values[i, j]

    return log_k_fields


# Plotting:

def plot_synthetic_gw(k_grid, x_gr, y_gr, point_loc=None, save_name=None):
    """
    Plots one RF, along with (optional) location of observation wells, with the corresponding values in each cell in the
    grid
    :param k_grid: <np.array[n_x, n_y] grid with RF values
    :param x_gr: <np.array[x_cells, y_cells]> with x location of each cell in the grid
    :param y_gr: <np.array[x_cells, y_cells]> with y location of each cell in the grid
    :param point_loc: <np.array[n_obs, 2]> with [x (col), y (row)] location of observation wells
    :param save_name: <string> with path and file_name.ext where to save resulting figure. If None, figure is not saved.
    :return: ---
    """
    fig, ax = plt.subplots()

    im = ax.pcolormesh(x_gr, y_gr, k_grid)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log(K)')

    if point_loc is not None:
        ax.scatter(point_loc[:, 1], point_loc[:, 0], s=40, facecolors='k', edgecolors='w')

    ax.set_aspect(1.0 / ax.get_data_ratio() * 1)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show(block=False)

