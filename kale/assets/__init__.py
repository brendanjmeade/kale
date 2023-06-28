import pathlib

import numpy as np
import pyvista as pv
import scipy.io as sio


def load_coastlines():
    path = pathlib.Path(__file__).parent / "WorldHiVectors.mat"
    WORLD_BOUNDARIES = sio.loadmat(path)
    n = len(WORLD_BOUNDARIES["lat"])
    z = np.zeros(n)
    points = np.c_[WORLD_BOUNDARIES["lon"], WORLD_BOUNDARIES["lat"], z]
    return pv.lines_from_points(points)
