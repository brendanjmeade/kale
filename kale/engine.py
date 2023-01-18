"""Kale Engine."""
from pathlib import Path

import numpy as np
import pyvista as pv
import xarray as xr


class Engine:
    def __init__(self, mesh_filename, data_filename, zscale=0.1):
        if not Path(mesh_filename).exists():
            raise ValueError(f"`{mesh_filename} does not exist.")
        if not Path(data_filename).exists():
            raise ValueError(f"`{data_filename} does not exist.")

        self._mesh = pv.read(mesh_filename)
        self._ds = xr.open_dataset(data_filename, engine="netcdf4")

        self._algorithm = None

        self._modified_callbacks = set()

        # Clear any data arrays in the mesh - only use data from HDF5 file
        self.mesh.clear_data()

        # Scale Z axis on mesh itself to avoid scaled-rendering issues
        self.mesh.points[:, -1] *= zscale

        # Set initial time step and populate mesh
        self.max_time_step = self.ds[self.keys[0]].shape[0] - 1
        self.time_step = 0

    def modified(self):
        for callback in self._modified_callbacks:
            callback()

    def add_modified_callback(self, callback):
        self._modified_callbacks.add(callback)

    def clear_modified_callbacks(self, callback):
        self._modified_callbacks = set()

    @property
    def mesh(self):
        return self._mesh

    @property
    def ds(self):
        return self._ds

    @property
    def keys(self):
        return list(self.ds.keys())

    def get_variable(self, name):
        """Returns variable array for current time step."""
        var = np.array(self.ds[name][self.time_step, :])
        if len(var) != self.mesh.n_cells:
            raise ValueError("Dimensional mismatch between data and mesh")
        return var

    def clim(self, name=None):
        var = np.array(self.ds[name or self.mesh.active_scalars_name or self.keys[0]])
        return np.nanmin(var), np.nanmax(var)

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, value):
        if not isinstance(value, int):
            raise TypeError("Time step must be an integer")
        if value < 0 or value > self.max_time_step:
            raise ValueError("Time step out of time range.")
        self._time_step = value
        for name in self.keys:
            self.mesh[name] = self.get_variable(name)
        self.modified()

    @property
    def algorithm(self):
        if self._algorithm is None:
            from kale.algorithms import EngineAlgorithm

            self._algorithm = EngineAlgorithm(self)
        return self._algorithm
