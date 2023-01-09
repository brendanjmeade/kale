"""Kale Engine."""
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


class Engine:
    def __init__(self, mesh_filename, data_filename, zscale=0.1):
        if not Path(mesh_filename).exists():
            raise ValueError(f"`{mesh_filename} does not exist.")
        if not Path(data_filename).exists():
            raise ValueError(f"`{data_filename} does not exist.")

        self._mesh = pv.read(mesh_filename)
        self._ds = h5py.File(data_filename, "r")

        self._modified_callbacks = set()

        # Clear any data arrays in the mesh - only use data from HDF5 file
        self.mesh.clear_data()

        # Scale Z axis on mesh itself to avoid scaled-rendering issues
        self.mesh.points[:, -1] *= zscale

        # Default to first key
        self._active_variable = self.keys[0]

        self._variable = None
        self._variable_invalidated = True

        # Set initial time step and populate mesh
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

    @property
    def active_variable(self):
        return self._active_variable

    @active_variable.setter
    def active_variable(self, name):
        if name not in self.keys:
            raise ValueError(f"{name} not present in dataset keys: {self.keys}")
        self._active_variable = name
        self._variable_invalidated = True
        self.mesh[self.active_variable] = self.variable[self.time_step, :]
        self.mesh.set_active_scalars(self.active_variable)
        self.modified()

    @property
    def variable(self):
        if self._variable_invalidated:  # caching of sorts
            var = np.array(self.ds[self.active_variable])
            if var.shape[1] != self.mesh.n_cells:
                raise ValueError("Dimensional mismatch between data and mesh")
            self._variable = var
            self._variable_invalidated = False
            self.modified()
        return self._variable

    @property
    def clim(self):
        var = self.variable
        return np.nanmin(var), np.nanmax(var)

    @property
    def time_step(self):
        return self._time_step

    @property
    def max_time_step(self):
        return self.variable.shape[0] - 1

    @time_step.setter
    def time_step(self, value):
        if not isinstance(value, int):
            raise TypeError("Time step must be an integer")
        if value < 0 or value > self.max_time_step:
            raise ValueError("Time step out of time range.")
        self._time_step = value
        self.mesh[self.active_variable] = self.variable[self._time_step, :]
        self.modified()
