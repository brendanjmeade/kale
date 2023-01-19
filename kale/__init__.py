"""Kale module."""
__version__ = "0.0.1"

from kale.algorithms import *
from kale.engine import Engine
from kale.widgets import time_controls

# VTK configurations to improve rendering
from pyvista._vtk import vtkMapper

vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)
