"""Kale module."""
__version__ = "0.0.1"

from kale.algorithms import *
from kale.engine import Engine
from kale.widgets import time_controls, show_ui

# VTK configurations to improve rendering
from pyvista._vtk import vtkMapper

vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)

# PyVista configurations
import pyvista as pv

# After https://github.com/pyvista/pyvista/pull/3870, we can
#   make a custom theme for your visual preferences
pv.set_plot_theme("document")

# Trame Defaults
pv.set_jupyter_backend("server")
pv.global_theme.trame.interactive_ratio = 2
pv.global_theme.trame.still_ratio = 2
