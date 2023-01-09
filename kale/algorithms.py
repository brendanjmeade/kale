import traceback

import pyvista
from pyvista import _vtk
from pyvista.errors import MissingDataError
from pyvista.utilities.algorithms import (
    active_scalars_algorithm,
    algorithm_to_mesh_handler,
    set_algorithm_input,
)

from kale.engine import Engine


class EngineAlgorithm(_vtk.VTKPythonAlgorithmBase):
    """vtkAlgorithm container for Engine."""

    def __init__(self, engine: Engine):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=0,
            nOutputPorts=1,
            outputType="vtkUnstructuredGrid",
        )
        self.engine = engine
        self.engine.add_modified_callback(self.Modified)

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            out = self.GetOutputData(outInfo, 0)
            out.ShallowCopy(self.engine.mesh)
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            raise e
        return 1


def contour_banded(
    self,
    n_contours,
    rng=None,
    scalars=None,
    component=0,
    clip_tolerance=1e-6,
    generate_contour_edges=True,
    scalar_mode="value",
    clipping=True,
):
    """Generate filled contours.

    Generates filled contours for vtkPolyData. Filled contours are
    bands of cells that all have the same cell scalar value, and can
    therefore be colored the same. The method is also referred to as
    filled contour generation.

    This filter implements `vtkBandedPolyDataContourFilter
    <https://vtk.org/doc/nightly/html/classvtkBandedPolyDataContourFilter.html>`_.

    Parameters
    ----------
    n_contours : int
        Number of contours.

    rng : Sequence, optional
        Range of the scalars. Optional and defaults to the minimum and
        maximum of the active scalars of ``scalars``.

    scalars : str, optional
        The name of the scalar array to use for contouring.  If ``None``,
        the active scalar array will be used.

    component : int, default: 0
        The component to use of an input scalars array with more than one
        component.

    clip_tolerance : float, optional
        Set/Get the clip tolerance.  Warning: setting this too large will
        certainly cause numerical issues. Change from the default value at
        your own risk. The actual internal clip tolerance is computed by
        multiplying ``clip_tolerance`` by the scalar range.

    generate_contour_edges : bool, default: True
        Controls whether contour edges are generated.  Contour edges are
        the edges between bands. If enabled, they are generated from
        polygons/triangle strips and returned as a second output.

    scalar_mode : str, default: 'value'
        Control whether the cell scalars are output as an integer index or
        a scalar value.  If ``'index'``, the index refers to the bands
        produced by the clipping range. If ``'value'``, then a scalar value
        which is a value between clip values is used.

    clipping : bool, default: True
        Indicate whether to clip outside ``rng`` and only return cells with
        values within ``rng``.

    Returns
    -------
    algorithm : vtkBandedPolyDataContourFilter
        Get edges with `alg.GetContourEdgesOutput()`

    """
    self, algo = algorithm_to_mesh_handler(self)

    if not isinstance(self, _vtk.vtkPolyData):
        surf_filter = _vtk.vtkDataSetSurfaceFilter()
        # surf_filter.SetPassThroughPointIds(True)
        # surf_filter.SetPassThroughCellIds(True)
        set_algorithm_input(surf_filter, algo or self, port=0)
        self, algo = algorithm_to_mesh_handler(surf_filter)

    # check active scalars
    if scalars is not None:
        if algo is not None:
            algo = active_scalars_algorithm(algo, scalars, preference="point")
            self, algo = algorithm_to_mesh_handler(algo)
        else:
            self.point_data.active_scalars_name = scalars
    else:
        if algo is not None:
            raise RuntimeError
        pyvista.set_default_active_scalars(self)
        if self.point_data.active_scalars_name is None:
            raise MissingDataError("No point scalars to contour.")

    if rng is None:
        rng = (self.active_scalars.min(), self.active_scalars.max())

    contour = _vtk.vtkBandedPolyDataContourFilter()
    contour.GenerateValues(n_contours, rng[0], rng[1])
    set_algorithm_input(contour, algo or self, port=0)
    # contour.SetInputConnection(algo.GetOutputPort())
    contour.SetClipping(clipping)
    if scalar_mode == "value":
        contour.SetScalarModeToValue()
    elif scalar_mode == "index":
        contour.SetScalarModeToValue()
    else:
        raise ValueError(
            f'Invalid scalar mode "{scalar_mode}". Should be either "value" or "index".'
        )
    contour.SetGenerateContourEdges(generate_contour_edges)
    contour.SetClipTolerance(clip_tolerance)
    contour.SetComponent(component)

    # Must rename array as VTK sets the active scalars array name to a nullptr.
    # if mesh.point_data and mesh.point_data.GetAbstractArray(0).GetName() is None:
    #     mesh.point_data.GetAbstractArray(0).SetName(self.point_data.active_scalars_name)
    # if mesh.cell_data and mesh.cell_data.GetAbstractArray(0).GetName() is None:
    #     mesh.cell_data.GetAbstractArray(0).SetName(self.cell_data.active_scalars_name)

    return contour
