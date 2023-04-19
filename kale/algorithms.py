import collections.abc
import traceback
import weakref

import numpy as np
import pyvista
from pyvista import _vtk
from pyvista.errors import MissingDataError
from pyvista.utilities.algorithms import (
    PreserveTypeAlgorithmBase,
    active_scalars_algorithm,
    algorithm_to_mesh_handler,
    cell_data_to_point_data_algorithm,
    extract_surface_algorithm,
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


class OutputPortAlgorithm(PreserveTypeAlgorithmBase):
    """vtkAlgorithm container for output ports.

    Work around for https://gitlab.kitware.com/vtk/vtk/-/issues/18776
    """

    def __init__(self, source, port):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
        )
        self.SetInputConnection(0, source.GetOutputPort(port))
        self.port = port

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            out = self.GetOutputData(outInfo, 0)
            out.ShallowCopy(self.GetInputData(inInfo, 0, 0))
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            raise e
        return 1


class RenameArrayAlgorithm(PreserveTypeAlgorithmBase):
    """vtkAlgorithm to rename an array."""

    def __init__(self, source_name, new_name):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
        )
        self.source_name = source_name
        self.new_name = new_name

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            inp = self.GetInputData(inInfo, 0, 0)
            out = self.GetOutputData(outInfo, 0)
            out.ShallowCopy(inp)
            for i in range(out.GetPointData().GetNumberOfArrays()):
                array = out.GetPointData().GetAbstractArray(i)
                name = array.GetName()
                if name == self.source_name:
                    array.SetName(self.new_name)
            for i in range(out.GetCellData().GetNumberOfArrays()):
                array = out.GetCellData().GetAbstractArray(i)
                name = array.GetName()
                if name == self.source_name:
                    array.SetName(self.new_name)
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            raise e
        return 1


def contour_banded(
    self,
    contours,
    scalars,
    rng=None,
    component=0,
    clip_tolerance=1e-6,
    # generate_contour_edges=True,
    scalar_mode="value",
    clipping=False,
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
    contours : int or Sequence
        Number of contours or a sequence of contour values to use.

     scalars : str
        The name of the scalar array to use for contouring.  If ``None``,
        the active scalar array will be used.

    rng : Sequence, optional
        Range of the scalars. Optional and defaults to the minimum and
        maximum of the active scalars of ``scalars``.

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

    clipping : bool, default: False
        Indicate whether to clip outside ``rng`` and only return cells with
        values within ``rng``.

    Returns
    -------
    algorithm : vtkBandedPolyDataContourFilter
        Get edges with `alg.GetContourEdgesOutput()`

    """
    if scalars is None:
        raise RuntimeError("Please set scalars")

    self, algo = algorithm_to_mesh_handler(self)

    if not isinstance(self, _vtk.vtkPolyData):
        algo = extract_surface_algorithm(algo or self)
        self, algo = algorithm_to_mesh_handler(algo)

    if algo is None:
        raise TypeError(
            "This version of the filter currently only supports algorithms."
        )

    if scalars not in self.point_data:
        algo = cell_data_to_point_data_algorithm(algo, pass_cell_data=False)
        self, algo = algorithm_to_mesh_handler(algo)
    if scalars not in self.point_data:
        raise ValueError("Scalars not present as POINT data.")
    algo = active_scalars_algorithm(algo, scalars, preference="point")
    self, algo = algorithm_to_mesh_handler(algo)

    if rng is None:
        rng = (self.active_scalars.min(), self.active_scalars.max())

    contour = _vtk.vtkBandedPolyDataContourFilter()
    if isinstance(contours, int):
        # generate values
        contour.GenerateValues(contours, rng)
    elif isinstance(contours, (np.ndarray, collections.abc.Sequence)):
        contour.SetNumberOfContours(len(contours))
        for i, val in enumerate(contours):
            contour.SetValue(i, val)
    else:
        raise TypeError("isosurfaces not understood.")

    set_algorithm_input(contour, algo or self, port=0)
    contour.SetClipping(clipping)
    if scalar_mode == "value":
        contour.SetScalarModeToValue()
    elif scalar_mode == "index":
        contour.SetScalarModeToIndex()
    else:
        raise ValueError(
            f'Invalid scalar mode "{scalar_mode}". Should be either "value" or "index".'
        )
    # contour.SetGenerateContourEdges(generate_contour_edges)
    contour.SetGenerateContourEdges(True)
    contour.SetClipTolerance(clip_tolerance)
    contour.SetComponent(component)

    # Must rename array as VTK sets the active scalars array name to a nullptr.
    # See upstream changes also
    rename = RenameArrayAlgorithm(None, scalars)
    set_algorithm_input(rename, contour, port=0)

    # GetOutputPort(1) are the edges
    return rename, OutputPortAlgorithm(contour, 1)


def subdivide_algorithm(inp, n):
    """Subdivide and smooth the data fields on mesh."""
    sfilter = _vtk.vtkLoopSubdivisionFilter()
    sfilter.SetNumberOfSubdivisions(n)
    set_algorithm_input(sfilter, inp)
    return sfilter


class ActiveScalarsOperationAlgorithm(PreserveTypeAlgorithmBase):
    """vtkAlgorithm to perform a user operation on the active scalars.

    The operation must be a callable that accepts the
    input numpy array of the input's active scalars.

    This assumes the type of the mesh is preserved
    through the operation.

    """

    def __init__(self, operation: callable, output_scalars_name=None):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
        )
        self.operation = operation
        self.output_scalars_name = output_scalars_name

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            inp = pyvista.wrap(self.GetInputData(inInfo, 0, 0))
            out = self.GetOutputData(outInfo, 0)
            result = inp.copy(deep=True)
            if self.output_scalars_name:
                name = self.output_scalars_name
            else:
                name = f"fn({result.active_scalars_name})"
            result[name] = self.operation(result[result.active_scalars_name])
            result.set_active_scalars(name)
            out.ShallowCopy(result)
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            raise e
        return 1


def scalars_operation_algorithm(inp, operation, output_scalars_name=None):
    operator = ActiveScalarsOperationAlgorithm(
        operation=operation, output_scalars_name=output_scalars_name
    )
    set_algorithm_input(operator, inp)
    return operator
