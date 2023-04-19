"""Requires https://github.com/pyvista/pyvista/pull/3870"""
import pyvista as pv
from pyvista._vtk import vtkMapper
from pyvista.themes import DocumentTheme

# VTK configurations to improve rendering
vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)

CONTOUR_LINE_WIDTH = 1
CONTOUR_LINE_COLOR = "black"
COLOR_MAPS = {
    "cumulative_slip": "CET_L19",  #'CET_C3',
    "geometric_moment": "CET_R3",
    "last_event_slip": "CET_L19",
    "loading_rate": "CET_C3",
}

COLORBAR_FONT_SIZE = 34
AXES_FONT_SIZE = 26
TIMESTEP_FONT_SIZE = 12

SCALAR_BAR_OPTS = dict(
    # Labels
    title_font_size=AXES_FONT_SIZE,  # 10
    label_font_size=AXES_FONT_SIZE,  # 10
    n_labels=3,
    italic=False,
    fmt="%.1f",
    font_family="arial",
    shadow=True,  # False
)
SCALAR_BAR_V = dict(vertical=True, **SCALAR_BAR_OPTS)
SCALAR_BAR_H = dict(vertical=False, **SCALAR_BAR_OPTS)


class KaleTheme(DocumentTheme):
    def __init__(self):
        """Initialize the theme."""
        super().__init__()
        self.background = "white"
        self.jupyter_backend = "server"

        # Use different colors as you add data to the scene
        # self.color_cycler = "default"  # Uses Matplotlibs default color cycler
        self.color = CONTOUR_LINE_COLOR

        self.cmap = "CET_L19"

        self.font.family = "arial"
        self.font.size = TIMESTEP_FONT_SIZE

        self.edge_color = CONTOUR_LINE_COLOR
        self.line_width = CONTOUR_LINE_WIDTH

        self.trame.interactive_ratio = 2
        self.trame.still_ratio = 2

        self.image_scale = 2  # upscales the saved screenshots/video frames

        # Default hide scalar bar - must explicitly enable it
        self.show_scalar_bar = False

        # Default orientation
        self.colorbar_orientation = "vertical"

        # Parameters for vertical
        self.colorbar_vertical.height = 0.20  # .50
        self.colorbar_vertical.width = 0.05  # .10
        self.colorbar_vertical.position_x = 0.05
        self.colorbar_vertical.position_y = 0.40

        # Parameters for horizontal
        self.colorbar_horizontal.height = 0.03
        self.colorbar_horizontal.width = 0.50
        self.colorbar_horizontal.position_x = 0.25
        self.colorbar_horizontal.position_y = 0.275

        # TODO: antialiassing


pv.set_plot_theme(KaleTheme())
