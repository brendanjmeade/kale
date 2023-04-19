import numpy as np

from kale import theme
from kale.algorithms import contour_banded


def add_contours(plotter, source, scalars, levels, **kwargs):
    n_colors = len(levels) - 1
    clim = [np.min(levels), np.max(levels)]
    contour, edges = contour_banded(
        source,
        levels,
        rng=clim,
        scalars=scalars,
    )
    actor_c = plotter.add_mesh(
        contour,
        clim=clim,
        scalars=scalars,
        n_colors=n_colors,
        **kwargs,
    )
    actor_e = plotter.add_mesh(
        edges, color=theme.CONTOUR_LINE_COLOR, line_width=theme.CONTOUR_LINE_WIDTH
    )
    return actor_c, actor_e
