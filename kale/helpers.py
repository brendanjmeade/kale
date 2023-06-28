import numpy as np
import pyvista as pv

from kale import theme
from kale.algorithms import contour_banded
from kale.assets import load_coastlines


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
    actor_e = pv.Plotter.add_mesh(
        plotter,
        edges,
        color=theme.CONTOUR_LINE_COLOR,
        line_width=theme.CONTOUR_LINE_WIDTH,
    )
    return actor_c, actor_e


def add_bounds(plotter, **kwargs):
    # Requires https://github.com/pyvista/pyvista/pull/3977
    args = dict(
        grid="back",
        location="outer",
        ticks="both",
        n_xlabels=2,
        n_ylabels=2,
        n_zlabels=2,
        xtitle="Longitude",
        ytitle="Latitude",
        ztitle="Elevation",
        fmt="%.2f",
    )
    args.update(kwargs)

    return pv.Plotter.show_bounds(plotter, **args)


def add_coastlines(plotter, **kwargs):
    """Run after adding all other data to the scene."""
    coasts = load_coastlines()
    roi = coasts.clip_box(bounds=plotter.bounds, invert=False)
    kwargs.setdefault("color", theme.COASTLINE_COLOR)
    pv.Plotter.add_mesh(plotter, roi, **kwargs)
