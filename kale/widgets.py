import ipywidgets as widgets
import pyvista as pv

from kale.engine import Engine


def time_controls(engine: Engine, plotter: pv.BasePlotter):
    def update_time_step(time_step):
        engine.time_step = time_step
        plotter.render()

    tmax = engine.max_time_step

    def set_time(change):
        step = change["new"]
        if step < 0:
            step = 0
        if step >= tmax:
            step = tmax - 1
        update_time_step(step)

    play = widgets.Play(
        value=engine.time_step,
        min=0,
        max=tmax,
        step=100,
        description="Time Step",
    )
    play.observe(set_time, "value")

    slider = widgets.IntSlider(min=0, max=tmax, continuous_update=True)
    widgets.jslink((play, "value"), (slider, "value"))
    return widgets.HBox([play, slider])


def show_ui(engine: Engine, plotter: pv.BasePlotter):
    iframe = plotter.show()
    controls = time_controls(engine, plotter)
    return widgets.VBox([iframe, controls])
