import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
import ipywidgets as wg
import ecco_v4_py as ecco
from IPython.display import display
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)

from widget import EccoSelector, Stage, make_adjusters
from plot import plot_from_widgets
from ecco_compatibility import ecco_dataset, ecco_variable

geometry = ecco_dataset('GEOMETRY_LLC0090GRID')
grid = ecco.get_llc_grid(geometry)


def plot_utility():
    """A wrapper around plot_select that picks the input arrays with widgets."""
    color = wg.Text(description='Color plot:', value='THETA')
    quiver_x = wg.Text(description='Arrow plot x:', value='UVELMASS')
    quiver_y = wg.Text(description='Arrow plot y:', value='VVELMASS')
    hbox1 = wg.HBox([color, quiver_x, quiver_y])
    start = wg.DatePicker(description='Start date:', value=dt.date(2017, 1, 1))
    end = wg.DatePicker(description='End date:', value=dt.date(2017, 1, 10))
    timing = wg.Dropdown(options=['Monthly', 'Daily', 'Snapshot'], value='Daily', description='Timing:')
    hbox2 = wg.HBox([start, end, timing])
    plt.ioff()
    figure = plt.figure()
    select_stage = Stage('Load data', 'Clear data',
                         lambda: on_load_button(color.value, quiver_x.value, quiver_y.value,
                                                start.value, end.value, timing.value, figure))
    display(hbox1, hbox2)
    select_stage.display()


def on_load_button(color: str, quiver_x: str, quiver_y: str,
                   start: dt.date, end: dt.date, timing: str, figure: plt.Figure) -> str:
    """
    Queries ECCO datasets and use them to call plot_select.
    :return: A status message.
    """
    if not (color or quiver_x or quiver_y):
        return 'Enter variable names above'
    if not (start and end):
        return 'Enter start and end dates'
    if start > end:
        return 'Start date must be before end date'
    if start < np.datetime64('1992-01-01'):
        return 'Start date must not be before 1992'
    if end >= np.datetime64('2018-01-01'):
        return 'End date must not be after 2017'
    c, x, y = None, None, None
    if color:
        try:
            c = ecco_variable(color, start, end, timing)
        except ValueError as e:
            return str(e)
    if quiver_x:
        try:
            x = ecco_variable(quiver_x, start, end, timing)
        except ValueError as e:
            return str(e)
    if quiver_y:
        try:
            y = ecco_variable(quiver_y, start, end, timing)
        except ValueError as e:
            return str(e)
    plot_select(c, x, y, figure)


def plot_select(c: xr.DataArray = None, u: xr.DataArray = None,
                v: xr.DataArray = None, figure: plt.Figure = None):
    """Makes a set of widgets to help plot up to three DataArrays on one figure.
    :param c: Plot using color
    :param u: Plot using the horizontal component of a quiver
    :param v: Plot using the vertical component of a quiver
    :param figure: If not provided, a new figure is created
    """
    plt.ioff()
    if figure is None:
        figure = plt.figure()
    # If there is no color plot, plot land vs. ocean instead
    if c is None:
        c = geometry.hFacC
        if (u is None or 'k' not in u.dims) and (v is None or 'k' not in v.dims):
            c = c.sel(k=0)
    # If one of the arrow components isn't used, make it zero
    if u is not None and v is None:
        v = xr.DataArray(0, coords=u.coords, dims=u.dims)
    if v is not None and u is None:
        u = xr.DataArray(0, coords=v.coords, dims=v.dims)
        print(u)
    # Merge variables into one dataset in order to perform uniform selection
    data = xr.Dataset({x_name: x for (x_name, x) in {'c': c, 'u': u, 'v': v}.items() if x is not None})
    if len(set(data.dims) - {'tile'}) < 2:
        raise ValueError('Must have at least two dimensions to make a plot')
    if any(len(data[dim]) == 0 for dim in data.dims):
        raise ValueError('Dimension with zero length')
    if {'i_g', 'j_g', 'k_l', 'k_u', 'k_p1'} & set(data.dims):
        grid_dims = {'i', 'i_g', 'j', 'j_g', 'tile'} & set(data.dims)
        if len(grid_dims) < 3 or any(len(data[dim]) < len(geometry[dim]) for dim in grid_dims):
            raise ValueError('In order for plotting to work correctly, you have to interpolate to grid cell centers before selecting along grid dimensions')
    selectors = EccoSelector(data, geometry)
    adjusters = make_adjusters()
    plot_stage = Stage('Plot', 'Clear plot',
                       lambda: on_plot_button(data, selectors, figure, adjusters))
    selectors.display()
    plot_stage.display()


def on_plot_button(data: xr.Dataset, selectors: EccoSelector, figure: plt.Figure,
                   adjusters: dict[str, wg.ValueWidget]) -> str:
    """
    Collects parameters from widget values to call plot_from_widgets.
    :return: A status message.
    """
    for adjuster in adjusters.values():
        adjuster.unobserve_all('value')
    selection = dict()
    if 'tile' not in data.dims or selectors.tile_selector.value >= 0:
        for selector in selectors.coord_selectors:
            if selector.axis.value == 'Choose a value:':
                selection[selector.dim] = selector.slider.value
        if 'tile' in data.dims:
            selection['tile'] = selectors.tile_selector.value

        xaxis_selectors = [selector for selector in selectors.coord_selectors
                           if selector.axis.value == 'Plot on x-axis']
        if len(xaxis_selectors) != 1:
            return 'One dimension must be selected to plot on the x-axis'
        if xaxis_selectors[0].non_uniform and xaxis_selectors[0].proportional.value:
            xaxis = xaxis_selectors[0].display_dim
        else:
            xaxis = xaxis_selectors[0].dim

        yaxis_selectors = [selector for selector in selectors.coord_selectors
                           if selector.axis.value == 'Plot on y-axis']
        if len(yaxis_selectors) != 1:
            return 'One dimension must be selected to plot on the y-axis'
        if yaxis_selectors[0].non_uniform and yaxis_selectors[0].proportional.value:
            yaxis = yaxis_selectors[0].display_dim
        else:
            yaxis = yaxis_selectors[0].dim
    else:
        for selector in selectors.coord_selectors:
            if selector.dim not in {'i', 'j'}:
                selection[selector.dim] = selector.slider.value
        xaxis, yaxis = 'i', 'j'
    if 'tile' not in data.dims or selectors.tile_selector.value >= 0:
        plot_from_widgets(figure, data, xaxis, yaxis, selection, grid, adjusters, None)
    elif selectors.tile_selector.value == -3:
        plot_from_widgets(figure, data, xaxis, yaxis, selection, grid, adjusters, 'atlantic')
    elif selectors.tile_selector.value == -2:
        plot_from_widgets(figure, data, xaxis, yaxis, selection, grid, adjusters, 'pacific')
    elif selectors.tile_selector.value == -1:
        plot_from_widgets(figure, data, xaxis, yaxis, selection, grid, adjusters, 'arctic')
