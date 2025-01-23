import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ipywidgets as widgets
import ecco_v4_py as ecco
import datetime as dt
from IPython.display import display
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)

from plot import plot_from_widgets
from ecco_compatibility import ecco_dataset, ecco_variable

geometry = ecco_dataset('GEOMETRY_LLC0090GRID')
grid = ecco.get_llc_grid(geometry)


def make_coords_widget(selection: widgets.Widget, coords: widgets.Widget) -> (widgets.Widget, bool):
    """Helper function for plot_select.
    :param selection: A dropdown widget to select whether to display a dimension
    on an axis, or to provide a specific coordinate
    :param coords: A slider widget to select a specific coordinate for a dimension
    :return: An output widget that conditionally displays coords depending on the
    value of selection, as well as the function that actually displays coords
    """
    output = widgets.Output()
    def show_coords(change):
        if change['new'] == 'Choose a value:':
            with output:
                display(coords)
        else:
            output.clear_output()
    selection.observe(show_coords, names='value')
    return output, show_coords


def plot_select(c: xr.DataArray = None, u: xr.DataArray = None, v: xr.DataArray = None):
    """Make a set of widgets to help plot up to three DataArrays on one figure.
    :param c: Plot using color
    :param u: Plot using the horizontal component of a quiver
    :param v: Plot using the vertical component of a quiver
    """
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
    # plt.close() # Close other open plots to avoid having too many plots open at once
    # Merge variables into one dataset in order to perform uniform selection
    data = xr.Dataset({x_name: x for (x_name, x) in {'c': c, 'u': u, 'v': v}.items() if x is not None})
    if len(set(data.dims) - {'tile'}) < 2:
        raise ValueError('Must have at least two dimensions to make a plot')
    if any(len(data[dim]) == 0 for dim in data.dims):
        raise ValueError('Dimension with zero length')
    if {'i_g', 'j_g', 'k_l', 'k_u', 'k_p1', 'time_l'} & set(data.dims):
        grid_dims = {'i', 'i_g', 'j', 'j_g', 'tile'} & set(data.dims)
        if len(grid_dims) < 3 or any(len(data[dim]) < len(geometry[dim]) for dim in grid_dims):
            raise ValueError(
                'In order for plotting to work correctly, you have to interpolate to grid cell centers before selecting along grid dimensions')
    selection_widgets = dict()
    selection_hboxes = []
    if 'tile' in data.dims:
        tile_options = [('Tile ' + str(tile), tile) for tile in data.tile.values]
        # Multi-tile plots only make sense if the data variables have both x- and y-coordinates
        if {'i', 'i_g'} & set(data.dims) and {'j', 'j_g'} & set(data.dims):
            tile_options = [('All tiles (Atlantic)', -1), ('All tiles (Pacific)', -2)] + tile_options
        tile_selection = widgets.Dropdown(description='Plot area:', options=tile_options)
        all_tiles_widgets = dict()
    if {'i', 'i_g'} & set(data.dims):
        i_selection = widgets.Dropdown(
            description='Tile x-coord:',
            options=['Plot on x-axis', 'Plot on y-axis', 'Choose a value:'],
            value='Plot on x-axis',
        )
        i_coords = widgets.IntSlider(min=0, max=89)
        i_output, i_show_coords = make_coords_widget(i_selection, i_coords)
        selection_widgets['i'] = [i_selection, i_coords, i_output, i_show_coords]
        selection_hboxes.append(widgets.HBox([i_selection, i_output]))
    if {'j', 'j_g'} & set(data.dims):
        j_selection = widgets.Dropdown(
            description='Tile y-coord:',
            options=['Plot on x-axis', 'Plot on y-axis', 'Choose a value:'],
            value='Plot on y-axis',
        )
        j_coords = widgets.IntSlider(min=0, max=89)
        j_output, j_show_coords = make_coords_widget(j_selection, j_coords)
        selection_widgets['j'] = [j_selection, j_coords, j_output, j_show_coords]
        selection_hboxes.append(widgets.HBox([j_selection, j_output]))
    if {'k', 'k_l', 'k_u', 'k_p1'} & set(data.dims):
        k_selection = widgets.Dropdown(
            description='Depth:',
            options=['Plot on x-axis', 'Plot on y-axis', 'Choose a value:'],
            value='Choose a value:',
        )
        k_coords = widgets.SelectionSlider(
            options=[(str(int(-k)) + ' m', i) for (i, k) in enumerate(geometry.Z.values)])
        k_proportional = widgets.Checkbox(description='Proportional axis', value=False)
        k_output = widgets.Output()

        def k_show_coords(change):
            if change['new'] == 'Choose a value:':
                k_output.clear_output()
                with k_output:
                    display(k_coords)
            elif change['old'] == 'Choose a value:':
                k_output.clear_output()
                with k_output:
                    display(k_proportional)

        k_selection.observe(k_show_coords, names='value')
        selection_widgets['k'] = [k_selection, k_coords, k_output, k_show_coords]
        selection_hboxes.append(widgets.HBox([k_selection, k_output]))
        if 'tile' in data.dims:
            all_tiles_widgets['k'] = widgets.SelectionSlider(description='Depth:', options=k_coords.options)
    for dim in {'time', 'time_l'}:
        if dim in data.dims:
            t_selection = widgets.Dropdown(
                description='Date:',
                options=['Plot on x-axis', 'Plot on y-axis', 'Choose a value:'],
                value='Choose a value:',
            )
            t_coords = widgets.SelectionSlider(options=data[dim].values)
            t_output, t_show_coords = make_coords_widget(t_selection, t_coords)
            selection_widgets['time'] = [t_selection, t_coords, t_output, t_show_coords]
            selection_hboxes.append(widgets.HBox([t_selection, t_output]))
            if 'tile' in data.dims:
                all_tiles_widgets['time'] = widgets.SelectionSlider(description='Date:', options=t_coords.options)
            break

    selection_output = widgets.Output()

    # 'change' means a change to the tile_selection widget's value (since tile_selection observes this function)
    def set_selection_widgets(change):
        selection_output.clear_output()
        if change['new'] < 0:
            with selection_output:
                display(*all_tiles_widgets.values())
        else:
            with selection_output:
                display(*selection_hboxes)
            for [selection, _, _, show_coords] in selection_widgets.values():
                # make coordinate sliders appear initially
                show_coords({'new': selection.value, 'old': 'Choose a value:'})

    set_selection_widgets({'new': tile_selection.value if 'tile' in data.dims else 0})
    if 'tile' in data.dims:
        tile_selection.observe(set_selection_widgets, names='value')

    plot_button = widgets.Button(description='Plot')
    clear_button = widgets.Button(description='Clear plot')
    plot_status = widgets.Label(value='')
    output = widgets.Output()
    fig = plt.figure()
    fig.set_size_inches(0.01, 0.01)

    def on_plot_button(_):
        plot_status.value = ''
        if 'tile' not in data.dims or tile_selection.value >= 0:
            selection = {dim: coords_widget.value
                         for (dim, [selection_widget, coords_widget, _, _]) in selection_widgets.items()
                         if selection_widget.value == 'Choose a value:'}
            if 'tile' in data.dims:
                selection['tile'] = tile_selection.value

            xaxis = [dim for (dim, [selection_widget, _, _, _]) in selection_widgets.items()
                     if selection_widget.value == 'Plot on x-axis']
            if len(xaxis) != 1:
                plot_status.value = 'One dimension must be selected to plot on the x-axis'
                return
            else:
                xaxis = xaxis[0]
            if xaxis == 'k' and k_proportional.value: xaxis = 'Z'

            yaxis = [dim for (dim, [selection_widget, _, _, _]) in selection_widgets.items()
                     if selection_widget.value == 'Plot on y-axis']
            if len(yaxis) != 1:
                plot_status.value = 'One dimension must be selected to plot on the y-axis'
                return
            else:
                yaxis = yaxis[0]
            if yaxis == 'k' and k_proportional.value: yaxis = 'Z'
        else:
            selection = {dim: widget.value for (dim, widget) in all_tiles_widgets.items()}
            xaxis, yaxis = 'i', 'j'
        output.clear_output()
        with output:
            if 'tile' not in data.dims or tile_selection.value >= 0:
                plot_from_widgets(fig, data, xaxis, yaxis, selection, grid, None)
            elif tile_selection.value == -1:
                plot_from_widgets(fig, data, xaxis, yaxis, selection, grid, 'atlantic')
            elif tile_selection.value == -2:
                plot_from_widgets(fig, data, xaxis, yaxis, selection, grid, 'pacific')

    def on_clear_button(_):
        output.clear_output()
        fig.clf()
        fig.set_size_inches(0.01, 0.01)

    plot_button.on_click(on_plot_button)
    clear_button.on_click(on_clear_button)
    if 'tile' in data.dims:
        display(tile_selection)
    display(selection_output, widgets.HBox([plot_button, clear_button, plot_status]), output)
    plt.show()


def plot_utility():
    """A wrapper around plot_select that picks the input arrays with widgets."""
    color = widgets.Text(description='Color plot:', value='THETA')
    quiver_x = widgets.Text(description='Arrow plot x:', value='UVELMASS')
    quiver_y = widgets.Text(description='Arrow plot y:', value='VVELMASS')
    hbox1 = widgets.HBox([color, quiver_x, quiver_y])
    start = widgets.DatePicker(description='Start date:', value=dt.date(2017, 1, 1))
    end = widgets.DatePicker(description='End date:', value=dt.date(2017, 1, 10))
    timing = widgets.Dropdown(options=['Monthly', 'Daily', 'Snapshot'], value='Daily', description='Timing:')
    hbox2 = widgets.HBox([start, end, timing])
    load_button = widgets.Button(description='Load data')
    clear_button = widgets.Button(description='Clear data')
    load_status = widgets.Label(value='')
    hbox3 = widgets.HBox([load_button, clear_button, load_status])
    output = widgets.Output()

    def on_load_button(_):
        if not (color.value or quiver_x.value or quiver_y.value):
            load_status.value = 'Enter variable names above'
        elif not (start.value and end.value):
            load_status.value = 'Enter start and end dates'
        elif start.value > end.value:
            load_status.value = 'Start date must be before end date'
        elif start.value < np.datetime64('1992-01-01'):
            load_status.value = 'Start date must not be before 1992'
        elif end.value >= np.datetime64('2018-01-01'):
            load_status.value = 'End date must not be after 2017'
        else:
            load_status.value = ''
            c, x, y = None, None, None
            monthly = True
            if color.value:
                try:
                    c = ecco_variable(color.value, start.value, end.value, timing.value)
                except ValueError as e:
                    load_status.value = str(e)
                    return
            if quiver_x.value:
                try:
                    x = ecco_variable(quiver_x.value, start.value, end.value, timing.value)
                except ValueError as e:
                    load_status.value = str(e)
                    return
            if quiver_y.value:
                try:
                    y = ecco_variable(quiver_y.value, start.value, end.value, timing.value)
                except ValueError as e:
                    load_status.value = str(e)
                    return
            output.clear_output()
            with output:
                plot_select(c, x, y)

    def on_clear_button(_):
        output.clear_output()

    load_button.on_click(on_load_button)
    clear_button.on_click(on_clear_button)
    display(hbox1, hbox2, hbox3, output)
