from typing import Callable

import xarray as xr
import ipywidgets as wg
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap

axis_options = ['Plot on x-axis', 'Plot on y-axis', 'Choose a value:']
all_tiles_options = [('All tiles (Atlantic)', -3), ('All tiles (Pacific)', -2),
                     ('All tiles (Arctic)', -1)]

# The colormap used to show land/ocean
land_mask = LinearSegmentedColormap.from_list('land_mask', ['#e0f0a0', '#ffffff'])

def make_adjusters() -> dict[str, wg.ValueWidget]:
    return {
        'title': wg.Text(description='Plot title:'),
        'cmap': wg.Dropdown(description='Color map:', options=[
            ('viridis', 'viridis'), ('inferno', 'inferno'), ('cividis', 'cividis'),
            ('gray', 'binary'), ('gray (inverted)', 'gray'), ('pale', 'pink'),
            ('heat', 'gist_heat'), ('red-blue', 'RdBu_r'), ('seismic', 'seismic'),
            ('spectral', 'Spectral'), ('land mask', land_mask)
        ]),
        'clabel': wg.Text(description='Color units:'),
        'uvlabel': wg.Text(description='Arrow units:'),
        'uvcolor': wg.Dropdown(description='Arrow color:', options=[('Black', 'k'), ('White', 'w')], value='k'),
    }


class Stage:
    """A widget used to split the display of other widgets into stages. A
    proceed button unlocks the next stage of widgets, and a correpsonding clear
    button removes the next stage from view."""

    def __init__(self, proceeed_msg: str, clear_msg: str, on_proceed: Callable[[], str]):
        """
        :param proceeed_msg: The description for the proceed button.
        :param clear_msg: The description for the clear button.
        :param on_proceed: Run when the proceed button is clicked; may return
            a status message to display to the user..
        """
        self.on_proceed = on_proceed
        self.proceed_button = wg.Button(description=proceeed_msg)
        self.proceed_button.on_click(self._proceed)
        self.clear_button = wg.Button(description=clear_msg)
        self.clear_button.on_click(self._clear)
        self.status = wg.Label()
        self.next_stage = wg.Output()

    def display(self):
        display(wg.HBox([self.proceed_button, self.clear_button, self.status]))
        display(self.next_stage)

    def _proceed(self, _):
        """Wrapper around self.on_proceed that sets the status."""
        self.next_stage.clear_output()
        with self.next_stage:
            self.status.value = self.on_proceed() or ''

    def _clear(self, _):
        """Clears the status and the next stage."""
        self.status.value = ''
        self.next_stage.clear_output()


class CoordinateSelector:
    """A widget used to specify how a dimension is represented on a plot."""

    def __init__(self, description: str, coords: xr.DataArray,
                 display_coords: xr.DataArray = None):
        """
        :param description: The dimension name to be displayed on the widget.
        :param coords: The underlying coordinate values used to select along
            the dimension.
        :param display_coords: Optional coordinate names displayed on the
            selection slider, if not the same as the underlying values.
        """
        self.non_uniform = (display_coords is not None)
        if self.non_uniform:
            if coords.ndim != 1 or display_coords.ndim != 1:
                raise ValueError('Coordinates must be 1D arrays')
            if coords.shape != display_coords.shape:
                raise ValueError('Coordinates and display coordinates must have equal lengths')
            self.display_dim = display_coords.name
            self.proportional = wg.Checkbox(description='Proportional axis')
            slider_options = zip(display_coords.values, coords.values)
        else:
            slider_options = coords.values
        self.dim = coords.name
        self.description = description
        self.axis = wg.Dropdown(description=description, options=axis_options)
        self.axis.observe(lambda change: self._axis_update(change['new']),
                          names='value')
        self.slider = wg.SelectionSlider(options=list(slider_options))
        self.dynamic_widget = wg.Output()

    def display(self):
        self.slider.description = ' '
        self._axis_update(self.axis.value)
        display(wg.HBox([self.axis, self.dynamic_widget]))

    def display_slider(self):
        self.slider.description = self.description
        display(self.slider)

    def _axis_update(self, axis: str):
        """
        Modifies which sub-widgets are displayed in self.dynamic_widget. Called
        by the axis selector when its value is updated.
        :param axis: The value of the axis selector.
        """
        if axis == 'Choose a value:':
            self.dynamic_widget.clear_output()
            with self.dynamic_widget:
                display(self.slider)
        elif axis in {'Plot on x-axis', 'Plot on y-axis'}:
            self.dynamic_widget.clear_output()
            if self.non_uniform:
                with self.dynamic_widget:
                    display(self.proportional)


class EccoSelector:
    """A widget consisting of one coordinate selector for each dimension in an
    ECCO dataset."""

    def __init__(self, data: xr.Dataset, geometry: xr.Dataset):
        """
        :param data: The ECCO dataset.
        :param geometry: Geometric metadata for the dataset (mostly superfluous)
        """
        self.coord_selectors: list[CoordinateSelector] = list()
        self.dynamic_widget = wg.Output()
        if {'i', 'i_g'} & set(data.dims):
            i_selector = CoordinateSelector('Tile x-coord:', geometry.i)
            i_selector.axis.value = 'Plot on x-axis'
            self.coord_selectors.append(i_selector)
        if {'j', 'j_g'} & set(data.dims):
            j_selector = CoordinateSelector('Tile y-coord:', geometry.j)
            j_selector.axis.value = 'Plot on y-axis'
            self.coord_selectors.append(j_selector)
        if {'k', 'k_l', 'k_u', 'k_p1'} & set(data.dims):
            depth = (-geometry.Z).astype(int).astype(str) # + ' m'
            k_selector = CoordinateSelector('Depth:', geometry.k, depth)
            k_selector.axis.value = 'Choose a value:'
            self.coord_selectors.append(k_selector)
        if 'time' in data.dims:
            time_selector = CoordinateSelector('Date:', data.time)
            time_selector.axis.value = 'Choose a value:'
            self.coord_selectors.append(time_selector)
        if 'tile' in data.dims:
            self.has_tiles = True
            tile_options = [('Tile ' + str(tile), tile) for tile in data.tile.values]
            if {'i', 'i_g'} & set(data.dims) and {'j', 'j_g'} & set(data.dims):
                tile_options = all_tiles_options + tile_options
            self.tile_selector = wg.Dropdown(
                description='Plot area:', options=tile_options)
            self.tile_selector.observe(
                lambda change: self._tile_selector_update(change['new']),
                names='value')
            self._tile_selector_update(self.tile_selector.value)
        else:
            self.has_tiles = False
            self._tile_selector_update(0) # Display coordinate selectors in full

    def display(self):
        if self.has_tiles:
            display(self.tile_selector)
        display(self.dynamic_widget)

    def _tile_selector_update(self, tile: int):
        """
        Determines how to display the coordinate selectors.
        :param tile: The value of the tile selector.
        """
        self.dynamic_widget.clear_output()
        if tile < 0: # Display all tiles
            with self.dynamic_widget:
                for selector in self.coord_selectors:
                    selector.display_slider()
        else: # Display one tile
            with self.dynamic_widget:
                for selector in self.coord_selectors:
                    selector.display()


