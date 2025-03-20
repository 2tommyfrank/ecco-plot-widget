import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import xgcm
import ipywidgets as wg
from IPython.display import display
from math import ceil

from .widget import land_mask

# Used to select i, j, i_g, and j_g for quiver plots to space out data
skip = range(2, 88, 5)

# subplots[i] is the index of tile #i in the array of subplots
subplots = {
    'atlantic': [(3, 2), (2, 2), (1, 2), (3, 3), (2, 3), (1, 3), (0, 2),
                 (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)],
    'pacific': [(3, 0), (2, 0), (1, 0), (3, 1), (2, 1), (1, 1), (0, 2),
                (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)],
    'arctic': [(2, 2), (3, 0), (1, 0), (2, 3), (3, 1), (2, 1), (1, 1),
                (1, 2), (1, 3), (3, 3), (0, 1), (0, 3), (3, 2)],
}

# rotations[i] is the orientation of tile #i, as a multiple of 90 degrees
rotations = {
    'atlantic': [0, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1],
    'pacific': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'arctic': [0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3],
}

# Used to label axes
dimension_descriptions = {'i': 'Tile x-coordinate', 'j': 'Tile y-coordinate', 'k': 'Tile z-coordinate',
                          'Z': 'Depth (m)', 'tile': 'Plot area', 'time': 'Date'}


def cos90(angle: int) -> int:
    """The cosine of 90 degrees times the input."""
    if angle % 4 == 0: return 1
    elif angle % 4 == 2: return -1
    else: return 0


def sin90(angle: int) -> int:
    """The sine of 90 degrees times the input."""
    if angle % 4 == 1: return 1
    elif angle % 4 == 3: return -1
    else: return 0


def interpolate(array: xr.DataArray, dim: str, grid: xgcm.Grid) -> xr.DataArray:
    """Interpolate a DataArray along a single spatial dimension."""
    if dim in {'i', 'i_g', 'XC', 'XG'}:
        array_interp = grid.interp(array.load(), 'X', keep_coords=True)
    elif dim in {'j', 'j_g', 'YC', 'YG'}:
        array_interp = grid.interp(array.load(), 'Y', keep_coords=True)
    elif dim in {'k', 'k_u', 'k_l', 'k_p1', 'Z', 'Zp1', 'Zu', 'Zl'}:
        array_interp = grid.interp(array.load(), 'Z', boundary='fill', fill_value=0, keep_coords=True)
    else:
        raise ValueError('Cannot interpolate along ' + str(dim))
    if 'time' in array.coords:
        array_interp = array_interp.assign_coords(time=array.time)
    return array_interp


def interpolate_2d(u: xr.DataArray, v: xr.DataArray, grid: xgcm.Grid) -> (xr.DataArray, xr.DataArray):
    """Interpolate two DataArrays on the u- and v- grid to the tracer grid."""
    if {'i', 'j_g'} & set(u.dims):
        raise ValueError('The first input to interpolate_2d must be on the u-grid')
    if {'i_g', 'j'} & set(v.dims):
        raise ValueError('The second input to interpolate_2d must be on the v-grid')
    uv_interp = grid.interp_2d_vector({'X': u.load(), 'Y': v.load()}, boundary='extend')
    u_interp, v_interp = uv_interp['X'], uv_interp['Y']
    if 'time' in u.coords:
        u_interp = u_interp.assign_coords(time=u.time)
    if 'time' in v.coords:
        v_interp = v_interp.assign_coords(time=v.time)
    return u_interp, v_interp


def select_interpolate(data: xr.Dataset, selection: dict[str, float],
                       grid: xgcm.Grid) -> xr.Dataset:
    """Selects data along dimensions, interpolating where necessary."""
    # Select time/depth if possible before interpolating
    for dim in {'time', 'k'}:
        if dim in selection and dim in data.dims:
            data = data.sel({dim: selection[dim]})
    # Interpolate variables to tracer grid cells
    variables = dict(data.astype(float).data_vars)
    for (name, var) in variables.items():
        for dim in {'i_g', 'j_g', 'k_u', 'k_l', 'k_p1'}:
            if dim in var.dims:
                # Don't need to interpolate_2d because quivers are not displayed at tile borders
                variables[name] = interpolate(var, dim, grid)
    data = xr.Dataset(variables)
    # Second pass selection after interpolation changes dimensions
    for (dim, val) in selection.items():
        if dim in data.dims:
            data = data.sel({dim: val})
    return data


def infer_colormap(data: xr.DataArray) -> (mpl.colors.Colormap, float, float):
    """Infers a matplotlib colormap and bounds that best display a DataArray."""
    cmin = np.nanpercentile(data, 10)
    cmax = np.nanpercentile(data, 90)
    if cmin < 0 < cmax:
        cmax = np.nanpercentile(np.abs(data), 90)
        cmin = -cmax
        cmap = 'RdBu_r'
    else:
        cmap = 'viridis'
    return cmap, cmin, cmax


def plot_from_widgets(figure: plt.Figure, data: xr.Dataset, x: str, y: str,
                      selection: dict[str, float], grid: xgcm.Grid,
                      adjusters: dict[str, wg.ValueWidget], ocean_focus: str = None):
    """
    Plots DataArrays on one figure using configuration provided by widgets.
    :param figure: The matplotlib figure to plot on.
    :param data: A DataArray dictionary with keys 'c' (color); optional 'u' and
        'v' (quiver).
    :param x: The dimension for the x-axis.
    :param y: The dimension for the y-axis.
    :param selection: Coordinate values for the data along certain dimensions.
    :param grid: Metadata used to interpolate the data in case of mismatched
        coordinates.
    :param adjusters: A dictionary of widgets to be used for adjusting the plot.
    :param ocean_focus: Either 'atlantic', 'pacific', 'arctic', or None.
    """
    data = select_interpolate(data, selection, grid)
    if 'Z' in (x, y): data['Z'] = -data['Z']

    # Determine which adjustment widgets should be displayed
    plot_color = (data.c.dtype.kind == 'f')
    if 'long_name' in data.c.attrs:
        if 'vertical open fraction' in data.c.attrs['long_name']:
            plot_color = True
    plot_quiver = ({'u', 'v'} <= data.data_vars.keys())
    displayed_adjusters = [adjusters['title']]
    if plot_color:
        displayed_adjusters.append(adjusters['clabel'])
    if plot_quiver:
        displayed_adjusters.append(adjusters['uvlabel'])
    if plot_color:
        displayed_adjusters.append(adjusters['cmap'])
    if plot_color and plot_quiver:
        displayed_adjusters.append(adjusters['uvcolor'])
    display(wg.HBox(displayed_adjusters))

    figure.clf()
    if 'tile' in data.dims:
        plot = MultiTilePlot(figure, adjusters, data.tile, ocean_focus)
    else:
        plot = SingleTilePlot(figure, adjusters)
    plot.draw_mesh(data.c, x, y, plot_color)
    if plot_color:
        plot.make_colorbar()
    if plot_quiver:
        plot.draw_quiver(data.u, data.v, x, y)
    plt.ion()
    plt.show()


class Plot:
    """An abstract superclass for SingleTilePlot and MultiTilePlot."""

    def __init__(self, figure: plt.Figure, adjusters: dict[str, wg.ValueWidget]):
        self.figure: plt.Figure = figure
        self.adjusters: dict[str, wg.ValueWidget] = adjusters
        self.colorbar: plt.Colorbar | None = None
        self.cmin: float = 0
        self.cmax: float = 1
        self.quiverkey: plt.QuiverKey | None = None
        self.quiver_x: xr.DataArray | None = None
        self.quiver_y: xr.DataArray | None = None
        self.uvmax: float = 0

    def draw_mesh(self, data: xr.DataArray, x: str, y: str, infer_colors: bool):
        """
        Makes a color (mesh) plot.
        :param data: The DataArray to plot using color.
        :param x: The name of the dimension to map to the x-axis.
        :param y: The name of the dimension to map to the y-axis.
        :param infer_colors: Whether to infer the colormap based on data values
            or use a default colormap (land mask).
        """
        if infer_colors:
            self.adjusters['cmap'].value, self.cmin, self.cmax = infer_colormap(data)
        else:
            self.adjusters['cmap'].value = land_mask

    def make_colorbar(self):
        """Makes a colorbar; requires having called self.draw_mesh first."""
        pass

    def draw_quiver(self, u: xr.DataArray, v: xr.DataArray, x: str, y: str):
        """
        Makes a quiver plot.
        :param u: The DataArray to plot as the x-coordinate of the quiver.
        :param v: The DataArray to plot as the y-coordinate of the quiver.
        :param x: The name of the dimension to map to the x-axis.
        :param y: The name of the dimension to map to the y-axis.
        """
        x_skip, y_skip = ceil(len(u[x]) / 20), ceil(len(v[y]) / 20)
        self.quiver_x = u[x][(x_skip // 2)::x_skip]
        self.quiver_y = v[y][(y_skip // 2)::y_skip]
        self.uvmax = max(np.nanpercentile(np.abs(u), 90), np.nanpercentile(np.abs(v), 90))

    def make_quiverkey(self, uvlabel: str):
        """Makes a quiver key; requiers having called self.draw_quiver first."""
        pass


class SingleTilePlot(Plot):
    """A customizable QuadMesh and Quiver on one figure, representing a Dataset
    with no tile dimension.
    """

    def __init__(self, figure: plt.Figure, adjusters: dict[str, wg.ValueWidget]):
        """
        :param figure: An empty figure used for this plot.
        :param adjusters: Widgets used to adjust this plot.
        """
        super().__init__(figure, adjusters)
        self.figure.set_size_inches(5, 5)
        self.ax: plt.Axes = figure.subplots()
        self.mesh: plt.QuadMesh | None = None
        self.quiver: plt.Quiver | None = None
        self.transpose: bool = False
        self.adjusters['title'].observe(
            lambda change: self.ax.set_title(change['new']), names='value')
        self.ax.set_title(self.adjusters['title'].value)

    def draw_mesh(self, data: xr.DataArray, x: str, y: str, infer_colors: bool):
        super().draw_mesh(data, x, y, infer_colors)
        self.ax.set_xlabel(dimension_descriptions[x])
        self.ax.set_ylabel(dimension_descriptions[y])
        self.transpose = (x != data.dims[1] and y != data.dims[0])
        if (y in {'k', 'Z'}) or (self.transpose and y == 'i'):
            self.ax.yaxis.set_inverted(True)
        values = data.values
        if self.transpose: values = values.T
        self.mesh = self.ax.pcolormesh(
            data[x], data[y], values, cmap=self.adjusters['cmap'].value,
            vmin=self.cmin, vmax=self.cmax)
        if x == 'time':
            self.ax.set_xticks(self.ax.get_xticks()[::3])

    def make_colorbar(self):
        self.figure.set_size_inches(6.5, 5)
        self.colorbar = self.figure.colorbar(self.mesh)
        self.adjusters['clabel'].observe(
            lambda change: self.colorbar.set_label(change['new']), names='value')
        self.adjusters['cmap'].observe(
            lambda change: self.mesh.set_cmap(change['new']), names='value')
        self.colorbar.set_label(self.adjusters['clabel'].value)

    def draw_quiver(self, u: xr.DataArray, v: xr.DataArray, x: str, y: str):
        super().draw_quiver(u, v, x, y)
        if (y in {'k', 'Z'}) or (self.transpose and y == 'i'):
            v = -v
        u = u.where(u[x].isin(self.quiver_x), drop=True)
        u = u.where(u[y].isin(self.quiver_y), drop=True)
        v = v.where(u[x].isin(self.quiver_x), drop=True)
        v = v.where(v[y].isin(self.quiver_y), drop=True)
        u_values, v_values = u.values, v.values
        if self.transpose:
            u_values, v_values = u_values.T, v_values.T
        self.quiver = self.ax.quiver(self.quiver_x, self.quiver_y, u_values, v_values,
                                     scale=20 * self.uvmax, width=0.006)
        if self.colorbar is not None:
            self.adjusters['uvcolor'].observe(
                lambda change: self.quiver.set_color(change['new']), names='value')
            self.quiver.set_color(self.adjusters['uvcolor'].value)
        self.adjusters['uvlabel'].observe(
            lambda change: self.make_quiverkey(change['new']), names='value')
        self.make_quiverkey(self.adjusters['uvlabel'].value)

    def make_quiverkey(self, uvlabel: str):
        if self.quiverkey is not None:
            self.quiverkey.remove()
        label = f'{2 * self.uvmax:.5g}'
        if len(uvlabel) > 0:
            label += ' ' + uvlabel
        self.quiverkey = self.ax.quiverkey(self.quiver, 0.95, 1.05,
                                           2 * self.uvmax, label)


class MultiTilePlot(Plot):
    """Customizable QuadMeshes and Quivers on one figure with 4x4 subplots,
    representing a Dataset with a tile dimension.
    """

    def __init__(self, figure: plt.Figure, adjusters: dict[str, wg.ValueWidget],
                 tiles: xr.DataArray, ocean_focus: str):
        """
        :param figure: An empty figure used for this plot.
        :param adjusters: Widgets used to adjust this plot.
        :param tiles: The tile numbers used by the dataset for this plot.
        :param ocean_focus: Either 'atlantic', 'pacific', or 'arctic'.
        """
        super().__init__(figure, adjusters)
        self.figure.set_size_inches(10, 10.1)
        axes = figure.subplots(4, 4)
        figure.subplots_adjust(wspace=0, hspace=0)
        for ax in axes.ravel():
            ax.axis('off')
        self.tiles: np.ndarray = tiles.values
        self.axes: list[plt.Axes] = [axes[row][col] for (row, col) in subplots[ocean_focus]]
        self.meshes: list[plt.QuadMesh | None] = [None] * 13
        self.quivers: list[plt.Quiver | None] = [None] * 13
        self.tile_rotations: list[int] = rotations[ocean_focus]
        if ocean_focus in {'atlantic', 'pacific'}:
            self.quiverkey_tile = 6
        elif ocean_focus == 'arctic':
            self.quiverkey_tile = 11
        self.adjusters['title'].observe(
            lambda change: self.figure.suptitle(change['new'], x=0.435, y=0.92),
            names='value')
        self.figure.suptitle(self.adjusters['title'].value, x=0.435, y=0.92)

    def draw_mesh(self, data: xr.DataArray, x: str, y: str, infer_colors: bool):
        super().draw_mesh(data, x, y, infer_colors)
        for tile in self.tiles:
            self.axes[tile].axis('on')
            self.axes[tile].set_aspect('equal')
            self.axes[tile].get_xaxis().set_visible(False)
            self.axes[tile].get_yaxis().set_visible(False)
            rotated = np.rot90(data.sel(tile=tile).load(), self.tile_rotations[tile])
            self.meshes[tile] = self.axes[tile].pcolormesh(
                data[x], data[y], rotated, cmap=self.adjusters['cmap'].value,
                vmin=self.cmin, vmax=self.cmax)

    def make_colorbar(self):
        self.figure.set_size_inches(12.5, 10.1)
        self.colorbar = self.figure.colorbar(self.meshes[self.tiles[0]], ax=self.axes)
        self.adjusters['clabel'].observe(
            lambda change: self.colorbar.set_label(change['new']), names='value')
        self.adjusters['cmap'].observe(
            lambda change: self._set_mesh_cmap(change['new']), names='value')
        self.colorbar.set_label(self.adjusters['clabel'].value)

    def draw_quiver(self, u: xr.DataArray, v: xr.DataArray, x: str, y: str):
        super().draw_quiver(u, v, x, y)
        for tile in self.tiles:
            u_rotated = np.rot90(u.sel({'tile': tile, x: self.quiver_x, y: self.quiver_y}),
                                 self.tile_rotations[tile])
            v_rotated = np.rot90(v.sel({'tile': tile, x: self.quiver_x, y: self.quiver_y}),
                                 self.tile_rotations[tile])
            u_adjusted = (u_rotated * cos90(self.tile_rotations[tile])
                          + v_rotated * sin90(self.tile_rotations[tile]))
            v_adjusted = (v_rotated * cos90(self.tile_rotations[tile])
                          - u_rotated * sin90(self.tile_rotations[tile]))
            self.quivers[tile] = self.axes[tile].quiver(
                self.quiver_x, self.quiver_y, u_adjusted, v_adjusted,
                scale=20 * self.uvmax, width=0.006, clip_on=False)
        if self.colorbar is not None:
            self.adjusters['uvcolor'].observe(
                lambda change: self._set_quiver_color(change['new']), names='value')
            for quiver in self.quivers:
                quiver.set_color(self.adjusters['uvcolor'].value)
        self.adjusters['uvlabel'].observe(
            lambda change: self.make_quiverkey(change['new']), names='value')
        self.make_quiverkey(self.adjusters['uvlabel'].value)

    def make_quiverkey(self, uvlabel: str):
        if self.quiverkey is not None:
            self.quiverkey.remove()
        label = f'{2 * self.uvmax:.5g}'
        if len(uvlabel) > 0:
            label += ' ' + uvlabel
        self.quiverkey = self.axes[self.quiverkey_tile].quiverkey(
            self.quivers[self.quiverkey_tile], 0.95, 1.05, 2 * self.uvmax, label)

    def _set_mesh_cmap(self, cmap: str):
        for mesh in self.meshes:
            mesh.set_cmap(cmap)

    def _set_quiver_color(self, uvcolor: str):
        for quiver in self.quivers:
            quiver.set_color(uvcolor)
