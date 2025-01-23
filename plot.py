import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import xgcm
import ipywidgets as widgets
from IPython.display import display

# The colormap used to show land/ocean
land_mask = mpl.colors.LinearSegmentedColormap.from_list('land_mask', ['#e0f0a0', '#ffffff'])

# Used to select i, j, i_g, and j_g for quiver plots to space out data
skip = range(2, 88, 5)

# subplots[i] is the index of tile #i in the array of subplots
subplots = {
    'pacific': [(3, 0), (2, 0), (1, 0), (3, 1), (2, 1), (1, 1), (0, 2),
                (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)],
    'atlantic': [(3, 2), (2, 2), (1, 2), (3, 3), (2, 3), (1, 3), (0, 2),
                 (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)],
}

# rotations[i] is the orientation of tile #i, as a multiple of 90 degrees
rotations = {
    'pacific': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'atlantic': [0, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1],
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


def infer_colormap(data: xr.DataArray) -> (mpl.colors.Colormap, float, float):
    """Infer a matplotlib colormap and bounds that best display a DataArray."""
    cmin = np.nanpercentile(data, 10)
    cmax = np.nanpercentile(data, 90)
    if cmin < 0 < cmax:
        cmax = np.nanpercentile(np.abs(data), 90)
        cmin = -cmax
        cmap = 'RdBu_r'
    else:
        cmap = 'viridis'
    return cmap, cmin, cmax


def plot_from_widgets(fig: plt.Figure, data: xr.Dataset, x: str, y: str, selection: dict[str, float],
                      grid: xgcm.Grid, ocean_focus: str | None = None):
    """Plot DataArrays on one figure using configuration provided by widgets.
    :param fig: The matplotlib figure to plot on
    :param data: A DataArray dictionary with keys 'c' (color); optional 'u' and 'v' (quiver)
    :param x: The dimension for the x-axis
    :param y: The dimension for the y-axis
    :param selection: Coordinate values for the data along certain dimensions
    :param grid: Metadata used to interpolate the data in case of mismatched coordinates
    :param ocean_focus: Either 'pacific', 'atlantic', or None
    """

    plot_quiver = ({'u', 'v'} <= data.data_vars.keys())

    # Make the adjustment widgets for the plot
    title = widgets.Text(description='Plot title:')
    cmap = widgets.Dropdown(description='Color map:', options=[
        ('viridis', 'viridis'), ('inferno', 'inferno'), ('cividis', 'cividis'), ('gray', 'binary'),
        ('gray (inverted)', 'gray'),
        ('pale', 'pink'), ('heat', 'gist_heat'), ('red-blue', 'RdBu_r'), ('seismic', 'seismic'),
        ('spectral', 'Spectral'),
        ('land mask', land_mask)
    ])
    clabel = widgets.Text(description='Color units:')
    uvlabel = widgets.Text(description='Arrow units:')
    acolor = widgets.Dropdown(description='Arrow color:', options=[('Black', 'k'), ('White', 'w')], value='k')

    # Determine which adjustment widgets should be displayed
    ckind = data.c.dtype.kind
    if 'long_name' in data.c.attrs and 'vertical open fraction' in data.c.attrs['long_name']:
        ckind = 'b'
    adjust_widgets = [title]
    if ckind == 'f':
        adjust_widgets.append(clabel)
    if plot_quiver:
        adjust_widgets.append(uvlabel)
    if ckind == 'f':
        adjust_widgets.append(cmap)
        if plot_quiver:
            adjust_widgets.append(acolor)
    display(widgets.HBox(adjust_widgets))

    # Select time/depth if possible before interpolating
    for dim in {'time', 'k'}:
        if dim in selection and dim in data.dims:
            data = data.sel({dim: selection[dim]})
    # Interpolate variables to tracer grid cells
    variables = dict(data.astype(float).data_vars)
    for (name, var) in variables.items():
        for dim in {'i_g', 'j_g', 'k_u', 'k_l', 'k_p1'}:
            if dim in var.dims:
                variables[name] = interpolate(var, dim, grid)
    data = xr.Dataset(variables)
    # Second pass selection after interpolation changes dimensions
    for (dim, val) in selection.items():
        if dim in data.dims:
            data = data.sel({dim: val})
    cmap.value, cmin, cmax = land_mask, 0, 1
    if ckind == 'f':
        cmap.value, cmin, cmax = infer_colormap(data['c'])
    if 'Z' in (x, y): data['Z'] = -data['Z']

    fig.clf()
    if 'tile' in data.dims:
        axes = fig.subplots(4, 4)
        if ckind == 'f':
            fig.set_size_inches(12.5, 10.1)
        elif ckind == 'b':
            fig.set_size_inches(10, 10.1)
        fig.subplots_adjust(wspace=0, hspace=0)
        for ax in axes.ravel():
            ax.axis('off')
        axes = [axes[row][col] for (row, col) in subplots[ocean_focus]]
        title.observe(lambda change: fig.suptitle(change['new'], x=0.435, y=0.92), names='value')
        meshes = []
        for tile, ax in enumerate(axes):
            if tile not in data.tile: continue
            ax.axis('on')
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            c_rotated = np.rot90(data.c.sel(tile=tile).load(), rotations[ocean_focus][tile])
            meshes.append(ax.pcolormesh(data[x], data[y], c_rotated, cmap=cmap.value, vmin=cmin, vmax=cmax))
        if ckind == 'f':
            cbar = fig.colorbar(meshes[0], ax=axes)
            clabel.observe(lambda change: cbar.set_label(change['new']), names='value')
            cmap.observe(lambda change: [mesh.set_cmap(change['new']) for mesh in meshes], names='value')

    else:
        ax = fig.subplots()
        if ckind == 'f':
            fig.set_size_inches(6.5, 5)
        elif ckind == 'b':
            fig.set_size_inches(5, 5)
        ax.set_xlabel(dimension_descriptions[x])
        ax.set_ylabel(dimension_descriptions[y])
        title.observe(lambda change: ax.set_title(change['new']), names='value')
        transpose = (x != data.c.dims[1] and y != data.c.dims[0])
        if (y in {'k', 'Z'}) or (transpose and y == 'i'):
            ax.yaxis.set_inverted(True)
            if 'v' in data.data_vars:
                data['v'] = -data['v']
        mesh_c = data.c.values
        if transpose: mesh_c = mesh_c.T
        mesh = ax.pcolormesh(data[x], data[y], mesh_c, cmap=cmap.value, vmin=cmin, vmax=cmax)
        if ckind == 'f':
            cbar = fig.colorbar(mesh)
            clabel.observe(lambda change: cbar.set_label(change['new']), names='value')
            cmap.observe(lambda change: mesh.set_cmap(change['new']), names='value')
        if x == 'time':
            ax.set_xticks(ax.get_xticks()[::3])

    if plot_quiver:
        from math import ceil
        x_skip, y_skip = ceil(len(data[x]) / 20), ceil(len(data[y]) / 20)
        quiver_x, quiver_y = data[x][(x_skip // 2)::x_skip], data[y][(y_skip // 2)::y_skip]
        uvmax = max(np.nanpercentile(np.abs(data.u), 90), np.nanpercentile(np.abs(data.v), 90))

        if 'tile' in data.dims:
            quivers = []
            for tile, ax in enumerate(axes):
                if tile not in data.tile: continue
                # Rotate head of each vector around the tile to the correct orientation
                u_rotated = np.rot90(data.u.sel({'tile': tile, x: quiver_x, y: quiver_y}), rotations[ocean_focus][tile])
                v_rotated = np.rot90(data.v.sel({'tile': tile, x: quiver_x, y: quiver_y}), rotations[ocean_focus][tile])
                # Rotate tail of each vector around the head by the same amount
                u_adjusted = u_rotated * cos90(rotations[ocean_focus][tile]) + v_rotated * sin90(
                    rotations[ocean_focus][tile])
                v_adjusted = v_rotated * cos90(rotations[ocean_focus][tile]) - u_rotated * sin90(
                    rotations[ocean_focus][tile])
                quivers.append(
                    ax.quiver(quiver_x, quiver_y, u_adjusted, v_adjusted, scale=20 * uvmax, width=0.006, clip_on=False))
            if ckind == 'f':
                [quiver.set_color(acolor.value) for quiver in quivers]
                acolor.observe(lambda change: [quiver.set_color(change['new']) for quiver in quivers], names='value')
            quiverkey = axes[6].quiverkey(quivers[6], 1.5, 0.5, 5 * uvmax, f'{5 * uvmax:.5g}')
            def set_quiverkey_label(change):
                nonlocal quiverkey
                quiverkey.remove()
                label = f'{5 * uvmax:.5g}'
                if len(change['new']) > 0:
                    label += ' ' + change['new']
                quiverkey = axes[6].quiverkey(quivers[6], 1.5, 0.5, 5 * uvmax, label)
            uvlabel.observe(set_quiverkey_label, names='value')

        else:
            quiver_u = data.u.where(data[x].isin(quiver_x), drop=True).where(data[y].isin(quiver_y), drop=True)
            quiver_v = data.v.where(data[x].isin(quiver_x), drop=True).where(data[y].isin(quiver_y), drop=True)
            quiver_u, quiver_v = quiver_u.values, quiver_v.values
            if transpose: quiver_u, quiver_v = quiver_u.T, quiver_v.T
            plot_quiver = ax.quiver(quiver_x, quiver_y, quiver_u, quiver_v, scale=20 * uvmax, width=0.006)
            if ckind == 'f':
                plot_quiver.set_color(acolor.value)
                acolor.observe(lambda change: plot_quiver.set_color(change['new']), names='value')
            quiverkey = ax.quiverkey(plot_quiver, 0.95, 1.05, 2 * uvmax, f'{2 * uvmax:.5g} ')
            def set_quiverkey_label(change):
                nonlocal quiverkey
                quiverkey.remove()
                label = f'{2 * uvmax:.5g}'
                if len(change['new']) > 0:
                    label += ' ' + change['new']
                quiverkey = ax.quiverkey(plot_quiver, 0.95, 1.05, 2 * uvmax, label)
            uvlabel.observe(set_quiverkey_label, names='value')
