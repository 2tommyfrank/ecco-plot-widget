# [ECCOv4](https://github.com/ECCO-GROUP/ECCOv4-py) plotting utility with Jupyter widgets

The `plot_select(c, u, v)` function produces widgets to make a customizable
plot of up to three xarray DataArrays on one figure. The `c` array gets plotted
using color, while the `u` and `v` arrays get plotted using arrows.

The `plot_utility()` function provides another layer of widgets to select the
inputs to `plot_select` directly from the ECCOv4 output on PODAAC.

Both functions require a login to NASA Earthdata in a .netrc file, and use the
`%matplotlib ipympl` backend.