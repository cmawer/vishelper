import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import interpn
from matplotlib.colors import Normalize

from vishelper.config import formatting
import vishelper.helpers as helpers


def scatter_density(x, y, ax=None, sort=True, bins=20, cmap='gnuplot2', colorbar_off=False, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, cmap=cmap, **kwargs)

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    if colorbar_off is False:
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.ax.set_ylabel('Density')

    return ax


def scatter(x, y, ax=None, color=None, size=None, alpha=None, logx=False, logy=False, aspect=None, density=False, colorbar_off=False, **kwargs):
    """Creates a scatter plot of (x, y)

    Args:
        x (`list` or :class:`numpy.ndarray`): x-coordinates for plotting. Must be same size as `y`
        y (`list` or :class:`numpy.ndarray`): y-coordinates for plotting. Must be same size as `x`
        ax (:class:`matplotlib.axes._subplots.AxesSubplot`): Matplotlib axes handle
        color: The marker color. Can be a olor, sequence, or sequence of color, optional. Defaults to the first value
               in `formatting["color.darks"]. Possible values:
                    A single color format string.
                    A sequence of color specifications of length n.
                    A sequence of n numbers to be mapped to colors using cmap and norm.
        alpha (`float`, optional): The alpha blending value, between 0 (transparent) and 1 (opaque).
                                Defaults to `formatting['alpha.single']`
        logx (`bool`): If True, the x-axis will be transformed to log scale
        logy (`bool`): If True, the y-axis will be transformed to log scale
        **kwargs: Additional keyword arguments are passed to `ax.scatter()`

    Returns:
        ax (:class:`matplotlib.axes._subplots.AxesSubplot`): Matplotlib axes handle with scatter plot on it

    """
    fig, ax = helpers.get_ax_fig(ax, kwargs=kwargs)

    if color is None:
        color = formatting['color.single']
    if size is None:
        size = formatting['markersize']

    if alpha is None:
        alpha = formatting['alpha.single']

    if logx:
        ax.set_xscale("log");

    if logy:
        ax.set_yscale("log");

    if aspect:
        ax.set_aspect(aspect)

    if density:
        scatter_density(x, y, ax=ax, s=size, alpha=alpha, colorbar_off=colorbar_off, **kwargs)
    else:
        ax.scatter(x, y, color=color, s=size, alpha=alpha, **kwargs)

    if fig is None:
        return ax
    else:
        return fig, ax