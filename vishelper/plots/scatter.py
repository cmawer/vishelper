import matplotlib.pyplot as plt
import numpy as np

from vishelper.config import formatting
import vishelper.helpers as helpers


def scatter(x, y, ax=None, color=None, size=None, alpha=None, logx=False, logy=False, **kwargs):
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

    ax.scatter(x, y, color=color, s=size, alpha=alpha, **kwargs)

    if fig is None:
        return ax
    else:
        return fig, ax