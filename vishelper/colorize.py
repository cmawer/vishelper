import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import vishelper.config as config


def get_plot_color(color_data, color):

    if color_data is not None:
        assert isinstance(color_data, list), 'color_data must be a list'
        if len(np.shape(color_data)) == 1:
            plot_color = [color_data]
        else:
            plot_color = color_data
    elif color is not None:
        plot_color = [color] if type(color) == str else color
    else:
        plot_color = config.formatting["color.all"]

    return plot_color


def color_continuous(df, column_to_color, new_color_column="color", clip=True, log10=False,
                     cmap=None, return_all=False, **kwargs):
    """Adds a column to a dataframe with colors assigned according to the continuous value in the `column_to_color`"""
    vmin = df[column_to_color].min() if "vmin" not in kwargs else kwargs["vmin"]
    vmax = df[column_to_color].max() if "vmax" not in kwargs else kwargs["vmax"]
    cmap = mpl.cm.OrRd if cmap is None else cmap

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if log10:
        df[new_color_column] = df[column_to_color].apply(
            lambda x: "#%02x%02x%02x" % tuple(int(255 * n) for n in mapper.to_rgba(np.log10(x))[:-1]))
    else:
        df[new_color_column] = df[column_to_color].apply(
            lambda x: "#%02x%02x%02x" % tuple(int(255 * n) for n in mapper.to_rgba(x)[:-1]))

    if return_all:
        return df, cmap, norm
    else:
        return df


def color_categorical(df, column_to_color, new_color_column="color", colors=None):
    """Adds a column to a dataframe with colors assigned according to the category in the `column_to_color`"""
    colors = config.formatting["color.darks"] if colors is None else colors
    categories = df[column_to_color].unique()
    n = len(categories)
    if len(colors) < n:
        raise ValueError("There are %i unique values but only %i colors were given" % (n, len(colors)))
    color_map = dict(zip(categories, colors[:n]))

    df[new_color_column] = df[column_to_color].apply(lambda x: color_map[x])

    return df


def create_colorbar(ax, cmap, norm, where="right", size='5%', pad=0.25, label=None):
    """Adds a color bar as defined by the provided `cmap` and `norm` """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(where, size=size, pad=pad)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                     norm=norm)
    if label is not None:
        cbar.set_label(label)
    return ax


def column_to_colors(df, column, colors=None):
    """Takes a column of categorical values and assigns a color to each category."""

    if colors is None:
        colors = config.formatting["color.all"]
    cats = df[column].unique()

    color_map = dict(zip(cats, colors[:len(cats)]))
    colors = df[column].map(color_map)
    return colors, color_map