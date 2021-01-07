import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import logging
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

formatting = {'font.size': 16,
              'tick.labelsize': 14,
              'tick.size': 10,
              'markersize': 48,
              'figure.figsize': [12.0, 8.0],
              'darks': ['#0067a0', '#53565a', '#009681','#87189d', '#c964cf'],
              'mediums': ['#0085ca', '#888b8d', '#00c389', '#f4364c', '#e56db1'],
              'lights': ['#00aec7', '#b1b3b3', '#2cd5c4', '#ff671f', '#ff9e1b'],
              'greens': ['#43b02a', '#78be20', '#97d700'],
              'axes.labelsize': 16,
              'axes.labelcolor': '#53565a',
              'axes.titlesize': 20,
              'lines.color': '#53565a',
              'lines.linewidth': 3,
              'legend.fontsize': 14,
              'legend.location': 'best',
              'legend.marker': 's',
              'text.color': '#53565a',
              'alpha.single': 0.8,
              'alpha.multiple': 0.7,
              'suptitle.x': 0.5,
              'suptitle.y': 1.025,
              'suptitle.size': 24}

cmaps = {'diverging': sns.diverging_palette(244.4, 336.7, s=71.2, l=41.6, n=20),
         'heatmap': sns.cubehelix_palette(8, start=.5, rot=-.75),
         'blues': sns.light_palette(formatting['darks'][0]),
         'reds': sns.light_palette(formatting['mediums'][4]),
         'teals': sns.light_palette(formatting["darks"][2]),
         'purples': sns.light_palette(formatting['darks'][4])}

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
map_to_days = dict(zip(range(7), days_of_week))


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
            lambda x: "#%02x%02x%02x" % tuple(int(255*n) for n in mapper.to_rgba(np.log10(x))[:-1]))
    else:
        df[new_color_column] = df[column_to_color].apply(
            lambda x: "#%02x%02x%02x" % tuple(int(255 * n) for n in mapper.to_rgba(x)[:-1]))

    if return_all:
        return df, cmap, norm
    else:
        return df


def color_categorical(df, column_to_color, new_color_column="color", colors=None):
    """Adds a column to a dataframe with colors assigned according to the category in the `column_to_color`"""
    colors = formatting["darks"] if colors is None else colors
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


def listify(l, multiplier=1, order=1):
    """Embeds a list in a list or replicates a list to meet shape requirements."""
    l = l if (isinstance(l, list) or isinstance(l, np.ndarray)) else [
                                                                         l] * multiplier

    while len(np.shape(l)) != order and not (isinstance(l[0], list) or isinstance(l[0], np.ndarray)):
        l = [l] * multiplier

    return l


def add_labels(ax, xlabel=None, ylabel=None, title=None, main_title=None):
    """Adds `xlabel`, `ylabel`, `title`, `main_title`, if provided to `ax` with size given by the `formatting` dict. """
    if xlabel:
        ax.set_xlabel(xlabel, size=formatting['axes.labelsize']);
    if ylabel:
        ax.set_ylabel(ylabel, size=formatting['axes.labelsize']);
    if title:
        ax.set_title(title, size=formatting['axes.titlesize']);

    if main_title:
        plt.suptitle(main_title, x=formatting['suptitle.x'],
                     y=formatting['suptitle.y'],
                     size=formatting['suptitle.size'])

    return ax


def adjust_lims(ax, xlim=None, ylim=None):
    """Adjusts the x-axis and y-axis view limits of `ax` if `xlim` and/or `ylim` are provided.

    Args:
        ax (:class:`matplotlib.axes._subplots.AxesSubplot`): Matplotlib axes handle
        xlim (`tuple`, optional): Tuple of (`x_min`, `x_max`) giving the range of x-values to view in the plot.
            If `xlim=None` (default), the x-axis view limits will not be changed.
        ylim (`tuple`, optional): Tuple of (`y_min`, `y_max`) giving the range of y-values to view in the plot.
            If `ylim=None` (default), the x-axis view limits will not be changed.

    Returns:
        ax (:class:`matplotlib.axes._subplots.AxesSubplot`): Matplotlib axes handle with adjusted xlim and ylim
    """
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


def add_hline(ax, y, **kwargs):
    """Adds a horizontal line to the axis `ax` at the provided `y` value."""
    xmin = ax.get_xlim()[0]
    xmax = ax.get_xlim()[1]
    ax.hlines(y, xmin, xmax, **kwargs)
    return ax


def add_vline(ax, x, **kwargs):
    """Adds a vertical line to the axis `ax` at the provided `x` value."""
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    ax.vlines(x, ymin, ymax, **kwargs)
    return ax


def add_dline(ax, **kwargs):
    """Adds a diagonal line with unit slope from `(x_min, x_min)` to `(x_max, x_max)`"""
    xmax = ax.get_xlim()[1]
    xmin = ax.get_xlim()[0]
    ax = ax.plot([xmin, xmax], [xmin, xmax], **kwargs)
    return ax


def labelfy(labels, label_map=None, replacements=None):
    if type(labels) == str:
        labels = [labels]
        return_single_label = True
    formatted = []
    for label in labels:
        if label_map is not None and label in label_map:
            formatted.append(label_map[label])
        else:
            if replacements is not None:
                formatted_label = label.capitalize()
                for replacement in replacements:
                    formatted_label = formatted_label.replace(replacement[0], replacement[1])

            else:
                formatted_label = label.capitalize()
            formatted.append(" ".join(formatted_label.split('_')))
    if return_single_label:
        formatted = formatted[0]
    return formatted


def column_to_colors(df, column, colors=None):
    """Takes a column of categorical values and assigns a color to each category."""

    if colors is None:
        colors = formatting["darks"] + formatting["mediums"]
    cats = df[column].unique()

    color_map = dict(zip(cats, colors[:len(cats)]))
    colors = df[column].map(color_map)
    return colors, color_map


def fake_legend(ax, legend_labels, colors, marker=None, size=None, fontsize=None, linestyle="",
                loc=None, bbox_to_anchor=None, where="best", **kwargs):
    """Adds a fake legend to the plot with the provided legend labels and corresponding colors and attributes.

    Args:
        ax (:class:`matplotlib.axes._subplots.AxesSubplot`): Matplotlib axes handle
        legend_labels (`list` of `str`): Labels for the legend items.
        colors (`list`): List of colors of the items in the legend.
        marker (`str` or `list` of `str`, optional): Marker for the items in the legend. Defaults
                                                     to `formatting['legend.marker']`
        size (`str` or `list` of `str`, optional): Marker size for the items in the legend. Defaults
                                                     to `formatting['markersize']`
        where (`str`, optional): Where to put the legend. Options are `right`, `below`, and `best`
        linestyle (`str` or `list` of `str`, optional): Line style for items in legend. Defaults to `""` (no line).
        loc ('str`, optional): Location for where to place the legend. Defaults to "best".
        bbox_to_anchor (`tuple`): Where to anchor legend, used to place legend outside plot. Default None.
        **kwargs: Keyword arguments passed to `ax.legend()`

    Returns:
        ax (:class:`matplotlib.axes._subplots.AxesSubplot`): Matplotlib axes handle with fake legend
    """

    locations = dict(right=dict(loc='center left', bbox_to_anchor=(1, 0.5)),
                     below=dict(loc='upper center', bbox_to_anchor=(0.5, -0.1)),
                     best=dict(loc="best"))

    if loc is None and bbox_to_anchor is None:
        location = locations[where]
    else:
        location = {}
        if loc is not None:
            location["loc"] =loc
        if bbox_to_anchor is not None:
            location["bbox_to_anchor"] = bbox_to_anchor

    marker = formatting['legend.marker'] if marker is None else marker
    size = formatting['markersize'] if size is None else size

    fontsize = formatting['legend.fontsize'] if fontsize is None else fontsize

    for col, lab in zip(colors, legend_labels):
        ax.plot([], linestyle=linestyle, marker=marker, c=col, label=lab, markersize=size);

    lines, labels = ax.get_legend_handles_labels();

    for k in location:
        kwargs[k] = location[k]

    ax.legend(lines[:len(legend_labels)],
              labels[:len(legend_labels)],
              fontsize=fontsize,
              **kwargs);

    return ax


def scatter(x, y, ax=None, color=None, size=None, alpha=None, logx=False, logy=False, **kwargs):
    """Creates a scatter plot of (x, y)

    Args:
        x (`list` or :class:`numpy.ndarray`): x-coordinates for plotting. Must be same size as `y`
        y (`list` or :class:`numpy.ndarray`): y-coordinates for plotting. Must be same size as `x`
        ax (:class:`matplotlib.axes._subplots.AxesSubplot`): Matplotlib axes handle
        color: The marker color. Can be a olor, sequence, or sequence of color, optional. Defaults to the first value
               in `formatting["darks"]. Possible values:
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
    if not ax:
        fig, ax = plt.subplots(figsize=formatting['figure.figsize'])
    else:
        fig = None

    if color is None:
        color = formatting['darks'][0]
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


def line(x, y, ax=None, color=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots(figsize=formatting['figure.figsize'])
    else:
        fig = None

    if color is None:
        color = formatting['darks'][0]

    if 'marker' not in kwargs.keys():
        marker = None
    else:
        marker = kwargs.pop('marker')
    if 'markersize' not in kwargs.keys():
        markersize = formatting['markersize']
    else:
        markersize = kwargs.pop('markersize')

    if 'linestyle' not in kwargs.keys():
        linestyle = "-"
    else:
        linestyle = kwargs.pop("linestyle")
    if 'linewidth' not in kwargs.keys():
        linewidth = formatting["lines.linewidth"]
    else:
        linewidth = kwargs.pop("linewidth")
    if 'alpha' not in kwargs.keys():
        alpha = formatting['alpha.single']
    else:
        alpha = kwargs.pop('alpha')

    ax.plot(x, y, color=color, linestyle=linestyle, alpha=alpha, marker=marker, markersize=markersize, linewidth=linewidth, **kwargs)

    if fig is None:
        return ax
    else:
        return fig, ax


def barh(x, y, ax=None, color=None, label=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots(figsize=formatting['figure.figsize'])
    else:
        fig = None

    if not color:
        color = formatting['darks'][0]

    if 'alpha' not in kwargs.keys():
        alpha = formatting['alpha.single']
    else:
        alpha = kwargs['alpha']

    ax.barh(range(len(y)), y, color=color, alpha=alpha, label=label)
    ax.set_yticks(range(len(y)))
    ax.set_yticklabels(x, size=formatting['tick.labelsize'])

    ax.set_ylim([-1, len(y)]);

    if fig is None:
        return ax
    else:
        return fig, ax

    return fig, ax


def hist(x, ax=None, color=None, logx=False, ignore_nan=True, **kwargs):
    """Plots a histogram based on values of `x`"""
    if ignore_nan:
        if "stacked" not in kwargs or not kwargs["stacked"]:
            original_len = len(x)
            x = [xi for xi in x if not np.isnan(xi)]
            final_len = len(x)
            if final_len != original_len:
                diff = original_len-final_len
                logging.warning("A total of %i NaN values out of %i observations were removed" % (diff, original_len))
        else:
            x_to_process = x
            x = []
            for j, subx in enumerate(x_to_process):
                original_len = len(subx)
                subx = [xi for xi in subx if not np.isnan(xi)]
                x.append(subx)
                final_len = len(subx)
                if final_len != original_len:
                    diff = original_len - final_len
                    logging.warning("A total of %i NaN values out of %i observations were removed from set %i" % (diff, original_len, j))
    if not ax:
        fig, ax = plt.subplots(figsize=formatting['figure.figsize'])
    else:
        fig = None

    if logx:
        if 'bins' in kwargs.keys() and isinstance(kwargs['bins'], int):
            kwargs['bins'] = np.logspace(np.log10(np.min(x)),
                                         np.log10(np.max(x)), kwargs['bins'])
        ax.set_xscale("log")

    if 'alpha' not in kwargs.keys():
        alpha = formatting['alpha.single']
    else:
        alpha = kwargs.pop('alpha')

    if color is None:
        color = formatting['darks'][0]
    if "stacked" in kwargs and kwargs["stacked"]:
        if len(color)!=len(x):
            color = formatting['darks'][:len(x)]
        # if "stackedlabels" in kwargs:
        #     kwargs["label"] = kwargs.pop("stackedlabels")

    ax.hist(x, color=color, alpha=alpha, **kwargs)

    if fig is None:
        return ax
    else:
        return fig, ax


def heatmap(df, ax=None,
            label_size=None,
            xticklabels=None, yticklabels=None, log10=False,
            xrotation=90, yrotation=0, **kwargs):
    label_size = formatting['tick.labelsize'] if label_size is None else label_size

    if ax is None:
        if 'figsize' in kwargs.keys():
            figsize = kwargs.pop('figsize')
        else:
            figsize = formatting['figure.figsize']
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False
    yticklabels = df.index.tolist() if yticklabels is None else yticklabels
    xticklabels = df.columns.tolist() if xticklabels is None else xticklabels

    if log10:

        log_norm = LogNorm(vmin=df.replace(0, 1).min(skipna=True).min(skipna=True), vmax=df.replace(0, 1).max(skipna=True).max(skipna=True))

        cbar_ticks = [np.float_power(10, i) for i in
                      range(int(np.floor(np.log10(df.replace(0, 1).min(skipna=True).min(skipna=True)))), 1 + int(np.ceil(np.log10(df.replace(0, 1).max(skipna=True).max(skipna=True)))))]
        kwargs["cbar_kws"] = {"ticks": cbar_ticks}
        kwargs["norm"] = log_norm

    if 'cmap' in kwargs:
        cmap = kwargs.pop("cmap")
        if isinstance(cmap, str) and cmap in cmaps:
            cmap = cmaps[cmap]
    else:
        cmap = cmaps['heatmap']

    ax = sns.heatmap(df, cmap=cmap, **kwargs);
    ax.set_xticklabels(xticklabels, rotation=xrotation, size=label_size);
    ax.set_yticklabels(yticklabels, rotation=yrotation, size=label_size);

    ax.set_ylim([0, len(df)])

    if return_fig:
        return fig, ax
    else:
        return ax


def boxplot(x, y, ax=None, palette="Set3", data=None, hue=None, white=False, color_map=None, xrotation=90, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=formatting['figure.figsize'])
    else:
        fig = None

    if "label" in kwargs:
        kwargs.pop("label")

    if data is not None and type(x)==list:
        x = x[0]
        y = y[0]
    ax = sns.boxplot(x=x, y=y, data=data, palette=palette, hue=hue, ax=ax, **kwargs)

    if white or color_map is not None:
        for i, box in enumerate(ax.artists):
            if color_map is not None:
                label = ax.get_xticklabels()[i].get_text()
                color = color_map[label]
            else:
                color = "white"
            box.set_edgecolor('black')
            box.set_facecolor(color)

            # iterate over whiskers and median lines
            for j in range(6 * i, 6 * (i + 1)):
                ax.lines[j].set_color('black')
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation)
    if fig is None:
        return ax
    else:
        return fig, ax


def plotxy(x, y, ax, plot_function, plot_color, df=None, labels=None, **kwargs):
    if y is not None:
        y = listify(y, order=2)
        x = listify(x, order=2, multiplier=np.shape(y)[0])
    else:
        x = listify(x, order=2)
    if df is not None:
        logger.debug(x)
        logger.debug(y)
        x = [df[x] for xi in x]
        y = [df[y] for yi in y]
    for j, onex in enumerate(x):

        if labels is None:
            label = 'Set ' + str(j)
        else:
            label = labels[j]
        if y is None:
            ax = plot_function(onex, ax=ax,
                               color=plot_color[j],
                               label=label, **kwargs)
        else:
            ax = plot_function(onex, y=y[j], ax=ax,
                               color=plot_color[j],
                               label=label, **kwargs)
    if j > 0:
        plot_legend = True
    else:
        plot_legend = False
    return ax, plot_legend


plot_functions = dict(hist=hist, scatter=scatter, barh=barh, line=line, boxplot=boxplot, heatmap=heatmap)


def plot(x=None, y=None, df=None, kind=None, plot_function=None, ax=None,
         xlabel=None, ylabel=None, title=None, legend=None, legend_kwargs=None, ticks=None,
         labels=None, color=None,  color_data=None, figsize=None, xlim=None, ylim=None, **kwargs):
    """

    Args:
        x (list): Data to be plotted in the x-axis or for univariate plots. If None, df must be provided.
        y (:obj:`list`, optional): Default None (implies univariate plot).
        kind (:obj:`str`, optional): Type of plot. Defaults to None, which implies `plot_function` should be given.
            Univariate continuous (y=None)
            * hist
            * barh

            Bivariate: continous x continuous
            * scatter
            * line

            Bivariate: categorical x continuous
            * boxplot
        df: Dataframe containing the data to be plotted
        plot_function: Default None. If "kind" is not given, can provide a plot function with the
            form plot_function(x, y, ax, **kwargs) or plot_function(x, ax, **kwargs)
        ax (:py:class:`matplotlib.axes._subplots.AxesSubplot`, optional): Matplotlib axes handle. Default is None
            and a new `ax` is generated along with the `fig`.
        xlabel (:obj:`str`, optional): Label for x axis. Default None.
        ylabel (:obj:`str`, optional): Label for y axis. Default None.
        title (:obj:`str`, optional): Plot title. Default None.
        legend (bool, optional): A legend will be displayed if legend=True or legend=None and more than one thing
            is being plot. Default None.
        legend_kwargs (:obj`dict`, optional): Dictionary of keyword arguments for legend (see `legend` for more when
            legend will be displayed). Default None.
        labels (:obj:`list` of :obj:`str`, optional): List of labels for legend. Default None. If legend is True,
            and no labels are provided, labels will be "Set N" where N is the ordering of the inputs.
        ticks (:obj:`list` of :obj:`str`, optional): List of tick labels. Default None.
        color (:obj:`str` or :obj:`list` of :obj:`str`, optional): Color to plot. List of colors if there is more than
            one thing being plot. Default is None, in which case, if color_data is also None, default colors
            from `vishelper.formatting["darks"]` and then `vishelper.formatting["lights"]` will be used.
        color_data (:obj:`list` of :obj:`str`, optional): If provided, list should be the same length of the data,
            providing an individual color for each data point. Default None in which case all points will be colored
            according to `color` argument.
        figsize (tuple, optional): Figure size. Default is None and plot will be sized based on
            `vishelper.formatting['figure.figsize']`.
        xlim (tuple, optional): Tuple of minimum and maximum x values to plot (xmin, xmax). Default is None and
            matplotlib chooses these values.
        ylim (tuple, optional): Tuple of minimum and maximum y values to plot (ymin, ymax). Default is None and
            matplotlib chooses these values.
        **kwargs: These kwargs are any that are relevant to the plot kind provided.

    Returns:
        fig (matplotlib.figure.Figure): Matplotlib figure object
        ax (:py:class:`matplotlib.axes._subplots.AxesSubplot`): Axes object with the plot(s)

    """



    if ax is None:
        fig, ax = plt.subplots(figsize=formatting['figure.figsize'] if figsize is None else figsize)
    else:
        fig = None

    if not plot_function:
        plot_function = plot_functions[kind]

    if color_data is not None:
        if len(x) == 1:
            plot_color = [color_data]
        else:
            plot_color = color_data
    elif color is not None:
        plot_color = [color] if type(color) == str else color
    else:
        plot_color = formatting["darks"] + formatting['lights']

    if x is None:
        assert df is not None, 'Must provide either x or df'
        ax = plot_function(df, ax=ax, **kwargs)
        plot_legend = False
    else:
        ax, plot_legend = plotxy(x, y, ax, plot_function, plot_color, df=df, labels=labels, **kwargs)

    ax = add_labels(ax, xlabel, ylabel, title)

    if ticks is not None:
        ax.xaxis.set_ticks(ticks);
    ax.tick_params(labelsize=formatting['tick.labelsize'],
                   size=formatting['tick.size'])

    if (plot_legend and legend is None) or legend:
        if legend_kwargs is None:
            ax.legend(loc=formatting['legend.location'],
                      fontsize=formatting['legend.fontsize'])
        else:
            ax.legend(**legend_kwargs)
    ax = adjust_lims(ax, xlim, ylim)

    if fig is None:
        return ax
    else:
        return fig, ax


def subplots(columns_to_plot, layout=None):
    if isinstance(columns_to_plot, str):
        columns_to_plot = [columns_to_plot]
    elif not isinstance(columns_to_plot, list):
        raise ValueError('`columns_to_plot` must be a string or list containing which columns to plot')
    num_plots = len(columns_to_plot)
    if layout is None:
        if num_plots % 2 != 0:
            layout = (num_plots, 1)
        else:
            layout = (int(np.ceil(num_plots / 2)), 2)
    elif len(layout) != 2:
        raise ValueError('please provide layout as (n_rows, n_cols)')
    elif layout[0] * layout[1] < num_plots:
        raise ValueError(
            'layout provide does not have enough space for all desired plots')