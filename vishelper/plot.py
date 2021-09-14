import logging

import numpy as np
import matplotlib.pyplot as plt

import vishelper.config as config
import vishelper.helpers as helpers
import vishelper.reformat as reformat
import vishelper.plots as plots
import vishelper.colorize as colorize

logger = logging.getLogger(__name__)


def plotxy(x, y, ax, plot_function, plot_color, df=None, labels=None, stacked=False, grouped=False, **kwargs):
    if y is not None:
        y = helpers.listify(y, order=2)
        x = helpers.listify(x, order=2, multiplier=np.shape(y)[0])
    else:
        x = helpers.listify(x, order=2)

    if df is not None:
        logger.debug(x)
        logger.debug(y)
        x = [df[x] for xi in x]
        y = [df[y] for yi in y]

    if stacked or grouped:
        ax = plot_function(x, y, ax=ax,
                           color=plot_color, stacked=stacked, grouped=grouped,
                           groups=labels, **kwargs)
        plot_legend = True
    else:
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


plot_functions = dict(hist=plots.hist, scatter=plots.scatter, barh=plots.barh, bar=plots.bar, line=plots.line,
                      boxplot=plots.boxplot, heatmap=plots.heatmap)


def plot(x=None, y=None, df=None, kind=None, plot_function=None, ax=None,
         xlabel=None, ylabel=None, title=None, legend=None, legend_kwargs=None, ticks=None,
         labels=None, color=None, color_data=None, figsize=None, xlim=None, ylim=None, tight_layout=False, **kwargs):
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
            from `vishelper.config.formatting["color.all"]` will be used.
        color_data (:obj:`list` of :obj:`str`, optional): If provided, list should be the same length of the data,
            providing an individual color for each data point. Default None in which case all points will be colored
            according to `color` argument.
        figsize (tuple, optional): Figure size. Default is None and plot will be sized based on
            `vishelper.config.formatting['figure.figsize']`.
        xlim (tuple, optional): Tuple of minimum and maximum x values to plot (xmin, xmax). Default is None and
            matplotlib chooses these values.
        ylim (tuple, optional): Tuple of minimum and maximum y values to plot (ymin, ymax). Default is None and
            matplotlib chooses these values.
        tight_layout (bool, optional): Envokes `plt.tight_layout()` to ensure enough space is given to plot elements
        **kwargs: These kwargs are any that are relevant to the plot kind provided.

    Returns:
        fig (matplotlib.figure.Figure): Matplotlib figure object
        ax (:py:class:`matplotlib.axes._subplots.AxesSubplot`): Axes object with the plot(s)

    """

    fig, ax = helpers.get_ax_fig(ax, figsize)

    if not plot_function:
        plot_function = plot_functions[kind]

    plot_color = colorize.get_plot_color(color_data, color)

    if x is None:
        assert df is not None, 'Must provide either x or df'
        ax = plot_function(df, ax=ax, **kwargs)
        plot_legend = False
    else:
        ax, plot_legend = plotxy(x, y, ax, plot_function, plot_color, df=df, labels=labels, **kwargs)

    ax = reformat.add_labels(ax, xlabel, ylabel, title)

    if ticks is not None:
        ax.xaxis.set_ticks(ticks);
    ax.tick_params(labelsize=config.formatting['tick.labelsize'],
                   size=config.formatting['tick.size'])

    ax = reformat.decide_legend(ax, legend, plot_legend, legend_kwargs)
    ax = reformat.adjust_lims(ax, xlim, ylim)

    if tight_layout:
        plt.tight_layout()
    if fig is None:
        return ax
    else:
        return fig, ax

