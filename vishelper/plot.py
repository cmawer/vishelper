import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import logging
from matplotlib.colors import LogNorm
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
         'blues': sns.light_palette(formatting['darks'][0]),
         'reds': sns.light_palette(formatting['mediums'][4]),
         'teals': sns.light_palette(formatting["darks"][2]),
         'purples': sns.light_palette(formatting['darks'][4])}

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
map_to_days = dict(zip(range(7), days_of_week))

def listify(l, multiplier=1, order=1):
    l = l if (isinstance(l, list) or isinstance(l, np.ndarray)) else [
                                                                         l] * multiplier

    while len(np.shape(l)) != order and not (isinstance(l[0], list) or isinstance(l[0], np.ndarray)):
        l = [l] * multiplier

    return l


def add_labels(ax, xlabel=None, ylabel=None, title=None, main_title=None):
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
    # self.logger.info("Hello " + self.name)
    return ax


def adjust_lims(ax, xlim=None, ylim=None):
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


def hist(x, ax=None, color=None, logx=False, ignore_nan=True, **kwargs):

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

    # if "label" in kwargs:
    #     ax.legend(loc=formatting['legend.location'],
    #               fontsize=formatting['legend.fontsize'])

    if fig is None:
        return ax
    else:
        return fig, ax


def scatter(x, y, ax=None, color=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots(figsize=formatting['figure.figsize'])
    else:
        fig = None

    if color is None:
        color = formatting['darks'][0]
    if 'size' not in kwargs.keys():
        size = formatting['markersize']
    else:
        size = kwargs.pop('size')
    if 'alpha' not in kwargs.keys():
        alpha = formatting['alpha.single']
    else:
        alpha = kwargs.pop('alpha')

    if "logx" in kwargs:
        logx = kwargs.pop("logx")
        if logx:
            ax.set_xscale("log");

    if "logy" in kwargs:
        logy = kwargs.pop("logy")
        if logy:
            ax.set_yscale("log");

    ax.scatter(x, y, color=color, s=size, alpha=alpha, **kwargs)

    #
    #     if logx:
    #         ax = ax.set_xscale("log")
    #
    # if "logy" in kwargs:
    #     logy = kwargs.pop("logy")
    #     if logy:
    #         ax = ax.set_yscale("log")

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


def fake_legend(ax, legend_labels, colors, marker=None, size=None, linestyle="", loc="best", **kwargs):
    marker = formatting['legend.marker'] if marker is None else marker
    size = formatting['markersize'] if size is None else size

    for col, lab in zip(colors, legend_labels):
        ax.plot([], linestyle=linestyle, marker=marker, c=col, label=lab, markersize=size);

    lines, labels = ax.get_legend_handles_labels();
    
    ax.legend(lines[:len(legend_labels)],
              labels[:len(legend_labels)],
              loc=loc, **kwargs);

    return ax


def heatmap(df, ax=None,
            xlabel=None, ylabel=None, title=None, label_size=None,
            xticklabels=None, yticklabels=None, log10=False,
            xrotation=90, yrotation=0, **kwargs):
    label_size = formatting['tick.labelsize'] if label_size is None else label_size
    if 'figsize' in kwargs.keys():
        figsize = kwargs.pop('figsize')
    else:
        figsize = formatting['figure.figsize']
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
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
    else:
        cmap = sns.cubehelix_palette(8, start=.5, rot=-.75)

    ax = sns.heatmap(df, cmap=cmap, **kwargs);
    ax.set_xticklabels(xticklabels, rotation=xrotation, size=label_size);
    ax.set_yticklabels(yticklabels, rotation=yrotation, size=label_size);
    ax = add_labels(ax, xlabel, ylabel, title)
    return fig, ax


def boxplot(x, y, ax=None, palette="Set3", data=None, hue=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=formatting['figure.figsize'])
    else:
        fig = None

    if data is not None and type(x)==list:
        x = x[0]
        y = y[0]
    ax = sns.boxplot(x=x, y=y, data=data, palette=palette, hue=hue, ax=ax)

    if fig is None:
        return ax
    else:
        return fig, ax


plot_functions = dict(hist=hist, scatter=scatter, barh=barh, line=line, boxplot=boxplot)


def plot(x, y=None, kind=None, plot_function=None, ax=None,
         xlabel=None, ylabel=None, title=None, legend=None, legend_kwargs=None, ticks=None,
         labels=None, color=None,  color_data=None, figsize=None, xlim=None, ylim=None, **kwargs):
    """

    Args:
        x (list): Data to be plotted in the x-axis or for univariate plots. If plotting a heatmap,
            x should be be a :py:class:`pandas.DataFrame()`
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

        plot_function: Default None. If "kind" is not given, can provide a plot function with the
            form plot_function(x, y, ax, **kwargs) or plot_function(x, ax, **kwargs)
        ax:
        xlabel:
        ylabel:
        title:
        legend:
        legend_kwargs:
        ticks:
        labels:
        color:
        color_data:
        figsize:
        xlim:
        ylim:
        **kwargs:

    Returns:

    """
    if y is not None:
        y = listify(y, order=2)
        x = listify(x, order=2, multiplier=np.shape(y)[0])
    else:
        x = listify(x, order=2)

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
        plot_color = color
    else:
        plot_color = formatting["darks"] + formatting['lights']

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
    ax = add_labels(ax, xlabel, ylabel, title)

    if ticks is not None:
        ax.xaxis.set_ticks(ticks);
    ax.tick_params(labelsize=formatting['tick.labelsize'],
                   size=formatting['tick.size'])

    if (j > 0 and legend is None) or legend:
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


def add_hline(ax, y, **kwargs):
    xmin = ax.get_xlim()[0]
    xmax = ax.get_xlim()[1]
    ax.hlines(y, xmin, xmax, **kwargs)
    return ax


def add_vline(ax, x, **kwargs):
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    ax.vlines(x, ymin, ymax, **kwargs)
    return ax


def add_dline(ax, **kwargs):

    xmax = ax.get_xlim()[1]
    xmin = ax.get_xlim()[0]
    ax = ax.plot([xmin, xmax], [xmin, xmax], **kwargs)
    return ax


def labelfy(labels):
    if isinstance(labels, str):
        labels = ' '.join(labels.split('_')).capitalize()
    elif isinstance(labels[0], str):
        labels = [' '.join(lab.split('_')).capitalize() for lab in labels]

    return labels
