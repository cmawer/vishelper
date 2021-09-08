import numpy as np
import matplotlib.pyplot as plt

from vishelper.config import formatting
import vishelper.helpers as helpers


def get_x_bars(x, grouped=False, width=0.8, spacing=0.2, n_groups=1):
    if grouped:
        # Need to plot # bars per group x # groups
        xbars, xticks = [], []
        start = 0
        for j in x:
            group_x = np.arange(start, start + (n_groups-1)*width +spacing, width)

            xbars += group_x.tolist()
            start = xbars[-1] + width + spacing

            mid_x = (group_x[0] + group_x[-1])/2
            xticks.append(mid_x)
    else:
        xbars = np.arange(0, len(x)*(width+spacing), width+spacing)
        xticks = xbars
    return xbars, xticks


def bar(x, y, ax=None, color=None, label=None, stacked=False, grouped=False, groups=None,
        width=0.8, spacing=0.2, **kwargs):

    fig, ax = helpers.get_ax_fig(ax, kwargs=kwargs)

    y = helpers.listify(y, order=2)
    n_groups = np.shape(y)[0] if grouped else 1

    x = x[0] if len(np.shape(x)) > 1 else x

    xbars, xticks = get_x_bars(x, grouped, width, spacing, n_groups)

    if not color and not stacked:
        color = formatting['color.single']
    elif not color and stacked:
        if len(y) > len(formatting['color.all']):
            raise ValueError("Not enough colors in default vh.formatting['color.all'], "
                             "please provide `color` or reduce number of categories")
        else:
            color = formatting['color.all'][:len(y)]

    if 'alpha' not in kwargs.keys():
        alpha = formatting['alpha.single']
    else:
        alpha = kwargs.pop('alpha')

    bottom = np.zeros(len(xticks))  # Assuming len(x) same for all x
    for j, oney in enumerate(y):
        lab = None if label is None and groups is None else groups[j] if label is None else label
        col = color if isinstance(color, str) else color[j]
        onex = xbars[j::n_groups] if grouped else xbars
        ax.bar(
            onex,
            oney,
            bottom=bottom,
            width=width,
            color=col,
            alpha=alpha,
            label=lab,
            **kwargs);
        if stacked:
            bottom = bottom + np.array(oney)

    ax.set_xticks(xticks)
    ax.set_xticklabels(x, size=formatting['tick.labelsize'])

    ax.set_xlim([-1*width - spacing, np.max(xticks) + 1.5*width + spacing]);

    if fig is None:
        return ax
    else:
        return fig, ax


def barh(x, y, ax=None, color=None, label=None, stacked=False, grouped=False, groups=None,
         width=0.8, spacing=0.2, **kwargs):

    fig, ax = helpers.get_ax_fig(ax, kwargs=kwargs)

    y = helpers.listify(y, order=2)
    n_groups = np.shape(y)[0] if grouped else 1

    x = x[0] if len(np.shape(x)) > 1 else x

    xbars, xticks = get_x_bars(x, grouped, width, spacing, n_groups)

    if not color:
        color = formatting['color.single']

    if 'alpha' not in kwargs.keys():
        alpha = formatting['alpha.single']
    else:
        alpha = kwargs.pop('alpha')

    left = np.zeros(len(xticks))  # Assuming len(x) same for all x
    for j, oney in enumerate(y):
        lab = None if label is None and groups is None else groups[j] if label is None else label
        col = color if isinstance(color, str) else color[j]
        onex = xbars[j::n_groups] if grouped else xbars
        ax.barh(
            onex,
            oney,
            left=left,
            color=col,
            alpha=alpha,
            label=lab,
            **kwargs);
        if stacked:
            left = left + np.array(oney)

    ax.set_yticks(xticks)
    ax.set_yticklabels(x, size=formatting['tick.labelsize'])

    ymargin = (-1*width - spacing, 1.5*width + spacing) if grouped else (-1*width, width)
    ax.set_ylim([ymargin[0], np.max(xticks) + ymargin[1]]);

    if fig is None:
        return ax
    else:
        return fig, ax
