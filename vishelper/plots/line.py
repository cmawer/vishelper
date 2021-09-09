
import vishelper.config as config
import vishelper.helpers as helpers


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


def add_dline(ax, m=1, b=0, **kwargs):
    """Adds a diagonal line with slope, m, and y-intercept, b, between `x_min` and `x_max`"""
    xmin = ax.get_xlim()[0]
    xmax = ax.get_xlim()[1]
    y0 = m*xmin + b
    y1 = m*xmax + b
    ax = ax.plot([xmin, xmax], [y0, y1], **kwargs)
    return ax


def line(x, y, ax=None, color=None, **kwargs):

    fig, ax = helpers.get_ax_fig(ax, kwargs=kwargs)

    if color is None:
        color = config.formatting['color.single']

    if 'marker' not in kwargs.keys():
        marker = None
    else:
        marker = kwargs.pop('marker')
    if 'markersize' not in kwargs.keys():
        markersize = config.formatting['markersize']
    else:
        markersize = kwargs.pop('markersize')

    if 'linestyle' not in kwargs.keys():
        linestyle = "-"
    else:
        linestyle = kwargs.pop("linestyle")
    if 'linewidth' not in kwargs.keys():
        linewidth = config.formatting["lines.linewidth"]
    else:
        linewidth = kwargs.pop("linewidth")
    if 'alpha' not in kwargs.keys():
        alpha = config.formatting['alpha.single']
    else:
        alpha = kwargs.pop('alpha')

    ax.plot(x, y, color=color, linestyle=linestyle, alpha=alpha, marker=marker, markersize=markersize,
            linewidth=linewidth, **kwargs)

    if fig is None:
        return ax
    else:
        return fig, ax
