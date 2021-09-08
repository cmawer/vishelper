import matplotlib.pyplot as plt

import vishelper.config as config


def add_labels(ax, xlabel=None, ylabel=None, title=None, main_title=None):
    """Adds `xlabel`, `ylabel`, `title`, `main_title`, if provided to `ax` with size given by the `formatting` dict. """
    if xlabel:
        ax.set_xlabel(xlabel, size=config.formatting['axes.labelsize']);
    if ylabel:
        ax.set_ylabel(ylabel, size=config.formatting['axes.labelsize']);
    if title:
        ax.set_title(title, size=config.formatting['axes.titlesize']);

    if main_title:
        plt.suptitle(main_title, x=config.formatting['suptitle.x'],
                     y=config.formatting['suptitle.y'],
                     size=config.formatting['suptitle.size'])

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


def labelfy(labels, label_map=None, replacements=None):
    if type(labels) == str:
        labels = [labels]
        return_single_label = True
    else:
        return_single_label = False

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
            location["loc"] = loc
        if bbox_to_anchor is not None:
            location["bbox_to_anchor"] = bbox_to_anchor

    marker = config.formatting['legend.marker'] if marker is None else marker
    size = config.formatting['markersize'] if size is None else size

    fontsize = config.formatting['legend.fontsize'] if fontsize is None else fontsize

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


def decide_legend(ax, legend, plot_legend, legend_kwargs):
    if (plot_legend and legend is None) or legend:
        if legend_kwargs is None:
            ax.legend(loc=config.formatting['legend.location'],
                      fontsize=config.formatting['legend.fontsize'])
        else:
            ax.legend(**legend_kwargs)
    return ax