import seaborn as sns

import vishelper.helpers as helpers


def boxplot(x, y, hue=None, data=None, orient='v', ax=None,
            palette="Set3", edge_color='black', face_color=None, color_map=None,
            xrotation=None, yrotation=None, **kwargs):
    """Draws boxplots to compare distributions of numerical variables for a set of categories.

    Args:
       x: If orient='v', categorical values. If orient='h', numerical values.
          List of values or column name of values if data is provided.
       y: If orient='v', numerical values. If orient='h', categorical values.
          List of values or column name of values if data is provided.
       hue: Categorical values to group box vishelper by. List of values or column name of values if data is provided.
       data: Optional, dataframe containing data to plot. Default: None
       orient: 'v' to plot box vishelper vertically, 'h' to plot horizontally. Default: 'v'
       ax (:py:class:`matplotlib.axes._subplots.AxesSubplot`, optional): Matplotlib axes handle. Default is None
            and a new `ax` is generated along with the `fig`.
       palette: Colors to use for the different levels of the hue variable. Should be something that can be interpreted
            by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
       edge_color: Color for edges of box plot. Default: 'black'. Used if color_map or pallet are provided.
       face_color: Color for boxes in box plot. Default: None. `pallet` or `color_map` is used if not provided.
       color_map: Dictionary of mapping between unique categorical values and colors. Default None.
         `pallet` or `face_color` used if not provided.
       xrotation (int): Degree to rotate x-label. Default: None
       yrotation (int): Degree to rotate y-label. Default None.
       **kwargs: Any additional keyword arguments that `seaborn.boxplot()` can take

    Returns:
        fig (matplotlib.figure.Figure): Matplotlib figure object. Only returned if ax is not provided as an argument.
        ax (:py:class:`matplotlib.axes._subplots.AxesSubplot`): Axes object with the plot(s)
    """

    fig, ax = helpers.get_ax_fig(ax, kwargs=kwargs)

    if "label" in kwargs:
        kwargs.pop("label")

    if data is not None and type(x) == list:
        x = x[0]
        y = y[0]
    ax = sns.boxplot(x=x, y=y, data=data, palette=palette, hue=hue, ax=ax, orient=orient, **kwargs)

    if color_map is not None or face_color is not None:
        for i, box in enumerate(ax.artists):
            if color_map is not None:
                if orient == 'v':
                    label = ax.get_xticklabels()[i].get_text()
                else:
                    label = ax.get_yticklabels()[i].get_text()
                face_color = color_map[label]

            box.set_edgecolor(edge_color)
            box.set_facecolor(face_color)

            # iterate over whiskers and median lines
            for j in range(6 * i, 6 * (i + 1)):
                ax.lines[j].set_color(edge_color)

    if xrotation is not None:
        _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation)
    if yrotation is not None:
        _ = ax.set_yticklabels(ax.get_yticklabels(), rotation=yrotation)

    if fig is None:
        return ax
    else:
        return fig, ax