import seaborn as sns

import vishelper.helpers as helpers


def boxplot(x, y, ax=None, palette="Set3", data=None, hue=None, white=False, color_map=None, xrotation=90, **kwargs):

    fig, ax = helpers.get_ax_fig(ax, kwargs=kwargs)

    if "label" in kwargs:
        kwargs.pop("label")

    if data is not None and type(x) == list:
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