import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

from vishelper.config import formatting, cmaps
import vishelper.helpers as helpers


def heatmap(df, ax=None,
            label_size=None,
            xticklabels=None, yticklabels=None, log10=False,
            xrotation=90, yrotation=0, **kwargs):
    label_size = formatting['tick.labelsize'] if label_size is None else label_size

    fig, ax = helpers.get_ax_fig(ax, kwargs=kwargs)

    yticklabels = df.index.tolist() if yticklabels is None else yticklabels
    xticklabels = df.columns.tolist() if xticklabels is None else xticklabels

    if log10:
        log_norm = LogNorm(vmin=df.replace(0, 1).min(skipna=True).min(skipna=True),
                           vmax=df.replace(0, 1).max(skipna=True).max(skipna=True))

        cbar_ticks = [np.float_power(10, i) for i in
                      range(int(np.floor(np.log10(df.replace(0, 1).min(skipna=True).min(skipna=True)))),
                            1 + int(np.ceil(np.log10(df.replace(0, 1).max(skipna=True).max(skipna=True)))))]
        if 'cbar_kws' not in kwargs:
            kwargs['cbar_kws'] = {}
        kwargs["cbar_kws"]['ticks'] = cbar_ticks
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

    if fig is None:
        return ax
    else:
        return fig, ax
