import logging

import numpy as np

import vishelper.helpers as helpers
from vishelper.config import formatting

logger = logging.getLogger(__name__)


def hist(x, ax=None, color=None, logx=False, ignore_nan=True, **kwargs):
    """Plots a histogram based on values of `x`"""
    if ignore_nan:
        if "stacked" not in kwargs or not kwargs["stacked"]:
            original_len = len(x)
            x = [xi for xi in x if not np.isnan(xi)]
            final_len = len(x)
            if final_len != original_len:
                diff = original_len - final_len
                logger.warning("A total of %i NaN values out of %i "
                               "observations were removed", diff, original_len)
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
                    logger.warning(
                        "A total of %i NaN values out of %i observations were"
                        " removed from set %i",
                        diff, original_len, j)

    fig, ax = helpers.get_ax_fig(ax, kwargs=kwargs)

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
        color = formatting['color.single']
    if "stacked" in kwargs and kwargs["stacked"]:
        if len(color) != len(x):
            color = formatting['color.darks'][:len(x)]

    ax.hist(x, color=color, alpha=alpha, **kwargs)

    if fig is None:
        return ax
    else:
        return fig, ax
