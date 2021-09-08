import numpy as np
import matplotlib.pyplot as plt

import vishelper.config as config


def get_formats(kwargs, *attributes):
    values = [config.formatting[a] if a not in kwargs else kwargs[a] for a in attributes]
    return values if len(values)>1 else values[0]


def get_ax_fig(ax, figsize=None, kwargs=None):
    if figsize is None and kwargs is not None:
        figsize = get_formats(kwargs, 'figsize')
    elif figsize is None:
        figsize = config.formatting['figure.figsize']

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    return fig, ax


def listify(l, multiplier=1, order=1):
    """Embeds a list in a list or replicates a list to meet shape requirements."""
    l = l if (isinstance(l, list) or isinstance(l, np.ndarray)) else [
                                                                         l] * multiplier

    while len(np.shape(l)) != order and not (isinstance(l[0], list) or isinstance(l[0], np.ndarray)):
        l = [l] * multiplier

    return l