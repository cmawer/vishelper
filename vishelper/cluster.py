import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from matplotlib import pyplot as plt

import vishelper as vh

mpl_update = {'font.size': 14, 'figure.figsize': [12.0, 8.0],
              'axes.labelsize': 20, 'axes.labelcolor': '#677385',
              'axes.titlesize': 20, 'lines.color': '#0055A7', 'lines.linewidth': 3,
              'text.color': '#677385'}

mpl.rcParams.update(mpl_update)

""" 
This file provides functions for visualizing principal components from PCA and
visualization of the features of clusters generated. For hierarchical
clustering, specifically, interactive_visualize() allows the visualization of
populations of the clusters that are formed at a number of cophenetic distances
(controlled by a slider) and the distribution of feature values of those
clusters.

The cophenetic distance is the "distance between the largest two clusters that
contain the two objects individually when they are merged into a single cluster
that contains both" (Wikipedia). By decreasing the cophenetic distance,
less points will be able to be merged together and more clusters will form.
Hierarchical clustering creates clusters up until the maximum cophenetic
distance and then that distance has to be decreased to break apart the cluster
until a chosen number of clusters is reached.

The metric used for the distance metric in the cophenetic distance is decided by 
the linkage type and distance metric used. 
"""


def reset_mpl_format():
    """Reapplies the matplotlib format update."""
    mpl.rcParams.update(mpl_update)


def heatmap_pca(V, normalize=True, n_feats=None, n_comps=None, cmap=None,
                feature_names=None, transpose=False, **kwargs):
    """
    Creates a heatmap of the composition of the principal components given
    by V. If normalize is left as default (True), the magnitude of the
    components in V will be normalized to give a percent composition of each
    feature in V.

    :param V: list of list. PCA components. N components x M features.
    :param normalize: optional boolean, default True, whether to normalize V
                      to relative weights.
    :param n_feats: optional int - number of features to include in figure.
    :param n_comps: optional int - number of components to include in figure.
    :param feature_names: optional list of strings to include feature names in
                          axis labels. Will default to 'Feature 1','Feature 2',
                          etc if not specified.

    Returns nothing, displays a figure.
     """

    if n_feats is None:
        n_feats = V.shape[1]

    if n_comps is None:
        n_comps = V.shape[0]

    if normalize:
        weights = [np.abs(v) / np.sum(np.abs(v)) for v in V]
    else:
        weights = V

    if feature_names is None:
        feature_names = ['Feature ' + str(j) for j in range(1, n_feats+1)]

    component_names = ['Component ' + str(j) for j in range(1, n_comps+1)]

    to_plot = pd.DataFrame(weights[:n_comps], index=component_names, columns=feature_names)
    if transpose:
        to_plot = to_plot.T

    if cmap is None:
        cmap = vh.cmaps["blues"]

    fig, ax = vh.heatmap(to_plot, cmap=cmap,
                         xlabel="Principal Component" if transpose else "Feature",
                         ylabel="Feature" if transpose else "Principal Component",
                         title="Relative weight of features' contribution to principal components", **kwargs
                         )
    return fig, ax, to_plot


def dendrogram(z, xlabel='Observations', thresh_factor=0.5, remove_ticks=False, **kwargs):
    """
    Creates a dendrogram from . Colors clusters below thresh_
    factor*max(cophenetic distance).

    :param z: The hierarchical clustering encoded with the matrix returned by Scipy's linkage function.
    :param xlabel: String for xlabel of figure
    :param thresh_factor: Colors clusters according those formed by cutting the dendrogram at
                          thresh_factor*max(cophenetic distance)

    Returns: R (see Scipy dendrogram docs).
    Displays: Dendrogram.
    """

    color_thresh = thresh_factor * max(z[:, 2])
    reset_mpl_format()
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize')
    else:
        figsize = (12, 12)

    plt.figure(figsize=figsize)
    R = hac.dendrogram(z, color_threshold=color_thresh, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel('Cophenetic distance')
    if remove_ticks:
        plt.xticks([])
    return R


def create_labels(z, features_in, levels, criteria='distance', feature_names=None):
    """
    Labels each observation according to what cluster number it would fall into under the specified
    criteria and level(s).

    :param z: The hierarchical clustering encoded with the matrix returned by Scipy's linkage function.
    :param features_in: list of features for each sample or dataframe
    :param levels: list of different levels to label samples according to. Will depend on criteria used.
                   If criteria = 'distance', clusters are formed so that observations in a given cluster
                   have a cophenetic distance no greater than the level(s) provided. If criteria =
                   'maxcluster', cuts the dendrogram so that no more than the level number of clusters
                   is formed. See
                   http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
                   for more options.
    :criteria: string referring to which criterion to use for creating clusters. "distance" and "maxclusters"
               are two more commonly used. See param levels above. Other options can be found at
               http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    :feature_names: list of labels corresponding to the features to create pandas dataframe for output if
                   dataframe is not provided.

    Returns: features: Pandas dataframe with feature values for each observation as well as assigned cluster
             number for each specified level. Cluster assignment columns are labeled by str(level).
    """
    if isinstance(features_in, pd.DataFrame):
        features = features_in.copy()
    else:
        features = pd.DataFrame(features_in, columns=feature_names)

    for j in levels:
        y = hac.fcluster(z, j, criterion=criteria)
        features[str(j)] = y

    return features


def separate(features, level, feature_name, minimum_population=10):
    """
    Separates features into lists based on their cluster label and separates
    out clusters less than the minimum population in size as outliers.

    :param features: Pandas dataframe with rows for each observation, columns for each feature value and
                     cluster label for each level (see create_labels()).
    :param level: Level at which you want to separate observations into groups.
    :param feature_name: Desired feature for grouping.
    :param minimum_population: Minimum population for which a labeled cluster can be considered a full
                               cluster. Any cluster which has a lower population will be considered
                               a group of outliers.
    Returns: sep_features = list of list of feature values for each labeled group greater
                            than min_population in size.
             outliers = feature values for any observations in cluster which has a size smaller than
                        minimum population.
     """

    sep_features = []
    outliers = []

    for j in range(1, features[str(level)].max() + 1):

        sep_feature = features[features[str(level)] == j][feature_name].tolist()

        if len(sep_feature) > minimum_population:
            sep_features.append(sep_feature)
        else:
            outliers += sep_feature

    return sep_features, outliers


def visualize_clusters(features, level, feature_names, bins=20, xlim=None, ylim=None, log=False):
    """ Plots a histogram of the number of samples assigned to each cluster at
    a given cophentic distance and the distribution of the features for each
    cluster. This assumes labels exist in the features dataframe in
    column str(level).  """

    reset_mpl_format()
    # sns.set_palette('deep')
    plt.figure(figsize=(16, 16))
    n_rows = int(np.ceil(float(len(feature_names)) / 2)) + 1
    plt.subplot(n_rows, 1, 1)

    plt.hist(features[str(level)].values, features[str(level)].max())
    plt.xlabel('Cluster number')
    plt.title('level = ' + str(level))

    for j, feat in enumerate(feature_names):

        plt.subplot(n_rows, 2, j + 3)

        sep_features, outliers = separate(features, level, feat)

        labels = ['Cluster ' + str(j) for j in range(1, len(sep_features) + 1)]

        if len(outliers) > 1:
            sep_features.append(outliers)
            labels.append('Outliers')

        plt.hist(sep_features, bins, stacked=True,
                 label=labels);
        if log:
            plt.xlabel('Log10 of ' + feat)
        else:
            plt.xlabel(feat)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()

