import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy
import matplotlib.cm as cm

import vishelper.format as format
import vishelper.config as config
import vishelper.plot as plot
import vishelper.helpers as helpers

class VisDF:
    """ Easily create typical visualizations of data in a pandas dataframe.

            This class performs a number of plotting tasks for pandas data frames
            including easy sub-plotting and consistent axes labeling.

            Args:
                df (pandas dataframe): Data to be plotted
                column_labels (:obj:`dict`): Dictionary containing mappings of
                    columns of dataframe and corresponding labels to be
                    used instead when plotting for axes labels and legend.
                    If None, column names  will have '_' removed and the first
                    letter capitalized (e.g. "path_length" --> "Path length")
                cluster_label (:obj:`str`, optional): If it exists, the name of
                    the column that gives cluster or group assignments (and is
                    not a feature).
                index (:obj:`str` or :obj:`list` of :obj:`str`, optional):
                    Name of identifying column such as customer id or transaction
                    id or other column that should not be analyzed.
                numeric_columns (:obj:`list` of :obj:`str`, optional): List of
                    column names corresponding to numeric fields. If not provided,
                    this list will be assessed by data type of each column and
                    will exclude the index and cluster label, if given.
                nonnumeric_columns (:obj:`list` of :obj:`str`, optional): List of
                    column names corresponding to non-numeric fields. If not provided,
                    this list will be assessed by data type of each column and
                    will exclude the index and cluster label, if given.
                color_dict (:obj:`dict`): Optional. Keys correspond to categorical
                    columns where consistent coloring by category is desired.
                    Each key has a dictionary as it's value with the category names
                    and corresponding colors to use. Dictionary structure is:
                    {'column_name':{'category1': '#colorx', 'category2': '#colory'}}
                columns_to_color (:obj:`list` of :obj:`str`, optional): List of
                    names of categorical columns to apply consistent coloring to.
                    Colors to assign to each category will be provided by the
                    `colors` attribute.
                colors (:obj:`list`): List of colors to cycle through in plotting (if
                    None provided, will use defaults defined in config file).
                univariate_ylabels (:obj:`dict`): Dictionary of univariate plot
                    types and corresponding y-labels to use. Default is
                    dict(hist='Count', barh='Count').
                scale (:bool:): Default True. If True, scales the numeric columns
                    of the dataframe and stores them in the `scaled` attribute.
                pca (:bool:): Default True. If True, calculates the  principal
                    components of the numeric data and stores them in the
                    `pca` attribute.
            """

    def __init__(self, df, column_labels=None, labels=None, cluster_label=None,
                 index=None, numeric_columns=None, nonnumeric_columns=None,
                 color_dict=None, columns_to_color=None,
                 colors=None, univariate_ylabels=None,
                 scale=False, pca=False):

        self.df = df
        if colors is None:
            colors = config.formatting['color.darks']

        if column_labels is None:
            column_labels = dict(zip(df.columns, format.labelfy(df.columns)))

        self.column_labels = column_labels

        self.color_dict = {} if color_dict is None else color_dict

        if columns_to_color is not None:
            for column_name in columns_to_color:
                self.add_color_dict(column_name)

        self.univariate_ylabels = dict(hist='Count', barh='Count') \
            if univariate_ylabels is None \
            else univariate_ylabels

        self.index = index
        index_column = self.df.index if self.index is None else self.df[index]

        self.cluster_label = cluster_label

        self.nonnumeric_columns = [col for col in list(df.select_dtypes(
            exclude=[np.number]).columns.values) if ((col != self.index) and (col != self.cluster_label))]

        if nonnumeric_columns is not None:
            self.nonnumeric_columns = list(set(self.nonnumeric_columns + nonnumeric_columns))
        if numeric_columns is None:
            self.numeric_columns = [col for col in list(df.select_dtypes(
                include=[np.number]).columns.values) if ((col != self.index) and (col != self.cluster_label) and (col not in self.nonnumeric_columns))]
        else:
            self.numeric_columns = numeric_columns

        if scale or pca:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.df[self.numeric_columns])
            scaled_data = pd.DataFrame(scaled_data,
                                       columns=self.numeric_columns,
                                       index=index_column)
            if cluster_label is not None:
                scaled_data[cluster_label] = self.df[cluster_label].values
        if pca:
            pca_model = PCA()
            pca_data = pd.DataFrame(pca_model.fit_transform(scaled_data[self.numeric_columns]))
            if cluster_label is not None:
                pca_data[cluster_label] = self.df[cluster_label].values

        self.scaled = None if not scale else scaled_data
        """Pandas dataframe of numeric data scaled. """

        self.pca = None if not pca else pca_data
        """Pandas dataframe of the principal components of the numeric
        columns of the data."""

        self.logger = logging.getLogger(__name__)

    def to_labels(self, columns):
        """ Convert column names to figure labels.

        Converts a list of column names into labels based on either the
        provided dictionary `column_labels` or if not provided, based on the
        `labelfy` function which replaces undersores with spaces and
        capitalizes the first letter of the string.

        Args:
            columns (:obj:`list` of :obj:`str`): List of column names to convert
                to labels.

        Returns:
            labels (:obj:`list` of :obj:`str`): List of labels corresponding to
                the provdied column names.


        """

        labels = [self.column_labels[col] if col in self.column_labels.keys() else format.labelfy(col) for col in columns]

        return labels

    def add_color_dict(self, column_name, colors=None):
        """ Assign colors to categories within a defined column of the data.

        Args:
            column_name (:obj:`str`): Name of categorical column in the data.
            colors (:obj:`list` of :obj:`str`): Optional. Colors to be assigned
                to the categories. If not provided, will use `self.colors`.

        Returns: Nothing

        """
        colors = config.formatting["darks"] if colors is None else colors

        categories = self.df[column_name].unique()

        num_cats = len(categories)

        if num_cats > len(colors):
            raise ValueError('More categories than colors provided')
        else:
            self.color_dict[column_name] = dict(zip(categories,
                                                    colors[:len(categories)]))

    def dict_to_colors(self, column_name, df=None):
        if column_name not in self.color_dict:
            self.add_color_dict(column_name)
        if df is None:
            df = self.df
        return [self.color_dict[column_name][j] for j in df[column_name]]

    def subplots(self, columns_to_plot, kind, color_by=None, sort_by=None, ascending=False,
                 layout=None, titles=None, main_title=None, xlim=None, ylim=None,
                 legend_labels=None, legend_order=None, figsize=(16, 10),
                 counts=False, top_counts=None, min_counts=None,
                 **kwargs):
        """Create figure with many subplots at once from dataframe.

        This method will create a figure with subplots based on the column names inputted.

        Args:
            columns_to_plot (:obj:`list` of [:obj:`str` or :obj:`lists`): The
                column(s) to plot in each figure.

                *Univariate only*: If only plotting univariate plots,
                    `columns_to_plot` will look like
                    ['column1','column2',..., 'columnN']  where 'column1' will
                    be plot in figure 1, 'column2' in figure 2, etc.

                **Bivariate only*: If only plotting bivariate plots,
                    `columns_to_plot` will look like
                    [['columnx1', 'columny1'],['columnx2', 'columny2], ..., ]
                    where 'columnx1' will be plotted vs 'columny1' in figure 1.

                *Mix of univariate and bivariate*: If plotting a mix of plot
                    types, `columns_to_plot` will look something like:
                    ['column1', ['columnx2', 'columny2'], 'column3',...]. Note
                    that  this will require a mix of plot types and currently
                    **kwargs cannot be provided that don't work in all
                    plot functions.

            kind (:obj:`str` or :obj:`list` of :obj:`str`): What type of plot
                to plot. If a string, the plot type is used for each subplot.
                If a list, it should be the same length as
                :py:attr:`.columns_to_plot` and describe what type of
                plot to use in each subplot.
            layout (:obj:`tuple`, optional):  # of rows x # columns. If not
                given, the layout will default to N x 2 where N is calculated
                based on length of `columns_to_plot`
            main_title (:obj:`str`, optional): Optional, title for the entire figure.
            titles (:obj:`list` of :obj:`str, optional): List of titles for each subplot
                corresponding to the order of `columns_to_plot`
            counts (:bool:): If True, plot :py:meth:`pd.DataFrame.value_counts()`
                is plotted rather than the data frame data. This is typically
                used for categorical fields.
            top_counts (:int:, optional): If `counts` is True and this argument
                is provided, only the first `top_counts` number of rows from the
                :py:meth:`pd.DataFrame.value_counts()` data when the dataframe
                is sorted from highest to lowest counts.
            min_counts (:int:, optional): If `counts` is True and this argument
                is provided, only the first `min_counts` number of rows from the
                :py:meth:`pd.DataFrame.value_counts()` data when the dataframe
                is sorted from lowest to highest counts.
            **kwargs: Any other key word arguments will be fed into the
                plotting function and should be arguments to the core
                :python:mod:`matplotlib` plotting function (e.g. `bin` for
                histograms). Currently cannot feed arguments that don't apply
                to all plot kinds being used (as they are automatically filled in).

                TO DO: something to allow for keyword arguments to only be fed
                functions that they apply to



        Returns:
            fig: :python:obj:matplotlib.figure.Figure`
            axes: :python:obj:`numpy.ndarray` of :python:obj:`numpy.ndarray` of :python:obj:`matplotlib.axes._subplots.AxesSubplot`

        """

        if sort_by is not None:
            df_to_plot = self.df.copy().sort_values(by=sort_by, ascending=ascending)

        else:
            df_to_plot = self.df.copy()
        if columns_to_plot is not None:
            num_plots = len(columns_to_plot)
        else:
            raise ValueError('No columns_to_plot given')

        if layout is None:
            if num_plots % 2 != 0:
                layout = (num_plots, 1)
            else:
                layout = (int(np.ceil(num_plots / 2)), 2)
        elif len(layout) != 2:
            raise ValueError('please provide layout as (n_rows, n_cols)')
        elif layout[0] * layout[1] < num_plots:
            raise ValueError(
                'layout provide does not have enough space for all desired plots')

        # Add colors to kwargs if column to color by provided
        if color_by is not None:
            kwargs["color"] = self.dict_to_colors(color_by, df_to_plot)

        fig, axes = plt.subplots(layout[0], layout[1],
                                 figsize=figsize)

        axes_list = axes.ravel() if num_plots > 1 else [axes]

        for j, (ax, cols) in enumerate(zip(axes_list, columns_to_plot)):
            kind_j = kind if isinstance(kind, str) else kind[j]
            xlimi = xlim if type(xlim) != list else xlim[j]
            ylimi = ylim if type(ylim) != list else ylim[j]

            if counts:
                count_df = self.df[cols].value_counts()

                x = count_df.index.tolist()
                y = count_df.values.tolist()

                if top_counts is not None:
                    x = x[:top_counts]
                    y = y[:top_counts]
                elif min_counts is not None:
                    x = x[-min_counts:]
                    y = y[-min_counts:]

                ax = plot(x, y, ax=axes_list[j],
                             kind='barh',
                             **kwargs)
                ax = format.add_labels(ylabel=self.column_labels[cols],
                                   xlabel=self.univariate_ylabels['barh'],
                                   title=None if titles is None else
                                   titles[j],
                                   ax=ax)
                # TO DO: add in color based on color_dict, add multi-line plotting...
            elif isinstance(cols, list) and len(cols) > 1:

                ax = plot(df_to_plot[cols[0]], df_to_plot[cols[1]], ax=ax,
                             kind=kind_j,
                             **kwargs)
                ax = format.add_labels(xlabel=self.column_labels[cols[0 if kind_j != 'barh' else 1]],
                                   ylabel=self.column_labels[cols[1 if kind_j != 'barh' else 0]],
                                   title=None if titles is None else
                                   titles[j],
                                   ax=ax)
            else:
                cols = cols[0] if isinstance(cols, list) else cols
                ax = plot(df_to_plot[cols], ax=ax,
                             kind=kind_j,
                             **kwargs)
                ax = format.add_labels(xlabel=self.column_labels[cols],
                                   ylabel=self.univariate_ylabels[kind_j],
                                   title=None if titles is None else
                                   titles[j],
                                   ax=ax)

            ax = format.adjust_lims(ax, xlimi, ylimi)

        if color_by is not None:

            if legend_order is None:
                labels = list(self.color_dict[color_by].keys())
            else:
                labels = legend_order

            colors = [self.color_dict[color_by][l] for l in labels]
            if legend_labels is not None:
                labels = [legend_labels[l] for l in labels]

            ax = format.fake_legend(ax, labels, colors)
        # TO DO: Add in functionality to pass dictionaries for each figure
        # instead of columns_to_plot
        # for j, (ax, plot_dict) in enumerate(zip(axes_list, plot_dicts)):
        #                 if 'kind' in plot_dict.keys():
        #                     k = plot_dict.pop('kind')
        #                 elif isinstance(kind, str):
        #                     k = kind
        #                 else:
        #                     k = kind[j]

        #                 if 'y' is in plot_dict.keys():
        #                     ax = plot([df[onex] for onex in plot_dict[x]],
        #                                  y=[df[oney] for oney in plot_dict[y]],
        #                                  kind=k, ax=ax)

        sup_x = config.formatting['suptitle.x'] \
            if 'sup_x' not in kwargs.keys() \
            else kwargs.pop('sup_x')
        sup_y = config.formatting['suptitle.y'] \
            if 'sup_y' not in kwargs.keys() \
            else kwargs.pop('sup_y')

        if main_title:
            plt.suptitle(main_title, x=sup_x, y=sup_y,
                         size=config.formatting['suptitle.size'])
        fig.tight_layout()
        return fig, axes

    def fscore_by_feature(self, category_column):
        """ Prioritize categorical x continuous interactions to investigate.

        The F-test in one-way analysis of variance is used to assess whether
        the expected values of the variable within the categories in the
        category column differ from each other. A higher f-value or lower
        p-value indicates a bigger difference between the categories in
        a given variable.

        This is **not** meant to be used to make any statistically validated
        claims about the variables as no assumptions have been considered.
        Moreover, no adjustment has been made for making doing multiple
        tests. This should *only * be used as a directional signal of
        which variables may interact most with the categorical column
        provided and which should be prioritized for visual investigation.

        Args:
            category: Column name of which variable to group observations by
             and compare distributes of variables across.

        Returns:
            fps: :python:obj:`pandas.core.frame.DataFrame` of variables and
                 corresponding f-score and p-value.

        """
        fps = []
        for var in self.numeric_columns:
            gb = self.df[[var, category_column]].dropna().groupby(category_column)
            f, p = scipy.stats.f_oneway(*gb[var].apply(list).values.tolist())
            fps.append([var, f, p])

        fps = pd.DataFrame(fps, columns=['variable', 'f', 'p']) \
            .sort_values('f', ascending=False)

        return fps

    def percent_above_below(self, category, quantile_threshold=0.5,
                            how='above', exclude=None, category_dict=None,
                            transpose=False, cluster=True, variables=None,
                            xlabel=' ', ylabel=' ', cat_order=None,
                            **kwargs):
        """

        Args:
            category:
            quantile_threshold:
            how:
            exclude:
            category_dict:
            transpose:
            cluster:
            variables:
            xlabel:
            ylabel:
            cat_order:
            **kwargs:

        Returns:

        """
        if exclude is None:
            exclude = [category]
        else:
            exclude = helpers.listify(exclude).append(category)
        # define the threshold
        threshold = self.df.quantile(q=quantile_threshold)
        # compute the
        if variables is None:
            comp_df = self.df.copy()
        else:
            comp_df = self.df[variables + [category]].copy()
        comp_df = comp_df.groupby(category).apply(
            lambda g: frac_outside_threshold(g, threshold, exclude=exclude)).T

        if cat_order is not None:
            comp_df = comp_df[cat_order]

        if 'title' not in kwargs.keys():
            group_name = 'cluster' if cluster else 'group'
            kwargs['title'] = 'Percent of %s %s median of population' % (group_name, how)
        category_labels = [category_dict[str(cat)] if category_dict is not None else str(cat) for cat in
                           comp_df.columns.tolist()]
        feature_labels = self.to_labels(comp_df.index.tolist())[::-1]

        if transpose:
            comp_df = comp_df.T

        fig, ax = plot(df=comp_df, kind='heatmap', xlabel=xlabel, ylabel=ylabel,
                             xticklabels=feature_labels if transpose else category_labels,
                             yticklabels=category_labels if transpose else feature_labels,
                             **kwargs)
        ax = format.add_labels(ax, xlabel=xlabel, ylabel=ylabel)

        return fig, ax

    def compare_categories(self, category, variables=None, measure=np.mean):
        if variables is None:
            variables = self.numeric_columns
        variables = [var for var in variables if var != category]
        if measure == np.median:
            population = self.df[variables].median()
            groups = self.df.groupby(category)[variables].median()
        else:
            population = self.df[variables].apply(measure)
            groups = self.df.groupby(category)[variables].apply(measure)
        lift = groups.div(population)
        percent_increase = (groups - population).div(population)
        absolute_difference = groups - population
        return dict(lift=lift,
                    percent_increase=percent_increase,
                    absolute_difference=absolute_difference,
                    actual=groups)

    def category_heatmap(self, category, variables=None, transpose=False,
                         measure=np.mean, metric='actual', category_dict=None,
                         xlabel=None, ylabel=None, cat_order=None, log10=False, **kwargs):
        comp_df = self.compare_categories(category, variables=variables,
                                          measure=measure)[metric]
        if cat_order is not None:
            comp_df = comp_df.loc[cat_order, :]
        category_labels = [category_dict[cat] if category_dict is not None else cat for cat in comp_df.index.tolist()]
        feature_labels = self.to_labels(comp_df.columns.tolist())

        if transpose:
            comp_df = comp_df.T

        if 'title' not in kwargs.keys():
            kwargs['title'] = metric.capitalize()

        fig, ax = plot(df=comp_df, kind='heatmap', xlabel=xlabel, ylabel=ylabel, log10=log10,
                             xticklabels=category_labels if transpose else feature_labels,
                             yticklabels=feature_labels if transpose else category_labels,
                             **kwargs)

        ax = format.add_labels(ax, xlabel=xlabel, ylabel=ylabel)
        return fig, ax

    def labeled_scatter(self, category=None, x=None, y=None, pca=True, **kwargs):
        '''Method for visualizing clusters in 2D.'''
        if category is None:
            category = self.cluster_label

        if pca:
            plot_df = self.pca.copy()
            x = 0
            y = 1
        else:
            plot_df = self.df.copy()

        n_clusters = self.df[category].nunique()
        if n_clusters <= len(config.formatting['color.darks']):
            color_dict = dict(zip(self.df[category].unique(), config.formatting['color.darks'][:n_clusters]))
            colors = [color_dict[label] for label in self.df[category].values]
        else:
            colors = cm.spectral(self.df[category].astype(float) / n_clusters)

        fig, ax = plot(plot_df.loc[:, x], plot_df.loc[:, y],
                          kind='scatter', color=colors, **kwargs)
        # determine the cluster centers
        centers = np.zeros((n_clusters, 2))
        for i, label in enumerate(plot_df[category].unique()):
            centers[i] = np.mean(plot_df[plot_df[category] == label].loc[:, [x, y]], axis=0)
        # Draw white circles at cluster centers
        ax.scatter(centers[:, 0], centers[:, 1],
                   marker='o', c="white", alpha=1, s=400)
        for i, c in enumerate(centers):
            ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=200, c='r')
        return fig, ax

    # def boxplot(self, variable, category, grid=False, xlim=None, ylim=None, **kwargs):
    #     figsize = config.formatting["figure.figsize"] if "figsize" not in kwargs else kwargs.pop("figsize")
    #     fig, ax = plt.subplots(figsize=figsize)
    #     subset = [variable] if type(variable) == str else variable
    #     subset += [category]
    #     # ax = self.df[subset].boxplot(by=category, ax=ax, grid=grid)
    #
    #     ylabel = category if category not in self.column_labels else self.column_labels[category]
    #     if type(ax) == np.ndarray:
    #         axes_list = ax.ravel()
    #         xlim = helpers.listify(xlim, order=2 if xlim is not None else 1, multiplier=np.shape(variable)[0])
    #         ylim = helpers.listify(ylim, order=2 if ylim is not None else 1, multiplier=np.shape(variable)[0])
    #
    #         for j, a in enumerate(axes_list):
    #             xlabel = variable[j] if variable[j] not in self.column_labels else self.column_labels[variable[j]]
    #             ax = self.df[[variable, category]].boxplot(by=category, ax=ax, grid=grid)
    #             a = format.add_labels(a, xlabel=xlabel, ylabel=ylabel, **kwargs)
    #             a.tick_params(labelsize=config.formatting['tick.labelsize'],
    #                        size=config.formatting['tick.size'])
    #
    #             a = format.adjust_lims(a, xlim[j], ylim[j])
    #     else:
    #         xlabel = variable[j] if variable not in self.column_labels else self.column_labels[variable]
    #         ax = format.add_labels(ax, xlabel=xlabel, ylabel=ylabel, main_title=" ", **kwargs)
    #         ax = format.adjust_lims(ax, xlim, ylim)
    #         ax.tick_params(labelsize=config.formatting['tick.labelsize'],
    #                        size=config.formatting['tick.size'])
    #     return ax


def frac_outside_threshold(group, thresholds, how='above', exclude=[]):
    """Helper function for computing the """
    columns = []
    results = []
    for column in group.columns:
        if column not in exclude + ['cluster_label']:
            columns.append(column)
            if how == 'above':
                results.append(
                    group[group[column] > thresholds[column]].shape[0] /
                    group.shape[0])
            elif how == 'below':
                results.append(
                    group[group[column] < thresholds[column]].shape[0] /
                    group.shape[0])
    return pd.Series(data=results, index=columns)
