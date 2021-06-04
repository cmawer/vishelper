from math import pi
import palettable
import numpy as np

from bokeh.io import output_file, save
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar, ColumnDataSource, FixedTicker
from bokeh.plotting import figure


def interactive_heatmap(df,
             save_path,
             ycolumn='dayofweek',
             xcolumn='weekof',
             value_column='value',
             x_range=None,
             y_range=None,
             colors=None,
             vmin=None,
             vmax=None,
             bokehtools="hover,save,pan,box_zoom,reset,wheel_zoom",
             title="",
             plot_width=900,
             plot_height=500,
             min_border_right=0,
             colorbar_format="%d lbs",
             x_axis_location="above",
             y_axis_location='left',
             toolbar_location='below',
             colorbar_orientation='vertical',
             colorbar_place='right',
             tooltips=None,
             label_font_size="10pt",
             xlabel_orientation=None,
             colorbar_label_standoff=20,
             colorbar_major_label_text_align='center',
             xlabel="",
             ylabel=""):
    """Creates an interactive heatmap with tooltips

    Args:
        df:
        save_path (str): Where to save the output
        ycolumn (str): Which column in the dataframe represents the column that indicates which row of the
            heatmap (default: 'dayofweek')
        xcolumn (str): Which column in the dataframe represents the column that indicates which column of the
            heatmap (default: 'weekof')
        value_column (str): Which column in the dataframe the intersection of the row and column should be colored
            according to.
        x_range (`list` or similar): The possible row values (e.g. Monday, Tuesday..). Defaults to the
            unique set of values in the `xcolumn`
        y_range (`list` or similar): The possible column values (e.g. Week of Jan 1, Week of Jan 8, ...). Defaults to
            the unique set of values in the `ycolumn`
        colors: Color scale to use. Defaults to palettable.colorbrewer.sequential.BuGn_9.hex_colors
        vmin:
        vmax:
        bokehtools:
        title:
        plot_width:
        plot_height:
        min_border_right (int): Minimum border left between right side of image and border of figure. Default 0. It is
            recommended to change to ~80 when setting colorbar_orientation to horizontal to allow room for x-axis
            labels which are oriented pi/3
        colorbar_format:
        x_axis_location: which side to put the x-axis (column) labels. Default: 'above'. Options: 'above', 'below'
        y_axis_location: which side to put the y-axis (row) labels. Default: 'left'. Options: 'left', 'right'
        colorbar_orientation (str): How to orient the colorbar, 'vertical' or 'horizontal'. Default: 'vertical'
        colorbar_place (str, optional) : where to add the colorbar (default: 'right')
                Valid places are: 'left', 'right', 'above', 'below', 'center'.
        toolbar_location:
        tooltips:
        label_font_size:
        xlabel_orientation (float): Orientation of labels on x-axis. If left as None, default is pi/3
        colorbar_label_standoff (int): How much space to leave between colorbar and colorbar labels. Default 20. It is
            recommended to set to ~5 for vertical color bars.
        colorbar_major_label_text_align (`str`): How to align tick labels to ticks. Default 'center'.
        xlabel (str): Label for x-axis. Default=""
        ylabel (str): Label for y-axis. Default=""

    Returns:

    """
    if colors is None:
        colors = palettable.colorbrewer.sequential.BuGn_9.hex_colors
    if vmin is None:
        vmin = df[value_column].min()
    if vmax is None:
        vmax = df[value_column].max()
    mapper = LinearColorMapper(palette=colors, low=vmin, high=vmax)

    output_file(save_path)

    if x_range is None:
        x_range = np.sort(df[xcolumn].unique()).tolist()

    if y_range is None:
        y_range = np.sort(df[ycolumn].unique()).tolist()

    x_range = [str(x) for x in x_range]
    y_range = [str(y) for y in y_range]

    if tooltips is not None:
        p = figure(
            title=title,
            x_range=x_range,
            x_axis_label=xlabel,
            y_axis_label=ylabel,
            y_range=list(reversed(y_range)),
            x_axis_location=x_axis_location,
            y_axis_location=y_axis_location,
            plot_width=plot_width,
            plot_height=plot_height,
            tools=bokehtools,
            toolbar_location=toolbar_location,
            tooltips=tooltips,
            min_border_right=min_border_right
        )
    else:
        p = figure(
        title=title,
        x_range=x_range,
        y_range=list(reversed(y_range)),
        x_axis_location=x_axis_location,
        y_axis_location=y_axis_location,
        plot_width=plot_width,
        plot_height=plot_height,
        tools=bokehtools,
        toolbar_location=toolbar_location,
        )
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis[1].major_label_text_font_size = label_font_size
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3 if xlabel_orientation is None else xlabel_orientation

    source = ColumnDataSource(df)
    p.rect(x=xcolumn,
           y=ycolumn,
           width=1,
           height=1,
           source=source,
           fill_color={
               'field': value_column,
               'transform': mapper
           },
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper,
                         major_label_text_font_size=label_font_size,
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format=colorbar_format),
                         label_standoff=colorbar_label_standoff,
                         orientation=colorbar_orientation,
                         major_label_text_align=colorbar_major_label_text_align,
                         border_line_color=None,
                         location=(0, 0))  # color_bar must be placed at (0,0) so not configurable
    p.add_layout(color_bar, colorbar_place)

    save(p)
    return p