from math import pi
import palettable
import numpy as np

from bokeh.io import output_file, save
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar, ColumnDataSource, FixedTicker
from bokeh.plotting import figure


def cal_heat(df,
             save_path,
             ycolumn='dayofweek',
             xcolumn='weekof',
             value_column='tot_wgt',
             x_range=None,
             y_range=None,
             colors=None,
             vmin=None,
             vmax=None,
             bokehtools="hover,save,pan,box_zoom,reset,wheel_zoom",
             title="",
             plot_width=900,
             plot_height=500,
             colorbar_format="%d lbs",
             x_axis_location="above",
             toolbar_location='below',
             tooltips=None,
             label_font_size="10pt",
             xlabel=None,
             ylabel=None):
    """Creates a calendar heatmap with rows representing days of the week,
    columns representing weeks of the year, colored according to `value_column`

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
        colorbar_format:
        x_axis_location:
        toolbar_location:
        tooltips:
        label_font_size:
        xlabel:
        ylabel:

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

    xlabel = xcolumn if xlabel is None else xlabel
    ylabel = ycolumn if ylabel is None else ylabel

    if tooltips is not None:
        p = figure(
            title=title,
            x_range=x_range,
            y_range=list(reversed(y_range)),
            x_axis_location=x_axis_location,
            plot_width=plot_width,
            plot_height=plot_height,
            tools=bokehtools,
            toolbar_location=toolbar_location,
            tooltips=tooltips,
        )
    else:
        p = figure(
        title=title,
        x_range=x_range,
        y_range=list(reversed(y_range)),
        x_axis_location=x_axis_location,
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
    p.xaxis.major_label_orientation = pi / 3

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
                         label_standoff=20,
                         border_line_color=None,
                         location=(0, 0))
    p.add_layout(color_bar, 'right')

    save(p)
    return p