import folium
import json
import vishelper as vh
import ipywidgets as widgets
import os
import logging
import matplotlib as mpl
import numpy as np

to_geo = lambda x: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "geo/")), x)
geos = dict(usstates=dict(path_to_geo=to_geo("us-states.json"), key_on='feature.properties.name'))

abspath_to_states = os.path.abspath(os.path.join(os.path.dirname(__file__), "geo/state-abbs.json"))
with open(to_geo("state-abbs.json"), 'r') as f:
    state_map = json.load(f)

geoformatting = {"zoom_start": 4,
                 "location_start": (38, -102),
                 "fill_color": 'GnBu',
                 "fill_opacity":1,
                 "line_opacity": 2,
                 "radius":15000,
                 "colors": vh.formatting["darks"],
                 "tiles":"OpenStreetMap"}


def to_state_name(df, column, new_column):
    df[new_column] = df[column].apply(lambda x: x if x not in state_map else state_map[x])
    non_states = list(set(df.fstate.unique()) - set(state_map.keys()))
    if len(non_states) > 0:
        logging.warning("The following values were not state names and were kept the same in the new column: %s" % ", ".join(non_states))
    return df


def basic_map(**kwargs):
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    fmap = folium.Map(location=getfmt("location_start"), zoom_start=getfmt("zoom_start"))
    return fmap


def plot_map(df, variable, geo_column, geo_type=None, path_to_geo=None, key_on=None, legend_name=None, reset=True, **kwargs):
    if geo_type is not None:
        path_to_geo = geos[geo_type]["path_to_geo"]
        key_on = geos[geo_type]["key_on"]

    if legend_name is None:
        legend_name = vh.labelfy(variable)

    fmap = basic_map(**kwargs)

    fmap.choropleth(
        geo_data=path_to_geo,
        data=df,
        columns=[geo_column, variable],
        key_on=key_on,
        reset=reset,
        fill_color=getfmt("fill_color"),
        fill_opacity=getfmt("fill_opacity"),
        line_opacity=getfmt("line_opacity"),
        legend_name=legend_name)
    fmap
    return fmap


def plot_subset(plot_function, df, option, option_column, kwargs):
    to_plot = df[df[option_column] == option]
    output = plot_function(to_plot, **kwargs)
    return output


def slider_interact(plot_function, df, options, option_column, initial_option=None, option_label=None,
                    disabled=False, button_style='', **kwargs):

    if option_label is None:
        option_label = vh.labelfy(option_column)
        
    option_slider = widgets.SelectionSlider(options=options,
                                            value=options[0] if initial_option is None else initial_option,
                                            description=option_label,
                                            disabled=disabled,
                                            button_style=button_style)

    widgets.interact(plot_subset, plot_function=widgets.fixed(plot_function), df=widgets.fixed(df),
                     option=option_slider, option_column=widgets.fixed(option_column),
                     kwargs=widgets.fixed(kwargs)
                     );


def color_continuous(df, color_col, clip=True, log10=False, cmap=None, **kwargs):
    vmin = df[color_col].min() if "vmin" not in kwargs else kwargs["vmin"]
    vmax = df[color_col].max() if "vmax" not in kwargs else kwargs["vmax"]
    cmap = mpl.cm.OrRd if cmap is None else cmap

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    if log10:
        df["color"] = df[color_col].apply(
            lambda x: "#%02x%02x%02x" % tuple(int(255*n) for n in mapper.to_rgba(np.log10(x))[:-1]))
    else:
        df["color"] = df[color_col].apply(
            lambda x: "#%02x%02x%02x" % tuple(int(255 * n) for n in mapper.to_rgba(x)[:-1]))
    return df


def color_categorical(df, color_col, colors=None):
    colors = vh.formatting["darks"] if colors is None else colors
    categories = df.color_col.unique()
    n = len(categories)
    if len(colors) < n:
        raise ValueError("There are %i unique values but only %i colors were given" % (n, len(colors)))
    color_map = dict(zip(categories, colors[:n]))

    df["color"] = df[color_col].apply(lambda x: color_map[x])

    return df


def add_lat_lng(df, lat_col="lat", lng_col="lng", color_col=None,
                color_how=None, colors=None, fmap=None, fill=True, **kwargs):
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    if color_col is not None:
        if color_how == "categorical":
            df = color_categorical(df, color_col, colors)
        elif color_how == "continuous":
            df = color_continuous(df, color_col, **kwargs)
        else:
            raise ValueError("Must specify color_how as 'categorical' or 'continuous' or change color_col to None")
    else:
        if colors is None:
            color = vh.formatting["darks"][0]
        elif type(colors) == list:
            color = colors[0]
        else:
            color = colors

    if fmap is None:
        fmap = folium.Map(location=getfmt("location_start"), zoom_start=getfmt("zoom_start"))

    for j in df.index:
        color = color if color_col is None else df.loc[j, "color"]
        folium.Circle(location=[df.loc[j, lat_col], df.loc[j, lng_col]], radius=getfmt("radius"),
                      fill=fill, color=color, fill_color=color).add_to(fmap)
    return fmap
