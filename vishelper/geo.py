import folium
import json
import vishelper as vh
import ipywidgets as widgets
import os
import logging
import matplotlib as mpl
import numpy as np
from selenium import webdriver
import time

to_geo = lambda x: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "geo/")), x)
geos = dict(usstates=dict(path_to_geo=to_geo("us-states.json"), key_on='feature.properties.name'))

abspath_to_states = os.path.abspath(os.path.join(os.path.dirname(__file__), "geo/state-abbs.json"))
with open(to_geo("state-abbs.json"), 'r') as f:
    state_map = json.load(f)

geoformatting = {"zoom_start": 4,
                 "location_start": [38, -96.5],
                 "fill_color": 'GnBu',
                 "fill_opacity":1,
                 "line_opacity": 2,
                 "opacity":1,
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
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]
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


def add_latlons(latlons, color=None, fmap=None, fill=True, lines=False, line_color="black", **kwargs):
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    if fmap is None:
        fmap = folium.Map(location=getfmt("location_start"), zoom_start=getfmt("zoom_start"))

    if len(np.shape(latlons)) == 1:
        latlons = [latlons] * len(latlons)

    assert np.shape(latlons)[1] == 2

    color = vh.formatting["darks"][0] if color is None else color

    if type(color) != list:
        color = [color] * len(latlons)

    assert len(color) == len(latlons)

    for latlon, col in zip(latlons, color):

        folium.Circle(location=latlon, radius=getfmt("radius"),
                      fill=fill, color=col, fill_color=col).add_to(fmap)
    if lines:
        if type(line_color) != list:
            line_color = [line_color] * (len(latlons) - 1)
        assert len(latlon) > 1
        assert len(line_color) == len(latlons) - 1
        for latlonA, latlonB, lcolor in zip(latlons[:-1], latlons[:-1], line_color):
            fmap = add_line(latlonA, latlonB, line_color=line_color, fmap=fmap)

    return fmap


def add_df_latlon(df, lat_col="lat", lng_col="lng", color_col=None, fill_opacity=1, radius=15000,
                color_how=None, colors=None, fmap=None, fill=True, color_continuous_kwargs=None, **kwargs):
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    if color_col is not None:
        if color_how == "categorical":
            df = vh.color_categorical(df, color_col, colors)
        elif color_how == "continuous":
            color_continuous_kwargs = {} if color_continuous_kwargs is None else color_continuous_kwargs
            df = vh.color_continuous(df, color_col, **kwargs)
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

    for i, j in enumerate(df.index):
        r = radius if type(radius) != list and type(radius) != np.ndarray else radius[i]
        color = color if color_col is None else df.loc[j, "color"]
        folium.Circle(location=[df.loc[j, lat_col], df.loc[j, lng_col]], radius=r,
                      fill=fill, color=color, fill_color=color, fill_opacity=fill_opacity, **kwargs).add_to(fmap)
    return fmap


def add_line(latlonA, latlonB, line_color="black", fmap=None, **kwargs):
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    if fmap is None:
        fmap = folium.Map(location=getfmt("location_start"), zoom_start=getfmt("zoom_start"))

    folium.PolyLine([[latlonA, latlonB]], color=line_color, **kwargs).add_to(fmap)

    return fmap


def html_to_png(htmlpath=None, pngpath=None, delay=5, width=2560, ratio=0.5625):
    if htmlpath is None and pngpath is None:
        raise ValueError("Must give at least htmlpath or pngpath")
    elif htmlpath is None:
        htmlpath = pngpath.split(".")[0] + ".html"
    else:
        pngpath = htmlpath.split(".")[0] + ".png"
    tmpurl = 'file://{htmlpath}'.format(htmlpath=htmlpath)
    browser = webdriver.Safari()
    browser.set_window_size(width, width * ratio)

    browser.get(tmpurl)
    # Give the map tiles some time to load
    time.sleep(delay)
    browser.save_screenshot(pngpath)
    browser.quit()


def save_map(fmap, htmlpath=None, pngpath=None, delay=5, width=2560, ratio=0.5625):
    if htmlpath is None and pngpath is None:
        raise ValueError("Must give at least htmlpath or pngpath")
    elif htmlpath is None:
        htmlpath = pngpath.split(".")[0] + ".html"
    else:
        pngpath = htmlpath.split(".")[0] + ".png"

    fmap.save(htmlpath)

    html_to_png(htmlpath=htmlpath, pngpath=pngpath, delay=delay, width=width, ratio=ratio)
