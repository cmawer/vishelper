import json
import logging
import os
import time

import folium
import ipywidgets as widgets
import numpy as np
import requests

import vishelper as vh

logger = logging.getLogger(__name__)
to_geo_dir = lambda x: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "data/")), x)

geos = dict(usstates=dict(geo_data=to_geo_dir("us-states.json"), key_on='feature.properties.name'),
            us_states=dict(geo_data=to_geo_dir("us-states.json"), key_on='feature.properties.name')
            )

abspath_to_states = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/state-abbs.json"))
with open(to_geo_dir("state-abbs.json"), 'r') as f:
    state_map = json.load(f)

geoformatting = {"zoom_start": 4,
                 "location_start": [38, -96.5],
                 "fill_color": 'GnBu',
                 "fill_color_single": vh.formatting["darks"][0],
                 "fill_opacity": 1,
                 "line_opacity": 1,
                 "opacity":1,
                 "radius":15000,
                 "colors": vh.formatting["darks"],
                 "tiles":"CartoDB positron"}


def to_state_name(df, state_column, new_column):
    """Creates a column with the state names corresponding to the abbreviations in the state_column provided.

    Args:
        df: Dataframe containing `column` name
        state_column: Name of column containing state abbreviations
        new_column: Name of column to create with state names

    Returns:
        df: Dataframe now containing `new_column`

    """
    df[new_column] = df[state_column].apply(lambda x: x if x not in state_map else state_map[x])
    non_states = list(set(df[state_column].unique()) - set(state_map.keys()))
    if len(non_states) > 0:
        logging.warning("The following values were not state names and were kept the same in the new column: %s" % ", ".join(non_states))
    return df


def basic_map(**kwargs):
    """

    Args:
        **kwargs:
            location_start (tuple): Lat/lon to center map on to begin with. Default given by vh.geo_formatting.
            zoom_start (int): Level of zoom, 1-10. Default given by vh.geo_formatting
            tiles (`str`): Name of folium tiles to use. Default given by vh.geo_formatting.

    Returns:
        fmap (folium.Map): folium map object

    """
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    fmap = folium.Map(location=getfmt("location_start"), zoom_start=getfmt("zoom_start"), tiles=getfmt("tiles"))
    return fmap


def filter_geojson(geojson_dict, df, geo_col, geojson_key='ZCTA5CE10', filter_geos=True):

    # We want to only include features that are in our dataset so we clear the features key and rebuild it.
    filtered_geojson_dict = geojson_dict.copy()
    filtered_geojson_dict['features'] = []
    geo_values = df[geo_col].unique()
    included_geos = []

    for feat in geojson_dict['features']:
        geojson_val = feat['properties'][geojson_key]
        if (filter_geos and geojson_val in geo_values) or not filter_geos:
            filtered_geojson_dict['features'].append(feat)
            included_geos.append(geojson_val)

    filtered_df = df[df[geo_col].isin(included_geos)]
    not_included_geos = df[~df[geo_col].isin(included_geos)][geo_col].unique()
    if len(not_included_geos) > 0:
        if len(not_included_geos) > 20:
            logger.warning('Over 20 geos were not in the provided geojson and will not be plotted.')
        else:
            logger.warning("The following geos were not in the provided geojson and will not be plotted: %s",
                           ','.join(not_included_geos))
    return filtered_geojson_dict, filtered_df


def read_geojson(geo_data):
    if type(geo_data) == dict:
        geojson_dict = geo_data
    elif type(geo_data) == str and geo_data.startswith('https'):
        geojson_dict = requests.get(geo_data).json()
    else:
        with open(geo_data, 'r') as f:
            geojson_dict = json.load(f)

    return geojson_dict


def plot_map(df=None, color_column=None, geo_column=None, geo_type=None, geo_data=None, key_on=None,
             legend_name=None, fmap=None, reset=True, fill_color=None, filter_geos=True, **kwargs):
    """Creates a choropleth map based on a dataframe. Colors regions in a map based on provided data.

    Must provide geo_type *or* geo_data and key_on.

    `geo_type` options:
        * 'us_states': will join to a column containing state names. See
            `to_state_name()` to convert state abbreviations to full state names


    Args:
        df: Dataframe containing `color_column` and `geo_column`. If None,
            all regions will be colored a single color.
        color_column (`str`): Name of column in dataframe containing data to color map
            according to. If None, df must be None as well. All regions will be colored
            a single color.
        geo_column (`str`): Name of column in dataframe containing geographic key that
            can join to `key_on` in geoJson. If None, df must be None as well. All regions
            will be colored a single color.
        geo_type (`str`): If provided, maps to a geo_data and key_on for a given geography
        type. Default None. Must provide this or geo_data and key_on. See geo_type options above.
        geo_data (`str`): URL, file path, or data (json, dict, geopandas, etc) to your
            GeoJSON geometries. Required if geo_type not provided.
        key_on (`str`): Variable in the geo_data GeoJSON file to bind the data to. Must start
            with ‘feature’ and be in JavaScript objection notation. Ex: ‘feature.id’ or
            ‘feature.properties.statename’. If not provided, must have geo_type specified.
        legend_name (`str`): Name of legend. If None, legend will be labeled with
            `color_column` name formatted witH underscores removed.
        fmap (folium.Map): map object. Default None - will create new map.
        reset (bool): Default True. Clears map data from fmap object.
        fill_color (`str`):Area fill color. Can pass a hex code, color name,
                or if you are binding data, one of the following color brewer palettes:
                ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’,
                ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’. Default given by vh.geo_formatting.
        filter_geos (`bool`): If True,  will filter out all geoJSON entries for regions not
            included in the dataframe, `df`. This is typically necessary as folium generally
            won't plot anything if there isn't a match for each region.
        **kwargs:
            location_start (tuple): Lat/lon to center map on to begin with. Default given by vh.geo_formatting.
            zoom_start (int): Level of zoom, 1-10. Default given by vh.geo_formatting
            tiles (`str`): Name of folium tiles to use. Default given by vh.geo_formatting.
            fill_opacity (`float`): Opacity of area fill, range 0-1. Default given by vh.geo_formatting.
            line_opacity (`float`): Opacity of line, range 0-1. Default given by vh.geo_formatting.
    Returns:
        fmap (folium.Map): Map with geographic regions colored according to provided data
    """

    # If not data is provided, a single color will be used to color in everything in the geo_data
    if df is None:
        assert color_column is None and geo_column is None, \
            'No data provided, color_column must be None as all regions will be colored the same'

        # The choropleth function will return nothing if a colormap name is given (e.g. GnBu) so must give one color
        fill_color = geoformatting['fill_color_single'] if fill_color is None else fill_color
    else:
        assert color_column in df.columns, 'color_column must be provided to indicate which column to use in coloring'
        assert geo_column in df.columns, 'geo_column must be provided to indicate which column contains geographic key'

        fill_color = geoformatting['fill_color'] if fill_color is None else fill_color

        # Use `color_column` as legend label but format nicely
        if legend_name is None:
            legend_name = vh.labelfy(color_column)

    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    if geo_type is not None:
        try:
            geo_data = geos[geo_type]["geo_data"]
            key_on = geos[geo_type]["key_on"]
        except KeyError:
            raise KeyError('Only options for geo_type are: %s' % ','.join(geos.keys()))
        except Exception as e:
            raise ValueError('Error with geo_type: %s' % e)

    geojson_dict = read_geojson(geo_data)

    if df is not None:
        n_geos_original = len(geojson_dict['features'])
        json_key = key_on.split('.')[-1]
        geojson_dict, filtered_df = filter_geojson(geojson_dict, df.copy(), geo_column, json_key, filter_geos)
        logger.info('geo_data reduced to %i regions from %i', len(geojson_dict['features']), n_geos_original)

    if fmap is None:
        fmap = basic_map(**kwargs)

    folium.Choropleth(
        geo_data=geojson_dict,
        data=df,
        columns=[geo_column, color_column],
        key_on=key_on,
        reset=reset,
        fill_color=fill_color,
        fill_opacity=getfmt("fill_opacity"),
        line_opacity=getfmt("line_opacity"),
        legend_name=legend_name).add_to(fmap)

    return fmap


def plot_subset(plot_function, df, option, option_column, **kwargs):
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


def add_df_latlon(df, lat_col="lat", lng_col="lng",  radius_col=None, radius=15000,
                  color_col=None, raw_color_col='color', color_how=None, colors=None,
                  fill=True, fill_opacity=1, color_continuous_kwargs=None,
                  popup_col=None,  max_popup_width=2650, fmap=None, **kwargs):
    """Add points to a map from data in a dataframe. Can color and size points based on data columns.

    Args:
        df (pandas.DataFrame()): Dataframe containing at least `lat_col` and `lng_col`. If coloring according to
            data or adding a popup, dataframe should contain color_col, popup_col, and/or raw_color_col.
        lat_col (`str`): Name of column containing latitude
        lng_col (`str`): Name of column containing longitude
        radius_col (`str`): Name of column containing values for the radius of each point. Optional. If not provided,
            `radius` will be used for all points.
        radius (`int`): Radius of point if constant for all points. Default 15000.
        color_col (`str`): Name of column containing data to color the points according to. Must provide `color_how` to
        raw_color_col (`str`): If color_how='raw', name of column containing raw color name for each point. If
            color_how = 'continuous' or 'categorical', color assignments will be put in a column named `raw_color_col`.
            Default is 'color'. It will not be used if `color_how` is not provided.
        color_how: How to color the points. Options include:
            * None: The color provided in `colors` will be used for all points. If colors contains a list, the first
                color in the list will be used.
            * Raw: The color in `raw_color_col` will be used as the color for each point.
            * Continuous: Points will be colored based on a colormap and the values of data in the `color_col`
        colors (`str` or `list`): If `color_col` and `raw_color_col` are None, value or first value of list are used
            to color each point. If `color_col` is not None and `color_how` = 'categorical', list of colors is used.
        fill (bool): If True, circles will be filled in with color. Default True.
        fill_opacity (float): Opacity of area fill, range 0-1.
        color_continuous_kwargs:
        popup_col (`str`): Name of column containing information to display in a hover popup on map. Optional.
        max_popup_width (`int`): Maximum width of popup box. Default 2650.
        fmap (folium.Map): If a map object has already been created, it can be provided and points will be added to it.
        **kwargs:

    Returns:

    """
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    if raw_color_col is not None and raw_color_col in df.columns:
        logger.debug('Using raw color in %s column', raw_color_col)
        if color_col is not None:
            logger.warning('%s column already in dataframe so %s column will not be used. Change `raw_color_col` '
                           'if you want a new color mapping to be created.', raw_color_col, color_col)
    elif color_col is not None:
        if color_how == "categorical":
            df = vh.color_categorical(df, color_col, new_color_column=raw_color_col, colors=colors)
        elif color_how == "continuous":
            color_continuous_kwargs = {} if color_continuous_kwargs is None else color_continuous_kwargs
            df = vh.color_continuous(df, color_col, new_color_column=raw_color_col, **color_continuous_kwargs)
        else:
            raise ValueError("Must specify color_how as 'categorical' or 'continuous' or change color_col to None")
    else:
        raw_color_col = None
        if colors is None:
            color = vh.formatting["darks"][0]
        elif type(colors) == list:
            color = colors[0]
        else:
            color = colors

    if fmap is None:
        fmap = folium.Map(location=getfmt("location_start"), zoom_start=getfmt("zoom_start"), tiles=getfmt('tiles'))
    if type(radius) != list:
        radius = [radius] * len(df)

    for i, j in enumerate(df.index):

        r = df.loc[j, radius_col] if radius_col is not None else radius[i]
        color = color if raw_color_col is None else df.loc[j, raw_color_col]
        if popup_col is not None:
            popup = df.loc[j, popup_col]
            popup = folium.Popup(popup, max_width=max_popup_width)
        else:
            popup = None
        lat, lon = df.loc[j, lat_col], df.loc[j, lng_col]

        folium.Circle(location=(lat, lon), radius=r, popup=popup,
                      fill=fill, color=color, fill_color=color, fill_opacity=fill_opacity, **kwargs).add_to(fmap)

    return fmap


def add_line(latlonA, latlonB, line_color="black", fmap=None, **kwargs):
    getfmt = lambda x: geoformatting[x] if x not in kwargs else kwargs[x]

    if fmap is None:
        fmap = folium.Map(location=getfmt("location_start"), zoom_start=getfmt("zoom_start"))

    folium.PolyLine([[latlonA, latlonB]], color=line_color, **kwargs).add_to(fmap)

    return fmap


def html_to_png(htmlpath=None, pngpath=None, delay=5, width=2560, ratio=0.5625, browser=None):

    if htmlpath is None and pngpath is None:
        raise ValueError("Must give at least htmlpath or pngpath")
    elif htmlpath is None:
        htmlpath = '%s.html' % pngpath.split(".")[0]
    elif pngpath is None:
        pngpath = '%s.png' % htmlpath.split(".")[0]

    tmpurl = 'file://{htmlpath}'.format(htmlpath=htmlpath)

    if browser is None:
        from selenium import webdriver
        browser = webdriver.Safari()

    browser.set_window_size(width, width * ratio)

    browser.get(tmpurl)

    # Give the map tiles some time to load
    time.sleep(delay)
    browser.save_screenshot(pngpath)
    browser.quit()


def save_map(fmap, htmlpath=None, pngpath=None, png=False, delay=5, width=2560, ratio=0.5625, browser=None):

    if htmlpath is None and pngpath is None:
        raise ValueError("Must give at least htmlpath or pngpath")
    elif htmlpath is None:
        htmlpath = '%s.html' % pngpath.split(".")[0]
    elif png and pngpath is None:
        pngpath = '%s.png' % htmlpath.split(".")[0]

    fmap.save(htmlpath)

    if png:
        html_to_png(htmlpath=os.path.abspath(htmlpath), pngpath=pngpath, delay=delay,
                    width=width, ratio=ratio, browser=browser)
