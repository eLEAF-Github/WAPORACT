"""
waporact package

plotting functions (stand alone/support functions)

DISCLAIMER: Different to the other tools, standardising and automating 
the creation of visualisations (plots and maps) via functions is always much harder. 
This is because there is so much variation in what and how a user 
wants to visualise their data for themselves or their client.

This is why we provide only a few examples below of how standardisation or automation 
could be done. Considering th ecomplexitly of plot sthe original packages used already 
automate alot of the features, especially plotly which we decided to focus on here. 
If you want to take your plots and maps further I reccomend you take  a look
at the source packages themselves such as matplotlib and plotly. Links to the original code
used for each plotting function can be found in the descriptions.  

"""
##########################
# import packages
import os
from typing import Union
import sys
from datetime import timedelta
from timeit import default_timer

import pandas as pd
import geopandas as gpd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as xp

import matplotlib.pyplot as plt
import contextily as cx

from waporact.scripts.tools import vector
from waporact.scripts.tools.raster import gdal_info, raster_to_array, reproject_extent


##########################
#  Plot support functions
##########################
def roundup_max(value):
    """
    Description:
        automatically rounds up the value given to the 
        closest absolute value 

    Args:
        value: value to round up

    Return:
        rounded up value
    
    """
    digit_count = len(str(int(value)))-1

    multiplier = 10**digit_count

    return value if value % multiplier == 0 else value + multiplier - value % multiplier

##########################
def calculate_max_and_min(
    dataframe: pd.DataFrame, 
    column: str):
    """
    Description:
        given a dataframe and the name of a column in the dataframe 
        generates a min and max value from the range of values in
        that column for use in say axes mapping.

        NOTE: the column given must contain ints or floats

    Args:
        dataframe: dataframe containing the column of values
        column: name of the column in the dataframe containing
        the values

    Return:
        tuple: tuple giving the range of values calculated 
    
    """
    if not isinstance(dataframe[column].iloc[0], (float,int)):
        raise TypeError('input column must contain int or float values')
    zmax = np.nanmax(dataframe[column])
    zmin = np.nanmin(dataframe[column])

    if zmax <= 1 and zmin >= 0:
        zmax = 1
        zmin = 0
    
    else:
        if zmin > 0:
            zmin = 0

        zmax = roundup_max(zmax)

    return zmin, zmax


##########################
def generate_hover_template_chloro(
    z_label: str,
    z_column: str,
    id_label: str,
    secondary_inputs: dict=None,
    label_text: str= None,
    decimal_places: int=2):
    """
    Description:
        generates a hovertemplate for a chloropleth_map for the user based on the inputs
        and adds secondary labels based on the input csv if provided usin a dictionary

    Args:
        z_label: main label for the hovertemplate and primary values shown
        z_column: column name in the csv holdign the values
        id_label: id label of the shapefile
        (also the union key columnlinking the shapefile and csv)
        secondary_inputs: dict containing inputs to add as labels
        label_text: if provided places this as the final line at the bottom of the hoverlabel.
        decimal_places: decimal places to round all entries too

    Return:
        str: hovertemplate string
    
    """
    if secondary_inputs:
        if isinstance(secondary_inputs, dict):
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in secondary_inputs.items()):
                raise AttributeError('secondary_inputs must contain only string inputs')
        else:
            raise TypeError('secondary_inputs must be a dict')

    hovertemplate = '<b>{}:</b><br><br>{}: %{{customdata}}<br>{}:%{{z:.{}f}}'.format(z_label, id_label, z_column, decimal_places)

    if secondary_inputs:
        for label, column in secondary_inputs.items():
            hovertemplate += '<br>{}:%{{{}:.{}f}}'.format(label, column,decimal_places)

    if label_text:
        hovertemplate += '<br>%{{{}}}</b>'.format(label_text)

    return hovertemplate

    
##########################
# Non Spatial plots
##########################
def spiderplot(
    input_dict: dict,
    title: str,
    fill: bool=True,
    range: list=None,
    show_figure: bool=False,
    output_png_path: str=None,
    output_html_path: str=None,
    decrease_size_output_html: bool=False,
    ):
    """
    Description:
        create a spiderplot ffrom the input dictionary
        using the keys as the columns/corners and the
        values as the values.

        for more details or variations on the plot see:
        https://plotly.com/python/radar-chart/

    Args:
        input_dict: dictionary containing the information to plot
        title: title of the plot
        fill: if true fills the spider web in the diagram,
        range: range in list format ([0,1]) within which  he plotted values fall,
        if not provided uses [0, max(values),
        show_figure: if true shows the figure made
        output_png_path: if provided outputs the generated file to static png
        output_html_path: if provided outputs the generated file to interactive html
        decrease_size_output_html: if true decreases the size of the output html  by ~3mb
        however an internet connection is required to view it

    Return:
        int: 0
    
    """
    # retrieve the spidergram labels and values from the input dict
    labels = []
    values = []
    for key, value in input_dict.items():
        labels.append(key)
        values.append(value)

    # set the range of values for the spider diagram if not provided
    if not range:
        range = [0, max(values)]

    # fromat dict to dataframe expected by plotly function
    df = pd.DataFrame(dict(
        r=values,
        theta=labels))

    # create spider diagram plot
    fig = xp.line_polar(
        df,
        r='r',
        theta='theta',
        line_close=True,
        range_r=range,
        title=title)

    if fill:
        fig.update_traces(fill='toself')

    # output to png if path is provided
    if output_png_path:
        fig.write_image(output_png_path)
    # output to html if path is provided
    if output_html_path:
        if decrease_size_output_html:
            print('WARNING: size of output html decreased by ~3mb but an internet connection is required to view it')
            fig.write_html(output_html_path,  include_plotlyjs='cdn')
        else:
            fig.write_html(output_html_path,  include_plotlyjs=True)

    # show map if true
    if show_figure:
        fig.show()

    return 0

##########################
def bargraph(
    input_table: Union[pd.DataFrame,str],
    x: str,
    y: str,
    title: str,
    x_label: str=None,
    y_label: str=None,
    color: str=None,
    y_line: float = None,
    autoset_y_axis: bool=False,
    barmode: str = 'relative',
    sep: str=';',
    sheet: int=0,
    show_figure: bool=False,
    output_html_path: str=None,
    output_png_path: str = None,
    decrease_size_output_html: bool=False,
    ):
    """
    Description:
        create a bar graph using plotly express function
        and export to html or png if you want.

        for more details or variations on the plot see:
        https://plotly.com/python/bar-charts/

    Args:
        input_table: dataframe or path to the file to create graph from
        x: column in the input_table to use as the x column
        y: column in the input table to use as y column
        title: title of the plot
        color: column to use as the third (z) variable if provided, can be categorical (str)
        or continous (float, int)
        x_label: label for the x axis, if not provided uses the x column name
        y_label: label for the y axis, if not provided uses the y column name
        y_line: if provided adds a horizontal line at the given height of y
        autoset_y_axis: if True auto adjusts the y scale/axis. plotly already
        sets this to y min and y max bu this further adjusts it
        barmode: barmode to use auto set to 'relative', options
        also include 'stack', 'group'
        sep: seperator to us ewhen reading the file
        sheet: sheet to use if providing an excel
        show_figure: if true shows the created figure
        output_png_path: if provided outputs the generated file to static png
        output_html_path: if provided outputs the generated file to interactive html
        decrease_size_output_html: if true decreases the size of the output html  by ~3mb
        however an internet connection is required to view it

    Return:
        int: 0
    """
    if not x_label:
        x_label = x
    if not y_label:
        y_label = y

    # retrieve records from the file
    if isinstance(input_table, str):
        input_dataframe = vector.file_to_records(
            table=input_table, 
            sep=sep, 
            sheet=sheet)
    else:
        input_dataframe=input_table

    # create graph
    if color:
        fig = xp.bar(
            data_frame=input_dataframe, 
            x=x,
            y=y,
            color=color,
            labels={x:x_label,y:y_label},
            #width=850,
            #height=400,
            title=title,
            barmode=barmode)
    else:
        fig = xp.bar(
            data_frame=input_dataframe, 
            x=x,
            y=y,
            width=850,
            height=400,
            title=title,
            barmode=barmode)    

    if autoset_y_axis:
        yrange = calculate_max_and_min(
            dataframe=input_dataframe,
            column=y
            )
        fig.update_yaxes(
            range=yrange
        )

    # add y (threshold) line if applicable
    if y_line:
        fig.add_shape(
            type="line",
            line_color="salmon",
            line_width=3,
            opacity=1,
            line_dash="dot",
            x0=0, 
            x1=1, 
            xref="paper",
            y0=y_line,
            y1=y_line,
            yref="y"
            )


    # output to png if path is provided
    if output_png_path:
        fig.write_image(output_png_path)
    # output to html if path is provided
    if output_html_path:
        if decrease_size_output_html:
            print('WARNING: size of output html decreased by ~3mb but an internet connection is required to view it')
            fig.write_html(output_html_path,  include_plotlyjs='cdn')
        else:
            fig.write_html(output_html_path,  include_plotlyjs=True)

    # show map if true
    if show_figure:
        fig.show()

    return 0


##########################
def scatterplot(
    input_table: Union[pd.DataFrame,str],
    x: str,
    y: str,
    title: str,
    x_label: str=None,
    y_label: str=None,
    color: str=None,
    size: str=None,
    sep: str=';',
    sheet: int=0,
    autoset_y_axis: bool=False,
    show_figure: bool=False,
    output_html_path: str=None,
    output_png_path: str = None,
    decrease_size_output_html: bool=False,
    ):
    """
    Description:
        create a scatter plot using plotly express function
        and export to html or png if you want.

        for more details or variations on the plot see:
        https://plotly.com/python/bar-charts/

    Args:
        input_table: dataframe or path to the file to create graph from
        x: column in the input_table to use as the x column
        y: column in the input table to use as y column
        title: title of the plot
        color: column to use as the third variable if provided, can be categorical (str)
        or continous (float, int)
        size: column to use as the fourth variable if provided, can be categorical (str)
        or continous (float, int)
        x_label: label for the x axis, if not provided uses the x column name
        y_label: label for the y axis, if not provided uses the y column name
        autoset_y_axis: if True auto adjusts the y scale/axis. plotly already
        sets this to y min and y max bu this further adjusts it
        sep: seperator to us ewhen reading the file
        sheet: sheet to use if providing an excel
        show_figure: if true shows the created figure
        output_png_path: if provided outputs the generated file to static png
        output_html_path: if provided outputs the generated file to interactive html
        decrease_size_output_html: if true decreases the size of the output html  by ~3mb
        however an internet connection is required to view it

    Return:
        int: 0
    """
    if not x_label:
        x_label = x
    if not y_label:
        y_label = y

    # retrieve records from the file
    if isinstance(input_table, str):
        input_dataframe = vector.file_to_records(
            table=input_table,
            sep=sep, 
            sheet=sheet)
    else:
        input_dataframe=input_table

    # create graph
    if color and size:
        fig = xp.scatter(
            data_frame=input_dataframe, 
            x=x,
            y=y,
            color=color,
            size=size,
            labels={x:x_label,y:y_label},
            title=title)

    elif color:
        fig = xp.scatter(
            data_frame=input_dataframe, 
            x=x,
            y=y,
            color=color,
            labels={x:x_label,y:y_label},
            title=title)

    elif size:
        fig = xp.scatter(
            data_frame=input_dataframe, 
            x=x,
            y=y,
            size=size,
            labels={x:x_label,y:y_label},
            title=title)

    else:
        fig = xp.scatter(
            data_frame=input_dataframe,
            x=x,
            y=y,
            labels={x:x_label,y:y_label},
            title=title)

    if autoset_y_axis:
        yrange = calculate_max_and_min(
            dataframe=input_dataframe,
            column=y
            )
        fig.update_yaxes(
            range=yrange
        )

    # output to png if path is provided
    if output_png_path:
        fig.write_image(output_png_path)
    # output to html if path is provided
    if output_html_path:
        if decrease_size_output_html:
            print('WARNING: size of output html decreased by ~3mb but an internet connection is required to view it')
            fig.write_html(output_html_path,  include_plotlyjs='cdn')
        else:
            fig.write_html(output_html_path,  include_plotlyjs=True)

    # show map if true 
    if show_figure:
        fig.show()

    return 0

##########################
def violinplot(
    input_table: Union[pd.DataFrame,str],
    y: str,
    x_categorical: str,
    title: str,
    x_label: str=None,
    y_label: str=None,
    color_categorical: str=None,
    sep: str=';',
    sheet: int=0,
    show_figure: bool=False,
    output_html_path: str=None,
    output_png_path: str = None,
    decrease_size_output_html: bool=False,
    ):
    """
    Description:
        create a violin plot using plotly express function
        and export to html or png if you want.

        for more details or variations on the plot see:
        https://plotly.com/python/violin/

    Args:
        input_table: dataframe or path to the file to create graph from
        y: column in the input table to use as y column (main column for violin plot)
        title: title of the plot
        x_categorical: column in the input_table to use as the x column, used to categorise
        the y values, must be categorical
        categorical_color: column to use as the third (z) variable if provided,
        uses it to group the y column by
        x_label: label for the x axis, if not provided uses the x column name
        y_label: label for the y axis, if not provided uses the y column name
        sep: seperator to us ewhen reading the file
        sheet: sheet to use if providing an excel
        show_figure: if true shows the created figure
        output_png_path: if provided outputs the generated file to static png
        output_html_path: if provided outputs the generated file to interactive html
        decrease_size_output_html: if true decreases the size of the output html  by ~3mb
        however an internet connection is required to view it

    Return:
        int: 0
    """
    if not x_label:
        x_label = x_categorical
    if not y_label:
        y_label = y

    # retrieve records from the file
    if isinstance(input_table, str):
        input_dataframe = vector.file_to_records(
            table=input_table,
            sep=sep,
            sheet=sheet)
    else:
        input_dataframe=input_table

    # create graph
    if x_categorical and not color_categorical:
        # if x column provide but no color column
        fig = xp.violin(
            data_frame=input_dataframe, 
            y=y,
            x=x_categorical,
            labels={y:y_label, x_categorical: x_label},
            title=title,
            box=True)
    
    elif not x_categorical and color_categorical:
        # if no x column provide but a color column is
        fig = xp.violin(
            data_frame=input_dataframe, 
            y=y,
            color=color_categorical,
            labels={y:y_label},
            title=title,
            box=True)
    
    elif x_categorical and color_categorical:
        # if both x column and color column provided
        fig = xp.violin(
            data_frame=input_dataframe, 
            y=y,
            x=x_categorical,
            color=color_categorical,
            labels={y:y_label, x_categorical: x_label},
            title=title,
            box=True)
    
    else:
        # if only y column provided
        fig = xp.violin(
            data_frame=input_dataframe, 
            y=y,
            labels={y:y_label},
            title=title,
            box=True)

    # output to png if path is provided
    if output_png_path:
        fig.write_image(output_png_path)
    # output to html if path is provided
    if output_html_path:
        if decrease_size_output_html:
            print('WARNING: size of output html decreased by ~3mb but an internet connection is required to view it')
            fig.write_html(output_html_path,  include_plotlyjs='cdn')
        else:
            fig.write_html(output_html_path,  include_plotlyjs=True)

    # show map if true
    if show_figure:
        fig.show()

    return 0

##########################
def piechart(
    input_table: Union[pd.DataFrame,str],
    names: str,
    values: str,
    title: str,
    color_discrete_map: dict=None,
    sep: str=';',
    sheet: int=0,
    show_figure: bool=False,
    output_html_path: str=None,
    output_png_path: str = None,
    decrease_size_output_html: bool=False,
    ):
    """
    Description:
        create a violin plot using plotly express function
        and export to html or png if you want.

        for more details or variations on the plot see:
        https://plotly.com/python/pie-charts/

    Args:
        input_table: dataframe or path to the file to create graph from
        names: column holding the names for the pie slices
        values: column holding the value for the slices
        title: title of the plot
        color_discrete_map: dictionary that if provided maps the names column (entries)
        as the keys to the specific colours provided as values
        sep: seperator to us ewhen reading the file
        sheet: sheet to use if providing an excel
        show_figure: if true shows the created figure
        output_png_path: if provided outputs the generated file to static png
        output_html_path: if provided outputs the generated file to interactive html
        decrease_size_output_html: if true decreases the size of the output html  by ~3mb
        however an internet connection is required to view it

    Return:
        int: 0
    """

    # retrieve records from the file
    if isinstance(input_table, str):
        input_dataframe = vector.file_to_records(
            table=input_table, 
            sep=sep, 
            sheet=sheet)
    else:
        input_dataframe=input_table

    # create pie chart
    if color_discrete_map:
        fig=xp.pie(
            data_frame = input_dataframe,   
            values= values,
            names = names,
            title=title, 
            template='seaborn',
            color_discrete_map=color_discrete_map)
    else:
        fig=xp.pie(
            data_frame = input_dataframe,
            values= values,
            names = names,
            title=title, 
            template='seaborn')

    # output to png if path is provided
    if output_png_path:
        fig.write_image(output_png_path)
    # output to html if path is provided
    if output_html_path:
        if decrease_size_output_html:
            print('WARNING: size of output html decreased by ~3mb but an internet connection is required to view it')
            fig.write_html(output_html_path,  include_plotlyjs='cdn')
        else:
            fig.write_html(output_html_path,  include_plotlyjs=True)

    # show map if true
    if show_figure:
        fig.show()

    return 0 

    
##########################
# Interactive Spatial Plots
##########################
def interactive_categorical_map(
    input_shapefile_path: str,
    input_csv_path: str,
    cat_column: str,
    cat_label: str,
    union_key: str='wpid',
    mapbox_style="carto-positron",
    colorscale: str = None,
    opacity: float=0.5,
    sep: str=';',
    show_figure: bool=False,
    output_html_path: str=None,
    decrease_size_output_html: bool=False,
    ):
    """
    Description:
        creates an interactive html categorical map of the
        inputs provided using plotly.

        mapping options mapbox basemaps:

            white-bg,
            open-street-map
            carto-positron
            carto-darkmatter
            stamen-terrain
            stamen-toner
            stamen-watercolor

        for more details or variations on the map see:
        https://plotly.com/python/mapbox-county-choropleth/

    Args:
        input_shapefile_path: path to the input shape,
        input_csv_path: path to the input csv
        z_column: name of the column in the csv to use for the z value
        z_label: lable for the z value column
        show_map: if true shows the map after running
        output_html_path: if a '.html' path is provided outputs a
        html version of the map to this path
        union_key: cloumn/key used to union the csv and shapefile attribute table
        mapbox_style: mapbox style to use for the basemap. seven free options listed above
        colorscale: colorscale to sue ofr the z axis (Viridis etc)
        opacity: opacity of the features
        sep: seperator used in the provided csv
        decrease_size_output_html: if true decreases the size of the output html  by ~3mb
        however an internet connection is required to view it

    Return:
        int: 0
    """
    if not colorscale:
        autocolorscale=True
    else:
        autocolorscale=False
    
    # retrieve the csv
    df = pd.read_csv(input_csv_path,sep=sep)
    
    # retrieve features from the input shape as a geodataframe
    gdf = vector.file_to_records(input_shapefile_path, output_crs=4326)

    # retrieve central mapping points from the geodataframe
    autozoom, xy = vector.get_plotting_zoom_level_and_central_coords_from_gdf(gdf)

    # convert geodatafrmae to geojson as plotly requires
    output_json_path = os.path.splitext(input_shapefile_path)[0] +'.json'
    gdf.to_file(output_json_path, driver='GeoJSON')

    # retrieve features in geojson format
    with open(output_json_path) as f:
        features = json.load(f)

    # create a template for the hover text
    hovertemplate='<b>{}:<br><br>{{locations}}: {{color}}<extra></extra>'.format(cat_label, union_key, cat_column)

    # create categorical map
    fig = xp.choropleth_mapbox(
        geojson=features,
        featureidkey="properties.{}".format(union_key),
        locations=df[union_key],
        color=df[cat_column],
        opacity=opacity,
        mapbox_style=mapbox_style,
        zoom=autozoom,
        center={"lat": xy[1], "lon": xy[0]},
        template=hovertemplate,
        )

    # output to html if path is provided
    if output_html_path:
        if decrease_size_output_html:
            print('WARNING: size of output html decreased by ~3mb but an internet connection is required to view it')
            fig.write_html(output_html_path,  include_plotlyjs='cdn')
        else:
            fig.write_html(output_html_path,  include_plotlyjs=True)
    # show map if true
    if show_figure:
        fig.show()

    return 0

##########################
def interactive_choropleth_map(
    input_shapefile_path: str,
    input_csv_path: str,
    z_column: str,
    z_label: str,
    show_map: bool=False,
    output_html_path: str=None,
    union_key: str='wpid',
    mapbox_style="carto-positron",
    colorscale: str = None,
    zmin: float=None,
    zmax: float=None,
    opacity: float=0.5,
    sep: str=';',
    decrease_size_output_html: bool=False,    
    secondary_hovertemplate_inputs: dict=None,
    secondary_hovertemplate_text: str= None,
    hovertemplate_decimal_places: int=2
    ):
    """
    Description:
        creates an interactive html chloropleth map of the
        inputs provided using plotly.

        mapping options mapbox:

            white-bg,
            open-street-map
            carto-positron
            carto-darkmatter
            stamen-terrain
            stamen-toner
            stamen-watercolor

        for more details or variations on the map see:
        https://plotly.com/python/mapbox-county-choropleth/

    Args:
        input_shapefile_path: path to the input shape,
        input_csv_path: path to the input csv
        z_column: name of the column in the csv to use for the z value
        z_label: label for the z value column
        show_map: if true shows the map after running
        output_html_path: if a '.html' path is provided outputs a
        html version of the map to this path
        union_key: cloumn/key used to union the csv and shapefile attribute table
        mapbox_style: mapbox style to use for the basemap. seven free options listed above
        colorscale: colorscale to sue ofr the z axis (Viridis etc)
        zmin: minimum value for the z color axes, auto set if not provided
        zmax: maximum value for the z color axes, auto set if not provided
        opacity: opacity of the features
        sep: seperator used in the provided csv
        decrease_size_output_html: if true decreases the size of the output html  by ~3mb
        however an internet connection is required to view it
        secondary_hovertemplate_inputs: dict containing inputs to add as labels
        seocndary_hovertemplate_text: if provided places this text as the final line at
        the bottom of the hoverlabel.
        hovertemplate_decimal_places: decimal places to round all entries
        too in the hovertemplate

    Return:
        int: 0
    """    
    if not colorscale:
        autocolorscale=True
    else:
        autocolorscale=False
    
    # retrieve the csv
    df = pd.read_csv(input_csv_path,sep=sep)
    
    # retrieve features from the input shape as a geodataframe
    gdf = vector.file_to_records(input_shapefile_path, output_crs=4326)

    # retrieve central mapping points from the geodataframe
    autozoom, xy = vector.get_plotting_zoom_level_and_central_coords_from_gdf(gdf)

    # convert geodatafrmae to geojson as plotly requires
    output_json_path = os.path.splitext(input_shapefile_path)[0] +'.json'
    gdf.to_file(output_json_path, driver='GeoJSON')

    # retrieve features in geojson format
    with open(output_json_path) as f:
        features = json.load(f)

    # calculate zmin and zmax for the colorbar
    if zmin is None or zmax is None:
        zmin, zmax = calculate_max_and_min(
            dataframe=df,
            column=z_column)

    # create a template for the hover text
    hovertemplate = generate_hover_template_chloro(
        z_label=z_label,
        z_column=z_column,
        id_label=union_key,
        secondary_inputs=secondary_hovertemplate_inputs,
        label_text=secondary_hovertemplate_text,
        decimal_places=hovertemplate_decimal_places)

    # create chloropleth map 
    fig = go.Figure(go.Choroplethmapbox(
        geojson=features,
        featureidkey="properties.{}".format(union_key),
        locations=df[union_key],
        z=df[z_column],
        customdata=df[union_key],
        zmax=zmax,
        zmin=zmin,
        colorscale=colorscale,
        autocolorscale=autocolorscale,
        marker_opacity=opacity,
        marker_line_width=1,
        hovertemplate=hovertemplate,
        # format the colorbar
        colorbar=dict(title=dict(
        side='top',
        text=z_label,
        ),
        thicknessmode='pixels',
        thickness=50,
        lenmode='fraction',
        len=0.6,
        y=0.8,
        yanchor='top'
        )
        ))

    # update figure layout
    fig.update_layout(
        mapbox_style=mapbox_style,
        mapbox_zoom=autozoom,
        mapbox_center={"lat": xy[1], "lon": xy[0]},
        margin={'r':0, 't':0, 'l':0, 'b':0})

    # output to html if paht is provided
    if output_html_path:
        if decrease_size_output_html:
            print('WARNING: size of output html decreased by ~3mb but an internet connection is required to view it')
            fig.write_html(output_html_path,  include_plotlyjs='cdn')
        else:
            fig.write_html(output_html_path,  include_plotlyjs=True)
    # show map if true 
    if show_map:
        fig.show()

    return 0

##########################
# Non Interactive Spatial Plots
##########################
def rasterplot(
    input_value_raster_path: str,
    output_plot_path: str,
    title: str,
    input_mask_raster_path: str=None,
    colorscale: str = 'RdYlGn',
    zmin: float=None,
    zmax: float=None,
    ):
    """
    Description:
        creates a matplotlib plot at the extent of the mask raster and showing the masked value
        raster. A standard basemap is placed underneath id specified

    Args:
        input_value_raster_path: path to the input raster containing the vlaues to plot
        input_mask_raster_path: path to the mask raster used to mask out the values raster 
        output_plot_path: path to output the plot too 
        title: title of the plot
        colorscale: standard colorscale to use (standard options)
        zmin: minimum value on the z axis, autoset based on available values if not provided
        zmax: minimum value on the z axis, autoset based on available values if not provided
        opacity: opacity of the shapes in the plot

    Return:
        str: path to the outputted png plot
    """
    # turn off interactive plots
    plt.ioff()

    assert colorscale in ['RdYlGn', 'Blues'], 'please specifiy one of the current available colorscales: RDYlGn or Blues'
   # convert the dataset into arrays
    value_array = raster_to_array(input_value_raster_path)
    if input_mask_raster_path:
        mask_array = raster_to_array(input_mask_raster_path) 

    # mask no data in original raster
    value_array = np.ma.masked_values(value_array, gdal_info(input_value_raster_path)['nodata'])

    if input_mask_raster_path:
        # mask the original raster with the masked raster
        masked_value_array = np.ma.masked_values(value_array, mask_array)
        value_array = masked_value_array

    # calculate zmin and zmax for the colorbar as needed
    if zmin is None or zmax is None:
        zmax = np.nanmax(value_array) 
        zmin = np.nanmin(value_array)

        if zmax <= 1 and zmin >= 0:
            zmax = 1
            zmin = 0
        
        else:
            if zmin > 0:
                zmin = 0

            zmax = roundup_max(zmax)

    # plot the raster
    ax = plt.gca()
    
    # set basic matplotlib variables
    plt.set_cmap(colorscale)
    plt.title(title)

    # plot the values
    plot = ax.imshow(value_array)
    plot.set_clim(zmin, zmax)

    cbar = plt.colorbar(plot, aspect=40)

    ax.ticklabel_format(useOffset=False, style='plain')
    plt.setp(ax.get_xticklabels(), rotation=90)

    # save plot to a output directory
    plt.savefig(output_plot_path)

    plt.close()
    plt.cla()
    plt.clf()

    return output_plot_path


##########################
def shapeplot(
    input_shape_path: str,
    output_plot_path: str,
    title: str,
    z_column: str,
    csv_path: str=None,
    join_column: str=None,
    colorscale: str = 'RdYlGn',
    zmin: float=None,
    zmax: float=None,
    opacity: float=0.5,
    coordinate_axis: bool=False,
    sep: str = ';'
    ):
    """
    Description:
        creates a matplotlib plot at the extent of the shapes in the given
        shape path. Plotting the values in the column specified by the z column.
        A standard basemap is placed underneath id specified  

    Args:
        input_shape_path: path to the input shape
        output_plot_path: path to output the plot too 
        title: title of the plot
        z_column: column in the shapefile/geojson to use as the z column/axis
        csv_path: if provided joins the csv to the shapefile using the join key/column
        join_column: if a csv is provided joins thecsv stats to the shapefile using this
        colorscale: standard colorscale to use (standard options)
        zmin: minimum value on the z axis, autoset based on available values if not provided
        zmax: minimum value on the z axis, autoset based on available values if not provided
        opacity: opacity of the shapes in the plot
        coordinate_axis: if false turns off the x,y axis labels for coordinates

    Return:
        str: path to the outputted png plot
    """
    # trun off interactive plotting
    plt.ioff()

    assert colorscale in ['RdYlGn', 'Blues'], 'please specifiy one of the current available colorscales: RDYlGn or Blues'

    vector_gdf = gpd.read_file(input_shape_path)

    if csv_path and join_column:
        # if a csv and join key are provided join the csv to the sahpefile as well
        stats_df = pd.read_csv(csv_path,sep=sep)
        vector_gdf = vector_gdf.join(stats_df.set_index(join_column), on=join_column)


    if z_column not in vector_gdf:
        print('The mentioned z column does not exists in the vector file, or joined csv file')

    if isinstance(vector_gdf[z_column], str):
        categorical = True
    else:
        categorical = False

    fig, ax = plt.subplots(figsize=(10, 10), alpha=0.5)
    
    if zmin is None or zmax is None:  
        # calculate zmin and zamx if not provided
        zmin, zmax = calculate_max_and_min(
            dataframe=vector_gdf,
            column=z_column)

    vector_gdf.plot(column=z_column,
                    categorical=categorical,
                    legend=True,
                    cmap=colorscale,
                    edgecolor="black",
                    ax=ax,
                    vmin=zmin,
                    vmax=zmax,
                    alpha=opacity)

    cx.add_basemap(ax, crs=vector_gdf.crs, zoom=12, source=cx.providers.OpenStreetMap.Mapnik)

    ax.set_title(title)

    if not coordinate_axis:
        ax.set_axis_off()

    plt.savefig(output_plot_path)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    return output_plot_path

##########################
if __name__ == "__main__":
    start = default_timer()
    args = sys.argv
