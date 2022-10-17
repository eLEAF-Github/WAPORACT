"""
waporact package

statistics functions (stand alone/support functions)
"""
##########################
# import packages
import os
import sys
from types import FunctionType
from datetime import datetime, timedelta
from timeit import default_timer

from typing import Union

import math
import fiona
import rasterio
from rasterio import features

from shapely.geometry import shape
import numpy as np
import pandas as pd

from waporact.scripts.tools import raster
from waporact.scripts.tools import vector


import logging

logger = logging.getLogger(__name__)

########################################################
# Functions used by waporact retrieval
########################################################

########################################################
# Dataframe Functions
########################################################
def dict_to_dataframe(in_dict: Union[dict, list]):
    """transfrom a dict to a dataframe

    Parameters
    ----------
    in_dict : Union[dict, list]
        dict to transform

    Returns
    -------
    pandas.DataFrame
        transformed dict as dataframe

    Raises
    ------
    AttributeError
        if the input is a list not containing dicts
    AttributeError
        if there is more then two levels in list and dict
    AttributeError
        if there is more then two levels in list and dict
    AttributeError
        if nested dicts provided all entries need to be dicts
    """
    temp_list = []

    # transform dictionary to dataframe
    if isinstance(in_dict, list):
        # check the list contains only dicitonaries
        if not all(isinstance(item, dict) for item in in_dict):
            raise AttributeError(
                "if you provide a list to dict_to_dataframe, it must contain "
                " dictionaries"
            )
        else:
            # check each dict in the list contains no dictionaries
            for item in in_dict:
                item_keys = item.keys()
                if any(isinstance(item[item_key], dict) for item_key in item_keys):
                    raise AttributeError(
                        "dict_to_dataframe accepts a list of dicts, a dict or nested dicts,\n"
                        " (dict of dicts), double nested dicts or a list of nested dicts or more are not accepted"
                    )

            temp_list = in_dict

    elif isinstance(in_dict, dict):
        # check if it is a single dict or nested dict
        dict_keys = in_dict.keys()
        if any(isinstance(in_dict[key], dict) for key in dict_keys):
            # if a nested dic make sure all values are dicts
            if all(isinstance(in_dict[key], dict) for key in dict_keys):
                # make sure no nested dicts have nested dicts
                for key in dict_keys:
                    subdict_keys = in_dict[key].keys()
                    if any(
                        isinstance(in_dict[key][subkey], dict)
                        for subkey in subdict_keys
                    ):
                        raise AttributeError(
                            "dict_to_dataframe accepts a list, dict or nested dict,"
                            " (dict of dicts), dobule nested dicts or more are not accepted"
                        )

                # no double nested dicts found, un-nesting now
                for key in dict_keys:
                    in_dict[key]["dict_key"] = key
                    temp_list.append(in_dict[key])

            else:
                raise AttributeError(
                    "if nested dicts are provided all entries"
                    " in the top level dict need to be dictionaries"
                )

        else:
            # assign singular dict to a list
            temp_list = [in_dict]

    # write list of dicts to dataframe
    out_dataframe = pd.DataFrame(temp_list)

    return out_dataframe


##########################
def output_table(
    table: Union[list, dict, pd.DataFrame],
    output_file_path: str,
    output_formats: list = [".csv", ".xlsx"],
    csv_seperator: str = ";",
):
    """output a list of dicts, dict or dataframe to file

    Parameters
    ----------
    table : Union[list, dict, pd.DataFrame]
        table to output
    output_file_path : str
        location to output too
    output_formats : list, optional
        formats to output as, by default [".csv", ".xlsx"]
    csv_seperator : str, optional
        if csv is output type the seperator to use, by default ";"

    Returns
    -------
    str
        path to the outputted table
    """
    # create output subfolders as needed
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pd_processes = {
        ".csv": pd.DataFrame.to_csv,
        ".xlsx": pd.DataFrame.to_excel,
        ".pkl": pd.DataFrame.to_pickle,
        ".json": pd.DataFrame.to_json,
        ".tex": pd.DataFrame.to_latex,
        ".pq": pd.DataFrame.to_parquet,
    }

    # set up output paths
    base_path, initial_ext = os.path.splitext(output_file_path)
    output_paths = [(output_file_path, initial_ext, pd_processes[initial_ext])]

    output_formats = [outf for outf in output_formats if outf != initial_ext]

    for outf in output_formats:
        output_paths.append((base_path + outf, outf, pd_processes[outf]))

    # process file
    if isinstance(table, pd.DataFrame):
        output_df = table
    else:
        output_df = dict_to_dataframe(table)

    for out_path, outf, out_process in output_paths:
        if outf == ".csv":
            output_df.to_csv(out_path, sep=csv_seperator, index=False)
        else:
            out_process(output_df, out_path)

    return output_file_path


########################################################
# Functions used outside wapor retrieval
########################################################

########################################################
# Mathematical Functions
########################################################
def latlon_dist(lat1: float, lat2: float, lon1: float, lon2: float):
    """calculates the distance in meters from lat lon inputs

    Parameters
    ----------
    lat1 : float
        latitude variable from first coordinates tuple
    lat2 : float
        latitude variable from second coordinates tuple
    lon1 : float
        longitude variable from first coordinates tuple
    lon2 : float
        longitude variable from second coordinates tuple

    Returns
    -------
    float
        distance
    """
    # radius of the earth
    R = 6373.0

    lat1 = math.radians(abs(lat1))
    lon1 = math.radians(abs(lon1))

    lat2 = math.radians(abs(lat2))
    lon2 = math.radians(abs(lon2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    # Haversine formula
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance


##########################
def ceiling_divide(a: float, b: float, cieling: float = 1):
    """divides a value by b and if the calculated value is
        higher than the cieling sets it to the cieling value

    Parameters
    ----------
    a : float
        value
    b : float
        value
    cieling : float, optional
        highest possible value, by default 1

    Returns
    -------
    float
        divided value
    """
    if b == 0:
        c = np.nan
    else:
        c = a / b
        if c > cieling:
            c = cieling
    return c


##########################
def floor_minus(a: float, b: float, floor: float = 0):
    """minus a value by b and if the calculated value is
        lower than the floor sets it to the floor value

    Parameters
    ----------
    a : float
        value
    b : float
        value
    floor : float, optional
        lowest possible value, by default 0

    Returns
    -------
    float
        substracted value
    """
    c = a - b
    if c < floor:
        c = floor
    return c


########################################################
# statistics sub Functions
########################################################
def generate_zonal_stats_column_and_function(input_name: str, statistic: str):
    """generates a column name for the zonal stat calculated using a combination og the input_name and
        statistic as well retrieving the numpy fucniton required to carry it out.

    Parameters
    ----------
    input_name : str
        name to use in the column
    statistic : str
        statistic keyword used in the input name as well as being used to retrieve the
        required numpy function


    Returns
    -------
    tuple
        column name, numpy function

    Raises
    ------
    KeyError
        if the given statistic can not be found among the numpy options
    """
    numpy_dict = {
        "sum": np.nansum,
        "mean": np.nanmean,
        "count": np.count_nonzero,
        "stddev": np.nanstd,
        "min": np.nanmin,
        "max": np.nanmax,
        "median": np.nanmedian,
        "percentile": np.nanpercentile,
        "variance": np.nanvar,
        "quantile": np.nanquantile,
        "cumsum": np.nancumsum,
        "product": np.nanprod,
        "cumproduct": np.nancumprod,
    }

    # generate numpy function
    if statistic not in numpy_dict.keys():
        raise KeyError(
            f"statistic: {statistic} not found among numpy options: {numpy_dict.keys()}"
        )

    else:
        numpy_function = numpy_dict[statistic]

    # generate column name
    column_name = statistic + "_" + input_name

    return column_name, numpy_function


########################################################
# zonal statistics Functions
########################################################
def calc_field_statistics(
    fields_shapefile_path: str,
    input_rasters: list,
    field_stats: list = ["min", "max", "mean", "sum", "stddev"],
    statistic_name: str = "",
    output_csv_path: str = None,
    id_key: str = "wpid",
    out_dict: bool = False,
):
    """calculate various statistics per field from a raster
    using the shapefile to identify the fields

    Parameters
    ----------
    fields_shapefile_path : str
        path to the shapefile defining the fields
    input_rasters : list
        list of rasters/vrts to carry out zonal statistics on
    field_stats : list, optional
        list of statistics to carry out, also used in the column names, by default ["min", "max", "mean", "sum", "stddev"]
    statistic_name : str, optional
        name/ identifier to give to the stat calculated (used in combo with each field stat calculated), by default ""
    output_csv_path : str, optional
        path to output the csv too if provided, by default None
    id_key : str, optional
        name of shapefile column/feature dictionary key providing the feature indices, by default "wpid"
    out_dict : bool, optional
        if true outputs a dictionary instead of a dataframe, by default False

    Returns
    -------
    Union[dict, pandas.DataFrame]
        statistics calculated
    """
    # check if column identifier exists in the shapefile
    vector.check_column_exists(shapefile_path=fields_shapefile_path, column=id_key)

    # check for multiple rasters or vrts
    multiple_rasters = False

    if any("vrt" in os.path.splitext(raster)[1] for raster in input_rasters):
        multiple_rasters = True

    if len(input_rasters) > 1:
        multiple_rasters = True

    if multiple_rasters:
        logger.info("attempting to calculate zonal stats for multiple rasters or a vrt")
        stats = multiple_raster_zonal_stats(
            input_shapefile_path=fields_shapefile_path,
            raster_path_list=input_rasters,
            statistic_name=statistic_name,
            field_stats=field_stats,
            out_dict=out_dict,
            id_key=id_key,
        )
    else:
        logger.info("attempting to calculate zonal stats for a single raster")
        stats = single_raster_zonal_stats(
            input_shapefile_path=fields_shapefile_path,
            input_raster_path=input_rasters[0],
            statistic_name=statistic_name,
            field_stats=field_stats,
            out_dict=out_dict,
            id_key=id_key,
        )

    # if a csv is wanted as output transfrom as needed and create it
    if output_csv_path:
        # create output subfolders as needed
        output_dir = os.path.dirname(output_csv_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if isinstance(stats, dict):
            stats_csv = dict_to_dataframe(in_dict=stats)
        else:
            stats_csv = stats
        stats_csv.to_csv(output_csv_path, sep=";", index=False)
        logger.info("sep used for the csv is: ;")

        logger.info(f"csv outputted too: {output_csv_path}")

    return stats


##########################
def multiple_raster_zonal_stats(
    input_shapefile_path: str,
    raster_path_list: list,
    field_stats: list = ["min", "max", "mean", "sum", "stddev"],
    statistic_name: str = "",
    out_dict: bool = False,
    id_key: str = "wpid",
):
    """rasterize the features in the shapefile according to the
        template raster provided and extract the indices corresponding
        to each feature

    Parameters
    ----------
    input_shapefile_path : str
        shapefile to retireve raster indices for per feature
    raster_path_list : list
        list of paths to rasters to analyse also accepts and loops through vrts inside the list
    field_stats : list, optional
        list of statistics to carry out, also used in the column names, by default ["min", "max", "mean", "sum", "stddev"]
    statistic_name : str, optional
        name/ identifier to give to the stat calculated (used in combo with each field stat calculated), by default ""
    out_dict : bool, optional
        if True exports a dict instead of a dataframe, by default False
    id_key : str, optional
        name of shapefile column/feature dictionary key providing the feature indices, by default "wpid"

    Returns
    -------
    Union[dict, pandas.DataFrame]
        calculated statistics
    """
    # check if column identifier exists in the shapefile
    vector.check_column_exists(shapefile_path=input_shapefile_path, column=id_key)

    # make sure the crs match
    vector.compare_raster_vector_crs(
        raster_path=raster_path_list[0], vector_path=input_shapefile_path
    )

    index_dict = raster.create_polygon_index_dict(
        template_raster_path=raster_path_list[0],
        input_vector_path=input_shapefile_path,
        id_key=id_key,
    )

    stats_df = equal_dimensions_zonal_stats(
        polygon_index_base_raster_path=raster_path_list[0],
        polygon_index_dict=index_dict,
        raster_path_list=raster_path_list,
        statistic_name=statistic_name,
        field_stats=field_stats,
        out_dict=out_dict,
        id_key=id_key,
    )

    return stats_df


##########################
def equal_dimensions_zonal_stats(
    polygon_index_dict: dict,
    polygon_index_base_raster_path: str,
    raster_path_list: list,
    field_stats: list = ["min", "max", "mean", "sum", "stddev"],
    statistic_name: str = "",
    out_dict: bool = False,
    id_key: str = "wpid",
):
    """takes a dictionary made using create_polygon_index_dict
        and an input_raster_path and calculates statistics per field
        in the raster using the index to identify the cells within each field

    Parameters
    ----------
    polygon_index_dict : dict
        dict providing the indices to analyse
    polygon_index_base_raster_path : str
        raster that formed the template for the polygon index_dict
    raster_path_list : list
        list of paths to rasters to analyse also accepts and loops through vrts inside the list
    field_stats : list, optional
        list of statistics to carry out, also used in the column names, by default ["min", "max", "mean", "sum", "stddev"]
    statistic_name : str, optional
        name/ identifier to give to the stat calculated, by default ""
    out_dict : bool, optional
        iif True exports a dict instead of a dataframe, by default False
    id_key : str, optional
        name of shapefile column/feature dictionary key providing the feature indices, by default "wpid"

    Returns
    -------
    union[dict, pandas.DataFrame]
        calculated statistics

    Raises
    ------
    KeyError
        if the id key is not found in the polygon index dict
    AttributeError
        if geotransform parameters do not match
    """
    # setup output dict
    try:
        stats = {key: {id_key: key} for key in list(polygon_index_dict.keys())}
    except KeyError:
        raise KeyError(
            f"column key: {id_key} not found in polygon_index_dict, incorrectly made check function for details on how to run it"
        )

    for raster_path in raster_path_list:
        if not raster.check_dimensions(
            raster_path_a=polygon_index_base_raster_path, raster_path_b=raster_path
        ):
            raise AttributeError(
                f"geotransform parameters of the index_base_raster: {polygon_index_base_raster_path} and raster: {raster_path} have to match"
                ": \n check the overlay of your input shapefile and input rasters"
            )

        # if a vrt loop through the bands
        if "vrt" in os.path.splitext(raster_path)[1]:
            for band in range(1, raster.gdal_info(raster_path)["band_count"]):
                array = raster.raster_to_array(raster_path, band_num=band)
                for key, value in polygon_index_dict.items():
                    for stat in field_stats:
                        (
                            column_name,
                            numpy_function,
                        ) = generate_zonal_stats_column_and_function(
                            input_name=f"{statistic_name}_{band}",
                            statistic=stat,
                        )
                        stats[key][column_name] = numpy_function(array[value])

        else:
            array = raster.raster_to_array(raster_path)
            for key, value in polygon_index_dict.items():
                for stat in field_stats:
                    (
                        column_name,
                        numpy_function,
                    ) = generate_zonal_stats_column_and_function(
                        input_name=f"{statistic_name}_{band}", statistic=stat
                    )
                    stats[key][column_name] = numpy_function(array[value])

    if not out_dict:
        stats = dict_to_dataframe(in_dict=stats)

    return stats


##########################
def single_raster_zonal_stats(
    input_raster_path: str,
    input_shapefile_path: str,
    field_stats: list = ["min", "max", "mean", "sum", "stddev"],
    statistic_name: str = "",
    id_key: str = "wpid",
    out_dict: bool = False,
):
    """carries out a zonal stats analysis and organises the results
        in a dictionary or dataframe according to the requirements

    Parameters
    ----------
    input_raster_path : str
         path to the raster containing the data being analysed
    input_shapefile_path : str
        shapefile to retrieve raster indices for per feature
    field_stats : list, optional
        list of statistics to carry out, also used in the column names, by default ["min", "max", "mean", "sum", "stddev"]
    statistic_name : str, optional
        name/ identifier to give to the stat calculated, by default ""
    id_key : str, optional
        name of shapefile column/feature dictionary key providing the feature indices, by default "wpid"
    out_dict : bool, optional
        if True exports a dict instead of a dataframe, by default False

    Returns
    -------
    Union[dict, pandas.DataFrame]
        statistics calculated

    Raises
    ------
    AttributeError
        if array sizes do not match
    """
    # check if column identifier exists in the shapefile
    vector.check_column_exists(shapefile_path=input_shapefile_path, column=id_key)
    # check that the crs match
    vector.compare_raster_vector_crs(
        raster_path=input_raster_path, vector_path=input_shapefile_path
    )

    # setup output dictionary
    stats = {}

    logger.info("calculating all feature statistics...")

    skipped_features = {}

    with fiona.open(input_shapefile_path, "r") as shp:
        geoms = [feature["geometry"] for feature in shp]
        indices = [feature["properties"][id_key] for feature in shp]
        for geom, tid in zip(geoms, indices):
            stats[tid] = {id_key: tid}
            bbox = shape(geom).bounds

            with rasterio.open(input_raster_path, "r") as src:
                raster_window = rasterio.windows.from_bounds(
                    left=bbox[0],
                    bottom=bbox[1],
                    right=bbox[2],
                    top=bbox[3],
                    transform=src.transform,
                )

                if any(
                    value < 0
                    for value in [raster_window.col_off, raster_window.row_off]
                ):
                    if raster_window.col_off < 0:
                        skipped_features[tid] = (tid, "feature col offset is negative")
                    if raster_window.row_off < 0:
                        skipped_features[tid] = (tid, "feature row offset is negative")
                    for stat in field_stats:
                        column_name, __ = generate_zonal_stats_column_and_function(
                            input_name=statistic_name, statistic=stat
                        )
                        stats[tid][column_name] = np.nan

                else:
                    window_transform = rasterio.windows.transform(
                        raster_window, src.transform
                    )
                    # set shape to minimum if smaller than 1
                    if raster_window.width < 1:
                        width = 1
                    else:
                        width = int(round(raster_window.width))

                    if raster_window.height < 1:
                        height = 1
                    else:
                        height = int(round(raster_window.height))

                    col_offset = int(round(raster_window.col_off))
                    row_offset = int(round(raster_window.row_off))

                    window_shape = (height, width)

                    window_array = src.read(
                        1,
                        window=rasterio.windows.Window(
                            col_offset, row_offset, width, height
                        ),
                    )
                    window_array = np.where(
                        window_array == src.nodata, np.nan, window_array
                    )

                    geom_array = features.rasterize(
                        [(geom, 1)],
                        out_shape=window_shape,
                        transform=window_transform,
                        all_touched=True,
                        fill=0,
                        dtype="uint8",
                    )

                    if geom_array.shape != window_array.shape:
                        raise AttributeError(
                            "rasterized polygon array calculated in single_raster_zonal_stats "
                            "should be the same as the raster window array: \n check the overlay of your input shapefile and input raster"
                        )

                    temp_array = np.where(geom_array == 1, window_array, np.nan)

                    for stat in field_stats:
                        (
                            column_name,
                            numpy_function,
                        ) = generate_zonal_stats_column_and_function(
                            input_name=statistic_name, statistic=stat
                        )
                        stats[tid][column_name] = numpy_function(temp_array)

    if skipped_features:
        logger.info(skipped_features)

    if not out_dict:
        stats = dict_to_dataframe(in_dict=stats)

    return stats


#################################
def raster_count_statistics(
    input_raster_path: str,
    output_csv: str = None,
    out_dict: bool = False,
    categories_dict: dict = None,
    category_name: str = "landcover",
):
    """count the occurrence of unique values in a raster
        and various other statistics around it for the
        the percentage of non nan cells that make up the raster

    Parameters
    ----------
    input_raster_path : str
        input raster to check
    output_csv : str, optional
        if the path to an output csv is provided then an csv and excel of the output is made, by default None
    out_dict : bool, optional
        if true outputs as dict otherwise as a dataframe, by default False
    categories_dict : dict, optional
        if a dict is provided uses the dic tto assign names/categories to the values found., by default None
    category_name : str, optional
        if a category dict is provided this is use dot name the new dict column made, by default "landcover"

    Returns
    -------
    tuple
        dataframe/dict: dict or dataframe of various statistics related to the counted raster values, path to the csv
    """
    counts_list = raster.count_raster_values(input_raster_path)

    if categories_dict:
        # reverse categories_dict
        categories_dict_reversed = {
            categories_dict[key]: key
            for key in categories_dict.keys()
            if not isinstance(categories_dict[key], list)
        }

        # create counts dictionary
        categories_list = []
        for count_dict in counts_list:
            count_dict[category_name] = categories_dict_reversed[count_dict["value"]]
            categories_list.append(count_dict)

        categories_stats = {_dict[category_name]: _dict for _dict in categories_list}

    else:
        categories_stats = {_dict["value"]: _dict for _dict in counts_list}

    if not out_dict:
        categories_stats = dict_to_dataframe(in_dict=categories_list)

    if output_csv:
        # create output subfolders as needed
        output_dir = os.path.dirname(output_csv)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_table(table=categories_stats, output_file_path=output_csv)

    return categories_stats, output_csv


########################################################
# raster/Array statistics Functions
########################################################
def calc_dual_array_statistics(
    a: Union[str, np.ndarray],
    b: Union[str, np.ndarray],
    calc_function: FunctionType,
    output_raster_path: str,
    template_raster_path: str = None,
    output_nodata: float = -9999,
    mask_to_template: bool = False,
):
    """given two rasters or arrays applies the given function to them and
    outputs the result as a new raster

    Parameters
    ----------
    a : Union[str, np.ndarray]
        path to raster a or array a to calculate a statistic from in combo with b
    b : Union[str, np.ndarray]
        path to raster b or array b to calculate a statistic from in combo with a
    calc_function : FunctionType
         function to use (a / b etc)
    output_raster_path : str
        path to output the calculated statistic too as a raster
    template_raster_path : str, optional
        path to the template raster providing the metadata for the output raster if used if not provided uses the input raster, by default None
    output_nodata : float, optional
        nodata for the output raster if used, by default -9999
    mask_to_template : bool, optional
        If True masks to the template raster and the output array will be masked to it, by default False

    Returns
    -------
    str
        path to the output raster

    Raises
    ------
    AttributeError
        if not a raster or array is provided for a
    AttributeError
        if not a raster or array is provided for b
    AttributeError
        if no template raster is provided
    AttributeError
        if no template raster is provided
    """
    # open inputs and extract array if needed
    if isinstance(a, np.ndarray):
        a = a

    elif os.path.exists(a):
        raster.check_gdal_open(a)
        a = raster.raster_to_array(input_raster_path=a)

    else:
        raise AttributeError("please provide either a raster path or an np.ndarray")

    if isinstance(b, np.ndarray):
        b = b

    elif os.path.exists(b):
        raster.check_gdal_open(b)
        b = raster.raster_to_array(input_raster_path=b)

    else:
        raise AttributeError("please provide either a raster path or an np.ndarray")

    # vectorize function and apply to the arrays
    vfunc = np.vectorize(calc_function)
    output = vfunc(a, b)

    # retrieve the needed metadata
    if template_raster_path:
        metadata = template_raster_path
    else:
        if isinstance(a, np.ndarray) & isinstance(b, np.ndarray):
            raise AttributeError(
                "if providing only arrays please provide a template_raster_path for the output or a raster as input"
            )

        elif os.path.exists(a):
            metadata = a
        else:
            raise AttributeError("please provide a template_raster_path")

    # output to raster
    raster.array_to_raster(
        metadata=metadata,
        output_raster_path=output_raster_path,
        input_array=output,
        output_nodata=output_nodata,
    )

    if mask_to_template:
        assert os.path.exists(
            template_raster_path
        ), "íf masking to the template raster please provide one"
        raster.mask_raster(
            mask_raster_path=template_raster_path,
            input_raster_path=output_raster_path,
            output_raster_path=output_raster_path,
            output_nodata=output_nodata,
        )

    raster.check_gdal_open(output_raster_path)

    return output_raster_path


##########################
def calc_single_array_numpy_statistic(
    input: Union[str, np.ndarray],
    numpy_function: FunctionType,
    output_raster_path: str = None,
    template_raster_path: str = None,
    output_nodata: float = -9999,
    mask_to_template: bool = False,
    **kwargs,
):
    """given a raster or an array calculates a statistic for the input
        using the given function, if an output raster path is specified
        it attaches the new value to all non nan cells and
        outputs to the path otherwise it returns the computed statistic.

    Parameters
    ----------
    input : Union[str, np.ndarray]
        input to calculate a statistic for can be a raster path or a numpy array
    numpy_function : FunctionType
        function to use (np.nanpercentile etc)
    output_raster_path : str, optional
        path to output the calculated statistic too as a raster if you so wish, by default None
    template_raster_path : str, optional
        path to the template raster providing the metadata for the output raster if used if not provided uses the input raster, by default None
    output_nodata : float, optional
        nodata for the output raster, by default -9999
    mask_to_template : bool, optional
        If True masks to the template raster, by default False

    Returns
    -------
    Union[str,float]
        path to the output raster or calculated statistics

    Raises
    ------
    AttributeError
        if not a raster or array is provided
    AttributeError
        if an array is provided but no template raster
    AttributeError
        if not a raster or array is provided
    """
    # open input and extract array if needed
    if isinstance(input, np.ndarray):
        item = input

    elif os.path.exists(input):
        raster.check_gdal_open(input)
        item = raster.raster_to_array(input_raster_path=input)

    else:
        raise AttributeError("please provide either a raster path or an np.ndarray")

    # calculate chosen statistic
    output = numpy_function(item, **kwargs)

    if output_raster_path:
        if template_raster_path:
            # if a template is available mask to the template and assign percentile value
            template = raster.raster_to_array(input_raster_path=template_raster_path)
            output = np.where(np.isnan(template), np.nan, output)
            metadata = template_raster_path

        else:
            # assign statistic where not nan
            output = np.where(np.isnan(item), np.nan, output)

            if isinstance(input, np.ndarray):
                raise AttributeError(
                    "if providing an array and outputting to raster please provide a template_raster_path for the output or a raster as input"
                )

            elif os.path.exists(input):
                metadata = input
            else:
                raise AttributeError(
                    "please provide either a raster path or an np.ndarray"
                )

        # output to raster
        raster.array_to_raster(
            metadata=metadata,
            output_raster_path=output_raster_path,
            input_array=output,
            output_nodata=output_nodata,
        )

    if mask_to_template:
        assert os.path.exists(
            template_raster_path
        ), "íf masking to the template raster please provide one"
        raster.mask_raster(
            mask_raster_path=template_raster_path,
            input_raster_path=output_raster_path,
            output_raster_path=output_raster_path,
            output_nodata=output_nodata,
        )

        raster.check_gdal_open(output_raster_path)

        output = output_raster_path

    return output


##########################
def calc_multiple_array_numpy_statistic(
    input: Union[str, list],
    numpy_function: FunctionType,
    output_raster_path: str,
    template_raster_path: str = None,
    output_nodata: float = -9999,
    mask_to_template: bool = False,
    axis: int = 0,
    **kwargs,
):
    """given a a vrt or a list of
        rasters or arrays calculates a per pixel statistic on the input(s)
        using the given function and on the specified axis and outputs to raster

    Parameters
    ----------
    input : Union[str, list]
        input to calculate a statistic for can be a vrt, list of arrays or raster_paths
    numpy_function : FunctionType
        function to use (np.nanpercentile etc)
    output_raster_path : str
        path to output the calculated statistic too as a raster if it is an array
    template_raster_path : str, optional
        path to the template raster providing the metadata for the output, by default None
    output_nodata : float, optional
        nodata for the output, by default -9999
    mask_to_template : bool, optional
        If True masks to the template raster, by default False
    axis : int, optional
        numpy optional argument required by multiple functions to determine the axis on which to apply the function, by default 0

    Returns
    -------
    str
        outputted raster path

    Raises
    ------
    AttributeError
        if a list input does not cotnain arrays or rasters
    AttributeError
        if the input is not a list, array or raster/vrt path
    AttributeError
        if ony one raster or array is provided
    AttributeError
        if providing arrays provide a template raster
    AttributeError
        if a list input does not cotnain arrays or rasters
    AttributeError
        if the input is not a list, array or raster/vrt path
    """
    array_list = []

    # open input and extract arrays list of arrays, list of rasters or a vrt or a raster is accepted
    if isinstance(input, list):
        for item in input:
            if isinstance(item, np.ndarray):
                array_list.append(item)
            elif os.path.exists(item):
                raster.check_gdal_open(item)
                array = raster.raster_to_array(item)
                array_list.append(array)
            else:
                raise AttributeError(
                    "if providing a list as input please provide either arrays or raster tif paths"
                )

    elif os.path.exists(input):
        raster.check_gdal_open(input)
        band_count = raster.gdal_info(input)["band_count"]

        for i in range(0, band_count):
            item = raster.raster_to_array(input_raster_path=input, band_num=i + 1)

            array_list.append(item)

    else:
        raise AttributeError(
            "please provide either a vrt or a list of rasters or arrays"
        )

    if len(array_list) < 2:
        raise AttributeError(
            "number of arrays is 1 please use calc_single_array_statistics instead"
        )

    # get output_info
    if template_raster_path:
        metadata = template_raster_path
    else:
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                raise AttributeError(
                    "if providing arrays please provide a template_raster_path for the output"
                )
            elif os.path.exists(input[0]):
                metadata = input[0]
            else:
                raise AttributeError(
                    "if providing a list as input please provide either arrays or raster tif paths"
                )
        elif os.path.exists(input):
            metadata = input
        else:
            raise AttributeError(
                "please provide either a vrt or a list of rasters or arrays"
            )
    # stack arrays
    array_stack = np.stack(array_list)

    # calculate chosen statistic
    try:
        output_array = numpy_function(array_stack, axis=axis, **kwargs)
    except:
        logger.info(
            "attempt to run the function with the axis argument failed, reattempting without"
        )
        output_array = numpy_function(array_stack, **kwargs)

    # output monthly potet as raster
    raster.array_to_raster(
        metadata=metadata,
        output_raster_path=output_raster_path,
        input_array=output_array,
        output_nodata=output_nodata,
    )

    if mask_to_template:
        assert os.path.exists(
            template_raster_path
        ), "íf masking to the template raster please provide one"
        raster.mask_raster(
            mask_raster_path=template_raster_path,
            input_raster_path=output_raster_path,
            output_raster_path=output_raster_path,
            output_nodata=output_nodata,
        )

    raster.check_gdal_open(output_raster_path)

    return output_raster_path


##########################
def mostcommonzaxis(array_stack, **kwargs):
    """find the most common/frequent element per cell on the z axis of a stack of arrays

    Parameters
    ----------
    array_stack : _type_
        stack of arrays to analyse

    Returns
    -------
    nimpy.ndarray
        2 dimensional array of most occurring value
    """
    # identify most occuring values
    values = list(set(list(np.unique(array_stack[~np.isnan(array_stack)]))))

    # create an empty output frame and set to nan
    output_array = np.empty(array_stack.shape[1:])
    output_array[:] = np.nan
    # count the nans exisitng across time for the baseline
    highest_occurrence = np.count_nonzero(np.isnan(array_stack), axis=0)

    for value in values:
        current_count = np.nansum(array_stack == value, axis=0)
        output_array = np.where(current_count > highest_occurrence, value, output_array)
        highest_occurrence = np.where(
            current_count > highest_occurrence, current_count, highest_occurrence
        )

    return output_array
