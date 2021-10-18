
"""
minbuza_waterpip project

statistics support functions
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

from waterpip.scripts.support import raster

########################################################
# Dataframe Functions
########################################################
def dict_to_dataframe(
    in_dict: Union[dict, list], 
    orient: str = 'index', 
    combine_keys: bool=False):
    """
    transform dict, nested dict or list of dicts to dataframe
    """
    # transform dictionary to dataframe
    if isinstance(in_dict, list):
        # list of dicts to dataframe
        out_dataframe = pd.DataFrame(in_dict)

    else:
        keys = list(in_dict.keys())
        if isinstance(in_dict[keys[0]], dict):
            if combine_keys:
                # nested dict to dataframe
                temp_dict = {(i,j): in_dict[i][j] for i in in_dict.keys() for j in in_dict[i].keys()}
            else: 
                temp_dict = in_dict
        else:
            # dict to dataframe
            temp_dict = in_dict

        out_dataframe = pd.DataFrame.from_dict(
            temp_dict,
            orient=orient)

    return out_dataframe

##########################
def output_table(
    table: Union[list,dict, pd.DataFrame],
    output_file_path: str, 
    output_formats: list = ['.csv', '.xlsx'],
    csv_seperator: str = ';',
    orient: str = 'columns'):
    """
    Description:
        takes a list of dicitonaries, a dictionary, a nested dictionary or 
        a dataframe and outputs a dataframe as a table to file usign the 
        given output_path. Auto outputs in all formats specified in the output file,
        as well as those in the output formats list 

    Args:
        table: table to process and output can be in various formats
        output_file_path: path to output the files too
        output_formats: other formats to output the file as
        csv_seperator: if output format includes csv then this sep
        is used
        orient: orientation to cosntruct the dataframe in when working from dict
        or nested dict

    Return:
        int: 0    
    """
    pd_processes = {'.csv': pd.DataFrame.to_csv, '.xlsx': pd.DataFrame.to_excel,
        '.pkl': pd.DataFrame.to_pickle, '.json': pd.DataFrame.to_json, 
        '.tex': pd.DataFrame.to_latex, '.pq': pd.DataFrame.to_parquet
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
        output_df = dict_to_dataframe(table, orient=orient)
    
    for out_path, outf, out_process in output_paths:
        if outf == '.csv':
            output_df.to_csv(out_path, sep=csv_seperator)
        else:
            out_process(output_df, out_path)
    
    return 0

########################################################
# Mathematical Functions
########################################################
def latlon_dist(lat1,lat2,lon1,lon2):

    #radius of the earth
    R = 6373.0

    lat1 = math.radians(abs(lat1))
    lon1 = math.radians(abs(lon1))

    lat2 = math.radians(abs(lat2))
    lon2 = math.radians(abs(lon2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2

    # Haversine formula
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance

##########################
def ceiling_divide(a,b):
    if b == 0:
        c = np.nan
    else:
        c = a / b
        if c > 1:
            c = 1
    return c

##########################
def floor_minus(a,b):
    c = a - b
    if c < 0:
        c = 0
    return c


########################################################
# Statistical Functions
########################################################
def calc_dual_array_statistics(
    a: Union[str, np.ndarray],
    b: Union[str, np.ndarray],
    calc_function: FunctionType,
    output_raster_path: str,
    template_raster_path: str=None,
    output_nodata: float=-9999,
    mask_to_template: bool=False):
    """
    Description:
    given two rasters or arrays applies the given function to them and 
    outputs the resutl as a new raster
    
    Args:
        a: path to raster a or array a to calculate a statistic from in combo with b
        b: path to raster b or array b to calculate a statistic from in combo with a
        calc_function: function to use (a / b etc)
        output_raster_path: path to output the calculated
        statistic too as a raster
        template_raster_path: path to the template raster 
        providing the metadata for the output raster if used, 
        if not provided uses the input raster
        output_nodata: nodata for the output raster if used
        mask_to_template: If True masks to the template raster 
        and the output array will be masked to it


    Return:
        str: path to the output raster
    """
    # open inputs and extract array if needed
    if isinstance(a,np.ndarray):
        a = a

    elif os.path.exists(a):
        raster.check_gdal_open(a)
        a = raster.raster_to_array(input_raster_path=a)

    else:
        raise AttributeError('please provide either a raster path or an np.ndarray')

    if isinstance(b,np.ndarray):
        b = b

    elif os.path.exists(b):
        raster.check_gdal_open(b)
        b = raster.raster_to_array(input_raster_path=b)

    else:
        raise AttributeError('please provide either a raster path or an np.ndarray')

    # vectorize function and apply to the arrays
    vfunc = np.vectorize(calc_function)
    output = vfunc(a,b)

    # retrieve the needed metadata
    if template_raster_path:
        metadata = template_raster_path
    else:
        if isinstance(a,np.ndarray) & isinstance(b,np.ndarray):
            raise AttributeError('if providing only arrays please provide a template_raster_path for the output or a raster as input')

        elif os.path.exists(a):
            metadata = a
        else:
            raise AttributeError('please provide a template_raster_path')

    # output to raster
    raster.array_to_raster(
        metadata=metadata,
        output_raster_path=output_raster_path,
        input_array=output,
        output_nodata=output_nodata)

    if mask_to_template:
        assert os.path.exists(template_raster_path), 'íf masking to the template raster please provide one'
        raster.mask_raster(
            mask_raster_path=template_raster_path,
            input_raster_path=output_raster_path,
            output_raster_path=output_raster_path,
            output_nodata=output_nodata
        )

    raster.check_gdal_open(output_raster_path)

    return output_raster_path

##########################
def calc_single_array_numpy_statistic(
    input: Union[str, np.ndarray],
    numpy_function: FunctionType,
    output_raster_path: str = None,
    template_raster_path: str=None,
    output_nodata: float=-9999,
    mask_to_template: bool=False,
    **kwargs):
    """
    Description:
        given a raster or an array calculates a statistic for the input 
        using the given function, if an output raster path is specified 
        it attaches the new value to all non nan cells and 
        outputs to the path otherwise it returns the computed statistic.

        Note: raster_to_array automatically sets the ouput nodata value to 
        numpy.nan so that numpy functions will work correctly, if providing 
        arrays you have to do this yourself
    
    Args:
        input: input to calculate a statistic for
        can be a raster path or a numpy array
        numpy_function: function to use (np.nanpercentile etc)
        output_raster_path: path to output the calculated
        statistic too as a raster if you so wish
        template_raster_path: path to the template raster 
        providing the metadata for the output raster if used, 
        if not provided uses the input raster
        output_nodata: nodata for the output raster if used
        kwargs: keyword arguments to use in the 
        specified numpy function
        mask_to_template: If True masks to the template raster 
        and the output array will be masked to it

    Return:
        Union[str,float] : outputted raster or calculated statistic
    """
    # open input and extract array if needed
    if isinstance(input,np.ndarray):
        item = input

    elif os.path.exists(input):
        raster.check_gdal_open(input)
        item = raster.raster_to_array(input_raster_path=input)

    else:
        raise AttributeError('please provide either a raster path or an np.ndarray')

    # calculate chosen statistic
    output = numpy_function(item,**kwargs)

    if output_raster_path:
        if template_raster_path:
            # if a template is available mask to the template and assign percentile value
            template = raster.raster_to_array(input_raster_path=template_raster_path)
            output = np.where(np.isnan(template), np.nan, output)
            metadata = template_raster_path

        else:
            # assign statistic where not nan
            output = np.where(np.isnan(item), np.nan, output)

            if isinstance(input,np.ndarray):
                raise AttributeError('if providing an array and outputting to raster please provide a template_raster_path for the output or a raster as input')

            elif os.path.exists(input):
                metadata = input
            else:
                raise AttributeError('please provide either a raster path or an np.ndarray')

        # output to raster
        raster.array_to_raster(
            metadata=metadata,
            output_raster_path=output_raster_path,
            input_array=output,
            output_nodata=output_nodata)

    if mask_to_template:
        assert os.path.exists(template_raster_path), 'íf masking to the template raster please provide one'
        raster.mask_raster(
            mask_raster_path=template_raster_path,
            input_raster_path=output_raster_path,
            output_raster_path=output_raster_path,
            output_nodata=output_nodata
        )

        raster.check_gdal_open(output_raster_path)

        output = output_raster_path

    return output

##########################
def calc_multiple_array_numpy_statistic(
    input: Union[str, list],
    numpy_function: FunctionType,
    output_raster_path: str,
    template_raster_path: str=None,
    output_nodata: float=-9999,
    mask_to_template: bool=False,
    **kwargs):
    """
    Description:
        given a a vrt or a list of 
        rasters or arrays calculates a per pixel statistic on the input(s) 
        using the given function and on the specified axis 

        Note: raster_to_array automatically sets the ouput nodata value to 
        numpy.nan so that numpy funcitons will work correctly, if providing 
        arrays you have to do this yourself
    
    Args:
        input: input to calculate a statistic for
        can be a vrt, list of arrays or raster_paths
        numpy_function: function to use (np.nanpercentile etc)
        output_raster_path: path to output the calculated
        statistic too as a raster if it is an array
        template_raster_path: path to the template raster 
        providing the metadata for the output
        output_nodata: nodata for the output
        kwargs: keyword arguments to use in the 
        specified numpy_array function
        mask_to_template: If True masks to the template raster 
        and the output array will be masked to it

    Return:
        int : 0
    """
    array_list = []

    # open input and extract arrays list of arrays, list of rasters or a vrt or a raster is accepted
    if isinstance(input,list):
        for item in input:
            if isinstance(item, np.ndarray):
                array_list.append(item)
            elif os.path.exists(item):
                raster.check_gdal_open(item)
                array = raster.raster_to_array(item)
                array_list.append(array)
            else:
                raise AttributeError('if providing a list as input please provide either arrays or raster tif paths')

    elif os.path.exists(input):
        raster.check_gdal_open(input)
        band_count = raster.gdal_info(input)['band_count']

        for i in range(0,band_count):
            item = raster.raster_to_array(
            input_raster_path=input,
            band_num=i+1)

            array_list.append(item)

    else:
        raise AttributeError('please provide either a vrt or a list of rasters or arrays')

    if len(array_list) < 2:
        raise AttributeError('number of arrays is 1 please use calc_single_array_statistics instead')

    # get output_info
    if template_raster_path:
        metadata = template_raster_path
    else:
        if isinstance(input,list):
            if isinstance(input[0], np.ndarray):
                raise AttributeError('if providing arrays please provide a template_raster_path for the output')
            elif os.path.exists(input[0]):
                metadata = input[0]
            else:
                raise AttributeError('if providing a list as input please provide either arrays or raster tif paths')
        elif os.path.exists(input):
            metadata = input
        else:
            raise AttributeError('please provide either a vrt or a list of rasters or arrays')
    # stack arrays
    array_stack = np.stack(array_list)

    # calculate chosen statistic
    output_array = numpy_function(array_stack, **kwargs)

    # output monthly potet as raster
    raster.array_to_raster(
        metadata=metadata,
        output_raster_path=output_raster_path,
        input_array=output_array,
        output_nodata=output_nodata)

    if mask_to_template:
        assert os.path.exists(template_raster_path), 'íf masking to the template raster please provide one'
        raster.mask_raster(
            mask_raster_path=template_raster_path,
            input_raster_path=output_raster_path,
            output_raster_path=output_raster_path,
            output_nodata=output_nodata
        )

    raster.check_gdal_open(output_raster_path)

    return 0

##########################
def mostcommonzaxis(array_stack, **kwargs):
    """
    Description:
        find the most common/frequent element per cell on the z axis of a 
        stack of arrays

        method:
        for all unique values compare the amount of times each one exists within a cell 
        compared to the previous value checked and  if it occurs more often set it to the 
        output array and update the count

    args:
        stack: stack of arrays to analyse

    Return:
        array: 2 dimensional array of most occurring value
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
        highest_occurrence = np.where(current_count > highest_occurrence, current_count, highest_occurrence)

    return output_array

##########################
def multiple_raster_zonal_stats(
    template_raster_path: str,
    input_shapefile_path: str,    
    raster_path_list: list,
    analyses: list,
    out_dict: bool=False,
    id_key: str = 'wpid',    
    **kwargs): 
    """
    Description:
        rasterize the features in the shapefile according to the
        template raster provided and extract the indices corresponding 
        to each feature

        uses the dictionary made and the input_raster_path and calculates statistics per field
        in the raster using the index to identify the cells within each field

    Args:
        template_raster_path: path to the raster contianing the metadata needed
        input_shapefile_path: shapefile to retireve raster indices for per feature
        id_key: name of shapefile column/feature dictionary key providing the feature indices 
        wpid is a reliable autogenerated index provided while making the crop mask
        (note: also handy for joining tables and the crop mask shape/other shapes back later) 
        raster_path_list: list of paths to rasters to analyse also accepts and loops 
        through vrts inside the list
        analyses: list of tuples containing statistic to calculate [0] and 
        numpy function that matches it [1]
        out_dict: if True exports a dict instead of a dataframe

    Return:        
        dict/dataframe: dictionary or dataframe of calculated stats, with the key being the field id
    """
    index_dict = raster.create_polygon_index_dict(
        template_raster_path=template_raster_path,
        input_shapefile_path=input_shapefile_path,
        id_key=id_key,    
    )

    stats_df = equal_dimensions_zonal_stats(
        polygon_index_base_raster_path=template_raster_path,
        polygon_index_dict=index_dict,
        raster_path_list=raster_path_list,
        analyses=analyses,
        out_dict=out_dict
    )

    return stats_df

##########################
def equal_dimensions_zonal_stats(
    polygon_index_dict: dict,
    polygon_index_base_raster_path: str,
    raster_path_list: list,
    analyses: list,
    out_dict: bool=False,
    **kwargs):
    """
    Description:
        takes a dictionary made using create_polygon_index_dict
        and an input_raster_path and calculates statistics per field
        in the raster using the index to identify the cells within each field

        NOTE: The bounds and resolution of the tif the polygon_index_dict
        is based on and the raster being analysed needs to be the same. 
        that is why you need to provide it so that it can be checked each time.

    Args:
        polygon_index_dict: dict providing the indices to analyse
        polygon_index_base_raster_path: raster that formed the template for the polygon
        index_dict
        raster_path_list: list of paths to rasters to analyse also accepts and loops through vrts inside
        the list
        analyses: list of tuples containing statistic to calculate [0] and 
        numpy function thatbmatches it [1]
        out_dict: if True exports a dict instead of a dataframe

    Return:
        dict/dataframe: dictionary or dataframe of calculated stats, with the key being the field id

    """
    stats = {key: {} for key in list(polygon_index_dict.keys())}


    for raster_path in raster_path_list:
        if not raster.check_dimensions(
            raster_path_a=polygon_index_base_raster_path,
            raster_path_b=raster_path):
            raise AttributeError('geotransform parameters of the index_base_raster: {} and raster: {} have to match'.format(polygon_index_base_raster_path, raster_path))

        raster_name = os.path.splitext(os.path.basename(raster_path))[0]

        # if a vrt loop through the bands
        if 'vrt' in os.path.splitext(raster_path)[1]:
            for band in range(1, raster.gdal_info(raster_path)['band_count']):
                band_name = raster.gdal_info(raster_path, band_num=band)['band_name']
                array = raster.raster_to_array(raster_path, band_num=band)
                for key, value in polygon_index_dict.items():
                    for statistic in analyses:
                        band_specific_statistic_key = band_name + '_' + statistic[0]
                        stats[key][band_specific_statistic_key] = statistic[1](array[value], **kwargs)

        else:
            array = raster.raster_to_array(raster_path)
            for key, value in polygon_index_dict.items():
                for statistic in analyses:
                    raster_specific_statistic_key = raster_name + '_' + statistic[0]
                    stats[key][raster_specific_statistic_key] = statistic[1](array[value], **kwargs)

    if not out_dict:
        stats = dict_to_dataframe(in_dict=stats)

    return stats

##########################
def single_raster_zonal_stats(
    input_raster_path: str,
    input_shapefile_path: str,
    analyses: list,
    id_key: str='wpid',
    out_dict: bool=False):
    """
    Description:
        carries out a zonal stats analysis and organises the results 
        in a dictionary or dataframe according to the requirements 
        
    Args:
        template_raster_path: path to the raster contianing the data being analysed
        input_shapefile_path: shapefile to retrieve raster indices for per feature
        id_key: name of shapefile column/feature dictionary key providing the feature indices 
        wpid is a reliable autogenerated index provided while making the crop mask
        (note: also handy for joining tables and the crop mask shape/other shapes back later) 
        analyses: list of tuples containing statistic to calculate [0] and 
        numpy function thatbmatches it [1]
        out_dict: if True exports a dict instead of a dataframe

    Return:
        dict\dataframe: dictionary ro dataframe of calculated stats, with the key being the field id
    """
    # setup output dictionary
    name = os.path.splitext(os.path.basename(input_raster_path))[0]
    stats = {}

    print('calculating all feature statistics...')

    skipped_features = {}

    with fiona.open(input_shapefile_path, 'r') as shp:
        geoms = [feature['geometry'] for feature in shp]
        indices = [feature['properties'][id_key] for feature in shp]
        for geom, tid in zip(geoms, indices):
            stats[tid] = {}
            bbox = shape(geom).bounds

            with rasterio.open(input_raster_path, 'r') as src:
                raster_window = rasterio.windows.from_bounds(
                    left = bbox[0],
                    bottom = bbox[1],
                    right=bbox[2],
                    top=bbox[3], 
                    transform=src.transform)

                if any(value < 0 for value in [raster_window.col_off, raster_window.row_off]): 
                    if raster_window.col_off < 0:
                        skipped_features[tid] = (tid,'feature col offset is negative')
                    if raster_window.row_off < 0:
                        skipped_features[tid] = (tid,'feature row offset is negative')
                    for statistic in analyses:
                        statistic_name = name + '_' + statistic[0]
                        stats[tid][statistic_name] = np.nan             
                else:
                    window_transform = rasterio.windows.transform(raster_window, src.transform)
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

                    window_array = src.read(1, window=rasterio.windows.Window(col_offset, row_offset, width, height))
                    window_array = np.where(window_array == src.nodata, np.nan, window_array)
                
                    geom_array = features.rasterize(
                        [(geom, 1)],
                        out_shape=window_shape,
                        transform=window_transform,
                        all_touched=True,
                        fill=0,
                        dtype='uint8')

                    if geom_array.shape != window_array.shape:
                        raise AttributeError('rasterized array in calc single raster zonal stats should be the same as the window array')

                    temp_array = np.where(geom_array == 1, window_array, np.nan)

                    for statistic in analyses:
                        statistic_name = name + '_' + statistic[0]
                        stats[tid][statistic_name] = statistic[1](temp_array)
    
    if skipped_features:
        print(skipped_features)
    
    if not out_dict:
        stats = dict_to_dataframe(in_dict=stats)

    return stats


if __name__ == "__main__":
    start = default_timer()
    args = sys.argv
    
    try:
        stats = single_raster_zonal_stats(
            input_raster_path=r"C:\Users\roeland\workspace\projects\waterpip\waterpip_dir\cotton_test\L3\04_results\L3_cotton_bf_20200305_20201006.tif",
            input_shapefile_path=r"C:\Users\roeland\workspace\projects\waterpip\testing\static\Selected4Analysis.shp",
            id_key='wpid',
            analyses=[('mean',np.nanmean), ('sum', np.nansum)],
            out_dict=False)

        print(stats)
    finally:
        end = default_timer()
        print('process duration: {}'.format(timedelta(seconds=round(end - start, 2))))

