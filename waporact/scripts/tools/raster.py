
"""
waporact package

raster functions (stand alone/support functions)
"""
##########################
# import packages
import os
from typing import Union
from datetime import timedelta
from timeit import default_timer

from posixpath import splitext
import shutil
from typing import Union
import copy 

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import rasterio
import fiona

import numpy as np

import math

########################################################
# General Functions
########################################################
def area_of_latlon_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel.

    Copied from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = math.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*math.sin(math.radians(f))
        zp = 1 + e*math.sin(math.radians(f))
        area_list.append(
            math.pi * b**2 * (
                math.log(zp/zm) / (2*e) +
                math.sin(math.radians(f)) / (zp*zm)))
    return pixel_size / 360. * (area_list[0] - area_list[1])

def check_gdal_open(
        file: str,
        return_ds: bool = False):
    """
    Description:
        Check if gdal can open a file. Raise RuntimeError when resulting dataset is None.

    Args:
        file: path to file.
        return_ds: whether to return the opened dataset. Default is False.
    
    Return: 
        str : gdal dataset in case return ds is True.

    Raise:
        FileNotFoundError: if the file fails to open
    """
    file = os.fspath(file)

    if not os.path.exists(file):
        raise FileNotFoundError(f'file {file} not found, check path')

    ds = gdal.OpenEx(file)
    if ds is None:
        error_message = 'corrupt file: ' + file 
        raise RuntimeError(error_message)

    if not return_ds:
        ds = None
    
    return ds

#################################
def set_band_descriptions(raster_file_path: str, band_names: list):
    """
    Description:
        add description to the raster file bands

    Args:
        raster_file_path: path to the raster file to edit
        band_names: list of names for the output bands in order of the 
        bands themselves, if a file path is passed it extracts the file name as
        the band name
    
    Return:
        0: once file is edited
    """
    assert os.path.exists(raster_file_path), 'raster_file_path must be an exisitng raster file'
    assert isinstance(band_names, list), 'band_names must be a list'

    bands = []
    # check if given band names arew files and if so retireve file name as band name and orer by band
    for band_num , _name in enumerate(band_names):
        if os.path.exists(_name):
            _name = os.path.splitext(os.path.basename(_name))[0]
        bands.append((band_num+1, _name))

    ds = gdal.Open(raster_file_path, gdal.GA_Update)
    for band, desc in bands:
        rb = ds.GetRasterBand(band)
        rb.SetDescription(desc)
    del ds

    return 0

#################################
def gdal_info(raster_path: str, band_num: int=1) -> dict:
    """
    Description:
        given a raster path builds a dictionary of the rasters metadata retrieved using gdal 
        python functionality

    Args:
        raster_path: path to the raster that has to be mined for metadata
        band_num: band to retrieve the metadata for

    Return:
        dict: dictionary of the rasters metadata
    """
    # open raster
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    Type = raster.GetDriver().ShortName
    if Type == 'HDF4' or Type == 'netCDF':
        raster = gdal.Open(raster.GetSubDatasets()[band_num][0])

    raster_band = raster.GetRasterBand(band_num)

    # build and fill metadata dictionary
    metadata = {}

    metadata['driver'] = raster.GetDriver()
    metadata['path'] = raster.GetDescription()
    metadata['proj'] = raster.GetProjection()
    metadata['crs'] = int(osr.SpatialReference(raster.GetProjection()).GetAttrValue("AUTHORITY", 1))
    metadata['geotransform'] = raster.GetGeoTransform()
    metadata['xsize'] = raster.RasterXSize
    metadata['ysize'] = raster.RasterYSize
    metadata['cell_count'] = metadata['xsize'] * metadata['ysize']
    metadata['xmin'] = metadata['geotransform'][0]
    metadata['ymax'] = metadata['geotransform'][3]
    metadata['xmax'] = metadata['xmin'] + (metadata['geotransform'][1] * metadata['xsize'])
    metadata['ymin'] = metadata['ymax'] + (metadata['geotransform'][5] * metadata['ysize'])
    metadata['xres'] = metadata['geotransform'][1]
    metadata['yres'] = metadata['geotransform'][5]
    if metadata['crs'] == 4326:
        central_lat = metadata['ymin'] + (metadata['ymax'] - metadata['ymin'])
        metadata['cell_area'] = area_of_latlon_pixel(pixel_size=metadata['xres'], center_lat=central_lat)
    else:
        metadata['cell_area'] = abs(metadata['xres']) * abs(metadata['yres'])
    metadata['cell_area_unit'] = 'meters_sq'
    metadata['gdal_data_type_code'] = raster_band.DataType
    metadata['gdal_data_type_name'] = gdal.GetDataTypeName(metadata['gdal_data_type_code'])
    metadata['nodata'] = raster_band.GetNoDataValue()
    metadata['project_window'] = (metadata['xmin'],  metadata['ymax'], metadata['xmax'],  metadata['ymin'])
    metadata['ogr_extent'] = (metadata['xmin'],  metadata['ymin'], metadata['xmax'],  metadata['ymax'])
    metadata['band_count'] = raster.RasterCount
    band_name = raster_band.GetDescription()
    if os.path.isfile(band_name):
        band_name = os.path.splitext(os.path.basename(band_name))[0]
    metadata['band_name'] = band_name

    raster = None

    return metadata

#################################
def check_dimensions(
    raster_path_a: str,
    raster_path_b: str):
    """
    Description:
        checks the dimension of the two given rasters to 
        make sure they match including crs

    Args:
        raster_path_a: path to the raster to compare
        raster_path_b path to the raster to compare

    return: 
        bool: True if they match, False if they dont
    
    """
    a_meta = gdal_info(raster_path_a)
    b_meta = gdal_info(raster_path_b)

    match = True

    if any( a != b for a,b in (
        (a_meta['crs'] , b_meta['crs']),
        (a_meta['xmin'] , b_meta['xmin']),
        (a_meta['ymax'] , b_meta['ymax']),
        (a_meta['xmax'] , b_meta['xmax']),
        (a_meta['ymin'] , b_meta['ymin']),
        (a_meta['xres'] , b_meta['xres']),
        (a_meta['yres'] , b_meta['yres']), 
        (a_meta['xsize'] , b_meta['xsize']), 
        (a_meta['ysize'] , b_meta['ysize']))): 

        match=False

    return match

#################################
def raster_to_array(input_raster_path: str, band_num: int=1, dtype:str = None, use_nan: bool=True) -> np.ndarray:
    """
    Description:
        Given a raster_path retrieve the raster array and output

    Args:
        input_raster_path: path to the input raster
        band_num: band to retrieve the array from        
        dtype: data type to retrieve the array in, optional if not provided attempts to 
        retrieve it from the raster metadata
        use_nan: if True transforms the input nodata values found in the array to np.nan

    Return:
        numpy.array: the array of the specified rasters band
    """
    datatypes = {"Byte": np.uint8, "uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16":  np.int16, "Int16":  np.int16, "uint32": np.uint32,
    "int32": np.int32, "float32": np.float32, "float64": np.float64, "complex64": np.complex64, "complex128": np.complex128,
    "Int32": np.int32, "Float32": np.float32, "Float64": np.float64, "Complex64": np.complex64, "Complex128": np.complex128,}
  
    input_dataset = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    Type = input_dataset.GetDriver().ShortName
    if Type == 'HDF4':
        input_band = gdal.Open(input_dataset.GetSubDatasets()[band_num][0])
    else:
        input_band = input_dataset.GetRasterBand(band_num)
    nodata = input_band.GetNoDataValue()

    if not dtype:
        dtype = gdal.GetDataTypeName(input_band.DataType)
    dtype = datatypes[dtype]

    array = input_band.ReadAsArray().astype(dtype)

    if use_nan:
        array = np.where(array == nodata, np.nan, array)

    input_dataset = input_band = None
    
    return array

#################################
def count_raster_values(
    input_raster_path: str
    ):
    """
    Description:
        count the occurrence of unique values in a raster
        and also calculate the percentage of non nan cells
        they make up

    Args:
        input_raster_path: input raster to check

    Return:
        list: list of tuples (value ,count)
    """
    check_gdal_open(input_raster_path)

    meta = gdal_info(input_raster_path)

    input_array = raster_to_array(input_raster_path)

    non_nan_cell_count = np.count_nonzero(~np.isnan(input_array))

    counts = np.unique(input_array[~np.isnan(input_array)], return_counts=True)

    counts_list = []

    for value, count in zip(list(counts[0]), list(counts[1])):
        # if a float is returned check if it cna be made an int
        if isinstance(value,float):
            if value.is_integer():
                value = int(value)
        percentage = round(count / non_nan_cell_count * 100, 3)
        counts_list.append({
            'value': value , 
            'count': count, 
            'percentage': percentage,
            'area': count * meta['cell_area'],
            'unit': meta['cell_area_unit']})

    input_raster_path = input_array = None

    return counts_list

#################################
def array_to_raster(
    metadata: Union[str, dict], 
    output_raster_path: str, 
    input_array: np.array, 
    band_num: int=1, 
    xsize: int = None,
    ysize: int=None,
    geotransform: list = None,
    input_crs: int=None,
    output_crs: int= None,
    output_nodata: float = None,
    output_gdal_data_type: int = None,
    creation_options: list=["TILED=YES", "COMPRESS=DEFLATE"],
    driver_format: str ='GTiff') -> str:
    """
    Description:
        output an input array as a raster using another raster as template
        metadata source (template must have the same dimensions as the input array)
    Args:
        metadata: can either be the path to the template raster
        containing the metadata needed to transform the raster
        or a dict containing the data directly.
        output_raster_path: path to output the new tiff too
        input_array: array too attach to the new raster
        band_num: band to attach the array too
        tries to use metadata if not provided
        xsize: output xsize, tries to use metadata if not provided
        ysize: output ysize,tries to use metadata if not provided
        geotransform: geotransform metadata list, 
        tries to use metadata if not provided
        input_crs: projection of the input array, tries to use metadata if not provided
        output_crs: projection of the output if different from the input
        output_nodata: nodata value of the the output, 
        output_gdal_data_type: output data type, uses the datatype of the 
        input array if not provided
        tries to use metadata if not provided
        creation_options: creation_options for gdal
        driver_format: driver for creation of the output, standard to Gtiff

    Return:
        str: path to the ouputted raster
    """
    datatypes = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 3, "Int16": 3, "uint32": 4,
    "int32": 5, "float32": 6, "float64": 7, "complex64": 10, "complex128": 11,
    "Int32": 5, "Float32": 6, "Float64": 7, "Complex64": 10, "Complex128": 11,}
    if isinstance(metadata, str):
        if os.path.exists(metadata):
            template_meta = gdal_info(metadata)
    elif isinstance(metadata,dict):
        template_meta=metadata
    else:
        if any(item is None for item in [xsize, ysize, geotransform, output_crs, output_nodata]):
            raise AttributeError('missing input, please provide all metadata inputs if using direct inputs')
        else:
            template_meta = {}
    
    for items in [('xsize',xsize),('ysize',ysize),('geotransform',geotransform),('proj',input_crs),('nodata',output_nodata)]:
        if items[1] is not None:
            template_meta[items[0]] = items[1]

    # set standard raster_driver
    template_meta['driver'] = gdal.GetDriverByName(driver_format)

    if not output_gdal_data_type:
        gdal_data_type = datatypes[str(np.array(list(input_array)).dtype)]
    else:
        gdal_data_type = output_gdal_data_type


    output_array = np.where(np.isnan(input_array), template_meta['nodata'], input_array)

    if output_crs:
        input_crs =  "EPSG:{}".format(template_meta['crs'])
        output_crs =  "EPSG:{}".format(output_crs)

        temp_raster_path = os.path.splitext(output_raster_path)[0] + '_temp.tif'

        if output_crs == input_crs:
            # check if output crs is the same as the input crs in that case transfromation is not needed
            temp_raster_path = output_raster_path
    
    else:
        temp_raster_path = output_raster_path

    output_raster = template_meta['driver'].Create(temp_raster_path,
                                            template_meta['xsize'],
                                            template_meta['ysize'],
                                            band_num,
                                            gdal_data_type,
                                            options=creation_options)

    output_raster.SetGeoTransform(template_meta['geotransform'])
    output_raster.SetProjection(template_meta['proj'])
    output_band = output_raster.GetRasterBand(band_num)
    output_band.SetNoDataValue(template_meta['nodata'])
    output_band.WriteArray(output_array)
    output_band.GetStatistics(0,1)

    output_raster = output_band = None

    check_gdal_open(temp_raster_path)

    if output_crs:
        if output_crs != input_crs:
            warped_raster = gdal.Warp(
                destNameOrDestDS=temp_raster_path, 
                srcDSOrSrcDSTab=output_raster_path,
                format='Gtiff',
                srcSRS=input_crs,
                dstSRS=output_crs,
                srcNodata=template_meta['nodata'],
                dstNodata=template_meta['nodata'],
                width=template_meta['xsize'],
                height=template_meta['ysize'],
                outputBounds=template_meta['ogr_extent'],
                outputBoundsSRS=input_crs,
                resampleAlg='near',
                options=creation_options
                )
            
            warped_raster = None
            check_gdal_open(output_raster_path)
            os.remove(temp_raster_path)

    set_band_descriptions(
        raster_file_path=output_raster_path,
        band_names=[output_raster_path])

    return output_raster_path


############################
def retrieve_raster_crs(raster_path: str) -> int:
    """
    Description:
        retrieve the crs number of the input raster

    Args:
        raster_path: path to the raster

    Return:
        int: number code of the crs
    """
    # open raster
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)

    # retrieve crs
    proj = raster.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    crs = int(srs.GetAttrValue("AUTHORITY", 1))

    # close raster
    raster = None

    # return code
    return crs

############################
def rasterize_shape(
    template_raster_path: str, 
    shapefile_path: str, 
    output_raster_path: str,
    reverse: bool = False,
    output_nodata=0,
    output_gdal_datatype = 1,
    column: str = None,
    all_touched: bool=False,
    creation_options: list=["TILED=YES", "COMPRESS=DEFLATE"]
    ):
    """
    Description:
        rasterizes a shapefile onto a template raster creating either a mask
        or an attribute (rasterized polygon) raster

    Args:
        template_raster_path: path to the raster on which the new 
        mask/ rasterized raster is based
        shapefile_path: path to the shape to rasterize
        output_path: path to output the mask too
        reverse: if True reverse the mask if masking
        output_nodata: nodata value to use for the output. if action=attribute 
        make sure you have a fitting nodatavalue
        output_gdal_datatype: datatype of the output raster. if action=attribute 
        make sure you have a fitting datatype (6: float, 1: int)
        column: if provided burns in (uses) the values in the column specified 
        in the shapefile otherwise 0,1
        all_touched: If True includes all cells touched by a geoemtry in 
        rasterization not just where the line goes through the center of the geom
        as a minimum 
        creation_options: gdal creation options

    Return:
        0: if the masked file is created correctly

    Raise:
        AttributeError: if inputs are specifed incorrectly
    """
    if all_touched:
        all_touched = 'TRUE'
    else:
        all_touched = 'FALSE'

    meta = gdal_info(template_raster_path)

    shape_object = ogr.Open(shapefile_path)
    shape_layer = shape_object.GetLayer()

    target_ds = gdal.GetDriverByName('GTiff').Create(
        output_raster_path, 
        meta['xsize'], 
        meta['ysize'], 
        1, 
        output_gdal_datatype,
        options=creation_options)

    target_ds.SetGeoTransform(meta['geotransform'])
    target_ds.SetProjection(meta['proj'])
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(output_nodata)
    band.FlushCache()

    if not column:
        gdal.RasterizeLayer(
            target_ds,
            [1],
            shape_layer,
            burn_values=[1], 
            options = ["ALL_TOUCHED={}".format(all_touched)])

    else:
        gdal.RasterizeLayer(
            target_ds,
            [1],
            shape_layer,
            options = ["ALL_TOUCHED={}".format(all_touched), "ATTRIBUTE={}".format(column)])

    band.GetStatistics(0,1)

    target_ds = band = None
    check_gdal_open(output_raster_path)

    return 0

############################
def create_polygon_index_dict(
    template_raster_path: str,
    input_shapefile_path: str,
    id_key: str = 'wpid',
    ):
    """
    Description:
        rasterize the features in the shapefile according to the
        template raster provided and extract the indices corresponding 
        to each feature

    Args:
        template_raster_path: path to the raster contianing the metadata needed
        input_shapefile_path: shapefile to retireve raster indices for per feature
        id_key: name of shapefile column/feature dictionary key providing the feature indices 
        wpid is a reliable autogenerated index provided while making the crop mask
        (note: also handy for joining tables and the crop mask shape/other shapes back later) 

    Return:
        dict: dictionary of raster indices per feature index
    """
    with rasterio.open(template_raster_path) as src:
        with fiona.open(input_shapefile_path, 'r') as shp:
            geoms = [feature['geometry'] for feature in shp]
            index = [feature['properties'][id_key] for feature in shp]

            polygon_index_dict = {}
            for idx, geom in zip(index, geoms):
                geom_rasterize = rasterio.features.rasterize([(geom, 1)],
                                        out_shape=src.shape,
                                        transform=src.transform,
                                        all_touched=True,
                                        fill=0,
                                        dtype='uint8')

                polygon_index_dict[idx] = np.where(geom_rasterize == 1)

    return polygon_index_dict

############################
def create_values_specific_mask(
    mask_values: list,
    input_raster_path: str,
    output_mask_raster_path: str,
    output_values_raster_path: str=None,
    output_crs= None,
    output_nodata: float=-9999,
    keep_values: bool= False
    ):
    """
    Description:
        masks to a list of specific values in the input raster
        setting it to 1 and all other cells to 0

    Args:        
        mask_values: values to mask the mask too 
        input_raster_path: path to the raster to mask
        output_mask_raster_path: path to output the 0,1 mask raster too
        can be the same as the input path
        output_values_raster_path: if an output path is provided will also output a
        masked raster that maintains the unmaksed values
        output_crs: if porvided outputs the mask to this projection
        and not that of the input raster
        output_nodata: output nodata used if creating a values raster

    Return:
        int: 0
    """
    check_gdal_open(input_raster_path)

    input_array = raster_to_array(input_raster_path)

    metadata = gdal_info(input_raster_path)

    values_array = np.isin(input_array,mask_values)
    
    count_occurence = values_array.sum()

    percentage_occurrence = count_occurence / metadata['cell_count'] * 100

    if count_occurence == 0:
        raise AttributeError('given values to mask too occurs zero times in the given raster, please specify another value')

    elif percentage_occurrence < 1:
        print("WARNING: given values to mask too cover less that 1 percent of the given raster: {}".format(percentage_occurrence))

    else:
        pass
    
    # create specific values array
    value_mask_array = np.where(values_array, input_array, np.nan)
    # create mask array
    mask_array = np.where(values_array, 1, np.nan)

    if output_crs:
        if metadata['crs'] == output_crs:
            output_crs = None

    if output_values_raster_path:
        array_to_raster(
            metadata=input_raster_path,
            input_array=value_mask_array,
            output_raster_path=output_values_raster_path,
            output_gdal_data_type=6,
            output_crs=output_crs,
            output_nodata=output_nodata)

        check_gdal_open(output_values_raster_path)

    array_to_raster(
        metadata=input_raster_path,
        input_array=mask_array,
        output_raster_path=output_mask_raster_path,
        output_gdal_data_type=1,
        output_crs=output_crs,
        output_nodata=0)

    check_gdal_open(output_mask_raster_path)

    return 0

############################
def mask_raster(
    mask_raster_path: str, 
    input_raster_path: str,
    output_raster_path: str,
    output_nodata: float):
    """
    Description
        mask a raster using a mask raster 
        setting no data value to specified one.
        Raster diemnsions have to match

    Args:
        mask_raster_path: path to the mask raster 
        input_raster_path: path to the raster to mask
        output_raster_path: path to output the masked raster too, can be the same as the input path
        output_nodata: output no data value

    Return:
        int: 0
    """
    check_gdal_open(mask_raster_path)

    if not check_dimensions(
        raster_path_a=mask_raster_path,
        raster_path_b=input_raster_path):

        raise AttributeError('geotransform parameters of the input: {} and mask: {} have to match'.format(mask_raster_path, input_raster_path))
        
    input_meta = gdal_info(input_raster_path)
    
    mask_array = raster_to_array(mask_raster_path)
    input_array = raster_to_array(input_raster_path)

    output_array = np.where(np.isnan(mask_array), np.nan, input_array)

    array_to_raster(
        metadata=input_meta,
        input_array=output_array,
        output_raster_path=output_raster_path,
        output_nodata=output_nodata)

    check_gdal_open(output_raster_path)

    return 0

############################
def match_raster(
    match_raster_path: str,
    input_raster_path: str, 
    output_raster_path: str,
    mask_raster_path: str = None,
    resample_method: str = 'near',
    output_crs: int = None,
    output_nodata: float = -9999,
    creation_options: list=["TILED=YES", "COMPRESS=DEFLATE"]):
    """
    Description:
        matches the input raster to the metadata of the match
        raster and outputs the result to the output_raster_path

        NOTE: if the input raster already matches the template raster it copies the input raster
        to the output location

        NOTE: if the path to a mask raster is provided it uses that to mask the input as
        well. this is recommended to be the same raster as the match_raster_path
    
    Args:
        match_raster_path: path to the raster providing the metadata to match 
        the input raster too
        input_raster_path: input raster to alter as needed
        output_raster_path: path to output the output raster too
        mask_raster_path: path to the raster to use to mask the input raster
        can be the same raster as the match raster (recommended) 
        resample_method: resample method to use if needed
        output_crs: output projection of the raster, 
        if not provided uses the nodata value of the template
        output_nodata: nodata of the output data, 
        if not provided uses the nodata value of the template
        mask_to_template: boolean if True also masks to the template,
        creation_options: creation options for gdal

    Return
        int: 0
    """
    check_gdal_open(match_raster_path)
    check_gdal_open(input_raster_path)

    if not os.path.exists(os.path.dirname(output_raster_path)):
        os.makedirs(os.path.dirname(output_raster_path))

    template_meta = gdal_info(match_raster_path)
    input_meta = gdal_info(input_raster_path)

    if not output_crs:
        output_crs =  "EPSG:{}".format(template_meta['crs'])

    # check if the projection wanted (crs), no data value and dimensions do not match (different to check dimensions)
    if any( a != b for a,b in (
        (output_crs , input_meta['crs']), 
        (output_nodata, input_meta['nodata']),
        (template_meta['xmin'] , input_meta['xmin']),
        (template_meta['ymax'] , input_meta['ymax']),
        (template_meta['xmax'] , input_meta['xmax']),
        (template_meta['ymin'] , input_meta['ymin']),
        (template_meta['xres'] , input_meta['xres']),
        (template_meta['yres'] , input_meta['yres']), 
        (template_meta['xsize'] , input_meta['xsize']), 
        (template_meta['ysize'] , input_meta['ysize']))): 

        output_crs =  "EPSG:{}".format(output_crs)
        input_crs =  "EPSG:{}".format(input_meta['crs'])
        bounds_crs = "EPSG:{}".format(template_meta['crs'])

        warped_raster = gdal.Warp(
            destNameOrDestDS=output_raster_path, 
            srcDSOrSrcDSTab=input_raster_path,
            format='Gtiff',
            srcSRS=input_crs,
            dstSRS=output_crs,
            srcNodata=input_meta['nodata'],
            dstNodata=output_nodata,
            width=template_meta['xsize'],
            height=template_meta['ysize'],
            outputBounds=template_meta['ogr_extent'],
            outputBoundsSRS=bounds_crs,
            resampleAlg=resample_method,
            options=creation_options
        )

        warped_raster = None

    else:
        shutil.copy2(src=input_raster_path,dst=output_raster_path)

    check_gdal_open(output_raster_path)

    if mask_raster_path:
        check_gdal_open(mask_raster_path)
        mask_raster(
            mask_raster_path=mask_raster_path,
            input_raster_path=output_raster_path,
            output_raster_path=output_raster_path,
            output_nodata=output_nodata)

    set_band_descriptions(
        raster_file_path=output_raster_path,
        band_names=[output_raster_path])

    check_gdal_open(output_raster_path)

    return 0

############################
def build_vrt(
    raster_list: list, 
    output_vrt_path: str, 
    action: str='space'):
    """
    Description:
        builds either a spatial or timeseries vrt 
        using the given rasterlist

    Args:
        raster_list: list of rasters to combine
        action: type of vrt to build
            space: combines the rasters together as tiles to make a single image
            (assumes they lie next to each other)
            time: makes a timeseries out of the rasters 
            (assumes they have the same extent and resolution)
        output_vrt_path: path to output the vrt too

    Return:
        0 : vrt is created
    """
    if action == 'space':
        vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic')

    elif action =='time':
        vrt_options = gdal.BuildVRTOptions(separate=True)

    else:
        raise AttributeError('type must be either space or time')

    out_vrt = gdal.BuildVRT(
        destName=output_vrt_path,
        srcDSOrSrcDSTab=raster_list, 
        options=vrt_options,
        overwrite=True)
        
    # integrate creation options with vrt options
    
    out_vrt = None

    check_gdal_open(output_vrt_path)

    set_band_descriptions(
        raster_file_path=output_vrt_path,
        band_names=raster_list)

    return 0


