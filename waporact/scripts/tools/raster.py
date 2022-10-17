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

import shutil
from typing import Union

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import rasterio
import fiona

import numpy as np

import math

import logging

logger = logging.getLogger(__name__)


########################################################
# Functions used by waporact retrieval
########################################################
def reproject_coordinates(x: float, y: float, in_proj: int, out_proj: int):
    """transform a set of coordinates (x and y) from one projection to another

    Parameters
    ----------
    x : float
         x coordinate to transform
    y : float
         y coordinate to transform
    in_proj : int
        crs code/value to transform from
    out_proj : int
        crs code/value to transform to

    Returns
    -------
    tuple
        transformed /p projected value x and y
    """
    in_srs = osr.SpatialReference()
    in_srs.ImportFromEPSG(in_proj)
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(out_proj)

    if int(gdal.VersionInfo("VERSION_NUM")) >= 3000000 and hasattr(
        gdal.osr, "OAMS_TRADITIONAL_GIS_ORDER"
    ):
        in_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        out_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transf = osr.CoordinateTransformation(in_srs, out_srs)
    x_reprojected, y_reprojected = transf.TransformPoint(x, y)[:2]

    return x_reprojected, y_reprojected


#################################
def reproject_geotransform(
    geotransform: list, xsize: int, ysize: int, in_proj: int, out_proj: int
):
    """transform a geotransform object from one projection to another


    Parameters
    ----------
    geotransform : list
        geotransform object
    xsize : int
        array width
    ysize : int
        array height
    in_proj : int
        crs code/value to transform from
    out_proj : int
        crs code/value to transform to

    Returns
    -------
    list
        reprojected geotransform
    """
    xmin = geotransform[0]
    ymax = geotransform[3]

    xmax = geotransform[0] + (geotransform[1] * xsize)
    ymin = geotransform[3] + (geotransform[5] * ysize)

    xmin_reprojected, ymax_reprojected = reproject_coordinates(
        x=xmin, y=ymax, in_proj=in_proj, out_proj=out_proj
    )

    xmax_reprojected, ymin_reprojected = reproject_coordinates(
        x=xmax, y=ymin, in_proj=in_proj, out_proj=out_proj
    )

    xres = (xmax_reprojected - xmin_reprojected) / xsize
    yres = -(ymax_reprojected - ymin_reprojected) / ysize

    return (
        xmin_reprojected,
        xres,
        geotransform[2],
        ymax_reprojected,
        geotransform[4],
        yres,
    )


#################################
def check_gdal_open(file: str, return_ds: bool = False):
    """check with gdal if a file will open

    Parameters
    ----------
    file : str
        file path to check
    return_ds : bool, optional
        if true return the opened dataset, by default False

    Returns
    -------
    gdal dataset/None
        gdal dataset object or nothing

    Raises
    ------
    FileNotFoundError
        if no file is found
    RuntimeError
        if the file cannot be opened, is corrupt
    """
    file = os.fspath(file)

    if not os.path.exists(file):
        raise FileNotFoundError(f"file {file} not found, check path")

    ds = gdal.OpenEx(file)
    if ds is None:
        error_message = "corrupt file: " + file
        raise RuntimeError(error_message)

    if not return_ds:
        ds = None

    return ds


#################################
def set_band_descriptions(raster_file_path: str, band_names: list):
    """set band descriptions for the input raster file

    Parameters
    ----------
    raster_file_path : str
        path to the raster
    band_names : list
        list of tuples containing the band num and the band name to apply to it

    Returns
    -------
    int
        0
    """
    assert os.path.exists(
        raster_file_path
    ), "raster_file_path must be an exisitng raster file"
    assert isinstance(band_names, list), "band_names must be a list"

    bands = []
    # check if given band names arew files and if so retireve file name as band name and orer by band
    for band_num, _name in enumerate(band_names):
        if os.path.exists(_name):
            _name = os.path.splitext(os.path.basename(_name))[0]
        bands.append((band_num + 1, _name))

    ds = gdal.Open(raster_file_path, gdal.GA_Update)
    for band, desc in bands:
        rb = ds.GetRasterBand(band)
        rb.SetDescription(desc)
    del ds

    return 0


#################################
def gdal_info(raster_path: str, band_num: int = 1) -> dict:
    """retrieve and create a dictionary of raster metadata retrieved via gdal

    Parameters
    ----------
    raster_path : str
        raster to mine for metadata
    band_num : int, optional
        band to retrieve metadata from, by default 1

    Returns
    -------
    dict
        dictonary of input raster metadata
    """
    # open raster
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    Type = raster.GetDriver().ShortName
    if Type == "HDF4" or Type == "netCDF":
        raster = gdal.Open(raster.GetSubDatasets()[band_num][0])

    raster_band = raster.GetRasterBand(band_num)

    # build and fill metadata dictionary
    metadata = {}

    metadata["driver"] = raster.GetDriver()
    metadata["path"] = raster.GetDescription()
    metadata["proj"] = raster.GetProjection()
    metadata["crs"] = int(
        osr.SpatialReference(raster.GetProjection()).GetAttrValue("AUTHORITY", 1)
    )
    metadata["geotransform"] = raster.GetGeoTransform()
    metadata["xsize"] = raster.RasterXSize
    metadata["ysize"] = raster.RasterYSize
    metadata["cell_count"] = metadata["xsize"] * metadata["ysize"]
    metadata["xmin"] = metadata["geotransform"][0]
    metadata["ymax"] = metadata["geotransform"][3]
    metadata["xmax"] = metadata["xmin"] + (
        metadata["geotransform"][1] * metadata["xsize"]
    )
    metadata["ymin"] = metadata["ymax"] + (
        metadata["geotransform"][5] * metadata["ysize"]
    )
    metadata["xres"] = metadata["geotransform"][1]
    metadata["yres"] = metadata["geotransform"][5]
    if metadata["crs"] == 4326:
        metadata["cell_area"] = abs(0.00001 * metadata["xres"]) * abs(
            0.00001 * metadata["yres"]
        )

    else:
        metadata["cell_area"] = abs(metadata["xres"]) * abs(metadata["yres"])
    metadata["cell_area_unit"] = "meters_sq"
    metadata["gdal_data_type_code"] = raster_band.DataType
    metadata["gdal_data_type_name"] = gdal.GetDataTypeName(
        metadata["gdal_data_type_code"]
    )
    metadata["nodata"] = raster_band.GetNoDataValue()
    metadata["ogr_extent"] = (
        metadata["xmin"],
        metadata["ymin"],
        metadata["xmax"],
        metadata["ymax"],
    )
    metadata["band_count"] = raster.RasterCount
    band_name = raster_band.GetDescription()
    if os.path.isfile(band_name):
        band_name = os.path.splitext(os.path.basename(band_name))[0]
    metadata["band_name"] = band_name

    raster = None

    return metadata


#################################
def check_dimensions(raster_path_a: str, raster_path_b: str):
    """compare and check two rasters dimensions to see if they match

    Parameters
    ----------
    raster_path_a : str
        raster a to check
    raster_path_b : str
        raster b to check

    Returns
    -------
    bool
        True if they match on all aspects
    """
    a_meta = gdal_info(raster_path_a)
    b_meta = gdal_info(raster_path_b)

    match = True

    if any(
        a != b
        for a, b in (
            (a_meta["crs"], b_meta["crs"]),
            (a_meta["xmin"], b_meta["xmin"]),
            (a_meta["ymax"], b_meta["ymax"]),
            (a_meta["xmax"], b_meta["xmax"]),
            (a_meta["ymin"], b_meta["ymin"]),
            (a_meta["xres"], b_meta["xres"]),
            (a_meta["yres"], b_meta["yres"]),
            (a_meta["xsize"], b_meta["xsize"]),
            (a_meta["ysize"], b_meta["ysize"]),
        )
    ):

        match = False

    return match


#################################
def raster_to_array(
    input_raster_path: str,
    band_num: int = 1,
    dtype: str = None,
    use_nan: bool = True,
):
    """retrieve the array from a raster

    Parameters
    ----------
    input_raster_path : str
        input raster to retrieve array from
    band_num : int, optional
        band to retrieve array from, by default 1
    dtype : str, optional
        datatype of the raster array, autosets if not provided, by default None
    use_nan : bool, optional
        if true sets raster metadata nodata value to nan in array, by default True

    Returns
    -------
    np.ndarray
        retrieved array
    """
    datatypes = {
        "Byte": np.uint8,
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "Int16": np.int16,
        "uint32": np.uint32,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
        "Int32": np.int32,
        "Float32": np.float32,
        "Float64": np.float64,
        "Complex64": np.complex64,
        "Complex128": np.complex128,
    }

    input_dataset = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    _type = input_dataset.GetDriver().ShortName
    if _type == "HDF4":
        input_band = gdal.Open(input_dataset.GetSubDatasets()[band_num][0])
    else:
        input_band = input_dataset.GetRasterBand(band_num)
    nodata = input_band.GetNoDataValue()

    if dtype is None:
        dtype = gdal.GetDataTypeName(input_band.DataType)
    dtype = datatypes[dtype]

    array = input_band.ReadAsArray().astype(dtype)

    if use_nan:
        array = np.where(array == nodata, np.nan, array)
        array = np.where(array == np.inf, np.nan, array)
        array = np.where(array == -np.inf, np.nan, array)

    input_dataset = input_band = None

    return array


#################################
def array_to_raster(
    metadata: Union[str, dict],
    output_raster_path: str,
    input_array: np.array,
    band_num: int = 1,
    xsize: int = None,
    ysize: int = None,
    geotransform: list = None,
    input_crs: int = None,
    output_crs: int = None,
    output_nodata: float = None,
    output_gdal_data_type: int = None,
    creation_options: list = ["TILED=YES", "COMPRESS=LZW"],
    driver_format: str = "GTiff",
) -> str:
    """write an array to raster

    Parameters
    ----------
    metadata : Union[str, dict]
        can either be the path to the template raster
        containing the metadata needed to transform the raster
        or a dict containing the data directly.
    output_raster_path : str
        path to output the raster too
    input_array : np.array
        array to write to raster
    band_num : int, optional
        raster band to write the array too, by default 1
    xsize : int, optional
        xsize of the raster overwrites metadata value if provided, by default None
    ysize : int, optional
        ysize of the raster overwrites metadata value if provided, by default None
    geotransform : list, optional
        geotransform of the raster overwrites metadata value if provided, by default None
    input_crs : int, optional
        input_crs of the raster overwrites metadata value if provided, by default None
    output_crs : int, optional
        output_crs of the raster overwrites metadata value if provided, by default None
    output_nodata : float, optional
        output_nodata of the raster overwrites metadata value if provided, by default None
    output_gdal_data_type : int, optional
        output gdal data type of the raster overwrites metadata value if provided, by default None
    creation_options : list, optional
        creation options to apply, by default ["TILED=YES", "COMPRESS=LZW"]
    driver_format : str, optional
        driver output type, by default "GTiff"

    Returns
    -------
    str
        path to the created raster

    Raises
    ------
    AttributeError
        if any of the required inputs are missing
    """
    # create output subfolders as needed
    output_dir = os.path.dirname(output_raster_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datatypes = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "Int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11,
        "Int32": 5,
        "Float32": 6,
        "Float64": 7,
        "Complex64": 10,
        "Complex128": 11,
    }
    if isinstance(metadata, str):
        if os.path.exists(metadata):
            template_meta = gdal_info(metadata)
    elif isinstance(metadata, dict):
        template_meta = metadata
    else:
        if any(
            item is None
            for item in [xsize, ysize, geotransform, output_crs, output_nodata]
        ):
            raise AttributeError(
                "missing input, please provide all metadata inputs if using direct inputs"
            )
        else:
            template_meta = {}

    for items in [
        ("xsize", xsize),
        ("ysize", ysize),
        ("geotransform", geotransform),
        ("proj", input_crs),
        ("nodata", output_nodata),
    ]:
        if items[1] is not None:
            template_meta[items[0]] = items[1]

    # set standard raster_driver
    template_meta["driver"] = gdal.GetDriverByName(driver_format)

    if not output_gdal_data_type:
        gdal_data_type = datatypes[str(np.array(list(input_array)).dtype)]
    else:
        gdal_data_type = output_gdal_data_type

    output_array = np.where(np.isnan(input_array), template_meta["nodata"], input_array)

    input_crs_code = f"EPSG:{template_meta['crs']}"
    if output_crs:
        output_crs_code = f"EPSG:{output_crs}"
    else:
        output_crs_code = input_crs_code

    temp_raster_path = os.path.splitext(output_raster_path)[0] + "_temp.tif"

    output_raster = template_meta["driver"].Create(
        temp_raster_path,
        template_meta["xsize"],
        template_meta["ysize"],
        band_num,
        gdal_data_type,
        options=creation_options,
    )

    output_raster.SetGeoTransform(template_meta["geotransform"])
    output_raster.SetProjection(template_meta["proj"])
    output_band = output_raster.GetRasterBand(band_num)
    output_band.SetNoDataValue(template_meta["nodata"])
    output_band.WriteArray(output_array)
    output_band.GetStatistics(0, 1)

    output_raster = output_band = None

    check_gdal_open(temp_raster_path)

    if output_crs_code != input_crs_code:
        warped_raster = gdal.Warp(
            destNameOrDestDS=output_raster_path,
            srcDSOrSrcDSTab=temp_raster_path,
            format="Gtiff",
            srcSRS=input_crs_code,
            dstSRS=output_crs_code,
            srcNodata=template_meta["nodata"],
            dstNodata=template_meta["nodata"],
            width=template_meta["xsize"],
            height=template_meta["ysize"],
            outputBounds=template_meta["ogr_extent"],
            outputBoundsSRS=input_crs_code,
            resampleAlg="near",
            options=creation_options,
        )

        warped_raster.FlushCache()
        check_gdal_open(output_raster_path)

    else:
        shutil.copy2(src=temp_raster_path, dst=output_raster_path)

    os.remove(temp_raster_path)

    set_band_descriptions(
        raster_file_path=output_raster_path, band_names=[output_raster_path]
    )

    return output_raster_path


############################
def retrieve_raster_crs(raster_path: str) -> int:
    """retrieve the crs int code of a raster

    Parameters
    ----------
    raster_path : str
        path to the raster to retrieve crs from

    Returns
    -------
    int
        crs code
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
def mask_raster(
    mask_raster_path: str,
    input_raster_path: str,
    output_raster_path: str,
    output_nodata: float,
):
    """mask a raster using another raster

    Parameters
    ----------
    mask_raster_path : str
        path to the mask raster
    input_raster_path : str
        path to the raster to mask
    output_raster_path : str
        path to output the masked raster too
    output_nodata : float
        output no data value

    Returns
    -------
    int
        0

    Raises
    ------
    AttributeError
        if the geotransform parameters of the two rasters do not match
    """

    check_gdal_open(mask_raster_path)

    if not check_dimensions(
        raster_path_a=mask_raster_path, raster_path_b=input_raster_path
    ):

        raise AttributeError(
            f"geotransform parameters of the input: {input_raster_path} and mask: {mask_raster_path} have to match"
        )

    input_meta = gdal_info(input_raster_path)

    mask_array = raster_to_array(mask_raster_path)
    input_array = raster_to_array(input_raster_path)

    output_array = np.where(np.isnan(mask_array), np.nan, input_array)

    array_to_raster(
        metadata=input_meta,
        input_array=output_array,
        output_raster_path=output_raster_path,
        output_nodata=output_nodata,
    )

    check_gdal_open(output_raster_path)

    return output_raster_path


############################
def match_raster(
    match_raster_path: str,
    input_raster_path: str,
    output_raster_path: str,
    mask_raster_path: str = None,
    resample_method: str = "near",
    output_crs: int = None,
    output_nodata: float = -9999,
    creation_options: list = ["TILED=YES", "COMPRESS=LZW"],
):
    """warp one raster too match the other in terms of
    geo parameters

    Parameters
    ----------
    match_raster_path : str
        raster to match
    input_raster_path : str
        raster that has to match
    output_raster_path : str
        path to output the matched raster too
    mask_raster_path : str, optional
        path to a mask raster if wanting to mask as well, can be the same as the match raster, by default None
    resample_method : str, optional
        resample method to use, by default "near"
    output_crs : int, optional
        output coordinate system, by default None
    output_nodata : float, optional
        output no data value, by default -9999
    creation_options : list, optional
        creation options to apply, by default ["TILED=YES", "COMPRESS=LZW"]

    Returns
    -------
    str
        optupt raster path
    """
    check_gdal_open(match_raster_path)
    check_gdal_open(input_raster_path)

    # create output subfolders as needed
    output_dir = os.path.dirname(output_raster_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    template_meta = gdal_info(match_raster_path)
    input_meta = gdal_info(input_raster_path)

    if not output_crs:
        output_crs = template_meta["crs"]

    # check if the projection wanted (crs), no data value and dimensions do not match (different to check dimensions)
    if any(
        a != b
        for a, b in (
            (output_crs, input_meta["crs"]),
            (output_nodata, input_meta["nodata"]),
            (template_meta["xmin"], input_meta["xmin"]),
            (template_meta["ymax"], input_meta["ymax"]),
            (template_meta["xmax"], input_meta["xmax"]),
            (template_meta["ymin"], input_meta["ymin"]),
            (template_meta["xres"], input_meta["xres"]),
            (template_meta["yres"], input_meta["yres"]),
            (template_meta["xsize"], input_meta["xsize"]),
            (template_meta["ysize"], input_meta["ysize"]),
        )
    ):

        output_geotransform = reproject_geotransform(
            input_meta["geotransform"],
            xsize=input_meta["xsize"],
            ysize=input_meta["ysize"],
            in_proj=input_meta["crs"],
            out_proj=output_crs,
        )

        output_crs_code = f"EPSG:{output_crs}"
        input_crs_code = f"EPSG:{input_meta['crs']}"

        output_xmax = output_geotransform[0] + (
            output_geotransform[1] * template_meta["xsize"]
        )
        output_ymin = output_geotransform[3] + (
            output_geotransform[5] * template_meta["ysize"]
        )

        warped_raster = gdal.Warp(
            destNameOrDestDS=output_raster_path,
            srcDSOrSrcDSTab=input_raster_path,
            format="Gtiff",
            srcSRS=input_crs_code,
            dstSRS=output_crs_code,
            srcNodata=input_meta["nodata"],
            dstNodata=output_nodata,
            xRes=output_geotransform[1],  # template_meta['xres'],
            yRes=output_geotransform[5],  # template_meta['yres'],
            outputBounds=[
                output_geotransform[0],
                output_ymin,
                output_xmax,
                output_geotransform[3],
            ],  # template_meta['ogr_extent'],
            outputBoundsSRS=output_crs_code,  # bounds_crs,
            resampleAlg=resample_method,
            options=creation_options,
        )

        warped_raster.FlushCache()

    else:
        shutil.copy2(src=input_raster_path, dst=output_raster_path)

    check_gdal_open(output_raster_path)

    if mask_raster_path:
        check_gdal_open(mask_raster_path)
        mask_raster(
            mask_raster_path=mask_raster_path,
            input_raster_path=output_raster_path,
            output_raster_path=output_raster_path,
            output_nodata=output_nodata,
        )

    set_band_descriptions(
        raster_file_path=output_raster_path, band_names=[output_raster_path]
    )

    check_gdal_open(output_raster_path)

    return output_raster_path


############################
def build_vrt(raster_list: list, output_vrt_path: str, action: str = "space"):
    """build a gdal vrt

    Parameters
    ----------
    raster_list : list
        list of rasters to combine
    output_vrt_path : str
        path to output the vrt too
    action : str, optional
        either mosiac in space or stack in time, by default "space"

    Returns
    -------
    str
        path to the outputted raster vrt

    Raises
    ------
    AttributeError
        if action is not space or time
    """
    # create output subfolders as needed
    output_dir = os.path.dirname(output_vrt_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if action == "space":
        vrt_options = gdal.BuildVRTOptions(resampleAlg="cubic")

    elif action == "time":
        vrt_options = gdal.BuildVRTOptions(separate=True)

    else:
        raise AttributeError("type must be either space or time")

    out_vrt = gdal.BuildVRT(
        destName=output_vrt_path,
        srcDSOrSrcDSTab=raster_list,
        options=vrt_options,
        overwrite=True,
    )

    out_vrt.FlushCache()

    check_gdal_open(output_vrt_path)

    set_band_descriptions(raster_file_path=output_vrt_path, band_names=raster_list)

    return output_vrt_path


########################################################
# Functions udes outside waporact retrieval
########################################################
def area_of_latlon_pixel(pixel_size: float, center_lat: float):
    """get the area of a lat lon pixel

    Parameters
    ----------
    pixel_size : float
        width of the pixel in degrees
    center_lat : float
        _descrilatitutude of the center of the pixel

    Returns
    -------
    float
        area of the pixel in m2
    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = math.sqrt(1 - (b / a) ** 2)
    area_list = []
    for f in [center_lat + pixel_size / 2, center_lat - pixel_size / 2]:
        zm = 1 - e * math.sin(math.radians(f))
        zp = 1 + e * math.sin(math.radians(f))
        area_list.append(
            math.pi
            * b ** 2
            * (math.log(zp / zm) / (2 * e) + math.sin(math.radians(f)) / (zp * zm))
        )
    return pixel_size / 360.0 * (area_list[0] - area_list[1])


#################################
def count_raster_values(input_raster_path: str):
    """count the unique raster values in a raster

    Parameters
    ----------
    input_raster_path : str
        raster to count values in

    Returns
    -------
    list
        list of dicts containing count statistics
    """
    check_gdal_open(input_raster_path)

    meta = gdal_info(input_raster_path)

    input_array = raster_to_array(input_raster_path)

    non_nan_cell_count = np.count_nonzero(~np.isnan(input_array))

    counts = np.unique(input_array[~np.isnan(input_array)], return_counts=True)

    counts_list = []

    for value, count in zip(list(counts[0]), list(counts[1])):
        # if a float is returned check if it cna be made an int
        if isinstance(value, float):
            if value.is_integer():
                value = int(value)
        percentage = round(count / non_nan_cell_count * 100, 3)
        counts_list.append(
            {
                "value": value,
                "count": count,
                "percentage": percentage,
                # "area": round((count * meta["cell_area"]), 3), currently not working well
                # "unit": meta["cell_area_unit"],
            }
        )

    if meta["crs"] == 4326:
        logger.warning(
            "calculated area is an estimate of area im meters_sq based of latlon"
        )

    input_raster_path = input_array = None

    return counts_list


############################
def rasterize_vector(
    template_raster_path: str,
    vector_path: str,
    output_raster_path: str,
    reverse: bool = False,
    output_nodata=0,
    output_gdal_datatype=1,
    column: str = None,
    all_touched: bool = False,
    creation_options: list = ["TILED=YES", "COMPRESS=DEFLATE"],
):
    """rasterize a vector

    Parameters
    ----------
    template_raster_path : str
        template raster to rasterize too , metadata source
    vector_path : str
        vector to rasterize
    output_raster_path : str
        pathy to output the new vector raster too
    reverse : bool, optional
        if true reverse and rasterize everything but the vectors, currently not implemented, by default False
    output_nodata : int, optional
        nodata value for the output raster, by default 0
    output_gdal_datatype : int, optional
        output data type, by default 1
    column : str, optional
        column in the vector if setting values to the raster, by default None
    all_touched : bool, optional
        if true rasterizes all cells touched by the vector, by default False
    creation_options : list, optional
        creation options for the output raster, by default ["TILED=YES", "COMPRESS=DEFLATE"]

    Returns
    -------
    str
        path to the outputted raster
    """
    # create output subfolders as needed
    output_dir = os.path.dirname(output_raster_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if all_touched:
        all_touched = "TRUE"
    else:
        all_touched = "FALSE"

    meta = gdal_info(template_raster_path)

    vector_object = ogr.Open(vector_path)
    vector_layer = vector_object.GetLayer()

    target_ds = gdal.GetDriverByName("GTiff").Create(
        output_raster_path,
        meta["xsize"],
        meta["ysize"],
        1,
        output_gdal_datatype,
        options=creation_options,
    )

    target_ds.SetGeoTransform(meta["geotransform"])
    target_ds.SetProjection(meta["proj"])
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(output_nodata)
    band.FlushCache()

    if not column:
        gdal.RasterizeLayer(
            target_ds,
            [1],
            vector_layer,
            burn_values=[1],
            options=[f"ALL_TOUCHED={all_touched}"],
        )

    else:
        gdal.RasterizeLayer(
            target_ds,
            [1],
            vector_layer,
            options=[
                f"ALL_TOUCHED={all_touched}",
                f"ATTRIBUTE={column}",
            ],
        )

    band.GetStatistics(0, 1)

    target_ds = band = None
    check_gdal_open(output_raster_path)

    return output_raster_path


############################
def create_polygon_index_dict(
    template_raster_path: str,
    input_vector_path: str,
    id_key: str = "wpid",
):
    """rasterize the features in the shapefile according to the
        template raster provided and extract the indices corresponding
        to each feature

    Parameters
    ----------
    template_raster_path : str
         path to the raster containing the metadata needed
    input_vector_path : str
        vector file to retrieve raster indices for per feature
    id_key : str, optional
        _description_, by default "wpid"

    Returns
    -------
    dict
        dictionary of raster indices per feature index
    """
    with rasterio.open(template_raster_path) as src:
        with fiona.open(input_vector_path, "r") as vec:
            geoms = [feature["geometry"] for feature in vec]
            index = [feature["properties"][id_key] for feature in vec]

            polygon_index_dict = {}
            for idx, geom in zip(index, geoms):
                geom_rasterize = rasterio.features.rasterize(
                    [(geom, 1)],
                    out_shape=src.shape,
                    transform=src.transform,
                    all_touched=True,
                    fill=0,
                    dtype="uint8",
                )

                polygon_index_dict[idx] = np.where(geom_rasterize == 1)

    return polygon_index_dict


############################
def create_values_specific_mask(
    mask_values: list,
    input_raster_path: str,
    output_mask_raster_path: str,
    output_values_raster_path: str = None,
    output_crs=None,
    output_nodata: float = -9999,
    keep_values: bool = False,
):
    """mask to specific values in a raster setting them to 1 and all other values to 0

    Parameters
    ----------
    mask_values : list
        values to mask too
    input_raster_path : str
        raster to mask values in
    output_mask_raster_path : str
        path to output the resulting mask too
    output_values_raster_path : str, optional
        path to output the values raster too if wanted, by default None
    output_crs : _type_, optional
        output crs, by default None
    output_nodata : float, optional
        output nodata, by default -9999
    keep_values : bool, optional
        if true keeps the values and does not set them to 1, by default False

    Returns
    -------
    str
        path to the output raster

    Raises
    ------
    AttributeError
        if the values to mask too do not occur in the raster
    """
    check_gdal_open(input_raster_path)

    input_array = raster_to_array(input_raster_path)

    metadata = gdal_info(input_raster_path)

    values_array = np.isin(input_array, mask_values)

    count_occurence = values_array.sum()

    percentage_occurrence = count_occurence / metadata["cell_count"] * 100

    if count_occurence == 0:
        raise AttributeError(
            "given values to mask too occurs zero times in the given raster, please specify another value"
        )

    elif percentage_occurrence < 5:
        logger.warning(
            f"WARNING: given values to mask too cover less that 5 percent of the given raster: {percentage_occurrence}"
        )

    else:
        pass

    # create specific values array
    value_mask_array = np.where(values_array, input_array, np.nan)
    # create mask array
    mask_array = np.where(values_array, 1, np.nan)

    if output_crs:
        if metadata["crs"] == output_crs:
            output_crs = None

    if output_values_raster_path:
        array_to_raster(
            metadata=input_raster_path,
            input_array=value_mask_array,
            output_raster_path=output_values_raster_path,
            output_gdal_data_type=6,
            output_crs=output_crs,
            output_nodata=output_nodata,
        )

        check_gdal_open(output_values_raster_path)

    array_to_raster(
        metadata=input_raster_path,
        input_array=mask_array,
        output_raster_path=output_mask_raster_path,
        output_gdal_data_type=1,
        output_crs=output_crs,
        output_nodata=0,
    )

    check_gdal_open(output_mask_raster_path)

    return output_mask_raster_path
