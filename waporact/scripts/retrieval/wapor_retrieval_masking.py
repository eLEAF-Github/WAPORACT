"""
waporact package

retrieval class (stand alone/support class and functions)

script for the makign vecotr based masks and wapor level 3 locational vecotr files
"""

import os
from datetime import datetime

import shutil
from shapely.geometry import shape
import fiona

import rtree

from waporact.scripts.tools import raster, vector, statistics

from waporact.scripts.retrieval.wapor_retrieval_support import (
    dissagregate_categories,
    wapor_lcc,
    check_categories_exist_in_categories_dict,
    check_categories_exist_in_count_dict,
)

import logging

logger = logging.getLogger(__name__)

#################################
def check_bbox_overlaps_l3_location(
    bbox_vector_path: str,
    l3_locations_vector_path: str,
):
    """check if the analysis bbox overlaps a l3 area

    Parameters
    ----------
    bbox_vector_path : str
        bbox vector to check
    l3_locations_vector_path : str
        l3 locational file to check against

    Returns
    -------
    str
        country code of the area overlapped if found

    Raises
    ------
    AttributeError
        if no overlap is found
    """
    code = None

    with fiona.open(l3_locations_vector_path, "r") as layer1:
        with fiona.open(bbox_vector_path, "r") as layer2:
            index = rtree.index.Index()
            for feat1 in layer1:
                fid = int(feat1["id"])
                geom1 = shape(feat1["geometry"])
                index.insert(fid, geom1.bounds)

            for feat2 in layer2:
                geom2 = shape(feat2["geometry"])
                for fid in list(index.intersection(geom2.bounds)):
                    if fid != int(feat2["id"]):
                        feat1 = layer1[fid]
                        geom1 = shape(feat1["geometry"])
                        if geom1.intersects(geom2):
                            # We retrieve the country code and break the loop
                            code = feat1["properties"]["value"]
                            break

    if not code:
        raise AttributeError(
            f"no overlap found between wapor_level 3 locations: {l3_locations_vector_path}"
            f"and the generated bbox: {bbox_vector_path} it is likely that there is no data available at level 3 for this area, check in qgis or similar to confirm"
        )

    return code


#################################
def create_raster_mask_from_shapefile_and_template_raster(
    input_vector_path: str,
    template_raster_path: str,
    mask_raster_path: str,
    mask_vector_path: str,
    output_crs: int = None,
):
    """create a raster mask from a vector and a template raster

    Parameters
    ----------
    input_vector_path : str
        vectors to rasterize
    template_raster_path : str
        raster to take metadata from as template
    mask_raster_path : str
        path to output the mask too
    mask_shape_path : str
        path to output the vector version of the mask too
    output_crs : int, optional
        crs num code of the output raster and vector,
        if not provided uses that of the template raster
    Returns
    -------
    tuple
        path of the mask raster outputted, path to mask vector file created
    """
    raster.rasterize_vector(
        template_raster_path=template_raster_path,
        vector_path=input_vector_path,
        output_raster_path=mask_raster_path,
    )

    current_crs = raster.gdal_info(mask_raster_path)["crs"]

    if not output_crs:
        output_crs = current_crs

    if current_crs != output_crs:
        temp_raster_path = os.path.splitext(mask_raster_path)[0] + "temp.tif"
        shutil.copy2(src=mask_raster_path, dst=temp_raster_path)

        raster.match_raster(
            match_raster_path=temp_raster_path,
            input_raster_path=temp_raster_path,
            output_raster_path=mask_raster_path,
            output_crs=output_crs,
            output_nodata=0,
        )

        os.remove(temp_raster_path)

    logger.info(f"mask raster made: {mask_raster_path}")

    raster.check_gdal_open(mask_raster_path)

    if not os.path.exists(mask_vector_path):
        # create a shapefile of the raw crop_mask
        vector.raster_to_polygon(
            input_raster_path=mask_raster_path,
            output_vector_path=mask_vector_path,
            mask_raster_path=mask_raster_path,
        )

        vector.check_add_wpid_to_shapefile(input_shapefile_path=mask_vector_path)

        logger.info(f"mask shapefile made and wpid id column added: {mask_vector_path}")

    else:
        logger.info("preexisting raster shape mask found skipping step")

    return mask_raster_path, mask_vector_path


#################################
def create_raster_mask_from_wapor_landcover_rasters(
    wapor_level: int,
    lcc_categories: list,
    wapor_landcover_rasters: list,
    most_common_lcc_raster_path: str,
    lcc_count_csv_path: str,
    raw_mask_values_raster_path: str,
    raw_mask_raster_path: str,
    raw_mask_shape_path: str,
    mask_shape_path: str,
    mask_raster_path: str,
    mask_values_raster_path: str,
    masked_lcc_count_csv_path: str,
    output_crs: int = 4326,
    area_threshold_multiplier: int = 1,
    output_nodata: float = -9999,
):
    """create a raster mask from wapor landcover rasters

    Parameters
    ----------
    wapor_level : int
        wapor level analysing for
    lcc_categories: list
            crops/land classification categories to mask too
    wapor_landcover_rasters : list
        wapor land cover rasters to create a mask from (timeseries stack)
        can be one
    most_common_lcc_raster_path : str
        path to output the intermediate most common in time
        landcover value raster
    lcc_count_csv_path : str
        path to output the intermediate lcc count csv too
    raw_mask_values_raster_path : str
        path to output the intermediate raw lcc values raster too
    raw_mask_raster_path : str
        path to output the intermediate mask raster too
    raw_mask_shape_path : str
        path to output the intermediate vectorized mask too
    mask_shape_path : str
        path to output the final vectorized mask too
    mask_raster_path : str
        path to output the final mask raster too
    mask_values_raster_path : str
        path to output the final lcc values raster too
    masked_lcc_count_csv_path : str
        path to output the final lcc count csv too
    output_crs : int, optional
        output crs, by default 4326
    area_threshold_multiplier : int, optional
        area threshold multiplier for choosing vectors to keep * cell size, by default 1
    output_nodata : float, optional
        output nodata, by default -9999

    Returns
    -------
    tuple
        path to the output mask raster, path to the output mask shape
    """
    lcc_dict = wapor_lcc(wapor_level=wapor_level)

    # check that the lcc category provided exists as an option
    check_categories_exist_in_categories_dict(
        categories_list=lcc_categories, categories_dict=lcc_dict
    )

    # disaggregate aggregate codes
    lcc_categories = dissagregate_categories(
        categories_list=lcc_categories, categories_dict=lcc_dict
    )

    # check that the lcc category provided exists as an option
    check_categories_exist_in_categories_dict(
        categories_list=lcc_categories, categories_dict=lcc_dict
    )

    # reverse lcc_codes and categories
    lcc_dict_reversed = {
        lcc_dict[key]: key
        for key in lcc_dict.keys()
        if not isinstance(lcc_dict[key], list)
    }

    if len(wapor_landcover_rasters) > 1:
        # if more than one raster exists the median (most common) land cover class across the period is assigned
        statistics.calc_multiple_array_numpy_statistic(
            input=wapor_landcover_rasters,
            numpy_function=statistics.mostcommonzaxis,
            output_raster_path=most_common_lcc_raster_path,
        )
    else:
        most_common_lcc_raster_path = wapor_landcover_rasters[0]

    # use the most common raster to produce the raw mask rasters
    lcc_count_dict, lcc_count_csv_path = statistics.raster_count_statistics(
        input_raster_path=most_common_lcc_raster_path,
        categories_dict=lcc_dict,
        category_name="landcover",
        output_csv=lcc_count_csv_path,
        out_dict=True,
    )

    check_categories_exist_in_count_dict(
        categories_list=lcc_categories, count_dict=lcc_count_dict
    )

    # create the raw crop mask raster
    mask_values = []
    for lcc_category in lcc_categories:
        mask_values.append(lcc_dict[lcc_category])

    raster.create_values_specific_mask(
        mask_values=mask_values,
        input_raster_path=most_common_lcc_raster_path,
        output_values_raster_path=raw_mask_values_raster_path,
        output_mask_raster_path=raw_mask_raster_path,
        output_crs=output_crs,
        output_nodata=output_nodata,
    )

    logger.info(f"raw mask raster made: {raw_mask_raster_path}")
    logger.info(f"raw mask values raster made: {raw_mask_values_raster_path}")

    raster.check_gdal_open(raw_mask_raster_path)
    raster.check_gdal_open(raw_mask_values_raster_path)

    if not os.path.exists(raw_mask_shape_path):
        # create a shapefile of the raw crop_mask
        vector.raster_to_polygon(
            input_raster_path=raw_mask_values_raster_path,
            output_vector_path=raw_mask_shape_path,
            column_name="lcc_val",
            mask_raster_path=raw_mask_raster_path,
        )

        # add the lcc categories to the crop mask
        vector.add_matched_values_to_shapefile(
            input_shapefile_path=raw_mask_shape_path,
            new_column_name="lcc_cat",
            union_key="lcc_val",
            value_type="str",
            values_dict=lcc_dict_reversed,
        )

        logger.info(f"raw lcc mask shape made: {raw_mask_shape_path}")

    if not os.path.exists(mask_shape_path):
        if wapor_level == 3:
            if area_threshold_multiplier <= 1.5:
                area_threshold_multiplier = 1.5
                logging.warning(
                    "for level 3 the area_threshold_multiplier is set to a minimum of 1.5 to prevent slow down, change value in the code to change this"
                )
        cell_area = raster.gdal_info(raw_mask_raster_path)["cell_area"]
        area_threshold = cell_area * area_threshold_multiplier
        logger.warning(
            "area threshold for polygons found is currently"
            f" set to {area_threshold_multiplier} * {cell_area} (cell area) = {area_threshold}"
        )

        vector.polygonize_cleanup(
            input_shapefile_path=raw_mask_shape_path,
            output_shapefile_path=mask_shape_path,
            area_threshold=area_threshold,
        )

        logger.info(f"mask shape made: {mask_shape_path}")

    if not os.path.exists(mask_raster_path) or not os.path.exists(
        mask_values_raster_path
    ):
        # create cleaned values raster
        raster.rasterize_vector(
            template_raster_path=raw_mask_values_raster_path,
            vector_path=mask_shape_path,
            output_raster_path=mask_values_raster_path,
            column="lcc_val",
            output_gdal_datatype=6,
            output_nodata=output_nodata,
        )

        # create cleaned mask raster
        raster.rasterize_vector(
            template_raster_path=raw_mask_raster_path,
            vector_path=mask_shape_path,
            output_raster_path=mask_raster_path,
        )

    raster.check_gdal_open(mask_raster_path)
    raster.check_gdal_open(mask_values_raster_path)

    (
        lcc_masked_count_dict,
        masked_lcc_count_csv_path,
    ) = statistics.raster_count_statistics(
        input_raster_path=mask_values_raster_path,
        categories_dict=lcc_dict,
        category_name="landcover",
        output_csv=masked_lcc_count_csv_path,
        out_dict=True,
    )

    check_categories_exist_in_count_dict(
        categories_list=lcc_categories, count_dict=lcc_masked_count_dict
    )

    return mask_raster_path, mask_shape_path
