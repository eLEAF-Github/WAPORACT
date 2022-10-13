"""
waporact package

vector functions (stand alone/support functions)
"""
##########################
# import packages
import os
from typing import Union
import sys
from datetime import timedelta
from timeit import default_timer

from osgeo import ogr
from osgeo import osr
from osgeo import gdal

import pandas as pd
import geopandas as gpd
import numpy as np

import fiona
from shapely.geometry import mapping, shape, MultiPolygon, Polygon
import shapely.ops as ops
import rtree
import pyproj
from functools import partial

from waporact.scripts.tools.raster import gdal_info, reproject_coordinates
from waporact.scripts.tools.statistics import dict_to_dataframe

import logging

logger = logging.getLogger(__name__)


########################################################
# Functions used by waporact retrieval
########################################################
def retrieve_vector_driver_from_file(vector_path: str):
    """retrieve the driver of a vector form the vectors file

    Parameters
    ----------
    vector_path : str
        path to the vector

    Returns
    -------
    str
        vecotr driver

    Raises
    ------
    KeyError
        if the vector driver could not be found in gdal internal list using the file ext
    """
    target_ext = os.path.splitext(vector_path)[1][1:]
    vector_drivers = dict()

    for i in range(gdal.GetDriverCount()):

        drv = gdal.GetDriver(i)
        md = drv.GetMetadata_Dict()

        d = [drv.ShortName, drv.LongName, drv.GetMetadataItem(gdal.DMD_EXTENSIONS)]

        if (
            "DCAP_VECTOR" in md
        ):  # note "if" not "elif" or "else" as some drivers can handle both raster and vector
            try:
                extensions = d[2].split(" ")
                for ext in extensions:
                    if ext not in vector_drivers.keys():
                        vector_drivers[ext] = d[0]
            except AttributeError:
                pass
    try:
        vector_driver = vector_drivers[target_ext]
    except:
        logger.info(f"available vector types: {vector_drivers}")
        raise KeyError(
            "vector driver could not be found in gdal internal list using given file path, please check file name or use a different file"
        )
    return vector_driver


############################
def retrieve_vector_crs(vector_path: str) -> int:
    """retrieve vector coordinate reference system

    Parameters
    ----------
    vector_path : str
        path to the vector

    Returns
    -------
    int
        crs code/number
    """
    driver_name = retrieve_vector_driver_from_file(vector_path=vector_path)
    driver = ogr.GetDriverByName(driver_name)
    datasource = driver.Open(vector_path)
    layer = datasource.GetLayer()

    # retrieve spatialref
    srs = layer.GetSpatialRef()
    crs = int(srs.GetAttrValue("AUTHORITY", 1))

    if crs == 6326:
        # geojson returning 6326 is the same for our purposes
        crs = 4326

    return crs


############################
def compare_raster_vector_crs(raster_path: str, vector_path: str):
    """compare the crs of a vecotr and a raster

    Parameters
    ----------
    raster_path : str
        path to the raster to compare
    vector_path : str
        path to the vector to compare

    Returns
    -------
    int
        0

    Raises
    ------
    AttributeError
        if the crs do not match
    """
    vector_crs = retrieve_vector_crs(vector_path)
    raster_crs = gdal_info(raster_path)["crs"]

    if raster_crs != vector_crs:
        raise AttributeError(
            f"crs of the given vector: ({vector_path},{vector_crs}), and the given raster: ({raster_path}, {raster_crs}), do not match"
        )

    return 0


############################
def vector_reprojection(
    vector_path: str, output_directory: str, crs: int, output_name: str = ""
) -> str:
    """reproject a vector to a different projection

    Parameters
    ----------
    vector_path : str
        path to the vecotr to reproject
    output_directory : str
        locaiton to output the reprojected vector
    crs : int
        crs number to reproject too
    output_name : str, optional
        name to output too, if not given uses existing name appended with reproject, by default ""

    Returns
    -------
    str
        path to the reprojected vector file
    """
    if output_name == "":
        output_name = os.path.splitext(os.path.basename(vector_path))[0]

    input_extension = os.path.splitext(vector_path)[1][1:]

    output_vector_file = os.path.join(
        output_directory, f"{output_name}_{crs}_reproject.{input_extension}"
    )

    if not os.path.exists(output_vector_file):

        # set output SpatialReference
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(int(crs))

        # input SpatialReference
        inSpatialRef = osr.SpatialReference()
        inSpatialRef.ImportFromEPSG(retrieve_vector_crs(vector_path=vector_path))

        driver_name = retrieve_vector_driver_from_file(vector_path=vector_path)
        driver = ogr.GetDriverByName(driver_name)

        # create the CoordinateTransformation
        coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

        # get the input layer
        inDataSet = driver.Open(vector_path)
        inLayer = inDataSet.GetLayer()

        # create the output layer
        outDataSet = driver.CreateDataSource(output_vector_file)
        outLayer = outDataSet.CreateLayer(
            output_name, outSpatialRef, geom_type=ogr.wkbMultiPolygon
        )

        # add fields
        inLayerDefn = inLayer.GetLayerDefn()
        for i in range(0, inLayerDefn.GetFieldCount()):
            fieldDefn = inLayerDefn.GetFieldDefn(i)
            outLayer.CreateField(fieldDefn)

        # get the output layer's feature definition
        outLayerDefn = outLayer.GetLayerDefn()

        # loop through the input features
        inFeature = inLayer.GetNextFeature()
        while inFeature:
            # get the input geometry
            geom = inFeature.GetGeometryRef()
            # reproject the geometry
            geom.Transform(coordTrans)
            # create a new feature
            outFeature = ogr.Feature(outLayerDefn)
            # set the geometry and attribute
            outFeature.SetGeometry(geom)
            for i in range(0, outLayerDefn.GetFieldCount()):
                outFeature.SetField(
                    outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i)
                )
            # add the feature to the shapefile
            outLayer.CreateFeature(outFeature)
            # dereference the features and get the next input feature
            outFeature = None
            inFeature = inLayer.GetNextFeature()

        # Save and close the shapefiles
        inDataSet = None
        outDataSet = None

    else:
        pass

    return output_vector_file


##########################
def delete_shapefile(shapefile_path: str):
    """delete all the parts of a shapefile

    Parameters
    ----------
    shape_path : str
        shapefile to delete

    Returns
    -------
    int
        0

    Raises
    ------
    FileNotFoundError
        if the file to delete cannot be found
    """

    if os.path.exists(shapefile_path):
        base_path = os.path.splitext(shapefile_path)[0]
        for ext in [
            ".shp",
            ".shx",
            ".dbf",
            ".sbn",
            ".sbx",
            ".fbn",
            "fbx",
            ".prj",
            ".xml",
            ".cpg",
        ]:
            file_path = base_path + ext
            if os.path.exists(file_path):
                os.remove(file_path)

    else:
        raise FileNotFoundError(f"file not found: {shapefile_path}")

    return 0


########################################
def retrieve_vector_file_bbox(
    input_vector_path,
    output_bbox_vector_file: str = None,
    output_driver: str = "GeoJSON",
    output_crs: int = None,
):
    """retrieve the bbox of a vector file

    Parameters
    ----------
    input_vector_path : _type_
        vecotr to retrieve for
    output_bbox_vector_file : str, optional
        location to output bbox vector file too if made, by default None
    output_driver : str, optional
        output driver, by default "GeoJSON"
    output_crs: int, optional
        crs to output bbox and file in if provided otherwise uses input crs, by default None

    Returns
    -------
    tuple
        bounding box tuple
    """
    driver_name = retrieve_vector_driver_from_file(vector_path=input_vector_path)
    driver = ogr.GetDriverByName(driver_name)
    inDataSource = driver.Open(input_vector_path)
    inLayer = inDataSource.GetLayer()

    bboxes = []
    for feature in inLayer:
        geom = feature.GetGeometryRef()
        bboxes.append(geom.GetEnvelope())

    minx = min([box[0] for box in bboxes])
    maxx = max([box[1] for box in bboxes])
    miny = min([box[2] for box in bboxes])
    maxy = max([box[3] for box in bboxes])

    # increase size by 0.01 total diameter to make sure data falls inside
    # when retrieved

    diffx = maxx - minx
    diffy = maxy - miny

    addx = 0.01 * diffx
    addy = 0.01 * diffy

    minx = minx - addx
    maxx = maxx + addx
    miny = miny - addy
    maxy = maxy + addy

    input_crs = retrieve_vector_crs(input_vector_path)

    if output_crs:
        minx, miny = reproject_coordinates(
            x=minx, y=miny, in_proj=input_crs, out_proj=output_crs
        )
        maxx, maxy = reproject_coordinates(
            x=maxx, y=maxy, in_proj=input_crs, out_proj=output_crs
        )

    else:
        output_crs = input_crs

    if isinstance(output_bbox_vector_file, str):
        coords = [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
            (minx, miny),
        ]

        write_vectors_to_file(
            vectors=[coords],
            output_vector_path=output_bbox_vector_file,
            driver_name=output_driver,
            crs=output_crs,
        )

    return (minx, miny, maxx, maxy)


########################################
def create_polygon(coords: list):
    """create a ogr polygon object from coords
    Parameters
    ----------
    coords : list
        coords as a list of tuple one x and y per tuple

    Returns
    -------
    str
        polgyon as wkt
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()


########################################
def write_vectors_to_file(
    vectors: list,
    output_vector_path: str,
    crs: int,
    values: list = None,
    driver_name: str = "GeoJSON",
):
    """write a vector given as coord list of tuples to file as a vector

    Parameters
    ----------
    vectors : list
        list of tuples to write to file , each tuple contains tuples of xy pairs defining a vector
    output_vector_path : str
        path to output the vector too
    values : list, optional
        list of values to add to each vector in the same order as vectors, only string for now, by default None
    crs : int
        crs number
    driver_name : str, optional
        driver to output too, by default "GeoJSON"

    Returns
    -------
    str
        path to the output vector
    """
    assert isinstance(
        vectors, (list, tuple)
    ), "vector coords must be a list of tuples of xy coord tuples"
    assert isinstance(
        vectors[0], (list, tuple)
    ), "vector coords must be a list of tuples of xy coord tuples"
    assert isinstance(
        vectors[0][0], (list, tuple)
    ), "vector coords must be a list of tuples of xy coord tuples"
    # create in memory shape
    driver = ogr.GetDriverByName(driver_name)
    data_source = driver.CreateDataSource(output_vector_path)
    # create the spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)

    # create the layer
    layer = data_source.CreateLayer("vector", srs, ogr.wkbPolygon)

    # Add the fields
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    if values:
        layer.CreateField(ogr.FieldDefn("value", ogr.OFTString))

    for _idx, vector_coords in enumerate(vectors):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the dict values
        feature.SetField("id", _idx)

        if values:
            feature.SetField("value", values[_idx])

        polygon = create_polygon(coords=vector_coords)

        polygon = ogr.CreateGeometryFromWkt(polygon)

        feature.SetGeometry(polygon)

        layer.CreateFeature(feature)

        feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None

    return output_vector_path


########################################################
# Functions used outside waporact retrieval
########################################################
def file_to_records(
    table: Union[str, pd.DataFrame],
    column_mapping: dict = None,
    default_values: dict = None,
    sep: str = ";",
    _filter: dict = None,
    input_crs=None,
    sheet=0,
    to_dict: bool = False,
):
    """retrieve records from a file/table

    Parameters
    ----------
    table : Union[str, pd.DataFrame]
        table to retrieve data from
    column_mapping : dict, optional
        mapping to map file columns to table columns if needed, by default None
    default_values : dict, optional
        default values to fill in if missing, by default None
    sep : str, optional
        sep used to read a csv if needed, by default ";"
    _filter : dict, optional
        if provided filters to records matching the filter, by default None
    input_crs : _type_, optional
        input crs if reading from vector file, by default None
    sheet : int, optional
        sheet to read from if excel, by default 0
    to_dict : bool, optional
        if true outputs a dict instead of a dataframe, by default False

    Returns
    -------
    Union[dict, pandas.DataFrame]
        talbe formatted as dict or dataframe

    Raises
    ------
    AttributeError
        if the input type is wrong
    """
    if isinstance(table, pd.DataFrame):
        records = table

    else:
        ext = os.path.splitext(table)[1]

        if ext in [".shp", ".geojson"]:
            df = gpd.read_file(table)
            if input_crs:
                df = df.to_crs(f"EPSG:{input_crs}")

            df["st_aswkt"] = df.geometry.to_wkt()

        elif ext == ".xlsx":
            df = pd.read_excel(table, sheet_name=sheet)

        elif ext == ".csv":
            df = pd.read_csv(table, sep=sep)

        elif ext == ".json":
            df = pd.read_json(table)

        else:
            raise AttributeError(
                "either a shapefile, csv, excel or json needs to be provided"
            )

        if to_dict:
            records = df.to_dict("records")

            if column_mapping:
                mapped_records = []
                for rec in records:
                    new_rec = rec.copy()
                    for k_new, k_old in column_mapping.items():
                        new_rec = {
                            k_new if k == k_old else k: v for k, v in new_rec.items()
                        }
                    mapped_records.append(new_rec)

                records = mapped_records

            if default_values:
                default_records = []
                for rec in records:
                    for k, v in default_values.items():
                        rec.setdefault(k, v)
                    default_records.append(rec)

                records = default_records

            if _filter:
                filtered_records = []
                for rec in records:
                    if any(rec[k] in v for k, v in _filter.items()):
                        filtered_records.append(rec)

                records = filtered_records

        else:
            records = df

    return records


##########################
def records_to_vector(
    field_records: Union[dict, pd.DataFrame],
    output_vector_path: str,
    fields_vector_path: str,
    union_key: str,
    output_crs: int = None,
):
    """write records to a file

    Parameters
    ----------
    field_records : Union[dict, pd.DataFrame]
        records to write to file
    output_vector_path : str
        path to output the vector too
    fields_vector_path : str
        template vector to attach records too
    union_key : str
        key joining the field vector and the table
    output_crs : int, optional
        crs to output too, by default None

    Returns
    -------
    str
        path to the outputted vector
    """
    gdf = gpd.read_file(fields_vector_path)

    # drop unneeded columns
    drop_columns = [
        col for col in list(gdf.columns) if col not in ["geom", "geometry", union_key]
    ]
    gdf.drop(columns=drop_columns)

    # if dict format to dataframe
    if isinstance(field_records, dict):
        field_records = dict_to_dataframe(field_records)

    # merge gdf and df
    out_gdf = gdf.merge(field_records, on=union_key, how="left")

    geodataframe_to_vector_file(
        geodataframe=out_gdf,
        output_vector_path=output_vector_path,
        output_crs=output_crs,
    )

    return output_vector_path


##########################
def geodataframe_to_vector_file(
    geodataframe: gpd.GeoDataFrame, output_vector_path: str, output_crs: int = None
):
    """write a geodataframe to vector file

    Parameters
    ----------
    geodataframe : gpd.GeoDataFrame
        geodatafrmae to write to file
    output_vector_path : str
        path to output the vector too
    output_crs : int, optional
        output crs of the file, by default None

    Returns
    -------
    str
        path to the outputted vector
    """
    # create output subfolders as needed
    output_dir = os.path.dirname(output_vector_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if output_crs:
        geodataframe = geodataframe.to_crs({"init": f"epsg:{output_crs}"})

    geodataframe.to_file(output_vector_path)

    logger.info(f"geodataframe outputted to vector: {output_vector_path}")

    return output_vector_path


##########################
def retrieve_geodataframe_bbox(geodataframe: gpd.GeoDataFrame):
    """retrieve the bbox of a geodataframe increased in size by
    0.01 * the width of the raster to include all vectors

    Parameters
    ----------
    geodataframe : gpd.GeoDataFrame
        geodataframe to query

    Returns
    -------
    tuple
        bbox tuple
    """
    # remove empty geometry
    valid_geom = geodataframe[
        geodataframe.geometry.map(lambda z: True if not z.is_empty else False)
    ]

    # explode (extract polygons from) multipolygons
    singlepart_geoms = valid_geom.geometry.apply(
        lambda geom: list(geom) if isinstance(geom, MultiPolygon) else geom
    ).explode()

    # get the bounds of each geometry
    bboxes = singlepart_geoms.map(lambda z: z.exterior.xy)

    minx = min([min(box[0]) for box in bboxes])
    miny = min([min(box[1]) for box in bboxes])
    maxx = max([max(box[0]) for box in bboxes])
    maxy = max([max(box[1]) for box in bboxes])

    # increase size by 0.01 total diameter to make sure data falls inside
    # when retrieved

    diffx = maxx - minx
    diffy = maxy - miny

    addx = 0.01 * diffx
    addy = 0.01 * diffy

    minx = minx - addx
    maxx = maxx + addx
    miny = miny - addy
    maxy = maxy + addy

    return (minx, miny, maxx, maxy)


##########################
def retrieve_geodataframe_central_coords(geodataframe: gpd.GeoDataFrame):
    """retrieve the central coords of a geodataframe

    Parameters
    ----------
    geodataframe : gpd.GeoDataFrame
        geodataframe to query

    Returns
    -------
    tuple
        x, y coords
    """
    # remove empty geometry
    valid_geom = geodataframe[
        geodataframe.geometry.map(lambda z: True if not z.is_empty else False)
    ]

    # explode (extract polygons from) multipolygons
    singlepart_geoms = valid_geom.geometry.apply(
        lambda geom: list(geom) if isinstance(geom, MultiPolygon) else geom
    ).explode()

    # get the bounds of each geometry
    bboxes = singlepart_geoms.map(lambda z: z.exterior.xy)

    minx = min([min(box[0]) for box in bboxes])
    miny = min([min(box[1]) for box in bboxes])
    maxx = max([max(box[0]) for box in bboxes])
    maxy = max([max(box[1]) for box in bboxes])

    x = minx + (maxx - minx) / 2
    y = miny + (maxy - miny) / 2

    return x, y


############################
def get_plotting_zoom_level_and_central_coords_from_gdf(gdf: gpd.GeoDataFrame):
    """get the plotting level zoom and central coords from a geodataframe
    for plotting purposes

    NOTE: linear sclar interpolation for the zoom levle is taken from
        https://community.plotly.com/t/dynamic-zoom-for-mapbox/32658/7

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        geodataframe to query

    Returns
    -------
    tuple
        zoom, (x, y)
    """
    bbox = retrieve_geodataframe_bbox(geodataframe=gdf)
    bbox_poly = create_bbox_polygon(bbox=bbox)
    area = calc_lat_lon_polygon_area(bbox_poly) / 10000

    interp_dict = {
        20: 0,
        19: 100,
        17: 1000,
        14: 10000,
        12: 50000,
        10: 500000,
        9: 1000000,
        7: 5000000,
        5: 5 ** 10,
        3: 6 * 10,
        1: 8 ** 10,
    }
    zooms = [key for key in interp_dict.keys()]
    areas = [value for value in interp_dict.values()]

    zoom = int(np.interp(x=area, xp=areas, fp=zooms))

    # retrieve central mapping points
    x, y = retrieve_geodataframe_central_coords(gdf)

    # Finally, return the zoom level and the associated boundary-box center coordinates
    return zoom, (x, y)


############################
def create_bbox_polygon(bbox: tuple):
    """create a polygon from a bbox using shapely

    Parameters
    ----------
    bbox : tuple
        tuple of bbox coords

    Returns
    -------
    shapely polygon
        bbox polygon object
    """
    # set bbox coordinates
    xy_list = [
        (bbox[0], bbox[3]),
        (bbox[2], bbox[3]),
        (bbox[2], bbox[1]),
        (bbox[0], bbox[1]),
    ]

    # set to polygon
    poly = shape({"type": "Polygon", "coordinates": [xy_list]})

    return poly


##########################
def copy_shapefile(input_shapefile_path: str, output_shapefile_path: str):
    """copy a shapefile using fiona

    Parameters
    ----------
    input_shapefile_path : str
        shapefile to copy
    output_shapefile_path : str
        path to copy it too

    Returns
    -------
    str
        path to the output shapefile
    """
    # create output subfolders as needed
    output_dir = os.path.dirname(output_shapefile_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the original Shapefile
    with fiona.open(input_shapefile_path, "r") as _input:
        # The output has the same schema
        output_schema = _input.schema.copy()

        # write a new shapefile
        with fiona.open(
            output_shapefile_path,
            "w",
            driver=_input.driver,
            crs=_input.crs,
            schema=output_schema,
        ) as output:
            for feature in _input:
                output.write(
                    {
                        "properties": feature["properties"],
                        "geometry": mapping(shape(feature["geometry"])),
                    }
                )

    return output_shapefile_path


##########################
def check_add_wpid_to_shapefile(input_shapefile_path: str, overwrite: bool = False):
    """check if a shapefile has the record wpid and if not add it as a new column

    Parameters
    ----------
    input_shapefile_path : str
        shapefile to check
    overwrite : bool, optional
        if true overwrites the original file , by default False

    Returns
    -------
    str
        path to the output file with wpid added
    """
    add_wpid = False
    # check for fid
    with fiona.open(input_shapefile_path, "r") as _input:
        schema = dict(_input.schema)
        if "wpid" not in schema["properties"].keys():
            add_wpid = True

    if overwrite:
        add_wpid = True

    if add_wpid:
        temp_shapefile_path = os.path.splitext(input_shapefile_path)[0] + "_delete.shp"
        # Read the original Shapefile
        with fiona.open(input_shapefile_path, "r") as _input:
            # The output has the same schema
            output_schema = _input.schema.copy()
            output_schema["properties"]["wpid"] = "int"

            # write a new shapefile
            with fiona.open(
                temp_shapefile_path,
                "w",
                driver=_input.driver,
                crs=_input.crs,
                schema=output_schema,
            ) as output:
                wpid = 1
                for feature in _input:
                    feature["properties"]["wpid"] = wpid
                    output.write(
                        {
                            "properties": feature["properties"],
                            "geometry": mapping(shape(feature["geometry"])),
                        }
                    )
                    wpid += 1

        copy_shapefile(
            input_shapefile_path=temp_shapefile_path,
            output_shapefile_path=input_shapefile_path,
        )

        delete_shapefile(shapefile_path=temp_shapefile_path)

    return input_shapefile_path


##########################
def add_matched_values_to_shapefile(
    input_shapefile_path: str,
    value_type: str,
    new_column_name: str,
    values_dict: dict,
    union_key: str,
):
    """adds matching values to a shapefile using a dict. where the keys in the
        dict are existing values in the shapefile matched to the right column
        using the union_key, the dict value gets inserted in

    Parameters
    ----------
    input_shapefile_path : str
        shapefile to add values too
    value_type : str
        type of the new values
    new_column_name : str
        column name for the new values
    values_dict : dict
        dict holdign the new values
    union_key : str
        key to join the values on

    Returns
    -------
    str
        path to the input shapefile
    """
    temp_shapefile_path = os.path.splitext(input_shapefile_path)[0] + "_delete.shp"
    # Read the original Shapefile
    with fiona.open(input_shapefile_path, "r") as _input:
        # The output has the same schema
        output_schema = _input.schema.copy()
        output_schema["properties"][new_column_name] = value_type

        # write a new shapefile
        with fiona.open(
            temp_shapefile_path,
            "w",
            driver=_input.driver,
            crs=_input.crs,
            schema=output_schema,
        ) as output:
            for feature in _input:
                feature["properties"][new_column_name] = values_dict[
                    feature["properties"][union_key]
                ]
                output.write(
                    {
                        "properties": feature["properties"],
                        "geometry": mapping(shape(feature["geometry"])),
                    }
                )

    copy_shapefile(
        input_shapefile_path=temp_shapefile_path,
        output_shapefile_path=input_shapefile_path,
    )

    delete_shapefile(shapefile_path=temp_shapefile_path)

    return input_shapefile_path


##########################
def check_column_exists(shapefile_path: str, column: str):
    """check if a column exists in a shapefile

    Parameters
    ----------
    shapefile_path : str
        path to the shapefile to check
    column : str
        column to check for

    Returns
    -------
    int
        0

    Raises
    ------
    KeyError
        if the column cannot be found
    """
    with fiona.open(shapefile_path, "r") as _input:
        schema = dict(_input.schema)
        if column not in schema["properties"].keys():
            raise KeyError(
                f"column: {column} not found in shapefile: {shapefile_path} , please provide an exisitng column"
            )

    return 0


####################################################
# spatial operations
####################################################
def create_spatial_index(feature_dict: dict, id_key: str = "wpid"):
    """create a set of spatial indices for spatial analysis

    Parameters
    ----------
    feature_dict : dict
        dict of dicts (features) to index
    id_key : str, optional
        id key to use to index, by default "wpid"

    Returns
    -------
    Index
        rtree spatial index object

    Raises
    ------
    AttributeError
        if the features are not a dictionary
    """
    logger.info("attempting to create spatial indices")
    if not isinstance(feature_dict, dict):
        raise AttributeError("features must be a dictionary")
    # create a spatial index object
    idx = rtree.index.Index()
    # populate the spatial index
    for key in feature_dict:
        geometry = shape(feature_dict[key]["geometry"])
        idx.insert(feature_dict[key]["properties"][id_key], geometry.bounds)

    return idx


###########################
def calc_lat_lon_polygon_area(polygon: dict, epsg: int = 4326):
    """calc the area of a lat lon polygon

    Parameters
    ----------
    polygon : dict
        polygon to check
    epsg : int, optional
        epsg of the polygon, by default 4326

    Returns
    -------
    float
        area of the polygon in m2
    """
    proj_geom = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(f"EPSG:{epsg}"),
            pyproj.Proj(proj="aea", lat_1=polygon.bounds[1], lat_2=polygon.bounds[3]),
        ),
        polygon,
    )

    return proj_geom.area


###########################
def polygon_area_drop(feature_dict: dict, area_threshold: float, epsg: int):
    """sub function used to find and drop features in a feature dict that are
        smaller than the threshold given in m2

    Parameters
    ----------
    feature_dict : dict
        dictioanry of feature dictionaries to check
    area_threshold : float
        threshold area under which the polygons are dropped
    epsg : int
        espg of the polygons

    Returns
    -------
    dict
        feature dict with too small polygons removed
    """
    logger.info(f"deleting features/polygons that are smaller than {area_threshold} m2")
    drop_small_ids = []
    for _id, feature in feature_dict.items():
        poly = shape(feature["geometry"])
        if isinstance(poly, MultiPolygon):
            pass
        if epsg == 4326:
            area = calc_lat_lon_polygon_area(poly)
        else:
            area = poly.area
        if area < area_threshold:
            drop_small_ids.append(_id)

    # drop too small features
    for _id in drop_small_ids:
        feature_dict.pop(_id)

    return feature_dict


###########################
def fill_small_polygon_holes(feature_dict: dict, area_threshold: float, epsg: int):
    """fill small holes in polygons

    Parameters
    ----------
    feature_dict : dict
        dict of feature dicts to check
    area_threshold : float
        threshold under which holes are filled
    epsg : int
        epsg of the features

    Returns
    -------
    dict
        dictionary of features with their holes filled
    """
    logger.info(f"filling holes in polygons that are smaller than {area_threshold} m2")

    for __, feature in feature_dict.items():
        list_interiors = []
        poly = shape(feature["geometry"])
        if len(poly.interiors) > 0:
            for interior in poly.interiors:
                if epsg == 4326:
                    area = calc_lat_lon_polygon_area(interior)
                else:
                    area = interior.area
                if area > area_threshold:
                    list_interiors.append(interior)

            feature["geometry"] = mapping(
                Polygon(poly.exterior.coords, holes=list_interiors)
            )

    return feature_dict


###########################
def check_for_overlap(
    spatial_indices: rtree.index,
    feature_dict: dict,
    feature: dict,
    id_key: str = "wpid",
):
    """check if polygons overlap

    Parameters
    ----------
    spatial_indices : rtree.index
        spatial index used to check overlap
    feature_dict : dict
        features to check
    feature : dict
        feature to check against all in the feature dicts
    id_key : str, optional
        to use for indexes auto set to the package wpid
        expected that the same ids are present in the spatial_indices, by default "wpid"

    Returns
    -------
    list
        list of instersecting ids

    Raises
    ------
    KeyError
        if id key not found
    """
    intersecting_ids = []
    check_geometry = shape(feature["geometry"])

    existing_ids = [feature_dict[key]["properties"][id_key] for key in feature_dict]

    if id_key not in feature["properties"].keys():
        raise KeyError(
            f"required id_key: {id_key} not found among the feature properties while checking for intersections"
        )

    # get list of ids where bounding boxes intersect
    ids = [int(i) for i in spatial_indices.intersection(check_geometry.bounds)]

    ids = [id for id in ids if id != int(feature["properties"][id_key])]

    ids = [id for id in ids if id in existing_ids]

    # access the features that those ids reference
    for id in ids:
        sub_feature = feature_dict[id]
        sub_geom = shape(sub_feature["geometry"])

        # check the geometries intersect, not just their bboxs
        if (
            check_geometry.overlaps(sub_geom)
            or check_geometry.crosses(sub_geom)
            and not check_geometry.touches(sub_geom)
        ):
            intersecting_ids.append(id)

    return intersecting_ids


###########################
def overlap_among_features(
    spatial_indices: rtree.index, feature_dict: dict, id_key: str = "wpid"
):
    """checks if any features overlap, takes a dictionary of features and
        checks if any of the features intersect

    Parameters
    ----------
    spatial_indices : rtree.index
        spatila idnices used to check
    feature_dict : dict
        feature dicts to check
    id_key : str, optional
        id_key: to use for indexes auto set to the package wpid
        expected that the same ids are present in the spatial_indices , by default "wpid"

    Returns
    -------
    bool
        True if any overlaps exists at all

    Raises
    ------
    KeyError
        if id key not found
    """
    overlap_exists = False

    existing_ids = [feature_dict[key]["properties"][id_key] for key in feature_dict]

    temp_key = next(iter(feature_dict))
    if id_key not in feature_dict[temp_key]["properties"].keys():
        raise KeyError(
            f"required id_key: {id_key} not found among the feature properties while checking for intersections"
        )

    for key in feature_dict:
        if overlap_exists:
            break
        check_geometry = shape(feature_dict[key]["geometry"])

        # get list of ids where bounding boxes intersect
        ids = [
            int(i) for i in spatial_indices.intersection(check_geometry.bounds)
        ]  # not the same as shapely intersection

        ids = [
            id for id in ids if id != int(feature_dict[key]["properties"][id_key])
        ]  # exclude yourself

        ids = [id for id in ids if id in existing_ids]

        # access the features that those ids reference
        for id in ids:
            sub_feature = feature_dict[id]
            sub_geom = shape(sub_feature["geometry"])

            # check the geometries intersect, not just their bboxs
            if (
                check_geometry.overlaps(sub_geom)
                or check_geometry.crosses(sub_geom)
                and not check_geometry.touches(sub_geom)
            ):
                overlap_exists = True
                break

    return overlap_exists


##########################
def union_and_drop(
    spatial_indices: rtree.index, feature_dict: dict, id_key: str = "wpid"
):
    """finds overlapping features and unions them dropping the originals

    Parameters
    ----------
    spatial_indices : rtree.index
         rtree based spatial index object
    feature_dict : dict
        fiona based feature dict organised
        by a specific id
    id_key : str, optional
         to use for indexes auto set to the package wpid
        expected that the same ids are present in the spatial_indices, by default "wpid"

    Returns
    -------
    dict
        fiona based feature dict updated
    """
    checked_ids = []
    drop_ids = []

    # reset index for security
    feature_dict = {
        feature_dict[key]["properties"][id_key]: feature_dict[key]
        for key in feature_dict
    }

    # edit the polygons
    for _id, feature in feature_dict.items():
        # check for intersections and union
        if _id not in checked_ids:
            overlapping_ids = check_for_overlap(
                spatial_indices=spatial_indices,
                feature_dict=feature_dict,
                feature=feature,
                id_key=id_key,
            )

            checked_ids.append(_id)
            overlapping_ids = [_id for _id in overlapping_ids if _id not in checked_ids]

            if overlapping_ids:
                # list geometries that overlap
                union_geoms = [
                    (_id, shape(feature_dict[_id]["geometry"]))
                    for _id in overlapping_ids
                ]
                union_geoms.append((_id, shape(feature_dict[_id]["geometry"])))
                # calculate largest geometry to maintain most relevant assumed) properties
                union_geoms = [(geom[0], geom[1], geom[1].area) for geom in union_geoms]
                # sort by size
                union_geoms.sort(key=lambda x: x[2])
                # union geoms
                unioned_geom = ops.unary_union([geom[1] for geom in union_geoms])
                # replace existing geometry
                feature_dict[_id]["geometry"] = mapping(unioned_geom)
                # replace existing properties with that of the largest geom
                feature_dict[_id]["properties"] = feature_dict[unioned_geom[0][0]][
                    "properties"
                ]
                # reset replaced id back to current id
                feature_dict[_id]["properties"][id_key] = _id
                drop_ids.extend(overlapping_ids)
                checked_ids.extend(overlapping_ids)

    # drop unionised features
    for _id in drop_ids:
        feature_dict.pop(_id)

    return feature_dict


##########################
def polygonize_cleanup(
    input_shapefile_path: str,
    output_shapefile_path: str,
    area_threshold: float = 0,
    id_key: str = "wpid",
):
    """clean up a bunch of messy polygons, filling holes,
    dropping to small ones and merging overlapping ones

    Parameters
    ----------
    input_shapefile_path : str
        shapefile to check
    output_shapefile_path : str
        shapefile to output cleaned up vectors too
    area_threshold : float, optional
        threshold under which vectors are dropped and holes are filled, by default 0
    id_key : str, optional
         id used to organise the features, by default "wpid"

    Returns
    -------
    str
        path to the outputted vector
    """
    # create output subfolders as needed
    output_dir = os.path.dirname(output_shapefile_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if id_key == "wpid":
        # check for and add possible missing identifier:
        check_add_wpid_to_shapefile(input_shapefile_path=input_shapefile_path)

    # retrieve feature_dict from the input file
    with fiona.open(input_shapefile_path, "r") as _input:
        output_driver = _input.driver
        output_crs = _input.crs
        output_schema = dict(_input.schema)
        output_schema["geometry"] = "Polygon"

        features = {feature["properties"][id_key]: feature for feature in _input}

        epsg_code = int(_input.crs["init"].replace("epsg:", ""))

    _input = None

    logger.info(f"attempting to clean up {len(features)} polygon features")

    # fill the holes that are too small
    features = fill_small_polygon_holes(
        feature_dict=features, area_threshold=area_threshold, epsg=epsg_code
    )

    # remove the too small polygons premeptively to speed up the process
    features = polygon_area_drop(
        feature_dict=features, area_threshold=area_threshold, epsg=epsg_code
    )

    logger.info(f"smallest features dropped {len(features)} polygon features remaining")

    # create spatial index
    idx = create_spatial_index(feature_dict=features, id_key=id_key)

    prev_count = 0
    current_count = 0
    no_change_in_count = 0

    while overlap_among_features(spatial_indices=idx, feature_dict=features):

        # union features
        features = union_and_drop(spatial_indices=idx, feature_dict=features)

        # fill the holes that are too small
        fill_small_polygon_holes(
            feature_dict=features, area_threshold=area_threshold, epsg=epsg_code
        )

        features = polygon_area_drop(
            feature_dict=features, area_threshold=area_threshold, epsg=epsg_code
        )

        logger.info(
            f"smallest features dropped {len(features)} polygon features remaining"
        )

        logger.info(f"polygon features unionised: {len(features)} remaining")
        current_count = len(features)
        if prev_count == current_count:
            no_change_in_count += 1

        if no_change_in_count >= 5:
            logger.info(
                "WARNING: features no longer unionising but still overlapping,"
                " breaking loop and moving on but be aware that polygons still overlap"
            )
            break
        else:
            prev_count = current_count

            # recreate spatial index
            idx = create_spatial_index(feature_dict=features, id_key=id_key)

    # write the edited polygons to file
    with fiona.open(
        output_shapefile_path,
        "w",
        driver=output_driver,
        crs=output_crs,
        schema=output_schema,
    ) as output:

        # write the input file to output
        for __, feature in features.items():
            poly = shape(feature["geometry"])
            if isinstance(poly, Polygon):
                output.write(
                    {"properties": feature["properties"], "geometry": mapping(poly)}
                )
            elif isinstance(poly, MultiPolygon):
                for subpoly in poly:
                    output.write(
                        {
                            "properties": feature["properties"],
                            "geometry": mapping(subpoly),
                        }
                    )

            else:
                logger.warning(
                    "non polygon feature found at the end of cleaning up polygons ... discarding geometry"
                )

    output = None

    # check for and add possible missing identifier:
    check_add_wpid_to_shapefile(
        input_shapefile_path=output_shapefile_path, overwrite=True
    )

    return output_shapefile_path


##########################
def raster_to_polygon(
    input_raster_path: str,
    output_vector_path: str,
    column_name: str = "value",
    column_type=ogr.OFTInteger,
    mask_raster_path: str = None,
):
    """vectorize a raster

    Parameters
    ----------
    input_raster_path : str
        raster to vectorize
    output_vector_path : str
        path to output the vector
    column_name : str, optional
        column name to use for values, by default "value"
    column_type : _type_, optional
        type of the columns values, by default ogr.OFTInteger
    mask_raster_path : str, optional
        path tot he mask raster that determines the vectors, by default None

    Returns
    -------
    str
        path to the vectorized raster vector

    Raises
    ------
    AttributeError
        if unable to open the source raster
    AttributeError

        if unable to open the mask raster
    """
    # create output subfolders as needed
    output_dir = os.path.dirname(output_vector_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    source_dataset = gdal.Open(input_raster_path)
    if source_dataset is None:
        raise AttributeError(f"Unable to open {input_raster_path}")

    source_band = source_dataset.GetRasterBand(1)

    if mask_raster_path:
        mask_dataset = gdal.Open(mask_raster_path)
        if mask_dataset is None:
            raise AttributeError(f"Unable to open {mask_raster_path}")
        mask_band = mask_dataset.GetRasterBand(1)
    else:
        mask_band = None

    vector_name = os.path.splitext(os.path.basename(output_vector_path))[0]
    driver_name = retrieve_vector_driver_from_file(vector_path=output_vector_path)
    driver = ogr.GetDriverByName(driver_name)
    vector_out = driver.CreateDataSource(output_vector_path)

    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(gdal_info(input_raster_path)["crs"])

    # create the layer
    vector_layer = vector_out.CreateLayer(vector_name, srs, geom_type=ogr.wkbPolygon)

    newField = ogr.FieldDefn(column_name, column_type)
    vector_layer.CreateField(newField)
    gdal.Polygonize(source_band, mask_band, vector_layer, 0, [])

    vector_out = vector_layer = None

    # check for and add possible misisng identifier if shapefile:
    if os.path.splitext(output_vector_path)[1] == ".shp":
        check_add_wpid_to_shapefile(input_shapefile_path=output_vector_path)

    return output_vector_path
