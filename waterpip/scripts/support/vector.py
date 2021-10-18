
##########################
# import packages
import os
from typing import Union

from osgeo import ogr
from osgeo import osr
from osgeo import gdal

import json
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import mapping, shape, MultiPolygon, Polygon
import shapely.ops as ops
import rtree

from waterpip.scripts.support.raster import gdal_info


##########################
def file_to_records(
    table: str = None,
    column_mapping: dict = None,
    default_values: dict = None,
    sep: str = ';',
    filter: dict = None,
    output_crs=None,
    sheet=0,
    to_dict: bool=False) -> dict:
    """
    Description:
        reads in a file (shapefile, excel, csv, json) and
        extracts them to  according to the specifications as 
        a dataframe or dict. If dict it filters the rows (values)
        out and stores them as dicts according to 
        key (column name), value (column, row item)

    Args:
        table: path to the file of interest
        column_mapping: mapping for the columns to dict keys
        default values: default values to use for keys if no
        value is found 
        sep: sep to use if reading from csv
        filter: dict of a key (column) and value to filter the 
        retrieved dicts by
        output_crs: output crs if retrieving from shapefile
        sheet: sheet to use if retrieving from excel
        to_dict: if Tue returns a list of dicts per geom/record

    Return:
        list, dataframe, geodataframe : extracted records mapped from the file   
    """

    ext = os.path.splitext(table)[1]

    if ext == '.shp':
        df = gpd.read_file(table)
        if output_crs:
            df = df.to_crs('EPSG:{}'.format(output_crs))
        
        df['st_aswkt'] = df.geometry.to_wkt()

    elif ext == '.xlsx':
        df = pd.read_excel(table,sheet_name=sheet)

    elif ext == '.csv':
        df = pd.read_csv(table,sep=sep)

    elif ext == '.json':
        df = pd.read_json(table)

    else:
        raise AttributeError('either a shapefile, csv, excel or json needs to be provided')

    if to_dict:
        records = df.to_dict('records')

        if column_mapping:
            mapped_records = []
            for rec in records:
                new_rec = rec.copy()
                for k_new, k_old in column_mapping.items():
                    new_rec = {k_new if k == k_old else k:v for k,v in new_rec.items()}
                mapped_records.append(new_rec)

            records = mapped_records

        if default_values:
            default_records = []
            for rec in records:
                for k, v in default_values.items():
                    rec.setdefault(k, v)
                default_records.append(rec)

            records = default_records

        if filter:
            filtered_records = []
            for rec in records:
                if any(rec[k] in v for k, v in filter.items()):
                    filtered_records.append(rec)
                
            records = filtered_records
    
    else:
        records = df

    return records

##########################
def dataframe_to_shapefile(
    geodataframe: gpd.GeoDataFrame, 
    output_location: str,
    output_name: str,
    output_crs: int = None):
    """
    Description:
        takes a geodataframe and outputs it to shapefile

    Args: 
        geodataframe: geodataframe to output
        output_location: location to output the shape too
        output_name: name of the outputted shapefile
        output_crs: if given transfroms the data to match this crs if it
        does not yet

    Return:
        shapefile: path to the outputted shapefile
    """
    output_file =os.path.join(output_location, output_name)

    if output_crs:
        geodataframe = geodataframe.to_crs({'init': 'epsg:{}'.format(output_crs)})

    geodataframe.to_file(output_file)

    print("geodataframe outputted to shapefile: {}".format(output_file))

    return output_file

##########################
def retrieve_geodataframe_bbox(geodataframe: gpd.GeoDataFrame):
    """
    Description:
        retrieves the boundingbox of all geometries in a geodataframe

    Args:
        geodataframe: geodataframe to retrieve boundingbox for        
    
    Return:
        tuple: bbox tuple
    """
    # remove empty geometry
    valid_geom = geodataframe[geodataframe.geometry.map(lambda z: True if not z.is_empty else False)]
    
    # get the bounds of each geometry
    bboxes = valid_geom.geometry.map(lambda z: z.exterior.xy)

    minx = min([min(box[0]) for box in bboxes])
    miny = min([min(box[1]) for box in bboxes])
    maxx = max([max(box[0]) for box in bboxes])
    maxy = max([max(box[1]) for box in bboxes])

    return (minx, miny, maxx, maxy)

############################
def retrieve_shapefile_crs(shapefile_path: str) -> int:
    """
    Description:
        retrieve the crs code/number of the given shapefile

    Args:
        shapefile_path: path to the shapefile

    Return:
        int: crs code/number of the input shapefile
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(shapefile_path)
    layer = datasource.GetLayer()

    # retrieve spatialref
    srs = layer.GetSpatialRef()
    crs = int(srs.GetAttrValue("AUTHORITY", 1))

    return crs

############################
def shape_reprojection(shapefile_path: str, output_directory: str, crs: int, output_name: str='') -> str:
    """
    Description:
        reproject the input shapefile to a different coordinate system using a given crs code

    Args:
        shapefile_path: path to the shapefile
        output_directory: folder to output too
        crs: code/ number of the crs (coordinate reference system)
        output_name: name for the output shapefile if not given the 
        original name is appended with '{_reproject'

    Return:
        str: the path to the reprojected shapefile
    """
    if output_name is '':
        output_name = os.path.splitext(os.path.basename(shapefile_path))[0]

    output_shp = os.path.join(output_directory, "{}_{}_reproject.shp".format(output_name, crs))

    if not os.path.exists(output_shp):
            
        # set output SpatialReference
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(int(crs))

        # input SpatialReference
        inSpatialRef = osr.SpatialReference()
        inSpatialRef.ImportFromEPSG(retrieve_shapefile_crs(shapefile_path=shapefile_path))

        driver = ogr.GetDriverByName('ESRI Shapefile')

        # create the CoordinateTransformation
        coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

        # get the input layer
        inDataSet = driver.Open(shapefile_path)
        inLayer = inDataSet.GetLayer()

        # create the output layer
        outDataSet = driver.CreateDataSource(output_shp)
        outLayer = outDataSet.CreateLayer(output_name, outSpatialRef, geom_type=ogr.wkbMultiPolygon)

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
                outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
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

    return output_shp


############################
def create_bbox_shapefile(
    output_shape_path: str,  
    bbox: tuple,
    crs:int=4326):
    """
    """
    #define schema
    schema = {
        'geometry':'Polygon',
        'properties':[('Name','str')]
        }

    #open a fiona object
    shp = fiona.open(output_shape_path, mode='w', driver='ESRI Shapefile',
        schema = schema, crs = "EPSG:{}".format(crs))

    #set bbox coordinates
    xy_list = [(bbox[0],bbox[3]),
    (bbox[2],bbox[3]),
    (bbox[2],bbox[1]),
    (bbox[0],bbox[1]),]

    #save record and close shapefile
    rowDict = {
    'geometry' : {'type':'Polygon',
                    'coordinates': [xy_list]}, 
    'properties': {'Name' : 'bbox'},
    }
    shp.write(rowDict)
    #close fiona object
    shp.close()

    return 0

##########################
def delete_shapefile(shapefile_path: str):
    """
    Description:
        little wrapper for deleting a shapefiles files
    
    Args:
        shapefile_path: path to the shapefile to delete

    Return:
        int: 0
    """
    if os.path.exists(shapefile_path):
        base_path = os.path.splitext(shapefile_path)[0]
        for ext in ['.shp','.shx','.dbf','.sbn','.sbx','.fbn','fbx','.prj','.xml','.cpg']:
            file_path = base_path+ext
            if os.path.exists(file_path):
                os.remove(file_path)

    else:
        raise FileNotFoundError('file not found: {}'.format(shapefile_path))

##########################
def copy_shapefile( 
    input_shapefile_path: str,
    output_shapefile_path: str):
    """
    Description:
        copies a shapefile and outputs it to a new location

    Args:
        input_shapefile_path: path to the shapefile to split
        output_shapefile_path: path to output the shapefile too
        add_fid: boolean if True adds an auto generated fid to each feature

    Return:
        int: 0
    """  
    # Read the original Shapefile
    with fiona.open(input_shapefile_path, 'r') as input:
        # The output has the same schema
        output_schema = input.schema.copy()

        # write a new shapefile
        with fiona.open(
            output_shapefile_path, 
            'w', 
            driver=input.driver,
            crs=input.crs,
            schema=output_schema) as output:
            for feature in input:
                output.write({'properties': feature['properties'],'geometry': mapping(shape(feature['geometry']))})

    return 0

##########################
def check_add_wpid_to_shapefile( 
    input_shapefile_path: str,
    overwrite:bool = False):
    """
    Description:
        checks for and if not present adds a wpid to a 
        shapefile and outputs it to the same location

        wpid: geometry unique identifer (the name is abritrary wpid = waterpip id)

    Args:
        input_shapefile_path: path to the shapefile to add fid to as
        organisational index
        overwrite: if true recreates the wpid index even if it is preexisting

    Return:
        int: 0
    """
    add_wpid = False
    # check for fid
    with fiona.open(input_shapefile_path, 'r') as input:
        schema = dict(input.schema)
        if 'wpid' not in schema["properties"].keys():
            add_wpid = True

    if overwrite:
        add_wpid = True

    if add_wpid:
        temp_shapefile_path = os.path.splitext(input_shapefile_path)[0] +'_delete.shp'
        # Read the original Shapefile
        with fiona.open(input_shapefile_path, 'r') as input:
            # The output has the same schema
            output_schema = input.schema.copy()
            output_schema['properties']['wpid'] = 'int'

            # write a new shapefile
            with fiona.open(
                temp_shapefile_path, 
                'w', 
                driver=input.driver,
                crs=input.crs,
                schema=output_schema) as output:
                wpid = 1
                for feature in input:
                    feature['properties']['wpid'] = wpid
                    output.write({'properties': feature['properties'],'geometry': mapping(shape(feature['geometry']))})
                    wpid +=1

        copy_shapefile(
            input_shapefile_path=temp_shapefile_path,
            output_shapefile_path=input_shapefile_path)

        delete_shapefile(shapefile_path=temp_shapefile_path)

    return 0


####################################################
# spatial operations
####################################################
def create_spatial_index(
    feature_dict: dict,
    id_key: str='wpid'):
    """
    Description:
        create a set of spatial indices for spatial analysis
    
    Args:
        feature_dict: dict to dicts (features) to index
        uses the keys in the top dict as the index value

    Return:
        spatial index object:  
    """
    if not isinstance(feature_dict, dict):
        raise AttributeError('features must be a dictionary')
    # create a spatial index object
    idx = rtree.index.Index()
    # populate the spatial index
    for key in feature_dict:
        geometry = shape(feature_dict[key]['geometry'])
        idx.insert(feature_dict[key]['properties'][id_key], geometry.bounds)

    return idx

###########################
def check_for_overlap(
    spatial_indices: rtree.index, 
    feature_dict: dict, 
    feature: dict,
    id_key: str = 'wpid'):
    """
    Description:
        takes a dictionary of features and a matching set of spatial indices and 
        a specific feature and finds if that feature intersects with any other

        meant to be used as a sub function

    Args:
        spatial_indices: rtree based spatial index object
        feature_dict: fiona based feature dict organised
        by a specific id
        feature: fiona based feature to check
        id_key: to use for indexes auto set to the package wpid
        expected that the same ids are present in the spatial_indeces

    Return: 
        list: intersecting features
    """
    intersecting_ids = []
    check_geometry = shape(feature['geometry'])

    existing_ids = [feature_dict[key]['properties'][id_key] for key in feature_dict]

    if id_key not in feature['properties'].keys():
        raise KeyError('required id_key: {} not found among the feature properties while checking for intersections'.format(id_key))

    # get list of ids where bounding boxes intersect
    ids = [int(i) for i in spatial_indices.intersection(check_geometry.bounds)]

    ids = [id for id in ids if id != int(feature['properties'][id_key])]

    ids = [id for id in ids if id in existing_ids]

    # access the features that those ids reference
    for id in ids:
        sub_feature = feature_dict[id]
        sub_geom = shape(sub_feature['geometry'])

        # check the geometries intersect, not just their bboxs
        if check_geometry.overlaps(sub_geom):
            intersecting_ids.append(id) 

    return intersecting_ids


###########################
def overlap_among_features(
    spatial_indices: rtree.index, 
    feature_dict: dict,
    id_key: str = 'wpid'):
    """
    Description:
        takes a dictionary of features and checks if any of the features intersect

        meant to be used as a sub function

    Args:
        spatial_indices: rtree based spatial index object
        feature_dict: fiona based feature dict organised
        by a specific id
        id_key: to use for indexes auto set to the package wpid
        expected that the same ids are present in the spatial_indices

    Return: 
        bool: True if intersection exists, False if Not
    """
    intersection_exists = False

    existing_ids = [feature_dict[key]['properties'][id_key] for key in feature_dict]

    temp_key = next(iter(feature_dict))
    if id_key not in feature_dict[temp_key]['properties'].keys():
        raise KeyError('required id_key: {} not found among the feature properties while checking for intersections'.format(id_key))

    for key in feature_dict:
        if intersection_exists:
            break
        check_geometry = shape(feature_dict[key]['geometry'])

        # get list of ids where bounding boxes intersect
        ids = [int(i) for i in spatial_indices.intersection(check_geometry.bounds)]

        ids = [id for id in ids if id != int(feature_dict[key]['properties'][id_key])]

        ids = [id for id in ids if id in existing_ids]

        # access the features that those ids reference
        for id in ids:
            sub_feature = feature_dict[id]
            sub_geom = shape(sub_feature['geometry'])

            # check the geometries intersect, not just their bboxs
            if check_geometry.overlaps(sub_geom):
                intersection_exists = True
                break

    return intersection_exists

##########################
def union_and_drop(
    spatial_indices: rtree.index, 
    feature_dict: dict, 
    id_key: str = 'wpid'):
    """
    Description:
        takes a colleciton of features and a matching set of spatial indices and 
        unions applicable features and drops the ones unioned from the dict

        meant to be used as a sub function

    Args:
        spatial_indices: rtree based spatial index object
        feature_dict: fiona based feature dict organised
        by a specific id
        id_key: to use for indexes auto set to the package wpid
        expected that the same ids are present in the spatial_indeces

    Return: 
        list: unioned and filtered features
    """
    checked_wpids = []
    drop_wpids =[]

    # reset index for security
    feature_dict = {feature_dict[key]['properties'][id_key]: feature_dict[key] for key in feature_dict}

    # edit the polygons
    for wpid, feature in feature_dict.items():
        # check for intersections and union
        if wpid not in checked_wpids:
            intersecting_wpids = check_for_overlap(
                spatial_indices=spatial_indices, 
                feature_dict=feature_dict, 
                feature=feature,
                id_key=id_key)

            checked_wpids.append(wpid)
            intersecting_wpids = [wpid for wpid in intersecting_wpids if wpid not in checked_wpids]

            if intersecting_wpids:
                union_geoms = [shape(feature_dict[wpid]['geometry']) for wpid in intersecting_wpids]
                union_geoms.append(shape(feature_dict[wpid]['geometry']))
                unioned_geom = ops.unary_union(union_geoms)
                feature_dict[wpid]['geometry'] = mapping(unioned_geom)
                drop_wpids.extend(intersecting_wpids)  
                checked_wpids.extend(intersecting_wpids)

    # drop unionised features
    for wpid in drop_wpids:
        feature_dict.pop(wpid)

    return feature_dict

##########################
def polygonize_cleanup(
    input_shapefile_path: str,
    output_shapefile_path: str,
    fill_holes: bool=True,
    area_threshold: float = 0,
    round_edges: bool=False):
    """
    Description:
        takes a set of polygons and cleans them up 
        with the holes filled and edges smoothed out

        built to be used on the output of gdal polygonize

    Args:
        input_shapefile_path: path to the shapefile to split
        output_shapefile_path: path to output the celaned shapefile too
        fill_holes: if True fills the holes in seperate polygons found
        using a buffer around the exterior
        round_edges: if true rounds the edges of the polygons smoothing out 
        the polygons
        area_threshold: filters out polygons with an area smaller than this 
        threshold in the input raster

    Return:
        int: 0
    
    """
    if round_edges:
        join_style = 1
    else:
        join_style=2

    # check for and add possible missing identifier:
    check_add_wpid_to_shapefile(input_shapefile_path=input_shapefile_path)

    # retrieve feature_dict from the input file
    with fiona.open(input_shapefile_path, 'r') as input:
        output_driver = input.driver
        output_crs = input.crs
        output_schema = dict(input.schema) 
        output_schema['geometry'] = "Polygon"

        features = {feature['properties']['wpid']: feature for feature in input}

    input = None

    # create spatial index
    idx = create_spatial_index(feature_dict=features)

    # fill the holes if wanted
    if fill_holes:
        for __, feature in features.items():
            poly = shape(feature['geometry'])
            poly = poly.buffer(0.01, join_style=join_style).buffer(-0.01, join_style=join_style)
            feature['geometry'] = mapping(poly)

    while overlap_among_features(
        spatial_indices= idx, 
        feature_dict= features):

        # union features
        features = union_and_drop(
            spatial_indices=idx, 
            feature_dict=features)

        # fill the holes if wanted
        if fill_holes:
            for __, feature in features.items():
                poly = shape(feature['geometry'])
                poly = poly.buffer(0.01, join_style=join_style).buffer(-0.01, join_style=join_style)
                feature['geometry'] = mapping(poly)

    # remove the remaining too small polygons
    drop_small_ids = []
    for wpid,feature in features.items():
        poly = shape(feature['geometry'])
        if isinstance(poly, MultiPolygon):
            pass
        if poly.area < area_threshold:
            drop_small_ids.append(wpid)

    # drop unionised features
    for wpid in drop_small_ids:
        features.pop(wpid) 

    # write the edited polygons to file
    with fiona.open(
        output_shapefile_path, 
        'w', 
        driver=output_driver,
        crs=output_crs,
        schema=output_schema) as output:
    
        # write the input file to output
        for fid, feature in features.items():
            poly = shape(feature['geometry'])
            if isinstance(poly, Polygon):
                    output.write({
                        'properties': feature['properties'],
                        'geometry': mapping(poly)
                    })
            elif isinstance(poly, MultiPolygon):
                for subpoly in poly:
                    output.write({
                        'properties': feature['properties'],
                        'geometry': mapping(subpoly)
                    })

            else:
                print('non polygon feature found at the end of cleaning up polygons ... discarding geometry')
                pass

    output = None

    # check for and add possible missing identifier:
    check_add_wpid_to_shapefile(input_shapefile_path=input_shapefile_path, overwrite=True)

    return 0

##########################
def raster_to_polygon(
    input_raster_path: str,
    output_shapefile_path: str,
    mask_raster_path : str = None):
    """
    Description:
        internal polygonize method built around gdal polygonize that 
        polygonizes/vectorises cells in a raster

    Args:
        input_raster_path: path to the raster to polygonize
        output_shapefile_path: path to output the made shapefile too
        mask_raster_path: if provided masks to that raster

    Return:
        int: 0
    """
    source_dataset = gdal.Open(input_raster_path)
    if source_dataset is None:
        raise AttributeError('Unable to open %s' % source_dataset)

    source_band = source_dataset.GetRasterBand(1)
    
    if mask_raster_path:
        mask_dataset = gdal.Open(mask_raster_path)
        if mask_dataset is None:
            raise AttributeError('Unable to open %s' % mask_dataset)
        mask_band = mask_dataset.GetRasterBand(1)
    else:
        mask_band = None

    shapefile_name = os.path.splitext(os.path.basename(output_shapefile_path))[0]
    drv = ogr.GetDriverByName("ESRI Shapefile")
    shapefile_out = drv.CreateDataSource(output_shapefile_path)

    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(gdal_info(input_raster_path)['crs'])

    # create the layer
    shapefile_layer = shapefile_out.CreateLayer(shapefile_name, srs, geom_type=ogr.wkbPolygon)
    
    newField = ogr.FieldDefn('ID', ogr.OFTInteger)
    shapefile_layer.CreateField(newField)
    gdal.Polygonize(source_band, mask_band, shapefile_layer, 0, [])

    shapefile_out = shapefile_layer = None

    # check for and add possible misisng identifier:
    check_add_wpid_to_shapefile(input_shapefile_path=output_shapefile_path)

    return 0
