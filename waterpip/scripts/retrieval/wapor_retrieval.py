"""
script for the retrieval of WAPOR data utilising the package WAPOROCW made by ITC DELFT 
"""

import os
import sys
import shutil
import datetime
from datetime import datetime, timedelta
from timeit import default_timer
import time


from ast import literal_eval
import numpy as np
import pandas as pd
from shapely.geometry import shape, mapping, Polygon
import fiona 
from fiona.crs import from_epsg
import rtree
import requests

from waterpip.scripts.retrieval.wapor_api import WaporAPI
from waterpip.scripts.retrieval.wapor_land_cover_classification_codes import WaporLCC
from waterpip.scripts.structure.wapor_structure import WaporStructure
from waterpip.scripts.support import raster, vector, statistics

def printWaitBar(i, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    This function will print a waitbar in the console

    Variables:

    i -- Iteration number
    total -- Total iterations
    fronttext -- Name in front of bar
    prefix -- Name after bar
    suffix -- Decimals of percentage
    length -- width of the waitbar
    fill -- bar fill

    Authors: Tim Hessels
            UNESCO-IHE 2017
    Contact: t.hessels@unesco-ihe.org
    Repository: https://github.com/wateraccounting/watools
    Module:Functions/Start
    """
    # Adjust when it is a linux computer
    if (os.name=="posix" and total==0):
        total = 0.0001

    percent = ("{0:." + str(decimals) + "f}").format(100 * (i / float(total)))
    filled = int(length * i // total)
    bar = fill * filled + '-' * (length - filled)

    sys.stdout.write('\r%s |%s| %s%% %s' %(prefix, bar, percent, suffix))
    sys.stdout.flush()

    if i == total:
        print()

def process_wapor_time_code(time_code: str):
    """
    Description:
        small function to process the wpaor row[time_code] entry
        into a standardised filename time_code

    Args:
        time_code: time_code to process

    Return:
        str: reformatted time code 
    """
    time_code = time_code.replace(',', '_')
    for item in ['[',']',')','(','-']:
        time_code = time_code.replace(item,'')

    return time_code

class WaporRetrieval(WaporAPI,WaporStructure):
    """
    Description:
        Retrieves rasters from the Wapor database given the appropriate inputs

    Args:
        waterpip_directory: directory to output downloaded and processed data too
        shapefile_path: path to the shapefile to clip downloaded data too if given
        rasters to the meta of this raster if not provided uses the shapefile to do the same
        and otherwise the combination of the generated bbox and output crs
        wapor_level: wapor_level integer to download data for either 1,2, or 3
        api_token: api token to use when downloading data 
        project_name: name of the location to store the retrieved data  
        period_start: datetime object specifying the start of the period 
        period_end: datetime object specifying the end of the period 
        datacomponents: wapor datacomponents (interception (I) etc.) to download
        return_period: return period code of the component to be downloaded (D (Dekadal) etc.)
        (retrieve from https://wapor.apps.fao.org/profile)
        version: WAPOR version to use (standardised to 2.0)
        silent: boolean, if True the more general messages in the class are not printed 
        (autoset to False)       
        )

    return: 
        WAPOR rasters matching the given information are retrieved and stored in the 
        specified project  
    """
    def __init__(
        self,
        waterpip_directory: str,
        shapefile_path: str,
        wapor_level: int,
        api_token: str,
        project_name: int = 'test',
        period_start: datetime = datetime.now() - timedelta(days=1),
        period_end: datetime = datetime.now(),
        datacomponents: list = ['ALL'],
        return_period: str = 'D',
        wapor_version: int = 2,
        silent: bool=False,

    ):
        self.period_start = period_start
        self.period_end = period_end
        self.waterpip_directory = waterpip_directory
        self.project_name = project_name
        self.wapor_level = wapor_level
        self.datacomponents = datacomponents
        self.return_period = return_period
        self.api_token = api_token
        self.wapor_version = wapor_version
        self.silent = silent

        WaporAPI.__init__(self,
            period_start=self.period_start,
            period_end=self.period_end,
            version=self.wapor_version
        )

        WaporStructure.__init__(self,
            return_period=self.return_period,
            waterpip_directory=self._project_directory,
            project_name=self.project_name,
            period_end=self.period_end,
            period_start=self.period_start,
            wapor_level=self.wapor_level
        )

        self.shapefile_path = shapefile_path
        self.output_crs = None
        self.bbox = None
        self.bbox_shapefile = None
        self.land_classification_codes = None
      
        assert self.wapor_level in [1,2,3] , "wapor_level (int) needs to be either 1, 2 or 3"

        # check all catalogs and reference shapefiles have been stored in the metadata folder
        if not self.silent:
            print('running check for all wapor wapor_level catalogues and downloading as needed:')
        for wapor_level in (1,2,3):
            self.getCatalog(wapor_level=wapor_level)

        # check for a wapor_level 3 location shapefile
        self.get_level_3_availability_shapefile(wapor_level=3)

        # if wapor_level 3 check if the given area falls within an available area:
        if self.wapor_level == 3:
            self.country_code = self.check_level_3_location()
        else:
            self.country_code = 'notlevel3notused'
        if not self.silent:
            print('loading wapor catalogue for this run:')
        # set instance catalog
        self.catalog = self.getCatalog(wapor_level=self.wapor_level)

        # check if given return period exists against the retrieved catalog
        self.check_return_period() 

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        """ 
        quick description:
            takes a shapefile and produces a bounding box tuple for the download
        """
        run = False
        if not value:
            run = True
        elif not isinstance(value, tuple):  
            run = True
        else:
            pass
        if run:         
            if not self.shapefile_path:
                raise AttributeError('please provide the path to a shapefile')
            else:
                if os.path.exists(self.shapefile_path):
                    df = vector.file_to_records(self.shapefile_path, output_crs=4326)
                    self._bbox = vector.retrieve_geodataframe_bbox(df)

                else:
                    raise AttributeError('please provide the path to a shapefile')


    @property
    def land_classification_codes(self):
        return self._land_classification_codes

    @land_classification_codes.setter
    def land_classification_codes(self,value):
        """
        quick description:
            takes the wapor_level and retrieves the 
            land classification codes dict for it
        """
        if isinstance(value, dict):
            self._land_classification_codes = value
        else:
            assert self.wapor_level in (1,2,3), 'land classification codes is set using the wapor_level' 
            self._land_classification_codes = WaporLCC(wapor_level=self.wapor_level)

    @property
    def bbox_shapefile(self):
        return self._bbox_shapefile

    @bbox_shapefile.setter
    def bbox_shapefile(self, value):
        """ 
        quick description:
            takes a bbox and produces a bounding box shapefile for comparisons
        """
        run = False
        # check if a bbox shapefile has been previously made
        bbox_shapefile_name = os.path.splitext(os.path.basename(self.shapefile_path))[0] + '_bbox.shp'
        bbox_shapefile_path = os.path.join(self.project['reference'],bbox_shapefile_name)
        if not os.path.exists(bbox_shapefile_path):
            run = True
        else:
            pass
        
        if run:         
            if not self.shapefile_path:
                raise AttributeError('please provide the path to a shapefile')
            else:
                if os.path.exists(self.shapefile_path):
                    df = vector.file_to_records(self.shapefile_path, output_crs=4326)
                    bbox = vector.retrieve_geodataframe_bbox(df)
                    bbox_shapefile_name = os.path.splitext(os.path.basename(self.shapefile_path))[0] + '_bbox.shp'
                    bbox_shapefile_path = os.path.join(self.project['reference'],bbox_shapefile_name)
                    vector.create_bbox_shapefile(
                        output_shape_path=bbox_shapefile_path,
                        bbox=bbox)

                    if not self.silent:
                        print('bbox shapefile based on the input shapefile made and outputted too: {}'.format(bbox_shapefile_path))
            
                    self._bbox_shapefile = bbox_shapefile_path

                else:
                    raise AttributeError('please provide the path to a shapefile')

        else:
            if not self.silent:
                print('bbox shapefile based on the input shapefile can be found at: {}'.format(bbox_shapefile_path))
            self._bbox_shapefile = bbox_shapefile_path

    @property
    def output_crs(self):
        return self._output_crs

    @output_crs.setter
    def output_crs(self, value):
        """ 
        quick description:
            checks if a project crs has been provided and if not attempts to 
            retrieve it from the shapefile provided
        """
        if not value or not isinstance(value, int):
            if not self.shapefile_path:
                raise AttributeError('please provide either the path to a shapefile or a crs directly, 4326 is accepted')
            else:
                if os.path.exists(self.shapefile_path):
                    self._output_crs = vector.retrieve_shapefile_crs(self.shapefile_path)

                else:
                    raise AttributeError('please provide either the path to a shapefile or a crs directly, 4326 is accepted')

    #################################
    # check functions
    #################################
    def check_return_period(self):
        """
        Description
            checks if the return period code given during class initialisation exists in the given catalog

            NOTE: auto limited to return period codes belonging to the wapor_level previously 
            specified during class initiation

        Args: 
            return_period: return period code to check for

        Return
            int: 0

        Raise:
            AttributeError: If return period does not match expected return period codes
        """
        codes = list(self.catalog['period_code'])
        desc = list(self.catalog['period_desc'])

        combos = [item for item in zip(codes,desc)]

        combos = list(set(combos)) # filter to unique combos

        if not self.return_period in codes:
            print('given return period could not be found among the expected codes, use one of the following and try again')
            print(combos)
            raise AttributeError

        return 0

    #################################
    def check_datacomponents(
        self, 
        datacomponents: list, 
        return_period: str= None):
        """
        Description
            checks if the datacomponents given during class inititialisation are real ones that can be retrieved 
            according to the catalog, if All is provided all datacomponents are returned.

            NOTE: auto limited to datacomponents belonging to the wapor_level previously 
            specified during class initiation

        Args: 
            datacomponents: list of datacomponents to check
            return_period: if provided overwrties class return period

        Return
            list: list of datacomponents that do exist

        Raise:
            raises error if no datacomponents exists
        """
        if not return_period:
            return_period = self.return_period
        catalog_components = list(set(list(self.catalog['component_code'])))
        catalog_codes = list(set(list(self.catalog['code'])))

        if self.wapor_level == 3:
            if datacomponents[0] is 'ALL':
                existing_datacomponents = [comp for comp in catalog_components if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) in catalog_codes] 
                existing_codes = ['L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) for comp in catalog_components if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) in catalog_codes] 
                missing_datacomponents = [comp for comp in catalog_components if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) not in catalog_codes] 
                missing_codes = ['L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) for comp in catalog_components if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) not in catalog_codes] 

            else:
                existing_datacomponents = [comp for comp in datacomponents if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) in catalog_codes]
                existing_codes = ['L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) for comp in datacomponents if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) in catalog_codes] 
                missing_datacomponents = [comp for comp in datacomponents if  'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) not in catalog_codes]
                missing_codes = ['L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) for comp in datacomponents if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) not in catalog_codes] 


        else:
            if datacomponents[0] is 'ALL':
                existing_datacomponents = [comp for comp in catalog_components if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) in catalog_codes] 
                existing_codes = ['L{}_{}_{}'.format(self.wapor_level,comp,return_period) for comp in catalog_components if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) in catalog_codes] 
                missing_datacomponents = [comp for comp in catalog_components if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) not in catalog_codes] 
                missing_codes = ['L{}_{}_{}'.format(self.wapor_level,comp,return_period) for comp in catalog_components if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) not in catalog_codes] 

            else:
                existing_datacomponents = [comp for comp in datacomponents if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) in catalog_codes]
                existing_codes = ['L{}_{}_{}'.format(self.wapor_level,comp,return_period) for comp in datacomponents if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) in catalog_codes] 
                missing_datacomponents = [comp for comp in datacomponents if  'L{}_{}_{}'.format(self.wapor_level,comp,return_period) not in catalog_codes]
                missing_codes = ['L{}_{}_{}'.format(self.wapor_level,comp,return_period) for comp in datacomponents if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) not in catalog_codes] 

        if missing_datacomponents:
            print('the following given datacomponents could not be found in the wapor_level catalog or were not available for the specified return period:\n {}'.format(missing_codes))
            print('continuing with the remainder')

        if not existing_datacomponents:
            raise AttributeError('none of the given datacomponents were found back in the wapor_level catalog, datacomponent codes not found: {}'.format(missing_codes))

        return existing_datacomponents

    #################################
    def check_level_3_location(self):
        """
        Description:
            takes the given shapefiel/boundingbox and checks if it 
            falls within the boundaries of a wapor_level 3 area by comparing with
            the level3 shapefile generated on initialisation

        Args:
            self: shapefile_path

        Return:
            str: area code of the area the shapefile falls within

        Raises:
            FileNotFoundError: if the shapefile/bbox does not fall within an available area
        """
        code = None

        with fiona.open(self.l3_locations_shapefile_path, 'r') as layer1:
            with fiona.open(self.bbox_shapefile, 'r') as layer2:
                index = rtree.index.Index()
                for feat1 in layer1:
                    fid = int(feat1['id'])
                    geom1 = shape(feat1['geometry'])
                    index.insert(fid, geom1.bounds)

                for feat2 in layer2:
                    geom2 = shape(feat2['geometry'])
                    for fid in list(index.intersection(geom2.bounds)):
                        if fid != int(feat2['id']):
                            feat1 = layer1[fid]
                            geom1 = shape(feat1['geometry'])
                            if geom1.intersects(geom2):
                                # We retrieve the country code and break the loop
                                code = feat1['properties']['code']
                                break

        if not code:
            raise AttributeError('no intersection between wapor_level 3 locations and the given shapefile (no data available for the given area')
        
        return code


    #################################
    # retrieval functions
    #################################
    def getCatalog(
        self,
        wapor_level: int = None, 
        cubeInfo=True):
        '''
        Get catalog from workspace
        ''' 
        retrieve=False
        catalogue_csv = os.path.join(self.project['meta'], 'wapor_catalogue_L{}.csv'.format(wapor_level))
        if not os.path.exists(catalogue_csv):
            retrieve = True
        
        else:
            st=os.stat(catalogue_csv)    
            if (time.time() - st.st_mtime) >= 5184000: # 60 days
                retrieve = True
        
        if retrieve:
            print('No or Outdated WaPOR catalog found for wapor_level: {}, retrieving...'.format(wapor_level))
            try:
                df = self._query_catalog(wapor_level)
            except:
                print('ERROR: data of the specified wapor_level could not be retrieved or there was a connection error (wapor_level: {})'.format(self.wapor_level))
            if cubeInfo:
                cubes_measure=[]
                cubes_dimension=[]
                for cube_code in df['code'].values:                
                    cubes_measure.append(self._query_cubeMeasures(cube_code,
                                                                       version=self.version))
                    cubes_dimension.append(self._query_cubeDimensions(cube_code,
                                                                       version=self.version))
                df['measure'] = cubes_measure
                df['dimension'] = cubes_dimension

            df['period_code'] = df['code'].str.split('_').str[-1]
            df['component_code'] = df['code'].str.split('_').str[-2]
            df['component_desc'] = df['caption'].str.split('(').str[0]

            df.loc[df['period_code'] == 'LT' ,'period_desc'] = 'Long Term'
            df.loc[df['period_code'] == 'A' ,'period_desc'] = 'Annual'
            df.loc[df['period_code'] == 'S' ,'period_desc'] = 'Seasonal'
            df.loc[df['period_code'] == 'M' ,'period_desc'] = 'Monthly'
            df.loc[df['period_code'] == 'D' ,'period_desc'] = 'Dekadal'
            df.loc[df['period_code'] == 'E' ,'period_desc'] = 'Daily'

            if wapor_level == 3:
                df['country_code'] = df['code'].str.split('_').str[1]
                df['country_desc'] = df['caption'].str.split('\(').str[-1].str.split('-').str[0]
            
            else:
                pass

            df.loc[df['code'].str.contains('QUAL'), 'component_code'] = 'QUAL_' + df['component_code'] 

            df = df.fillna('NaN')

            statistics.output_table(
                table=df,
                output_file_path=catalogue_csv,
                csv_seperator=';')
        
            print("outputted table of the WaPOR catalogue for wapor_level: {}".format(wapor_level))
            print(catalogue_csv)
            print("on running the retrieval class again the catalogue will be auto replaced if the catalogue becomes outdated after a 60 day period or if is not found")
            
        else:
            df = pd.read_csv(catalogue_csv, sep=';')
            df['measure'] = df['measure'].apply(lambda x: literal_eval(x))
            df['dimension'] = df['dimension'].apply(lambda x: literal_eval(x))

        self.catalog=df
        
        if not self.silent:
            print('Loading WaPOR catalog for wapor_level: {}'.format(wapor_level))
            print('catalogue location: {}'.format(catalogue_csv))

        return self.catalog       

    #################################
    def get_level_3_availability_shapefile(
        self, 
        wapor_level: int = None):
        """
        Description:
            generates a shapefile of available areas at wapor_level 3 
            that can be used for testing if data is available

        Args:


        Return:
            0 : the generated shapefile is stored in the standard metadata folder

        """
        retrieve = False
        l3_locations_shapefile_path = os.path.join(self.project['meta'], 'wapor_L{}_locations.shp'.format(wapor_level))

        # check if the file already exists
        if not os.path.exists(l3_locations_shapefile_path):
            retrieve = True
        else:
            # check how old the file is
            st=os.stat(l3_locations_shapefile_path)    
            if (time.time() - st.st_mtime) >= 5184000: # 60 days
                retrieve = True

        if retrieve:
            print('creating wapor_level {} locations shapefile'.format(wapor_level))
            # set temporary date variables
            api_per_start = datetime(2010,1,1).strftime("%Y-%m-%d")
            api_per_end = datetime.now().strftime("%Y-%m-%d")
            api_period = '{},{}'.format(api_per_start, api_per_end)

            # retrieve country codes
            temp_catalog = self.getCatalog(wapor_level=wapor_level) 
            country_codes = zip(list(temp_catalog['country_code']),list(temp_catalog['country_desc']))
            codes = list(set([(x,y) for x,y in country_codes]))
            _data = []
            # loop through countries check data availability and retrieve the bbox
            for code in codes:
                cube_code= f"L3_{code[0]}_T_D"

                try:
                    df_avail=self.getAvailData(cube_code,time_range=api_period)
                except:
                    print('ERROR: cannot get list of available data')
                    raise AttributeError('check the cube code against the L3 catalogue: {}'.format(cube_code))
                
                bbox = df_avail.iloc[0]['bbox'][1]['value']
                _data.append(({
                    'code':code[0],
                    'country': code[1],
                    'srid':df_avail.iloc[0]['bbox'][1]['srid'],
                    'bbox': ', '.join(map(str, bbox))},
                    [   [bbox[0],bbox[3]],
                        [bbox[2],bbox[3]],
                        [bbox[2],bbox[1]],
                        [bbox[0],bbox[1]]]))

            # Define shp file schema
            schema = { 'geometry': 'Polygon', 
                       'properties': { 'code': 'str' ,
                                       'country': 'str',
                                       'srid': 'str',
                                       'bbox': 'str'} }

            # Create shp file
            with fiona.open(l3_locations_shapefile_path, "w", "ESRI Shapefile", schema, crs=from_epsg(4326)) as output:
                # Loop through the dict and populate shp file
                for d in _data:
                    # Write output
                    output.write({
                        'properties': d[0], 
                        'geometry': mapping(Polygon(d[1]))
                    })


            print('wapor_level 3 locations file made')
        
        else:
            if not self.silent:
                print('wapor_level 3 location shapefile exists skipping retrieval')

        if not self.silent:
            print('wapor_level 3 location shapefile: {}'.format(l3_locations_shapefile_path))

        self.l3_locations_shapefile_path = l3_locations_shapefile_path

        return 0

    #################################
    def create_crop_mask_from_shapefile(
        self,
        crop: str,
        input_shapefile_path: str = None,
        template_raster_path: str = None,
        output_name: str = None,
        period_start=datetime(2020,1,1),
        period_end=datetime(2020,1,2)):
        """
        Description:
            creates a crop mask for further anaylsis
            from the shapefile provided using either the whole shape
            or specific geometries defined in a shapefile column
        
        Args:
            input_shapefile_path: shapefile to create the crop mask with, if not provided 
            uses the class shapefile (this is recommended)
            output_name: name of the output raster and shapefile. If not provided 
            auto generates them. Location is fixed
            template_raster_path: raster providing the metadata for the output raster
            if not provided retrieves a raster from WAPOR to use as the template
            crop: crop name for the output raster
            period_start: standard value used to grab a template raster if  one is not provided, 
            can be ignored if the function works
            period_end: standard value used to grab a template raster if  one is not provided, 
            can be ignored if the function works
            return_period: standard value used to grab a template raster if  one is not provided, 
            can be ignored if the function works

        Return:
            tuple: path to the crop mask raster created, path to the crop mask shapefile created
        """
        if not input_shapefile_path:
            input_shapefile_path = self.shapefile_path

        crop = crop.lower().replace(' ', '_')
        print('crop was autocorrected to :{}'.format(crop))

        if not output_name:
            output_name = '{}_mask'.format(crop)

        crop_mask_raster_path = os.path.join(self.project['reference'], '{}.tif'.format(output_name))
        crop_mask_shape_path = os.path.join(self.project['reference'], '{}.shp'.format(output_name))

        if not os.path.exists(crop_mask_shape_path) or not os.path.exists(crop_mask_raster_path):
            if not template_raster_path:
                for rp in ['D', 'M', 'A']:
                    # get a raster as template
                    wapor_list = self.retrieve_wapor_download_info(
                        datacomponents=['ALL'],
                        return_period=rp,
                        period_start=period_start,
                        period_end=period_end,
                    )

                    if len(wapor_list) > 0:
                        break

                wapor_list = [wapor_list[0]]

                wapor_rasters = self.retrieve_wapor_rasters(
                    wapor_list=wapor_list,
                    create_vrt=False,
                )
                template_raster_path = wapor_rasters['T']['raster_list'][0]
            
            if not os.path.exists(crop_mask_raster_path):
                raster.rasterize_shape(
                    template_raster_path=template_raster_path,
                    shapefile_path=input_shapefile_path,
                    output_raster_path=crop_mask_raster_path
                    )

                print("crop mask raster made: {}".format(crop_mask_raster_path))

    
            else:
                if not self.silent:
                    print("preexisting crop mask raster found skipping step")
                raster.check_gdal_open(crop_mask_raster_path)

            if not os.path.exists(crop_mask_shape_path):
                vector.copy_shapefile(
                    input_shapefile_path=input_shapefile_path,
                    output_shapefile_path=crop_mask_shape_path)

                vector.check_add_wpid_to_shapefile(input_shapefile_path=crop_mask_shape_path)

                print("crop mask shapefile made: {}".format(crop_mask_shape_path))

            else:
                if not self.silent:
                    print("preexisting crop mask shape found skipping step")

        else:
            if not self.silent:
                print("preexisting crop mask and shape found skipping step")
            raster.check_gdal_open(crop_mask_raster_path)

        return crop_mask_raster_path, crop_mask_shape_path

    #################################
    def retrieve_crop_mask_from_WAPOR(
        self,
        crop: str,
        period_start: datetime=None,
        period_end: datetime=None,
        output_crop_count_csv: bool=True,
        area_threshold: int = 0):
        """
        Description:
            creates a crop mask raster and shapefile for further anaylsis
            using the bbox as defined by the shapefile and crop initially
            provided and the land cover classification rasters 
            that can be found on the WAPOR database
            if the period defined covers more than one raster it
            combines them all into one for the entire period.
            keeping the classification most common across the entire period

            the crop mask retrieved from WAPOR is considered the raw one as is the 
            shapefile based on it the crop mask is further clipped to an edited version
            of the initial crop mask the user can pick which one they use  
        
        Args:
            crop: crop to mask too has to match the name used in the wapor database  
            classification codes          
            period_start: period for which to retrieve the land cover raster
            period_end: period for which to retrieve the land cover raster,
            output_crop_count_csv: bool if True outputs a csv of the crops found in the 
            area and their occurences
            area_threshold: area threshold with which to filter out too small polygons  
            (single cell size * threshold makes the threshold) 

            uses the value defined during class intialisation if period_start,
            period_end or return_period or input_shapefile_path is not provided

        Return:
            tuple: path to the crop mask raster created, path to the crop mask shape created
        """
        if not period_start:
            period_start=period_start
        if not period_end:
            period_end = self.period_end

        crop = crop.lower().replace(' ', '_')
        print('crop was autocorrected to :{}'.format(crop))

        date_dict = self.generate_dates_dict(
            period_start=period_start,
            period_end=period_end,
            return_period='D',
            )   

        # check that the lcc exists in the given dataset
        if not crop in list(self.land_classification_codes.keys()):
            print('crop given for the wapor crop mask must be at least one of the following: {}'.format(list(self.land_classification_codes.keys())))
            raise KeyError('another check will be carried out to see if the given crop exists in the specified area after retrieving data, \
            for now please provide an exisitng crop')

        
        # create the file paths (as the LLC retrieved can differ depending on date the crop mask is period specific)
        crop_mask_shape_path = os.path.join(self.project['reference'], '{}_{}_{}_mask.shp'.format(crop, date_dict['per_start_str'], date_dict['per_end_str']))
        crop_mask_raster_path = os.path.join(self.project['reference'], '{}_{}_{}_mask.tif'.format(crop, date_dict['per_start_str'], date_dict['per_end_str']))
        crop_mask_shape_path_raw = os.path.join(self.project['reference'], '{}_{}_{}_raw_mask.shp'.format(crop, date_dict['per_start_str'], date_dict['per_end_str']))
        crop_mask_raster_path_raw = os.path.join(self.project['reference'], '{}_{}_{}_raw_mask.tif'.format(crop, date_dict['per_start_str'], date_dict['per_end_str']))
        crop_count_csv = os.path.join(self.project['reference'], '{}_{}_{}_crop_count.csv'.format(crop, date_dict['per_start_str'], date_dict['per_end_str']))
        most_common_crop_path = os.path.join(self.project['analysis'], 'LCC_{}_{}_mostcommon.tif'.format(crop, date_dict['per_start_str'], date_dict['per_end_str']))

        if not os.path.exists(crop_mask_shape_path_raw) or not os.path.exists(crop_mask_raster_path_raw):
            if not os.path.exists(crop_mask_raster_path_raw):
                # retrieve the lcc rasters and find the most common class per cell across the given period
                if not os.path.exists(most_common_crop_path):
                    if self.wapor_level < 3:
                        wapor_list = self.retrieve_wapor_download_info(
                            datacomponents=['LCC'],
                            return_period='A',
                            period_start=period_start,
                            period_end=period_end,
                        )
                    else:
                        wapor_list = self.retrieve_wapor_download_info(
                            datacomponents=['LCC'],
                            return_period='D',
                            period_start=period_start,
                            period_end=period_end,
                        )

                    wapor_rasters = self.retrieve_wapor_rasters(
                        wapor_list=wapor_list,
                        create_vrt=False,
                    )
                    if len(wapor_rasters['LCC']['raster_list']) > 1:
                        # if more than one raster exists the median (most common) land cover class across the period is assigned
                        statistics.calc_multiple_array_numpy_statistic(
                            input=wapor_rasters['LCC']['raster_list'],
                            numpy_function=statistics.mostcommonzaxis,
                            output_raster_path=most_common_crop_path)
                    else:
                        shutil.copy2(src= wapor_rasters['LCC']['raster_list'][0], dst=most_common_crop_path)
                
                # retrieve and sort the counts of lcc classes
                counts_list = raster.count_unique_values_raster(
                    input_raster_path=most_common_crop_path)

                # create counts dictionary
                key_list = list(self.land_classification_codes.keys())
                val_list = list(self.land_classification_codes.values())
                
                lcc_list = []
                for count_dict in counts_list:
                    position = val_list.index(count_dict['value'])
                    name = key_list[position]
                    count_dict['landcover'] = name                
                    lcc_list.append(count_dict)

                if output_crop_count_csv:
                    statistics.output_table(
                        table=lcc_list,
                        output_file_path=crop_count_csv,
                        orient='index')

                lcc_dict = {_dict['landcover']: _dict for _dict in lcc_list}

                if crop in lcc_dict.keys():
                    print('percentage of occurrence of your chosen crop in the raster according to WAPOR is: {}'.format(lcc_dict[crop]['percentage']))

                else:
                    print('Provided below is a dictionary of the crops that are found in the area:')
                    print(lcc_dict)
                    raise KeyError('your given crop was not found oin the area according to wapor')
                

                # create the rawe crop maks raster
                mask_value = self.land_classification_codes[crop]

                raster.create_value_specific_mask(
                    mask_value=mask_value,
                    input_raster_path= most_common_crop_path,
                    output_raster_path=crop_mask_raster_path_raw,
                    output_crs=self.output_crs)
                
                print("raw crop mask raster made: {}".format(crop_mask_shape_path_raw))

            else:
                if not self.silent:
                    print("preexisting raw crop mask raster found skipping step, raw crop mask raster can be found at: {}".format(crop_mask_raster_path_raw))
                raster.check_gdal_open(crop_mask_raster_path_raw)

            if not os.path.exists(crop_mask_shape_path_raw):
                # create a shapefile of the raw crop_mask
                vector.raster_to_polygon(
                    input_raster_path=crop_mask_raster_path_raw,
                    output_shapefile_path=crop_mask_shape_path_raw,
                    mask_raster_path=crop_mask_raster_path_raw)

                print("raw crop mask shape made: {}".format(crop_mask_shape_path_raw))

            else:
                if not self.silent:
                    print("preexisting raw crop mask shape found skipping step, raw crop mask shape can be found at: {}".format(crop_mask_shape_path_raw))

        else:
            if not self.silent:
                print("preexisting raw crop masks found skipping step, raw crop masks can be found at: {}".format(self.project['reference']))
            raster.check_gdal_open(crop_mask_raster_path_raw)

        if not os.path.exists(crop_mask_shape_path) or not os.path.exists(crop_mask_raster_path):
            # create the cleaned shapefile
            if self.wapor_level == 3:
                if area_threshold < 2:
                    print('WARNING: at wapor level 3 the area threshold is set to a minimum of 2')
                    print('turn this of in the code to set it lower than this or work with the raw shapefile directly')
                    area_threshold = 2

            area_threshold = raster.gdal_info(crop_mask_raster_path_raw)['cell_size'] * area_threshold

            vector.polygonize_cleanup(
                input_shapefile_path=crop_mask_shape_path_raw,
                output_shapefile_path=crop_mask_shape_path,
                area_threshold=area_threshold)

            print("crop mask shape made: {}".format(crop_mask_shape_path))

            # create final crop mask
            crop_mask_raster_path, crop_mask_shape_path = self.create_crop_mask_from_shapefile(
                input_shapefile_path = crop_mask_shape_path,
                template_raster_path = crop_mask_raster_path_raw,
                output_name = os.path.splitext(os.path.basename(crop_mask_raster_path))[0],
                crop=crop)

        else:
            if not self.silent:
                print("preexisting crop mask and shape found skipping step, crop masks can be found at: {}".format(self.project['reference']))
            raster.check_gdal_open(crop_mask_raster_path)

        return crop_mask_raster_path, crop_mask_shape_path

    #################################
    def retrieve_wapor_download_info(self, 
        datacomponents: list=None, 
        period_start: datetime=None, 
        period_end: datetime=None,
        return_period: str = None):
        """
        Description:
            retrieves data from the WAPOR API according to the class inputs provided
            and generates a download dict for retrieving and manipulating the WAPOR rasters

            NOTE: works in combination with retrieve_wapor_rasters

        Args:
            self: datacomponents, wapor_level, return_period, out_dir, shapefile_path, period_start (see class for details)
            datcomponents: datacomponents if you want to repeat the function using non self functions,
            period_start: datacomponents if you want to repeat the function using non self functions,
            period_end: datacomponents if you want to repeat the function using non self functions,
            return_period: if provided overrides the return period of the class

        Return:
            list: list of dicts containing variables that can be used to retrieve and store a specific wapor raster
        """
        if not datacomponents:
            datacomponents = self.datacomponents
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period

        dates_dict = self.generate_dates_dict(
            period_start=period_start,
            period_end=period_end,
            return_period=return_period)
        
        # setup output list
        wapor_list = []
        datacomponents = self.check_datacomponents(datacomponents, return_period)
        for component in datacomponents:
            wapor_dict = {}
            regionless_component = component # for ease of use region is kept out of folder names
            print('retrieving download info for component: {}'.format(regionless_component))
            if self.wapor_level == 3:
                component = '{}_{}'.format(self.country_code, component)
                print('retrieving download info for wapor_level 3 region: {}'.format(self.country_code))

            # generate wapor cube code for retrieving data
            cube_code=f"L{self.wapor_level}_{component}_{return_period}"

            # check for and if needed create sub folders
            folder_cube_code = f"L{self.wapor_level}_{regionless_component}_{return_period}"
            download_component_dir = os.path.join(self.project['download'], folder_cube_code)
            if not os.path.exists(download_component_dir):
                os.makedirs(download_component_dir)
            processed_component_dir = os.path.join(self.project['processed'], folder_cube_code)
            if not os.path.exists(processed_component_dir):
                    os.makedirs(processed_component_dir)
            try:
                cube_info=self.getCubeInfo(cube_code)
                multiplier=cube_info['measure']['multiplier']
            except:
                print('ERROR: Cannot get cube info. Check if WaPOR version has cube %s'%(cube_code))
                break
            
            download_success=False
            attempt = 1
            while not download_success:
                try:
                    df_avail=self.getAvailData(cube_code,time_range=dates_dict['api_period'])
                    download_success = True   
                except requests.exceptions.RequestException:
                    time.sleep(10)  # wait 10 sec before retry
                    attempt +=1
                    if attempt > 300:
                        raise ConnectionError('retrieving data available statistics from WAPOR \
                        attempted 300 times and failed adjust the attempts or sleep time to try for longer')

                except:
                    print('cannot retrieve data for cube code: {}'.format(cube_code))
                    print('cannot retrieve data for period: {}'.format(dates_dict['api_period']))
                    raise AttributeError('cannot get list of available data for given cube code and period')
        
            dates = [df_avail['time_code'][row] for row in range(0,df_avail.shape[0])]
            dates = [process_wapor_time_code(date) for date in dates]

            print('attempting to retrieve donwload info for {} rasters from wapor'.format(len(df_avail)))  
            # set up waitbar
            total_amount = len(df_avail)
            amount = 0
            printWaitBar(
                amount, 
                total_amount, 
                prefix = 'Download info Progress:', 
                suffix = 'Complete: {} out of {}'.format(amount,total_amount), 
                length = 50)

            for __ ,row in df_avail.iterrows():
                # construct file names
                processed_filename = '{}.tif'.format(row['raster_id'])
                download_filename = '{}.tif'.format(row['raster_id'])
                preprocessed_filename = '{}_temp.tif'.format(row['raster_id'])

                if self.wapor_level == 3:
                    country_code_in_name = '_{}'.format(self.country_code)
                    processed_filename = processed_filename.replace(country_code_in_name,'')
                    preprocessed_filename = preprocessed_filename.replace(country_code_in_name,'')

                period_start_str, period_end_str = row['time_code'].split(',')
                for i in ['[',']','(',')','-']:
                    period_start_str = period_start_str.replace(i,'')
                    period_end_str = period_end_str.replace(i,'')
                
                # construct  wapor download dict
                wapor_dict = {}
                wapor_dict['component'] = regionless_component
                wapor_dict['cube_code'] = cube_code
                wapor_dict['period_str'] = period_start_str + '_' + period_end_str
                wapor_dict['period_start'] = datetime.strptime(period_start_str,'%Y%m%d')
                wapor_dict['period_end'] = datetime.strptime(period_end_str,'%Y%m%d')
                wapor_dict['return_period'] = return_period
                wapor_dict['raster_id'] = row['raster_id']
                wapor_dict['multiplier'] = multiplier
                wapor_dict['download_file'] = os.path.join(download_component_dir,download_filename) 
                wapor_dict['download'] = True 
                wapor_dict['preprocessed_file'] = os.path.join(processed_component_dir,preprocessed_filename)
                wapor_dict['preprocess'] = True      
                wapor_dict['processed_file'] = os.path.join(processed_component_dir,processed_filename)   
                wapor_dict['process'] = True          
                wapor_dict['url'] = None    

                if not os.path.exists(wapor_dict['processed_file']):
                    if not os.path.exists(wapor_dict['preprocessed_file']):
                        if not os.path.exists(wapor_dict['download_file']):
                            ### get download url
                            attempt = 1
                            connection_attempt = 1
                            while wapor_dict['url'] is None:
                                try:
                                    wapor_dict['url'] =self.getCropRasterURL(
                                        self.bbox,
                                        cube_code,
                                        row['time_code'],
                                        row['raster_id'],
                                        self.api_token,
                                        print_job=False) 
                                    download_success = True   
                                except requests.exceptions.RequestException:
                                    time.sleep(10)  # wait 10 sec before retry
                                    connection_attempt +=1
                                    if connection_attempt > 300:
                                        raise ConnectionError('retrieving data available statistics from WAPOR \
                                        attempted 300 times every 10 sec and failed adjust the attempts or sleep time to try for longer')
                                else:
                                    time.sleep(5)
                                    attempt +=1
                                    if attempt > 30: 
                                        raise AttributeError('not a request error: no url found, tried every 5 sec 30 times check where your shapefile lies and if the region is available at this time')

                        else:
                            wapor_dict['url'] = None
                            wapor_dict['download'] = False 

                    else:
                        wapor_dict['url'] = None
                        wapor_dict['download'] = False 
                        wapor_dict['preprocess'] = False

                else:
                    wapor_dict['url'] = None
                    wapor_dict['download'] = False 
                    wapor_dict['preprocess'] = False
                    wapor_dict['process'] = False


                wapor_list.append(wapor_dict)

                amount += 1
                printWaitBar(amount, total_amount, 
                                            prefix = 'Download Info Progress:', 
                                            suffix = 'Complete: {} out of {}'.format(amount,total_amount), 
                                            length = 50)
    
        return wapor_list

                #print('attempting to retrieve for dates: {}'.format(dates))  
    
    #################################
    def retrieve_wapor_rasters(
        self, 
        wapor_list: list, 
        create_vrt: bool=True,
        template_raster_path: str = None,
        mask_to_template: bool = False,
        output_nodata: float=-9999) -> dict:
        """
        Description:
            retrieves data from the WAPOR API according to the dictionary inputs retrieved using the 
            class function retrieve_wapor_download_info

            NOTE: works in combination with retrieve_wapor_download_info

        Args:
            self: (see class for details)
            wapor_list: list of dicts containing the download info to use when retrieving the rasters
            create_vrt: if True (auto set to true) creates a vrt of the downloaded files after downloading 
            otherwise this step is skipped
            template_raster_path: if provided uses the template as the source for the metadata and matches rasters too 
            it and masks them too match it too
            mask_to_template: if True also masks all rasters to the template
            output_nodata: nodata value to use for the retrieved data

        Return:
            dict: dictionary of dictionaries ordered by datacomponent each containing a list of rasters retrieved and the path to the compiled vrt        
        """
        assert isinstance(wapor_list,list), 'please provide a list constructed using retrieve_wapor_download_info'
        assert isinstance(wapor_list[0],dict), 'please provide a list constructed using retrieve_wapor_download_info'

        # start retrieving data using the wapor dicts
        print('attempting to retrieve {} rasters from wapor'.format(len(wapor_list)))  
        # set up waitbar
        total_amount = len(wapor_list)
        amount = 0
        printWaitBar(
            amount, 
            total_amount, 
            prefix = 'Download Raster Progress:', 
            suffix = 'Complete: {} out of {}'.format(amount,total_amount), 
            length = 50)

        # retrieve data per wapor download dict
        for wapor_dict in wapor_list:
            if wapor_dict['preprocess']:
                if wapor_dict['download']:
                    if wapor_dict['url']:
                        download_success=False
                        attempt=1
                        while not download_success:
                            try:
                                resp=requests.get(wapor_dict['url'])
                                download_success = True   
                            except requests.exceptions.RequestException:
                                time.sleep(5)  # wait 5 sec before retry
                                attempt +=1
                                if attempt > 600:
                                    raise ConnectionError('retrieval from WAPOR using given download url \
                                    attempted 600 times every 5 sec and failed adjust the attempts or sleep time to try longer')

                        open(wapor_dict['download_file'],'wb').write(resp.content) 
                    else:
                        raise AttributeError('wapor_dict url missing which should not be possible at this stage check out retrieve_wapor_download_info')

                # preprocess the download raster and output if the preprocessed raster does not exist yet
                if wapor_dict['return_period'] == 'dekadal':
                    ### number of days
                    startdate=wapor_dict['period_start']
                    enddate=wapor_dict['period_end']
                    ndays=(enddate.timestamp()-startdate.timestamp())/86400

                else:
                    ndays = 1
                        
                # correct raster with multiplier and number of days in dekad if applicable
                array = raster.raster_to_array(wapor_dict['download_file'])
                array=np.where(array < 0 ,0 , array) # mask out flagged value -9998
                corrected_array=array*wapor_dict['multiplier']*ndays

                raster.array_to_raster(
                    output_raster_path=wapor_dict['preprocessed_file'],
                    metadata=wapor_dict['download_file'],
                    input_array=corrected_array,
                    output_nodata=output_nodata)
                    
                raster.check_gdal_open(wapor_dict['preprocessed_file'])
                os.remove(wapor_dict['download_file'])

            else:
                print('\n preprocessed file already exists skipping: {}'.format(wapor_dict['preprocessed_file']))
        
            amount += 1
            printWaitBar(amount, total_amount, 
                                        prefix = 'Download Raster Progress:', 
                                        suffix = 'Complete: {} out of {}'.format(amount,total_amount), 
                                        length = 50)

        # refresh waitbar again
        amount = 0
        printWaitBar(amount, total_amount, 
                    prefix = 'Processing\Warping Progress:', 
                    suffix = 'Complete: {} out of {}'.format(amount,total_amount), 
                    length = 50)  

        # provide template raster for final processing so as to match all rasters
        if not template_raster_path:
            template = wapor_list[0]['preprocessed_file']
        else:
            template = template_raster_path

        # match all preprocessed rasters in the wapor dicts and output to processed raster locations using gdal warp and the template raster
        for wapor_dict in wapor_list:
            if wapor_dict['process']:
                raster.match_raster(
                    template_raster_path=template,
                    input_raster_path=wapor_dict['preprocessed_file'],
                    output_raster_path=wapor_dict['processed_file'],
                    output_crs= self.output_crs,
                    mask_to_template=mask_to_template,
                    output_nodata=output_nodata)

            amount += 1

            printWaitBar(amount, total_amount, 
                        prefix = 'Processing\Warping Progress:', 
                        suffix = 'Complete: {} out of {}'.format(amount,total_amount), 
                        length = 50) 

        for wapor_dict in wapor_list:  
            if wapor_dict['process']:  
                os.remove(wapor_dict['preprocessed_file'])
        
        # per datacomponent create a vrt
        datacomponent_retrieval_dict = {}
        datacomponent_list =  list(set([d['component'] for d in wapor_list]))
        
        for comp in datacomponent_list:
            raster_list = [d['processed_file'] for d in wapor_list if d['component'] == comp]
            
            if create_vrt:
                period_start = sorted([d['period_start'] for d in wapor_list if d['component'] == comp])[0].strftime("%Y%m%d")
                period_end = sorted([d['period_end'] for d in wapor_list if d['component'] == comp])[-1].strftime("%Y%m%d")

                vrt_filename = '{}_{}_{}_{}.vrt'.format(
                    self.wapor_level, comp, period_start, period_end)
                vrt_path = os.path.join(os.path.dirname(raster_list[0]), vrt_filename)

                if not os.path.exists(vrt_path):
                    # combine outputted rasters into a temporal vrt
                    raster.build_vrt(
                        raster_list=raster_list, 
                        output_vrt_path=vrt_path,
                        action='time') 
                
                out_files = {'raster_list': raster_list, 'vrt_path': vrt_path}

            else:
                out_files = {'raster_list': raster_list, 'vrt_path': None}

            datacomponent_retrieval_dict[comp] = out_files

        return datacomponent_retrieval_dict

if __name__ == "__main__":
    print('main')

