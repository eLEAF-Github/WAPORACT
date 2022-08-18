
"""
waporact package

retrieval class (stand alone/support class and functions)

script for the retrieval of WAPOR data utilising the class WAPORAPI from the package WAPOROCW made by ITC DELFT
"""
##########################
# import packages
import os
import sys
import shutil
import datetime
from datetime import datetime, timedelta
from timeit import default_timer
import time


import numpy as np
import pandas as pd
from ast import literal_eval
from shapely.geometry import shape, mapping, Polygon
import fiona
from fiona.crs import from_epsg
import rtree
import requests

from waporact.scripts.retrieval.wapor_api import WaporAPI
from waporact.scripts.retrieval import wapor_land_cover_classification_codes as lcc
from waporact.scripts.structure.wapor_structure import WaporStructure
from waporact.scripts.tools import raster, vector, statistics

#################################
# stand alone functions
#################################
def printWaitBar(
    i, 
    total, 
    prefix = '',
    suffix = '',
    decimals = 1,
    length = 100,
    fill = '█'):
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

    sys.stdout.write('\r%s |%s| %s%% %s ' %(prefix, bar, percent, suffix))
    sys.stdout.flush()

    if i == total:
        print()

#################################
# retrieval class
#################################
class WaporRetrieval(WaporStructure):
    """
    Description:
        Retrieves rasters from the Wapor database given the appropriate inputs

        inherits/built on the WaporStructure class which is needed for setting
        class/ self parameters and the automated folder structure

    Args:
        wapor_directory: directory to output downloaded and processed data too
        shapefile_path: path to the shapefile to clip downloaded data too if given
        wapor_level: wapor wapor_level integer to download data for either 1,2, or 3
        api_token: api token to use when downloading data
        project_name: name of the location to store the retrieved data
        period_start: datetime object specifying the start of the period
        period_end: datetime object specifying the end of the period
        return_period: return period code of the component to be downloaded (D (Dekadal) etc.)
        silent: boolean, if True the more general messages in the class are not printed
        (autoset to False)
        )

    return:
        WAPOR rasters matching the given information are retrieved and stored in the
        specified project
    """
    def __init__(
        self,
        waporact_directory: str,
        shapefile_path: str,
        wapor_level: int,
        period_start: datetime,
        period_end: datetime,
        api_token: str,
        return_period: str = 'D',
        project_name: int = 'test',
        wapor_version: int = 2,
        silent: bool=False,

    ):
        # set verbosity (feedback) parameter
        self.silent = silent

        # set waporapi parameters
        self.api_token = api_token
        self.wapor_version = wapor_version

        # attach and setup the waporAPI class
        self.wapor_api = WaporAPI(
            version=self.wapor_version
        )

        # retrieve and set the catalogue for the wapor level being analysed
        self.catalogue = (wapor_level, waporact_directory)
        
        self.components = list(set(list(self.catalogue['component_code'])))
        self.cube_codes = list(set(list(self.catalogue['code'])))
        self.period_codes = list(set(list(self.catalogue['period_code'])))
        self.country_codes = list(set([(country_code, country_desc) for  country_code, country_desc in zip(self.catalogue['country_code'],self.catalogue['country_desc'])]))

        # inherit and initialise the WaporStructure class
        super().__init__(
            waporact_directory=waporact_directory,
            project_name=project_name,
            wapor_level=wapor_level,
            period_end=period_end,
            period_start=period_start,
            return_period = return_period,
        )

        # set and generate shapefile parameters
        self.shapefile_path = shapefile_path


        # run basic data availability checks
        if self.return_period not in self.period_codes:
            raise AttributeError('given return period {} not found amongst available options at wapor level 3: {}'.format(self.period_codes))

        if self.wapor_level == 3:
            # check if the bbox falls within an available level 3 area:
            self.country_code = self.check_bbox_overlaps_l3_location()
        else:
            self.country_code = 'notlevel3notused'

        print('WaporRetrieval class initiated and ready for WaPOR data retrieval')

    #################################
    # properties
    #################################
    @property
    def catalogue(self):
        return self._catalogue

    @catalogue.setter
    def catalogue(self, value: tuple):
        """
        Description:
            retrieves the catalogue for the given
            wapor level and sets it to self, if available retrieves it from
            file otherwise from the wapor api

            NOTE: only meant to be set once per class intialisation

        Args: 
            value: wapor level as integer and the catalogue directory as a tuple

        Return:
            dataframe: wapor level catalogue dataframe
        """
        if value[0] not in [1,2,3]:
            raise AttributeError("wapor_level (int) given as value needs to be either 1, 2 or 3")

        if not isinstance(value[1], str):
            raise AttributeError('please provide a directory path for the waporact directory')
        
        # check waporact metadata directory for existing catalogue (mirrors waporstructure)
        catalogue_folder = os.path.join(value[1], 'metadata')
        if not os.path.exists(catalogue_folder):
            print('waporact metadata directory provided does not exist attempting to make it now')
            try:
                os.makedirs(catalogue_folder)
            except Exception as e:
                print('failed to make the waporact metadata directory: {}'.format(catalogue_folder))
                raise e

        print('retrieving the wapor catalogue for wapor level: {}'.format(value[0]))
        catalogue = self.retrieve_and_store_catalogue(
            catalogue_output_folder=catalogue_folder,
            wapor_level=value[0],
            cubeInfo=True)

        self._catalogue = catalogue

   #################################
    @property
    def shapefile_path(self):
        return self._shapefile_path

    @shapefile_path.setter
    def shapefile_path(self, value: str):
        """ 
        Description:
            checks it the shapefile path provided exists
            and if yes sets it to shapefile path
            and uses it to create a bbox and
            bbox shapefile_path attached to self

        Args:
            value: path to a shapefile
        """
        if hasattr(self, 'shapefile_path'):
            raise AssertionError('shapefile_path already found in class instance, should only be generated once on class activation, code logic error')

        else:
            if not isinstance(value, str):
                raise AttributeError('please provide the path to a shapefile as string')
            if not os.path.exists(value):
                raise FileNotFoundError('shapefile not found at given location')
            
            bbox_shapefile_name = os.path.splitext(os.path.basename(value))[0] + '_bbox.shp'
            bbox_shapefile_path = os.path.join(self.project['reference'],bbox_shapefile_name)

            try:
                print('retrieving the input shapefile crs and setting it to the class instance: output_crs')
                output_crs = vector.retrieve_shapefile_crs(value)
            except Exception as e:
                print('failed to retrieve crs from the input shapefile, please check the error message and or your input shapefile and try again')
                raise e

            try:
                print('retrieving the  input shapefile bbox in latlon and setting it to the class instance: bbox')
                df = vector.file_to_records(value, output_crs=4326)
                bbox = vector.retrieve_geodataframe_bbox(df)
            except Exception as e:
                print('failed to retrieve bbox in latlon from the input shapefile, please check the error message and or your input shapefile and try again')
                raise e

            try:
                print('writing the latlon bbox to shapefile and setting it to the class instance: bbox_shapefile_path')
                vector.create_bbox_shapefile(
                    output_shape_path=bbox_shapefile_path,
                    bbox=bbox)
            except Exception as e:
                print('failed to write latlon bbox to shapefile, please check the error message and or your input shapefile and try again')
                raise e

            self._shapefile_path = value
            self.output_crs = output_crs
            self.bbox_shapefile_path = bbox_shapefile_path
            self.bbox = bbox

            print('shapefile, output_crs, bbox and bbox_shapefile set to class instance')

    #################################
    # class functions
    #################################
    @classmethod
    def wapor_connection_attempts_dict(cls):
        """
        Description:
            creates a dictionary for setting the limit and keeping count of the
            connection attempts made to the wapor site

        Args:
            0

        Return:
            dict: dictionary with settings to attempt data retrieval from wapor
        """
        return {
            'connection_attempts':0,
            'connection_sleep': 3,
            'connection_attempts_limit': 20
        }

    #################################
    @classmethod
    def deconstruct_wapor_time_code(
        cls,
        time_code: str):
        """
        Description:
            deconstruct the wapor time code into useful components and store in a dict

        Args:
            time_code: wapor generated time code

        Return:
            dict: deconstructed time code
        """
        wapor_time_dict = {}
        period_start_str, period_end_str = time_code.split(',')
        for i in ['[',']','(',')','-']:
            period_start_str = period_start_str.replace(i,'')
            period_end_str = period_end_str.replace(i,'')

        wapor_time_dict['period_string'] = period_start_str + '_' + period_end_str
        wapor_time_dict['period_start'] = datetime.strptime(period_start_str,'%Y%m%d')
        wapor_time_dict['period_end'] = datetime.strptime(period_end_str,'%Y%m%d')

        return wapor_time_dict

    #################################
    @classmethod
    def wapor_raster_processing(
        cls,
        input_raster_path: str,
        output_raster_path: str,
        wapor_multiplier: float,
        return_period: str,
        period_start: datetime,
        period_end: datetime,
        delete_input: bool=False,
        output_nodata: float=-9999
        ):
        """
        Description:
            process a downloaded wapor raster according to a standard process

        Args:
            input_raster_path: url used to retrieve the wapor raster
            output_raster_path: path to output the retrieved raster too
            wapor_multiplier: multiplier used opn wapor rasters
            output_nodata: output_nodata value
            return_period: return period between rasters
            period_start: start of the period the raster covers
            period_end: end of the period the raster covers
            delete_input: if true deletes the input file on completion

        Return:
            0 : processed raster is written to the given location
        """
        # process the downloaded raster and write to the processed folder
        if return_period == 'dekadal':
            ### number of days
            ndays=(period_end.timestamp()-period_start.timestamp())/86400
        else:
            ndays = 1
                        
        # correct raster with multiplier and number of days in dekad if applicable
        array = raster.raster_to_array(input_raster_path)
        array=np.where(array < 0 ,0 , array) # mask out flagged value -9998
        corrected_array=array*wapor_multiplier*ndays

        raster.array_to_raster(
            output_raster_path=output_raster_path,
            metadata=input_raster_path,
            input_array=corrected_array,
            output_nodata=output_nodata)
            
        raster.check_gdal_open(output_raster_path)
        if delete_input:
            os.remove(input_raster_path)

        return 0

    #################################
    # check functions
    #################################
    def check_datacomponent_availability(
        self,
        datacomponents: list,
        return_period: str=None):
        """
        Description
            checks if the combination of  wapor level, datacomponents, return period and
            country code (at level 3) that make up a cube code are available per
            datacomponent
            
            NOTE: if All is provided all available datacomponents are returned.

        Args:
            datacomponents: list of datacomponents to check
            return_period: if provided overwrites return period set on class activation

        Return
            list: list of datacomponents, cube_codes that do exist

        Raise:
            raises error if no cube_code exist for any datacomponent specified
        """
        self.return_period = return_period
        
        if datacomponents[0] == 'ALL':
            if self.wapor_level == 3:
                available_datacomponents = [comp for comp in self.components if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,self.return_period) in self.cube_codes] 
            else:
                available_datacomponents = [comp for comp in self.components if 'L{}_{}_{}'.format(self.wapor_level,comp,self.return_period) in self.cube_codes] 

        else:
            if self.wapor_level == 3:
                available_datacomponents = [comp for comp in datacomponents if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,self.return_period) in self.cube_codes]
                missing_datacomponents = [comp for comp in datacomponents if  'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,self.return_period) not in self.cube_codes]
            else:
                available_datacomponents = [comp for comp in datacomponents if 'L{}_{}_{}'.format(self.wapor_level,comp,self.return_period) in self.cube_codes]
                missing_datacomponents = [comp for comp in datacomponents if  'L{}_{}_{}'.format(self.wapor_level,comp,self.return_period) not in self.cube_codes]
               
            if missing_datacomponents:
                print('Available datacomponents of those specified: {}'.format(available_datacomponents))
                raise AttributeError('at wapor level: {} and return period: {} , the following datacomponents could not be \
                found and may not exist (at level 3 region may also affect availibility): {}'.format(self.wapor_level, return_period, missing_datacomponents))
        
        if not available_datacomponents:
            raise AttributeError('at wapor level: {} and return period: {} , no datacomponents could be \
            found, this is unlikely'.format(self.wapor_level, return_period))

        return available_datacomponents

    #################################
    def check_bbox_overlaps_l3_location(self):
        """
        Description:
            takes the given shapefile/boundingbox and checks if it
            falls within the boundaries of a wapor_level 3 area by comparing with
            the level 3 availability shapefile generated/retrieved

        Return:
            str: area code of the area the shapefile falls within

        Raises:
            FileNotFoundError: if the shapefile/bbox does not fall within an available area
        """
        l3_locations_shapefile_path = self.retrieve_level_3_availability_shapefile()
        code = None

        with fiona.open(l3_locations_shapefile_path, 'r') as layer1:
            with fiona.open(self.bbox_shapefile_path, 'r') as layer2:
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
            raise AttributeError('no overlap found between wapor_level 3 locations shapefile: {} and the generated bbox shapefile: {}\
            it is likely that there is no data available at level 3 for this area, check in qgis or similar to confirm'.format(
                l3_locations_shapefile_path, self.bbox_shapefile_path))
        
        return code
    
    #################################
    # retrieval functions
    #################################
    def generate_wapor_cube_code(
        self,
        component: str,
        return_period:str=None,
        ):
        """
        Description:
            format and return the cube code for querying for wapor data
            adding the region code for l3 if needed

        Args:
            component: wapor component code to query for a wapor cube code with
            return_period: return period code to for a wapor cube code with
        """
        if self.wapor_level == 3:
            component = '{}_{}'.format(self.country_code, component)
        cube_code=f"L{self.wapor_level}_{component}_{return_period}"

        if cube_code not in self.cube_codes:
            raise AttributeError('cube code generated: {}, not found in available list: {}'.format(cube_code, self.cube_codes))

        return cube_code

    #################################
    def retrieve_and_store_catalogue(
        self,
        catalogue_output_folder: str,
        wapor_level: int,
        cubeInfo=True):
        '''
        Description:
            retrieves the wapor catalogue fromWaPOR and formats it as a dataframe for use
            directly and stores a copy in the standard storage location a sub directory of
            the waporact directory you set on class activation. On retrieval creates a
            dataframe and sets it to self.catalogue for use.

            if the catalogue was not found or the version found was too old attempts to retrieve
            it from the WAPOR database.

            NOTE: based on the WaporAPI class function getCatalog

        Args:
            wapor_level: level to retrieve the catalogue for
            cube_info: if true also retrieves and formats the cube info from the catalogue into
            the output dataframe

        Return:
            dataframe: dataframe of the retrieved catalogue
        '''
        if wapor_level not in [1,2,3]:
            raise AttributeError("wapor_level (int) needs to be either 1, 2 or 3")
        retrieve=False
        catalogue_csv = os.path.join(catalogue_output_folder, 'wapor_catalogue_L{}.csv'.format(wapor_level))
        if not os.path.exists(catalogue_csv):
            retrieve = True
        else:
            st=os.stat(catalogue_csv)
            if (time.time() - st.st_mtime) >= 5184000: # 60 days
                retrieve = True
        
        if retrieve:
            print('No or Outdated WaPOR catalogue found for wapor_level: {}, retrieving now this may take a min...'.format(wapor_level))

            catlogue_df = self.wapor_api.retrieve_catalogue_as_dataframe(
                wapor_level=wapor_level,
                cubeInfo=cubeInfo
            )
            
            statistics.output_table(
                table=catlogue_df,
                output_file_path=catalogue_csv,
                csv_seperator=';')
        
            print("outputted table of the WaPOR catalogue for wapor_level: {}".format(wapor_level))
            print(catalogue_csv)
            print('on running the retrieval class again the catalogue will be reused if found or auto replaced '
                'if the catalogue becomes outdated after a 60 day period')
        else:
            catlogue_df = pd.read_csv(catalogue_csv, sep=';')
            catlogue_df['measure'] = catlogue_df['measure'].apply(lambda x: literal_eval(x))
            catlogue_df['dimension'] = catlogue_df['dimension'].apply(lambda x: literal_eval(x))
        
        if not self.silent:
            print('Loading WaPOR catalogue for wapor_level: {}'.format(wapor_level))
            print('catalogue location: {}'.format(catalogue_csv))

        return catlogue_df

    #################################
    def retrieve_level_3_availability_shapefile(
        self):
        """
        Description:
            retrieves and sets the path to the wapor level 3 locations shapefile
            and if it does not exist or is outdated creates it

            filepath generated is set to self.l3_locations_shapefile_path
            in the standard sub folder of the waporact directory set by the
            user on class activation

        Return:
            str: path to the generated L3 location shapefile
        """
        retrieve = False
        l3_locations_shapefile_path = os.path.join(self.project['meta'], 'wapor_L3_locations.shp')

        # check if the file already exists
        if not os.path.exists(l3_locations_shapefile_path):
            retrieve = True
        else:
            # check how old the file is
            st=os.stat(l3_locations_shapefile_path)
            if (time.time() - st.st_mtime) >= 10368000: # 120 days
                retrieve = True

        if retrieve:
            print('creating wapor_level 3 locations shapefile, this may take a min ...')
            # set temporary date variables
            api_per_start = datetime(2010,1,1).strftime("%Y-%m-%d")
            api_per_end = datetime.now().strftime("%Y-%m-%d")
            api_period = '{},{}'.format(api_per_start, api_per_end)

            # retrieve country codes
            _data = []
            # loop through countries check data availability and retrieve the bbox
            for code in self.country_codes:
                cube_code= f"L3_{code[0]}_T_D"

                df_avail = self.retrieve_wapor_data_availability(
                    cube_code=cube_code,
                    time_range=api_period)
                
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

        return l3_locations_shapefile_path

    #################################
    def create_raster_mask_from_shapefile(
        self,
        mask_name: str,
        input_shapefile_path: str = None,
        template_raster_path: str = None):
        """
        Description:
            creates a crop mask for further anaylsis
            from the shapefile provided using either the whole shape
            or specific geometries defined in a shapefile column
        
        Args:
            input_shapefile_path: shapefile to create the crop mask with, if not provided
            uses the class shapefile (this is recommended)
            mask_name: aoi or mask name of the mask raster created from the shapefile. If not provided
            auto set to input_shapefile_name plus mask (mask is the area of interest being investigated)
            template_raster_path: raster providing the metadata for the output raster
            if not provided retrieves a raster from WAPOR to use as the template

        Return:
            tuple: path to the mask raster created, path to the mask shapefile created
        """
        # store time parameters as temp variables used below
        save_return_period = self.return_period
        save_period_start = self.period_start
        save_period_end = self.period_end

        if not input_shapefile_path:
            input_shapefile_path = self.shapefile_path

        if not mask_name:
            base_name = os.path.splitext(os.path.basename(input_shapefile_path))[0]
            mask_name = base_name

        mask_raster_path = self.generate_output_file_path(
            description=mask_name,
            output_folder='reference',
            aoi_name=mask_name,
            ext='.tif')

        mask_shape_path = self.generate_output_file_path(
            description=mask_name,
            output_folder='reference',
            aoi_name=mask_name,
            ext='.shp')

        # retrieval tests (to make sure we get something)
        dekadal = ('D', datetime(2020,1,1), datetime(2020,1,2))
        monthly = ('M', datetime(2020,1,1), datetime(2020,2,1))
        annual = ('A', datetime(2020,1,1), datetime(2021,1,1))
        
        if not os.path.exists(mask_raster_path) or not os.path.exists(mask_shape_path):
            if not template_raster_path:
                for rp in [dekadal, monthly, annual]:
                    # get a raster as template
                    datacomponents = self.check_datacomponent_availability(datacomponents=['ALL'], return_period=rp[0])
                    for datacomp in datacomponents:
                        wapor_download_list = self.retrieve_wapor_download_info(
                            datacomponents=[datacomp],
                            return_period=rp[0],
                            period_start=rp[1],
                            period_end=rp[2],
                            )

                        if len(wapor_download_list) > 0:
                            break
                    if len(wapor_download_list) > 0:
                        break

                wapor_download_list = [wapor_download_list[0]]

                wapor_rasters = self.retrieve_actual_wapor_rasters(
                    wapor_download_list=wapor_download_list,
                )
                template_raster_path = wapor_rasters[datacomp]['raster_list'][0]
            
            if not os.path.exists(mask_raster_path):
                raster.rasterize_shape(
                    template_raster_path=template_raster_path,
                    shapefile_path=input_shapefile_path,
                    output_raster_path=mask_raster_path,
                    )

                print("mask raster made: {}".format(mask_raster_path))

            else:
                if not self.silent:
                    print("preexisting mask raster found skipping step")
                raster.check_gdal_open(mask_raster_path)

            if not os.path.exists(mask_shape_path):
                # create a shapefile of the raw crop_mask
                vector.raster_to_polygon(
                    input_raster_path=mask_raster_path,
                    output_shapefile_path=mask_shape_path,
                    mask_raster_path=mask_raster_path)

                vector.check_add_wpid_to_shapefile(input_shapefile_path=mask_shape_path)

                print("mask shapefile made and wpid id column added: {}".format(mask_shape_path))

            else:
                if not self.silent:
                    print("preexisting raster mask and shape mask found skipping step")

        else:
            if not self.silent:
                print("preexisting raster mask and shape mask found skipping step")
            raster.check_gdal_open(mask_raster_path)

        # restore time parameters 
        self.return_period = save_return_period
        self.period_start =save_period_start
        self.period_end = save_period_end 

        return mask_raster_path, mask_shape_path

    #################################
    def create_raster_mask_from_wapor_lcc(
        self,
        mask_name: str,
        lcc_categories: list = [],
        period_start: datetime=None,
        period_end: datetime=None,
        area_threshold_multiplier: int = 1,
        output_nodata: float = -9999):
        """
        Description:
            creates a mask raster and shapefile for further anaylsis
            using the bbox as defined by the shapefile and lcc categories initially
            provided and the land cover classification rasters
            that can be found on the WAPOR database
            if the period defined covers more than one raster it
            combines them all into one for the entire period.
            keeping the classification most common across the entire period

            the mask retrieved from WAPOR is considered the raw one as is the
            shapefile based on it the crop mask is further clipped to an edited version
            of the initial mask the user can pick which one they use
        
        Args:
            mask_name: aoi or mask name of the mask for the output file and aoi (mask) sub folder
            lcc_categories: crops/land classification categories to mask too
            has to match the names used in the wapor database classification codes
            period_start: period for which to retrieve the land cover raster
            period_end: period for which to retrieve the land cover raster,
            area_threshold_multiplier: area threshold with which to filter out too small polygons
            (single cell area * area_threshold_multiplier sets the threshold)
            output_nodata: nodata value for the output rasters that arte not 0,1 masks

            uses the value defined during class intialisation if period_start,
            period_end or return_period or input_shapefile_path is not provided

        Return:
            tuple: path to the mask raster created, path to the mask shape created
        """
        self.period_start = period_start
        self.period_end = period_end

        # store return period 
        save_return_period = self.return_period

        lcc_dict = lcc.wapor_lcc(wapor_level=self.wapor_level)

        # check that the lcc category provided exists as an option
        lcc.check_categories_exist_in_categories_dict(
            categories_list=lcc_categories,
            categories_dict=lcc_dict
            )
        
        # disaggregate aggregate codes
        lcc_categories = lcc.dissagregate_categories(
            categories_list=lcc_categories,
            categories_dict=lcc_dict
            )

        # check that the lcc category provided exists as an option
        lcc.check_categories_exist_in_categories_dict(
            categories_list=lcc_categories,
            categories_dict=lcc_dict
            )

        # reverse lcc_codes and categories 
        lcc_dict_reversed = {lcc_dict[key]: key for key in lcc_dict.keys() if not isinstance(lcc_dict[key], list)} 

        # create the file paths (as the LLC retrieved can differ depending on date the mask is period specific)
        mask_raster_path = self.generate_output_file_path(
                description=mask_name,
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.tif')

        mask_values_raster_path = self.generate_output_file_path(
                description='{}-values'.format(mask_name),
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.tif')

        mask_shape_path = self.generate_output_file_path(
                description=mask_name,
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.shp')

        raw_mask_raster_path = self.generate_output_file_path(
                description='{}-raw'.format(mask_name),
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.tif')

        raw_mask_values_raster_path = self.generate_output_file_path(
                description='{}-raw-values'.format(mask_name),
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.tif')

        raw_mask_shape_path = self.generate_output_file_path(
                description='{}-raw'.format(mask_name),
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.shp')

        lcc_count_csv_path = self.generate_output_file_path(
                description='{}-lcc-count'.format(mask_name),
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.csv')

        masked_lcc_count_csv_path = self.generate_output_file_path(
                description='{}-lcc-count-masked'.format(mask_name),
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.csv')

        most_common_lcc_raster_path = self.generate_output_file_path(
                description='{}-lcc-median'.format(mask_name),
                output_folder='reference',
                period_start=period_start,
                period_end=period_end,
                aoi_name=mask_name,
                ext='.tif')       

        # check for and produce if needed the raw mask rasters
        if not os.path.exists(raw_mask_raster_path) or not os.path.exists(raw_mask_values_raster_path):
            # check for and produce if needed the most common class raster
            # retrieve the lcc rasters and find the most common class per cell across the given period
            if not os.path.exists(most_common_lcc_raster_path):
                if self.wapor_level == 3:
                    rp = 'D'
                else:
                    rp ='A'
                    if period_end - period_start < 365: 
                        # adjust period start to make sure it retieves something
                        period_start = period_start - timedelta(days=365)

                # create the base mask based on the input shapefile first
                base_mask_raster_path, __ = self.create_raster_mask_from_shapefile(
                    mask_name='{}_base'.format(mask_name),
                )

                wapor_rasters = self.download_wapor_rasters(
                    datacomponents=['LCC'],
                    template_raster_path=base_mask_raster_path,
                    return_period=rp,
                    period_start=self.period_start,
                    period_end=self.period_end,
                    output_nodata=output_nodata,
                    aoi_name=mask_name
                    )
               
                if len(wapor_rasters['LCC']['raster_list']) > 1:
                    # if more than one raster exists the median (most common) land cover class across the period is assigned
                    statistics.calc_multiple_array_numpy_statistic(
                        input=wapor_rasters['LCC']['raster_list'],
                        numpy_function=statistics.mostcommonzaxis,
                        output_raster_path=most_common_lcc_raster_path)
                else:
                    shutil.copy2(src= wapor_rasters['LCC']['raster_list'][0], dst=most_common_lcc_raster_path)
            
            # use the most common raster to produce the raw mask rasters
            lcc_count_dict, lcc_count_csv_path = statistics.raster_count_statistics(
                input_raster_path=most_common_lcc_raster_path,
                categories_dict=lcc_dict,
                category_name='landcover',
                output_csv=lcc_count_csv_path,
                out_dict=True)

            lcc.check_categories_exist_in_count_dict(
                categories_list=lcc_categories,
                count_dict=lcc_count_dict
                )
                
            # create the raw crop mask raster
            mask_values = []
            for lcc_category in lcc_categories:
                mask_values.append(lcc_dict[lcc_category])

            raster.create_values_specific_mask(
                mask_values=mask_values,
                input_raster_path= most_common_lcc_raster_path,
                output_values_raster_path=raw_mask_values_raster_path,
                output_mask_raster_path=raw_mask_raster_path,
                output_crs=self.output_crs,
                output_nodata=output_nodata)

            print("raw mask raster made: {}".format(raw_mask_raster_path))
            print("raw mask values raster made: {}".format(raw_mask_values_raster_path))

        raster.check_gdal_open(raw_mask_raster_path)
        raster.check_gdal_open(raw_mask_values_raster_path)

        if not os.path.exists(raw_mask_shape_path):
            # create a shapefile of the raw crop_mask
            vector.raster_to_polygon(
                input_raster_path=raw_mask_values_raster_path,
                output_shapefile_path=raw_mask_shape_path,
                column_name='lcc_val',
                mask_raster_path=raw_mask_raster_path)
            
            # add the lcc categories to the crop mask
            vector.add_matched_values_to_shapefile(
                input_shapefile_path=raw_mask_shape_path,
                new_column_name='lcc_cat',
                union_key='lcc_val',
                value_type='str',
                values_dict=lcc_dict_reversed)

            print("raw lcc mask shape made: {}".format(raw_mask_shape_path))

        if not os.path.exists(mask_shape_path):
            # create the cleaned shapefile
            if self.wapor_level == 3:
                if area_threshold_multiplier <= 1.5:
                    area_threshold_multiplier = 1.5
                    print('for level 3 the area_threshold_multiplier is set to a minimum of 1.5, change value in the code to change this')
            cell_area = raster.gdal_info(raw_mask_raster_path)['cell_area']
            area_threshold = cell_area * area_threshold_multiplier
            print('WARNING: area threshold for polygons found is currently'
                ' set to {} * {} (cell area) = {}'.format(area_threshold_multiplier, cell_area, area_threshold))

            vector.polygonize_cleanup(
                input_shapefile_path=raw_mask_shape_path,
                output_shapefile_path=mask_shape_path,
                area_threshold=area_threshold)

            print("mask shape made: {}".format(mask_shape_path))
    
        if not os.path.exists(mask_raster_path) or not os.path.exists(mask_values_raster_path):
            # create cleaned values raster
            raster.rasterize_shape(
                template_raster_path=raw_mask_values_raster_path,
                shapefile_path=mask_shape_path,
                output_raster_path=mask_values_raster_path,
                column='lcc_val',
                output_gdal_datatype=6,
                output_nodata=output_nodata
                )

            # create cleaned mask raster
            raster.rasterize_shape(
                template_raster_path=raw_mask_raster_path,
                shapefile_path=mask_shape_path,
                output_raster_path=mask_raster_path
                )

        raster.check_gdal_open(mask_raster_path)
        raster.check_gdal_open(mask_values_raster_path)

        lcc_masked_count_dict , masked_lcc_count_csv_path = statistics.raster_count_statistics(
            input_raster_path=mask_values_raster_path,
            categories_dict=lcc_dict,
            category_name='landcover',
            output_csv=masked_lcc_count_csv_path,
            out_dict=True)

        lcc.check_categories_exist_in_count_dict(
            categories_list=lcc_categories,
            count_dict=lcc_masked_count_dict
        )

        # restore return period
        self.return_period = save_return_period

        return mask_raster_path, mask_shape_path

    #################################
    def retrieve_wapor_cube_info(
        self,
        cube_code: str,
        ):
        """
        Description:
            wrapper for WaporAPI getCubeInfo that runs it
            multiple times in an attempt to force a connection
            and retrieve the cube info
            from the WAPOR database in a more robust fashion

        Args:
            cube_code: cube code used to retrieve the raster

        Return:
            dict: cube info

        Raise:
            ConnectionError: if a connection cannot be established after multiple attempts
            TimeoutError: if the error is not a request error (unknown) and it fails for another reason
        """
        # reset connection attempts
        self.wapor_connection_attempts = WaporRetrieval.wapor_connection_attempts_dict()
        cube_info = None
        while cube_info is None:
            # attempt to retrieve cube_info
            try:
                cube_info=self.wapor_api.getCubeInfo(
                    cube_code=cube_code,
                    wapor_level=self.wapor_level,
                    catalogue=self.catalogue)
            except Exception as e:
                self.wapor_retrieval_connection_error(
                    description='retrieving cube info',
                    _exception=e)

        return cube_info

    #################################
    def retrieve_wapor_data_availability(
        self,
        cube_code: str,
        time_range: str
        ):
        """
        Description:
            wrapper for WaporAPI getAvailData that runs it
            multiple times in an attempt to force a connection
            and retrieve the cube info
            from the WAPOR database in a more robust fashion

        Args:
            cube_code: cube code used to retrieve the raster
            time_range: time range for retrieval formatted as required by the API

        Return:
            dataframe: data availability info

        Raise:
            ConnectionError: if a connection cannot be established after multiple attempts
            TimeoutError: if the error is not a request error (unknown) and it fails for another reason
        """
        # reset connection attempts
        self.wapor_connection_attempts = WaporRetrieval.wapor_connection_attempts_dict()
        data_availability = None
        while data_availability is None:
            # attempt to retrieve cube_info
            try:
                data_availability=self.wapor_api.getAvailData(
                    cube_code=cube_code,
                    time_range=time_range,
                    wapor_level=self.wapor_level,
                    catalogue=self.catalogue)
                    
            except Exception as e:
                self.wapor_retrieval_connection_error(
                    description='retrieving data avialbility info',
                    _exception=e)

        return data_availability

    #################################
    def retrieve_crop_raster_url(
        self,
        bbox: tuple,
        cube_code: str,
        time_code: str,
        raster_id: str,
        api_token:str
        ):
        """
        Description:
            wrapper for WaporAPI getCropRasterURL that runs it 
            multiple times in an attempt to force a connection
            and retrieve the raster url from the WAPOR database
            in a more robust fashion

        Args:
            bbox: bbox used to crop to the raster section
            cube_code: cube code used to retrieve the raster
            time_code: wapor data availability time code
            raster_id: wapor data availability raster_id
            api_token: api token used to retrieve the data

        Return:
            str: crop raster url

        Raise:
            ConnectionError: if a connection cannot be established after multiple attempts
            TimeoutError: if the error is not a request error (unknown) and it fails for another reason
        """
        # reset connection attempts
        self.wapor_connection_attempts = WaporRetrieval.wapor_connection_attempts_dict()
        url = None
        while url is None:
            try:
                ### attempt to retrieve the download url
                url = self.wapor_api.getCropRasterURL(
                    bbox=bbox,
                    cube_code=cube_code,
                    time_code=time_code,
                    rasterId=raster_id,
                    APIToken=api_token,
                    print_job=False,
                    wapor_level=self.wapor_level,
                    catalogue=self.catalogue)

            except Exception as e:
                self.wapor_retrieval_connection_error(
                    description='retrieving crop raster url',
                    _exception=e)

        return url

    #################################
    def wapor_raster_request(
        self,
        wapor_url: str,
        output_file_path: str
        ):
        """
        Description:
            wrapper function that attempts to retrieve the raster
            from the wpaor database using a stanrd api request and stores it

        Args:
            wapor_url: url used to retrieve the wapor raster
            output_file_path: path to output the retrieved raster too

        Return:
            0 : raster is written to the given location

        Raise:
            AttributeError: if url is not provided
            ConnectionError: if a connection cannot be established after multiple attempts
            TimeoutError: if the error is not a request error (unknown) and it fails for another reason
        """
        if not wapor_url:
            raise AttributeError('wapor_dict url missing which should not be possible at this stage check out retrieve_wapor_download_info')
        else:
            # reset connection attempts
            self.wapor_connection_attempts = WaporRetrieval.wapor_connection_attempts_dict()
            #initiate retrieval variables
            wapor_raster_result = None
            while wapor_raster_result is None:
                try:
                    ### attempt to retrieve the download url
                    wapor_raster_result = requests.get(wapor_url)

                except Exception as e:
                    self.wapor_retrieval_connection_error(
                        description='retrieving crop raster url',
                        _exception=e)
      
        open(output_file_path,'wb').write(wapor_raster_result.content) 

        return 0

    #################################
    def wapor_retrieval_connection_error(
        self,
        description: str,
        _exception: Exception,
        ):
        """
        Description:
            used if a connection error occurs:

            keeps track of the amount of connection/request attempts
            and increases the count on running the function.
            and if the limit of attempts is reached while running
            raises an error

            limit is set in wapor retrieval class
            self.wapor_connection_attempts

        Args:
            description: description of the 
            activity being undertaken
            Exception: eexception that was captured

        Return:
            int: 0

        Raise:
            ConnectionError: if the limit of attempts is reached and still no connection
            was established due to a connection error

            TimeoutError: if the limit of attempts is reached and still no connection
            was established due to an unknown error
        """
        time.sleep(self.wapor_connection_attempts['connection_sleep']) 
        self.wapor_connection_attempts['connection_attempts'] += 1
        if self.wapor_connection_attempts['connection_attempts'] == abs(self.wapor_connection_attempts['connection_attempts_limit']/2):
            print('{} failed,  {} connection errors noted, will continue connection attempts'.format(description, self.wapor_connection_attempts['connection_attempts'])) 
        if self.wapor_connection_attempts['connection_attempts'] >= self.wapor_connection_attempts['connection_attempts_limit']:
            if isinstance(_exception,requests.exceptions.RequestException):
                error_statement = ('{} from WAPOR '
                    'attempted to request data {} times every {} sec and failed due to request/connection error, adjust the self.wapor_connection_attempts or sleep'
                    'time to try for longer, there may also be no data available for your combination of return period, period_start, period_end,'
                    'and datacomponent'.format(description, self.wapor_connection_attempts['connection_attempts'], self.wapor_connection_attempts['connection_sleep']))
                raise ConnectionError(error_statement)

            else:
                error_statement = ('{} from WAPOR '
                    'attempted to request data {} times every {} sec and failed due to unknown error, adjust the self.wapor_connection_attempts or sleep'
                    ' time to try for longer, there may also be no data available for your combination of return period, period_start, period_end,'
                    'and datacomponent'.format(description, self.wapor_connection_attempts['connection_attempts'], self.wapor_connection_attempts['connection_sleep']))
                raise TimeoutError(error_statement)

        return 0

   #################################
    def retrieve_wapor_download_info(
        self, 
        datacomponents: list,
        period_start: datetime, 
        period_end: datetime,
        return_period: str,
        aoi_name: str = 'nomask'):
        """
        Description:
            WAPOR download works in two phases. the retrieval and setup of the download info and the 
            actual download of that info.
            
            this subfunction carries out the setup of the download info. It retrieves data from the WAPOR API 
            according to the class inputs provided and generates a download dict for retrieving and 
            manipulating the WAPOR rasters

            NOTE: This is actually the longest part of the download process as generating the download url
            is a slow process

            NOTE: works in combination with retrieve_actual_wapor_rasters

            NOTE: aoi_name (mask name) if supplied should match that in retrieve_actual_wapor_rasters

            NOTE: does not use clas sinstance period_start, period_end or return period as it assumes 
            these inputs are provided by download_wapor_rasters

        Args:
            datcomponents: list of datacomponents to retrieve,
            period_start: start of period to retrieve data for 
            period_end: end of period to retrieve data for
            return_period: return period/interval to retrieve data for
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided

        Return:
            list: list of dicts containing variables that can be used to retrieve and store a specific wapor raster
        """
        self.return_period = return_period
        
        assert isinstance(period_start, datetime), 'period_start must be a datetime object'
        assert isinstance(period_end, datetime), 'period_end must be a datetime object'

        if not isinstance(datacomponents, list):
            raise TypeError('datacomponents provided should be formatted as a list of strings')

        # generate dates for filenames
        dates_dict = WaporStructure.generate_dates_dict(
            period_start=period_start,
            period_end=period_end,
            return_period=self.return_period)
        
        # setup output list
        wapor_download_list = []

        #check if the datacomponents are available
        datacomponents = self.check_datacomponent_availability(datacomponents, self.return_period)

        # retrieve download info per available datacomponent
        for component in datacomponents:
            if self.wapor_level==3:
                print('retrieving download info for wapor_level 3 region: {}'.format(self.country_code))
            print('retrieving download info for component: {}'.format(component))
        
            wapor_dict = {}

            # construct the wapor_cube_code
            cube_code = self.generate_wapor_cube_code(
                component=component,
                return_period=self.return_period,
                )
            
            # attempt to retrieve cube code info
            cube_info = self.retrieve_wapor_cube_info(
                cube_code=cube_code
            )
            multiplier=cube_info['measure']['multiplier']
            
            # attempt to download data availability from WAPOR
            df_avail = self.retrieve_wapor_data_availability(
                    cube_code=cube_code,
                    time_range=dates_dict['api_period']
                    )
           
            print('attempting to retrieve download info for {} rasters from wapor'.format(len(df_avail)))  
            
            # set up waitbar
            count = 0
            total_count = len(df_avail)
            printWaitBar(
                i=count, 
                total= total_count, 
                prefix = 'Retrieving Raster Urls:', 
                suffix = 'Complete: {} out of {}'.format(count,total_count), 
                length = 50)

            # retrieve data
            for __ ,row in df_avail.iterrows():
                wapor_time_dict = WaporRetrieval.deconstruct_wapor_time_code(
                    time_code = row['time_code'])
                
                # construct  wapor download dict
                wapor_dict = {}
                wapor_dict['component'] = component
                wapor_dict['cube_code'] = cube_code
                wapor_dict['period_string'] = wapor_time_dict['period_string'] 
                wapor_dict['period_start'] = wapor_time_dict['period_start'] 
                wapor_dict['period_end'] = wapor_time_dict['period_end']
                wapor_dict['return_period'] = self.return_period
                wapor_dict['raster_id'] = row['raster_id']
                wapor_dict['multiplier'] = multiplier

                # construct input file paths
                for folder_key in ['temp', 'download']:
                    # create and attach input paths including intermediaries
                    wapor_dict[folder_key] = self.generate_input_file_path(
                        component=component,
                        raster_id=row['raster_id'],
                        return_period=self.return_period,
                        input_folder=folder_key,
                        ext='.tif')

                # create masked folder entry
                wapor_dict['masked'] = self.generate_output_file_path(
                        description=component,
                        output_folder='masked',
                        period_start=wapor_dict['period_start'],
                        period_end=wapor_dict['period_end'],
                        aoi_name=aoi_name,
                        ext='.tif')
       
                wapor_dict['url'] = None  

                # check if files in a downward direction exist and if not note for downloading and processing as needed  
                if os.path.exists(wapor_dict['masked']):
                    wapor_dict['processing_steps'] = 0
                    print('masked file found skipping raster url download')

                elif os.path.exists(wapor_dict['download']):
                    wapor_dict['processing_steps'] = 1
                    print('downloaded file found skipping raster url download')

                elif os.path.exists(wapor_dict['temp']):
                    wapor_dict['processing_steps'] = 2
                    print('temp file found skipping raster url download')

                else:
                    ### attempt to retrieve the download url
                    wapor_dict['url'] = self.retrieve_crop_raster_url(
                        bbox=self.bbox,
                        cube_code=cube_code,
                        time_code=row['time_code'],
                        raster_id=row['raster_id'],
                        api_token=self.api_token)

                    wapor_dict['processing_steps'] = 3
                  
                wapor_download_list.append(wapor_dict)

                count += 1
                printWaitBar(count, total_count, 
                                            prefix = 'Retrieving Raster Urls:', 
                                            suffix = 'Complete: {} out of {}'.format(count,total_count), 
                                            length = 50)
    
        return wapor_download_list

    #################################
    def retrieve_actual_wapor_rasters(
        self, 
        wapor_download_list: list, 
        template_raster_path: str = None,
        aoi_name: str = 'nomask',
        output_nodata: float=-9999) -> dict:
        """
        Description:
            WAPOR download works in two phases. the retrieval and setup of the download info and the 
            actual download of that info.
            
            this subfunction carries out the actual downloa using previously setup download info. It 
            retrieves data from the WAPOR API according to the dictionary inputs retrieved using the 
            class function retrieve_wapor_download_info and carries out the standardised processing
            steps required according to the step count provided in the dict

            NOTE: works in combination with retrieve_wapor_download_info

            NOTE: aoi_name (mask name) if supplied should match that in retrieve_wapor_download_info

        Args:
            self: (see class for details)
            wapor_download_list: list of dicts containing the download info to use when retrieving the rasters
            template_raster_path: if provided uses the template as the source for the metadata and matches rasters too 
            it and masks them too match it too
            aoi_name: name for the mask subfolder if not provided writes too nomask folder and possibly 
            overwrites what is there
            output_nodata: nodata value to use for the retrieved data

        Return:
            dict: dictionary of dictionaries ordered by datacomponent each containing a list 
            of rasters retrieved and the path to the compiled vrt        
        """
        assert isinstance(wapor_download_list,list), 'please provide a list constructed using retrieve_wapor_download_info'
        assert isinstance(wapor_download_list[0],dict), 'please provide a list constructed using retrieve_wapor_download_info'

        # start retrieving data using the wapor dicts
        print('attempting to retrieve {} rasters from wapor'.format(len(wapor_download_list)))  
        # set up waitbar
        total_count = len(wapor_download_list)
        count = 0
        printWaitBar(
            count, 
            total_count, 
            prefix = 'Download/Process Raster Progress:', 
            suffix = 'Complete: {} out of {} '.format(count,total_count), 
            length = 50)

        # retrieve and process data per wapor download dict as needed
        for wapor_dict in wapor_download_list:
            if wapor_dict['processing_steps'] >= 3:
                # retrieve the raster and write to the download folder
                self.wapor_raster_request(
                    wapor_url=wapor_dict['url'],
                    output_file_path=wapor_dict['temp']
                    )

            if wapor_dict['processing_steps'] >= 2:
                # Process the downloaded raster and write to the process folder
                    WaporRetrieval.wapor_raster_processing(
                            input_raster_path=wapor_dict['temp'],
                            output_raster_path=wapor_dict['download'],
                            wapor_multiplier=wapor_dict['multiplier'],
                            return_period=wapor_dict['return_period'],
                            period_start=wapor_dict['period_start'],
                            period_end=wapor_dict['period_end'],
                            delete_input=True,
                            output_nodata=output_nodata
                            )

            count += 1
            printWaitBar(count, total_count, 
                prefix = 'Download/Process Raster Progress:', 
                suffix = 'Complete: {} out of {} '.format(count,total_count), 
                length = 50)

        # prepare the template for processing all rasters to match
        if not template_raster_path:
            template_raster_path = wapor_download_list[0]['download']
            mask_raster_path = None
            print('no mask provided all files will be downloaded again')
        else:
            template_raster_path = template_raster_path
            mask_raster_path = template_raster_path

        # reset count and initiate processing and masking
        count = 0
        printWaitBar(count, total_count, 
            prefix = 'Process and Mask Raster Progress:', 
            suffix = 'Complete: {} out of {}' .format(count,total_count), 
            length = 50)

        for wapor_dict in wapor_download_list: 
            if wapor_dict['processing_steps'] >= 1:
                # process the processed rasters to match their proj, dimensions and mask them as needed and write to the masked folder
                # if no mask is provided this step is always carried out
                raster.match_raster(
                    match_raster_path=template_raster_path,
                    input_raster_path=wapor_dict['download'],
                    output_raster_path=wapor_dict['masked'],
                    output_crs=self.output_crs,
                    mask_raster_path=mask_raster_path,
                    output_nodata=output_nodata)

                count += 1
                printWaitBar(count,total_count, 
                    prefix = 'Process/Mask Raster Progress:', 
                    suffix = 'Complete: {} out of {} '.format(count,total_count), 
                    length = 50)

        # set output dictionary
        retrieved_rasters_dict = {}
        # reorganise the files per datacomponent and create a vrt as needed
        datacomponent_list =  list(set([d['component'] for d in wapor_download_list]))
        
        for comp in datacomponent_list:
            # find the all processed rasters of a certain datacomponent
            masked_raster_list = [d['masked'] for d in wapor_download_list if d['component'] == comp]
            
            # retrieve the total temporal range they cover
            period_start = sorted([d['period_start'] for d in wapor_download_list if d['component'] == comp])[0]
            period_end = sorted([d['period_end'] for d in wapor_download_list if d['component'] == comp])[-1]

            # generate the vrt file name 
            vrt_path = self.generate_output_file_path(
                description=comp,
                period_start=period_start,
                period_end=period_end,
                output_folder='masked',
                aoi_name=aoi_name,
                ext='.vrt')

            # if the vrt does not already exist create it
            if not os.path.exists(vrt_path):
                raster.build_vrt(
                    raster_list=masked_raster_list, 
                    output_vrt_path=vrt_path,
                    action='time') 
            
            out_files = {'raster_list': masked_raster_list, 'vrt_path': vrt_path}

            retrieved_rasters_dict[comp] = out_files

        return retrieved_rasters_dict

    #################################
    def download_wapor_rasters(
        self, 
        datacomponents: list=None, 
        period_start: datetime=None, 
        period_end: datetime=None,
        return_period: str = None,
        template_raster_path: str = None,
        aoi_name: str = 'nomask',
        output_nodata: float=-9999
        ):
        """
        Description:
            wrapper function for retrieve_wapor_download_info and retrieve_actual_wapor_rasters
            
            WAPOR download works in two phases. the retrieval and setup of the download info and the 
            actual download of that info. The two subfunctions take care of this respectively. This wrapper
            combines them into one easier to use function. The combined process retrieves the 
            required download info for all the rasters and downloads and processes each raster.
            please see the specifics in each function for details

        Args:
            self: (see class for details) 
            period_start: start of period to retrieve data for, if not provided uses class version
            period_end: end of period to retrieve data for, if not provided uses class version
            return_period: return period/interval to retrieve data for, if not provided uses class version
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            template_raster_path: if provided uses the template as the source for the metadata and matches rasters too 
            it and masks them too match it too
            output_nodata: nodata value to use for the retrieved data

        Return:
            dict: dictionary of dictionaries ordered by datacomponent each containing a list of rasters 
            downloaded, a list of yearly vrts and the path to the full period vrt        
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        if not isinstance(datacomponents, list):
            raise TypeError('datacomponents provided should be formatted as a list of strings')

        # setup download to be carried out per year depending on the requested dates
        date_tuples = WaporStructure.wapor_organise_request_dates_per_year(
            period_start=self.period_start,
            period_end=self.period_end,
            return_period=self.return_period
        )

        # setup download variables
        num_of_downloads = len(date_tuples)
        current_download = 1
        download_dict = {}

        print('attempting to download raster data for {} periods'.format(num_of_downloads))
        for dt in date_tuples:
            print('downloading rasters for time period: {} to {}, period {} out of {}'.format(dt[0],dt[1], current_download, num_of_downloads))
            # retrieve the download info
            retrieval_info = self.retrieve_wapor_download_info( 
                datacomponents=datacomponents, 
                period_start=dt[0], 
                period_end=dt[1],
                return_period=self.return_period,
                aoi_name=aoi_name)

            # retrieve and process the rasters        
            retrieved_rasters_dict = self.retrieve_actual_wapor_rasters(
                wapor_download_list=retrieval_info,
                template_raster_path=template_raster_path,
                aoi_name=aoi_name,
                output_nodata=output_nodata)

            current_download +=1

            # update download_dict with the yearly retrieved rasters dict
            for datacomponent in retrieved_rasters_dict:
                if not datacomponent in download_dict:
                    download_dict[datacomponent] = {}
                    download_dict[datacomponent]['raster_list'] = []
                    download_dict[datacomponent]['vrt_list'] = []
                
                download_dict[datacomponent]['raster_list'].extend(retrieved_rasters_dict[datacomponent]['raster_list'])
                download_dict[datacomponent]['vrt_list'].append(retrieved_rasters_dict[datacomponent]['vrt_path'])            
            
        # generate whole period vrts
        for datacomponent in retrieved_rasters_dict:
            # generate the vrt file name 
            complete_vrt_path = self.generate_output_file_path(
                description=datacomponent,
                period_start=self.period_start,
                period_end=self.period_end,
                output_folder='masked',
                aoi_name=aoi_name,
                ext='.vrt')

            raster.build_vrt(
                raster_list=download_dict[datacomponent]['raster_list'], 
                output_vrt_path=complete_vrt_path,
                action='time') 

            download_dict[datacomponent]['vrt_path'] =complete_vrt_path

            
        return download_dict

if __name__ == "__main__":
    start = default_timer()
    args = sys.argv
