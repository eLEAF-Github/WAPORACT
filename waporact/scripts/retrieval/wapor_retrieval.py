
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


from ast import literal_eval
import numpy as np
import pandas as pd
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

        inherits the waporstructure class folder structure

    Args:
        wapor_directory: directory to output downloaded and processed data too
        shapefile_path: path to the shapefile to clip downloaded data too if given
        wapor_level: wapor wapor_level integer to download data for either 1,2, or 3
        api_token: api token to use when downloading data
        project_name: name of the location to store the retrieved data
        period_start: datetime object specifying the start of the period
        period_end: datetime object specifying the end of the period
        return_period: return period code of the component to be downloaded (D (Dekadal) etc.)
        datacomponents: wapor datacomponents (interception (I) etc.) to download
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
        api_token: str,
        project_name: int = 'test',
        period_start: datetime = datetime.now() - timedelta(days=30),
        period_end: datetime = datetime.now(),
        datacomponents: list = ['ALL'],
        return_period: str = 'D',
        wapor_version: int = 2,
        silent: bool=False,

    ):  
        # set and generate the parameters
        self.period_start = period_start
        self.period_end = period_end
        self.waporact_directory = waporact_directory
        self.project_name = project_name
        self.wapor_level = wapor_level
        self.api_token = api_token
        self.wapor_version = wapor_version
        self.silent = silent
        self.return_period = return_period

        # inherit and initialise the WaporStructure class
        super().__init__( 
            waporact_directory=self.waporact_directory,
            return_period = self.return_period,
            project_name=self.project_name,
            period_end=self.period_end,
            period_start=self.period_start,
            wapor_level=self.wapor_level
        )

        # attach and setup the waporAPI class
        self.wapor_api = WaporAPI(
            period_start=self.period_start,
            period_end=self.period_end,
            version=self.wapor_version
        )

        # set and generate the remaining parameters
        self.shapefile_path = shapefile_path
        self.output_crs = None
        self.bbox = None
        self.bbox_shapefile = self.bbox

        # check all catalogs and reference shapefiles have been stored in the metadata folder
        if not self.silent:
            print('running check for all wapor wapor_level catalogues and downloading as needed:')
        for wapor_level in (1,2,3):
            self.retrieve_catalog(wapor_level=wapor_level)

        if self.wapor_level == 3:
            # check for a wapor_level 3 location shapefile
            self.set_level_3_availability_shapefile()

            # if wapor_level 3 check if the given area falls within an available area:
            self.country_code = self.check_level_3_location()
            self.country_code = self.country_code
        else:
            self.country_code = 'notlevel3notused'
            self.country_code = self.country_code

        if not self.silent:
            print('loading wapor catalogue for this run:')

        # set instance catalog
        self.catalog = self.retrieve_catalog(wapor_level=self.wapor_level)

        # check if return period exists against the retrieved catalog
        self.check_return_period(return_period)

        # check if datacomponents exist against the retrieved catalog and set them
        self.datacomponents = self.check_datacomponents(datacomponents)

    #################################
    # properties
    #################################
    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value: tuple):
        """ 
        Description:
            if no value is provided takes the self.shapefile_path
            and produces a bbox tuple using it

        Args:
            value: existing bbox tuple or nothing

        Return:
            tuple: calculated or existing bbox
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
                    raise FileNotFoundError('shapefile not found please provide a new path to a shapefile')
    
    #################################
    @property
    def bbox_shapefile(self):
        return self._bbox_shapefile

    @bbox_shapefile.setter
    def bbox_shapefile(self, value: tuple = None):
        """ 
        Description:
            if a bbox shapefile does not already exist in the expected location 
            takes the given bbox tuple and produces one, if no
            tuple is provided attempts to do it using the self.shapefile_path
        """
        run = False
        # check if a bbox shapefile has been previously made
        bbox_shapefile_name = os.path.splitext(os.path.basename(self.shapefile_path))[0] + '_bbox.shp'
        bbox_shapefile_path = os.path.join(self.project['reference'],bbox_shapefile_name)
        if not os.path.exists(bbox_shapefile_path):
            run = True
        else: 
            if not self.silent:
                print('bbox shapefile can be found at: {}'.format(bbox_shapefile_path))
            self._bbox_shapefile = bbox_shapefile_path
            pass
        
        if run:
            if isinstance(value, tuple):
                try:
                    vector.create_bbox_shapefile(
                        output_shape_path=bbox_shapefile_path,
                        bbox=value)
                    self._bbox_shapefile = bbox_shapefile_path
                    run = False
                
                    if not self.silent:
                        print('bbox shapefile based on the bbox tuple made and outputted too: {}'.format(bbox_shapefile_path))
                except:
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
                    raise FileNotFoundError('shapefile not found please provide a new path to a shapefile')

    #################################
    @property
    def output_crs(self):
        return self._output_crs

    @output_crs.setter
    def output_crs(self, value: int):
        """ 
        Description:
            checks if a project crs has been provided and if not attempts to
            retrieve it from the shapefile provided

        Args:
            value: int representation of the crs (epsg code)

        Return:
            int: provided or calculated crs (epsg code)
        """
        if not value or not isinstance(value, int):
            if not self.shapefile_path:
                raise AttributeError('please provide the path to a shapefile')
            else:
                if os.path.exists(self.shapefile_path):
                    self._output_crs = vector.retrieve_shapefile_crs(self.shapefile_path)

                else:
                    raise FileNotFoundError('shapefile not found please provide a new path to a shapefile')

    #################################
    @property
    def wapor_connection_attempts(self):
        return self._wapor_connection_attempts

    @wapor_connection_attempts.setter
    def wapor_connection_attempts(self, value: int = None):
        """
        Description:
            creates a dictionary setting the limit and keeping count of the
            connection attempts made ot the wapor site

        Args:
            value: any value not int intialises the wapor connection dict

        Return:
            dict: dictionary with settings to attempt connection/data retrieval from wapor
        """
        if not value or not isinstance(value, int):
            value = {
                'connection_error':0,
                'connection_sleep': 2,
                'connection_limit': 50
            }

            self._wapor_connection_attempts = value

        else:
            pass


    #################################
    # class functions
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
    def wapor_organise_request_dates_per_year(
        cls,
        period_start: str,
        period_end: str,
        return_period: str,
        ):
        """
        Description:
            class subfunction to organise dates for the function download_wapor_rasters
            so that rasters can be downloaded per year. Also carries out a check to see if the period
            specified is long enough for the return period specified

        Args:
            period_start: datetime object start of period to organise
            period_end: datetime object end of period to organise

        Return:
            list: list of tuples with the period start and period split between years
        """
        # return period in days to check
        return_period_dict = {
                        'D': (10, 'Dekadal'),
                        'M': (30, 'Monthly'),
                        'S': (100, 'Seasonal'),
                        'A': (365, 'Annual'),
                        'LT': (365, 'Long Term'),
                    }

        # check if period given is long enough for return_period given
        return_period_length = return_period_dict[return_period][0]
        num_days = (period_end - period_start).days

        if not num_days >= return_period_length:
            raise AssertionError('num_days between given period_start: {} and period_end: {} \
            are not long enough for the given return_period: {} to assure data retrieval'.format(
                period_start, period_end, return_period_dict[return_period][1]))

        # check if the period is longer than a year
        start_year = period_start.year
        end_year = period_end.year
        num_calendar_years = end_year - start_year + 1

        if num_calendar_years == 0:
            date_tuples = [(period_start, period_end)] 
        
        else:
            # filter through the years and create datetime period tuples to mine data for
            days_in_start_year = (datetime(period_start.year,12,31) - period_start).days
            days_in_end_year = (period_end - datetime(period_end.year,1,1)).days
            current_year = start_year
            date_tuples = []
            skip_year = False # only used in case the first year is combined with the next and the first year is not also the next year
            for i in range(0, num_calendar_years):
                if skip_year:
                    skip_year = False
                    continue
                if num_calendar_years == 2: 
                    # only two calendar years in the period
                    if days_in_start_year >= return_period_length and days_in_end_year >= return_period_length:
                        # split the download between the two periods within the two calendar years
                        date_tuples =  [(period_start, datetime(period_start.year,12,31)), 
                                        (datetime(period_end.year,1,1), period_end)] 
                    else:
                        # the period in either of the two years is to short so keep the original period
                        date_tuples = [(period_start, period_end)] 

                    break

                else:        
                    if i == 0: # the first year
                        if not days_in_start_year >= return_period_length:
                            # skip a year as the first calendar year is to short so combined with the following year
                            current_year += 1 
                            start = period_start
                            end = datetime(current_year,12,31) 
                            skip_year = True
                            
                        else:
                            start = period_start
                            end = datetime(period_start.year,12,31) 
                        
                    elif i == num_calendar_years: # if the last year 
                        if not days_in_end_year >= return_period_length:
                            #  combine the last two years as the last calendar year is to short so combined with the previous year
                            current_year -= 1 
                            if len(date_tuples) == 1:
                                start = period_start
                            else:
                                start = datetime(current_year,1,1)
                            
                            end = period_end
                            date_tuples.pop()

                        else:
                            start = datetime(period_end.year,1,1)
                            end = period_end
                    
                    else:
                        # its an inbetween year
                        start = datetime(current_year,1,1) 
                        end = datetime(current_year,12,31) 

                    date_tuples.append((start,end))
                    current_year += 1
                
        return date_tuples

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
    def check_return_period(self, return_period: str):
        """
        Description
            checks if the return period code given exists in the given catalog

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
        if not return_period in codes:
            desc = list(self.catalog['period_desc'])
            combos = [item for item in zip(codes,desc)]
            combos = list(set(combos)) # filter to unique combos
            raise AttributeError('given return period could not be found among the available return periods, \
            use one of the following and try activating the class again: {}'.format(combos))

        return 0


    #################################
    def check_datacomponents(
        self,
        datacomponents: list,
        return_period: str=None):
        """
        Description
            checks if the datacomponents given are real ones that can be retrieved
            according to the catalog, if All is provided all available datacomponents are returned.

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
        # check given return period exists first
        self.check_return_period(return_period)

        # retrieve all components from the catalog at class given wapor level
        catalog_components = list(set(list(self.catalog['component_code'])))
        catalog_codes = list(set(list(self.catalog['code'])))

        # retrieve all datacomponents for given combination of level, return period and country code (if applicable)
        if self.wapor_level == 3:
            available_datacomponents = [comp for comp in catalog_components if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) in catalog_codes] 
        else:
            available_datacomponents = [comp for comp in catalog_components if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) in catalog_codes] 

        if datacomponents[0] is 'ALL':
            out_datacomponents = available_datacomponents
        else:
            if self.wapor_level == 3:
                existing_datacomponents = [comp for comp in datacomponents if 'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) in catalog_codes]
                missing_datacomponents = [comp for comp in datacomponents if  'L{}_{}_{}_{}'.format(self.wapor_level,self.country_code,comp,return_period) not in catalog_codes]
            else:
                existing_datacomponents = [comp for comp in datacomponents if 'L{}_{}_{}'.format(self.wapor_level,comp,return_period) in catalog_codes]
                missing_datacomponents = [comp for comp in datacomponents if  'L{}_{}_{}'.format(self.wapor_level,comp,return_period) not in catalog_codes]
               
            if missing_datacomponents:
                print('Available datacomponents for the given inputs: {}'.format(available_datacomponents))
                raise AttributeError('at wapor level: {} and return period: {} , the following datacomponents could not be \
                found and may not exist: {}'.format(self.wapor_level, return_period, missing_datacomponents))
            elif not existing_datacomponents:
                raise AttributeError('at wapor level: {} and return period: {} , no datacomponents could  be \
                found'.format(self.wapor_level, return_period))
            else:
                out_datacomponents = existing_datacomponents

        return out_datacomponents

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
            raise AttributeError('no overlap found between wapor_level 3 locations shapefile: {} and the generated bbox shapefile: {}\
            it is likely that there is no data available at level 3 for this area, check in qgis or similar'.format(
                self.l3_locations_shapefile_path, self.bbox_shapefile))
        
        return code
    
    #################################
    # retrieval functions
    #################################
    def retrieve_catalog(
        self,
        wapor_level: int = None,
        cubeInfo=True):
        '''
        Description:
            retrieves the wapor catalog from the standard storage location a sub directory of
            the waporact directory you set on class activation. On retrieval creates a
            dataframe and sets it to self.catalog for use.

            if the catalog was not found or the version found was too old attempts to retrieve
            it from the WAPOR database.

            NOTE: based on the WaporAPI class function getCatalog

        Args:
            wapor_level: level to retrieve the catalog for
            cube_info: if true also retrieves and formats the cube info from the catalog into
            the output dataframe

        Return:
            dataframe: dataframe of the retrieved catalog
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
                df = self.wapor_api._query_catalog(wapor_level)
            except:
                print('ERROR: data of the specified wapor_level could not be retrieved'
                ' or there was a connection error (wapor_level: {})'.format(self.wapor_level))
            
            if cubeInfo:
                cubes_measure=[]
                cubes_dimension=[]
                for cube_code in df['code'].values:                
                    cubes_measure.append(self.wapor_api._query_cubeMeasures(cube_code,
                                                                       version=self.wapor_api.version))
                    cubes_dimension.append(self.wapor_api._query_cubeDimensions(cube_code,
                                                                       version=self.wapor_api.version))
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
            print('on running the retrieval class again the catalogue will be auto replaced ' 
                'if the catalogue becomes outdated after a 60 day period or if is not found')    
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
    def set_level_3_availability_shapefile(
        self):
        """
        Description:
            sets the path to the wapor level 3 locations shapefile
            and if it does not exist or is outdated creates it

            filepath generated is set to self.l3_locations_shapefile_path
            in the standard sub folder of the waporact directory set by the
            user on class activation

        Return:
            int: 0

        """
        retrieve = False
        l3_locations_shapefile_path = os.path.join(self.project['meta'], 'wapor_L3_locations.shp')

        # check if the file already exists
        if not os.path.exists(l3_locations_shapefile_path):
            retrieve = True
        else:
            # check how old the file is
            st=os.stat(l3_locations_shapefile_path) 
            if (time.time() - st.st_mtime) >= 5184000: # 60 days
                retrieve = True

        if retrieve:
            print('creating wapor_level 3 locations shapefile')
            # set temporary date variables
            api_per_start = datetime(2010,1,1).strftime("%Y-%m-%d")
            api_per_end = datetime.now().strftime("%Y-%m-%d")
            api_period = '{},{}'.format(api_per_start, api_per_end)

            # retrieve country codes
            temp_catalog = self.retrieve_catalog(wapor_level=3) 
            country_codes = zip(list(temp_catalog['country_code']),list(temp_catalog['country_desc']))
            codes = list(set([(x,y) for x,y in country_codes]))
            _data = []
            # loop through countries check data availability and retrieve the bbox
            for code in codes:
                cube_code= f"L3_{code[0]}_T_D"

                try:
                    df_avail=self.wapor_api.getAvailData(cube_code,time_range=api_period)
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
                    datacomponents = self.check_datacomponents(datacomponents=['ALL'], return_period=rp[0])
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
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end

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
                    period_start=period_start,
                    period_end=period_end,
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
        self.wapor_connection_attempts()
        cube_info = None
        while cube_info is None:
            # attempt to retrieve cube_info
            try:
                cube_info=self.wapor_api.getCubeInfo(cube_code)
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
        self.wapor_connection_attempts()
        data_availability = None
        while data_availability is None:
            # attempt to retrieve cube_info
            try:
                data_availability=self.wapor_api.getAvailData(cube_code, time_range=time_range)
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
        self.wapor_connection_attempts()
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
                    print_job=False)   

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
            self.wapor_connection_attempts()
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
        self.wapor_connection_attempts['connection_error'] += 1
        if self.wapor_connection_attempts['connection_error'] == abs(self.wapor_connection_attempts['connection_limit']/2):
            print('{} failed,  {} connection errors noted, will continue connection attempts'.format(description, self.wapor_connection_attempts['connection_error'])) 
        if self.wapor_connection_attempts['connection_error'] >= self.wapor_connection_attempts['connection_limit']:
            if isinstance(_exception,requests.exceptions.RequestException):
                error_statement = ('{} from WAPOR '
                    'attempted to request data {} times every {} sec and failed due to request/connection error, adjust the self.wapor_connection_attempts or sleep'
                    'time to try for longer'.format(description, self.wapor_connection_attempts['connection_error'], self.wapor_connection_attempts['connection_sleep']))
                raise ConnectionError(error_statement)

            else:
                error_statement = ('{} from WAPOR '
                    'attempted to request data {} times every {} sec and failed due to unknown error, adjust the self.wapor_connection_attempts or sleep'
                    'time to try for longer'.format(description, self.wapor_connection_attempts['connection_error'], self.wapor_connection_attempts['connection_sleep']))
                raise TimeoutError(error_statement)

        return 0 

   #################################
    def retrieve_wapor_download_info(self, 
        datacomponents: list=None, 
        period_start: datetime=None, 
        period_end: datetime=None,
        return_period: str = None,
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

        Args:
            self: datacomponents, wapor_level, return_period, out_dir, shapefile_path, period_start (see class for details)
            datcomponents: datacomponents if you want to repeat the function using non self functions,
            period_start: datacomponents if you want to repeat the function using non self functions,
            period_end: datacomponents if you want to repeat the function using non self functions,
            return_period: if provided overrides the return period of the class
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided

        Return:
            list: list of dicts containing variables that can be used to retrieve and store a specific wapor raster
        """
        if not return_period:
            return_period = self.return_period
        if not datacomponents:
            datacomponents = self.datacomponents
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end

        # generate dates for filenames
        dates_dict = self.generate_dates_dict(
            period_start=period_start,
            period_end=period_end,
            return_period=return_period)
        
        # setup output list
        wapor_download_list = []

        #check if the datacomponents are available
        datacomponents = self.check_datacomponents(datacomponents, return_period)

        # retrieve download info per available datacomponent
        for component in datacomponents:
            if self.wapor_level==3:
                print('retrieving download info for wapor_level 3 region: {}'.format(self.country_code))
            print('retrieving download info for component: {}'.format(component))
        
            wapor_dict = {}

            # construct the wapor_cube_code
            cube_code = self.generate_wapor_cube_code(
                component=component,
                return_period=return_period,
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
                wapor_time_dict = self.deconstruct_wapor_time_code(
                    time_code = row['time_code'])
                
                # construct  wapor download dict
                wapor_dict = {}
                wapor_dict['component'] = component
                wapor_dict['cube_code'] = cube_code
                wapor_dict['period_string'] = wapor_time_dict['period_string'] 
                wapor_dict['period_start'] = wapor_time_dict['period_start'] 
                wapor_dict['period_end'] = wapor_time_dict['period_end']
                wapor_dict['return_period'] = return_period
                wapor_dict['raster_id'] = row['raster_id']
                wapor_dict['multiplier'] = multiplier

                # construct input file paths
                for folder_key in ['temp', 'download']:
                    # create and attach input paths including intermediaries
                    wapor_dict[folder_key] = self.generate_input_file_path(
                        component=component,
                        raster_id=row['raster_id'],
                        return_period=return_period,
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
                WaporRetrieval.wapor_raster_request(
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
            datcomponents: datacomponents if you want to repeat the function using non self functions,
            period_start: datacomponents if you want to repeat the function using non self functions,
            period_end: datacomponents if you want to repeat the function using non self functions,
            return_period: if provided overrides the return period of the class
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            template_raster_path: if provided uses the template as the source for the metadata and matches rasters too 
            it and masks them too match it too
            output_nodata: nodata value to use for the retrieved data

        Return:
            dict: dictionary of dictionaries ordered by datacomponent each containing a list of rasters 
            downloaded, a list of yearly vrts and the path to the full period vrt        
        """
        if not datacomponents:
            datacomponents = self.datacomponents
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period

        # setup download to be carried out per year depending on the requested dates
        date_tuples = WaporRetrieval.wapor_organise_request_dates_per_year(
            period_start=period_start,
            period_end=period_end,
            return_period=return_period
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
                return_period=return_period,
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
                period_start=period_start,
                period_end=period_end,
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

