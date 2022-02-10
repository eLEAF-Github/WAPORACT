"""
waporact package

Performance Area Indicator Calculation (example pipeline class) 
"""
##########################
# import packages
import os
import shutil
import sys
from types import FunctionType
import numpy as np
from datetime import datetime, timedelta
from timeit import default_timer

from typing import Union

from waporact.scripts.tools import raster, statistics
from waporact.scripts.tools.plots import interactive_choropleth_map
from waporact.scripts.retrieval.wapor_retrieval import WaporRetrieval

##########################
class WaporPAI(WaporRetrieval):
    """
    Description:
        Given rasters and a shapefile calculates standardised statistics for
        Performance Area Indicators and stores them in the specific shapefile 
        according to to the structure given in WaporStructure

    Args:
        wapor_directory: directory to output downloaded and processed data too
        shapefile_path: path to the shapefile to clip downloaded data too if given 
        wapor_level: wapor wapor_level integer to download data for either 1,2, or 3
        api_token: api token to use when downloading data 
        project_name: name of the location to store the retrieved data  
        period_start: datetime object specifying the start of the period 
        period_end: datetime object specifying the end of the period 
        return_period: return period code of the component to be donwloaded (D (Dekadal) etc.)
        silent: boolean, if True the more general messages in the class are not printed 
        (autoset to False)       
        )

    return: 
        Statisitics calculated on the basis of the WAPOR rasters retrieved stored in
        a shapefile, other mediums to come)
    """
    def __init__(        
        self,
        waporact_directory: str,
        shapefile_path: str,
        api_token: str,
        wapor_level: int,
        project_name: int = 'test',
        period_start: datetime = datetime.now() - timedelta(days=1),
        period_end: datetime = datetime.now(),
        return_period: str = 'D',
        silent: bool = False
    ):
        self.waporact_directory = waporact_directory
        self.project_name = project_name
        self.shapefile_path = shapefile_path
        self.wapor_level = wapor_level
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period
        self.api_token = api_token
        self.silent = silent

        super().__init__(
            waporact_directory=self.waporact_directory,
            project_name=self.project_name,
            shapefile_path=self.shapefile_path,
            wapor_level=self.wapor_level,
            return_period=self.return_period,
            api_token=self.api_token,
            silent=self.silent,
            )


    ########################################################
    # Sub Dataframe functions
    ########################################################


    ########################################################
    # Sub Raster functions
    ########################################################
    def retrieve_and_analyse_period_of_wapor_rasters(
        self, 
        datacomponent: str,
        numpy_function: FunctionType,
        mask_raster_path: str,
        mask_folder: str,
        statistic: str, 
        retrieval_list: dict=None,       
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = None,
        output_nodata:float = -9999):
        """
        Description:
            retrieve and analyse a set of rasters
            from the wapor database for a given period using 
            a specific numpy statistic and if you want 
            mask to an area.

            useful for producing seasonal or annual sum or average 
            rasters for AETI or transpiration etc

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            datacomponent: wapor datacomponent to retrieve and analyse
            numpy_function: numpy function being called/ used to analyse the 
            set of rasters retrieved
            statistic: statistics being calculated used in the  output name, 
            should be related to the numpy funciton being used
            mask_folder: name to use for the mask folder auto set to nomask if not provided
            mask_raster_path: path to the crop mask defining the area for analysis
            output_nodata: nodata value to use on output
            retrieval_list: if you provide a retrieval list produced by 
            retrieve_wapor_download_info you can skip the preceding steps.
        
        Return:
            str: path to the produced raster     
        """
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period

        if not retrieval_list:
            output_raster_path =self.structure.generate_output_file_path(
                description='{}-{}'.format(datacomponent, statistic),
                period_start=period_start,
                period_end=period_end,
                output_folder='analysis',
                mask_folder=mask_folder,
                ext='.tif'   
            )

        else:
            # create standardised file name from retrieval list                
            period_start = sorted([d['period_start'] for d in retrieval_list])[0]
            period_end = sorted([d['period_end'] for d in retrieval_list])[-1]

            output_raster_path = self.structure.generate_output_file_path(
                description='{}_{}'.format(datacomponent, statistic),
                period_start=period_start,
                period_end=period_end,
                output_folder='analysis',
                mask_folder=mask_folder,
                ext='.tif'   
            )

        if not os.path.exists(output_raster_path):
            if not retrieval_list:
                print('retrieving {} data between {} and {} for masked data: {}'.format(
                    datacomponent, period_start, period_end, mask_folder))

                # retrieve the download info
                retrieval_info = self.retrieve_wapor_download_info(
                    period_start=period_start,
                    period_end=period_end,
                    return_period=return_period,
                    datacomponents=[datacomponent],
                    mask_folder=mask_folder)

            else:           
                retrieval_info = retrieval_list
            
            # download the rasters
            retrieved_data = self.retrieve_wapor_rasters(
                wapor_download_list=retrieval_info,
                template_raster_path=mask_raster_path,
                mask_folder=mask_folder)

            # calculate the aggregate raster using the specified numpy statistical function
            if len(retrieved_data[datacomponent]['raster_list']) > 1:
                statistics.calc_multiple_array_numpy_statistic(
                    input=retrieved_data[datacomponent]['vrt_path'],
                    numpy_function=numpy_function,
                    template_raster_path=mask_raster_path,
                    output_raster_path=output_raster_path,
                    axis=0,
                    output_nodata=output_nodata,
                    mask_to_template=True)

            else:
                print('only one raster found for the given period, summing process skipped and file copied instead')
                shutil.copy2(src=retrieved_data[datacomponent]['raster_list'][0], dst=output_raster_path)

        else:
            print('preexisting raster found skipping retrieval and analysis of: {}'.format(os.path.basename(output_raster_path)))

        return output_raster_path

    ##########################
    def calc_potential_raster(
        self, 
        input_raster_path: str,
        percentile: int,
        mask_raster_path: str,
        output_nodata:float=-9999):
        """
        Description:
            calculate a potential raster by selecting the 95% percentile 
            value within the input raster and assigning it to 
            all cells. Requires a mask to identify which cells to include in the analysis

        Args:
            self: (see class for details)
            evapotranspiration_raster_path: path to the evapotranspiration raster
            percentile: percentile to choose as the potential value
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            mask_folder: name to use for the mask folder auto set to nomask if not provided
            output_nodata: nodata value to use on output
        
        Return:
            str: path to the potential raster
        """
    	# create standardised filename
        file_parts = self.structure.deconstruct_output_file_path(
            output_file_path=input_raster_path
        )

        potential_raster_path = self.structure.generate_output_file_path(
            description='{}-pot'.format(file_parts['description']),
            period_start=file_parts['period_start'],
            period_end=file_parts['period_end'],
            output_folder='analysis',
            mask_folder=file_parts['mask_folder'],
            ext='.tif'
        )

        if not os.path.exists(potential_raster_path):
            # calculate the potet for the given date 
            statistics.calc_single_array_numpy_statistic(
                input=input_raster_path,
                numpy_function=np.nanpercentile,
                output_raster_path=potential_raster_path,
                template_raster_path=mask_raster_path,
                output_nodata=output_nodata,
                q=percentile,
                mask_to_template=True)
        else:
            print('preexisting raster found skipping analysis of: {}'.format(os.path.basename(potential_raster_path)))

        return potential_raster_path

    ########################################################
    # Main Raster functions
    ########################################################


    ########################################################
    # Performance Indicators Raster functions
    ########################################################
    def calc_relative_evapotranspiration(
        self, 
        mask_raster_path: str,
        mask_folder: str,   
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = None,
        percentile: int = 95,
        output_nodata:float = -9999,
        fields_shapefile_path: str=None,
        field_stats: list = ['mean'],
        id_key: str= 'wpid',
        out_dict: bool=False,
        ):
        """
        Description:
            calculate the relative evapotranspiration score to test for adequacy 
            per cell for the given period and area as defined by the class shapefile

            relative evapotranspiration: Sum of Evapotranspiration / Potential evapotranspiration

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            mask_folder: name to use for the mask folder auto set to nomask if not provided
            percentile: percentile of evapotranspiration values to choose as the 
            potential evapotranspiration value
            output_nodata: nodata value to use on output
            fields_shapefile_path: if the path to the fields shapefile path is provided
            then the field level statistics are also calculated
            field_stats: list of statistics to carry out during the field level analysis, 
            also used in the column names  
            id_key: name of shapefile column/feature dictionary key providing the feature indices 
            wpid is a reliable autogenerated index provided while making the mask
            (note: also handy for joining tables and the mask shape/other shapes back later) 
            out_dict: if true outputs a dictionary instead of a shapefile and does not
            write to csv.

        Return:
            tuple: path to the relative evapotranspiration raster,  (dataframe/dict, csv of field statistics)
        """        
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period

        # create standardised relative evapotranspiration file name
        relative_evapotranspiration_raster_path = self.structure.generate_output_file_path(
            description='ret',
            period_start=period_start,
            period_end=period_end,
            output_folder='results',
            mask_folder=mask_folder,
            ext='.tif',
        )
       
        if not os.path.exists(relative_evapotranspiration_raster_path):
            # retrieve and calculate sum of evapotranspiration for the given period
            evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                    period_start=period_start,
                    period_end=period_end,
                    datacomponent= 'AETI',
                    numpy_function=np.nansum,
                    mask_raster_path=mask_raster_path,
                    mask_folder=mask_folder,
                    statistic='sum',
                    return_period=return_period,
                    output_nodata=output_nodata)  

            # calculate potential evapotranspiration for the given period
            potential_evapotranspiration = self.calc_potential_raster(
                input_raster_path=evapotranspiration,
                percentile=percentile,
                mask_raster_path=mask_raster_path,
                output_nodata=output_nodata)

            # calculate relative evapotranspiration for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics( 
                a=evapotranspiration,
                b=potential_evapotranspiration,
                calc_function=statistics.ceiling_divide,
                output_raster_path=relative_evapotranspiration_raster_path,
                template_raster_path=mask_raster_path,
                output_nodata=output_nodata,
                mask_to_template=True)

            print('Relative evapotranspiration raster calculated: ret (Adequacy PAI)')
        
        else:
            print('Previously created relative evapotranspiration raster found: ret (Adequacy PAI)')
            
        if fields_shapefile_path:
            print('Calculating ret field statistics...')

            ret_csv_filepath = self.structure.generate_output_file_path(
                description='ret',
                period_start=period_start,
                period_end=period_end,
                output_folder='results',
                mask_folder=mask_folder,
                ext='.csv',
            )
            
            ret_field_stats = statistics.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=[relative_evapotranspiration_raster_path],
                output_csv_path=ret_csv_filepath,
                field_stats=field_stats,
                id_key=id_key,
                out_dict=out_dict,
                waporact_files=True)

            print('Relative evapotranspiration field statistics calculated: ret (Adequacy PAI)')

        else:
            ret_field_stats = None

        return (relative_evapotranspiration_raster_path, ret_field_stats)
    
    ########################################################
    def calc_temporal_variation_of_relative_evapotranspiration(
        self, 
        mask_raster_path: str,
        mask_folder: str,    
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = 'D',
        percentile: int = 95,
        output_nodata:float = -9999,
        fields_shapefile_path: str=None,
        field_stats: list = ['mean'],
        id_key: str= 'wpid',
        out_dict: bool=False,
        ):
        """
        Description:
            calculate the relative evapotranspiration score per dekad for the given period to test for reliability 
            per cell for the given period and area as defined by the class shapefile 

            temporal relative evapotranspiration: Sum of Evapotranspiration / Potential evapotranspiration per dekad as a time series

            tret = mean ret

        Args:
            self: (see class for details) 
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for
            autoset to dekad
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            mask_folder: name to use for the mask folder auto set to nomask if not provided
            percentile: percentile of evapotranspiration values to choose as the 
            potential evapotranspiration value
            output_nodata: nodata value to use on output
            fields_shapefile_path: if the path to the fields shapefile_path is provided
            then the field level statistics are also calculated            
            field_stats: list of statistics to carry out during the field level analysis, 
            also used in the column names 
            id_key: name of shapefile column/feature dictionary key providing the feature indices 
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later) 
            out_dict: if true outputs a dictionary instead of a shapefile and does not
            write to csv.

        Return:
            tuple: (path to the temporal variation in relative evapotranspiration raster, path to the vrt), (dataframe/dict, csv of field statistics)
        """        
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end

        # create standardised temporal relative evapotranspiration file name
        temporal_relative_evapotranspiration_raster_path = self.structure.generate_output_file_path(
            description='tret',
            period_start=period_start,
            period_end=period_end,
            output_folder='results',
            mask_folder=mask_folder,
            ext='.tif',
        )

        temporal_relative_evapotranspiration_vrt_path = self.structure.generate_output_file_path(
            description='tret',
            period_start=period_start,
            period_end=period_end,
            output_folder='results',
            mask_folder=mask_folder,
            ext='.vrt',
        )
      
        if not any(os.path.exists(file) for file in [temporal_relative_evapotranspiration_raster_path, temporal_relative_evapotranspiration_vrt_path]):
            # retrieve all available dekadal data for the period

            # retrieve the download info
            retrieval_info = self.retrieve_wapor_download_info(
                period_start=period_start,
                period_end=period_end,
                datacomponents=['AETI'])

            print("calculating {} sets of relative evapotranspiration".format(len(retrieval_info)))
            output_rasters = []

            for retrieval_dict in retrieval_info:
                # create standardised relative evapotranspiration file name            
                relative_evapotranspiration_raster_path = self.structure.generate_output_file_path(
                    description='ret',
                    period_start=retrieval_dict['period_start'],
                    period_end=retrieval_dict['period_end'],
                    output_folder='analysis',
                    mask_folder=mask_folder,
                    ext='.tif',
                )

                # retrieve and calculate sum of evapotranspiration for the given period
                evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                        datacomponent= 'AETI',
                        numpy_function=np.nansum,
                        mask_raster_path=mask_raster_path,
                        mask_folder=mask_folder,
                        statistic='sum',
                        retrieval_list=[retrieval_dict],
                        return_period=return_period,
                        output_nodata=output_nodata)  

                # calculate potential evapotranspiration for the given period
                potential_evapotranspiration = self.calc_potential_raster(
                    input_raster_path=evapotranspiration,
                    percentile=percentile,
                    mask_raster_path=mask_raster_path,
                    output_nodata=output_nodata)

                # calculate relative evapotranspiration for the given period (AETI/POTET)
                statistics.calc_dual_array_statistics( 
                    a=evapotranspiration,
                    b=potential_evapotranspiration,
                    calc_function=statistics.ceiling_divide,
                    output_raster_path=relative_evapotranspiration_raster_path,
                    template_raster_path=mask_raster_path,
                    output_nodata=output_nodata,
                    mask_to_template=True)

                output_rasters.append(relative_evapotranspiration_raster_path)

            raster.build_vrt(
                raster_list=output_rasters,
                output_vrt_path=temporal_relative_evapotranspiration_vrt_path,
                action='time'
            )

            statistics.calc_multiple_array_numpy_statistic(
                        input=temporal_relative_evapotranspiration_vrt_path,
                        numpy_function=np.nanmean,
                        template_raster_path=mask_raster_path,
                        output_raster_path=temporal_relative_evapotranspiration_raster_path,
                        axis=0,
                        output_nodata=output_nodata,
                        mask_to_template=True)

            # add average to calculate relaibility (average adequacy)

            print('Temporal variation in relative evapotranspiration raster(s) calculated: tret (Reliability PAI)')

        else:
            print('Previously created temporal variation in relative evapotranspiration raster(s) found: tret (Reliability PAI)')

        if fields_shapefile_path:
            print('Calculating tret field statistics...')

            tret_csv_filepath = self.structure.generate_output_file_path(
                description='tret',
                period_start=period_start,
                period_end=period_end,
                output_folder='results',
                mask_folder=mask_folder,
                ext='.csv',
            )

            tret_field_stats = statistics.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=[temporal_relative_evapotranspiration_raster_path, temporal_relative_evapotranspiration_vrt_path],
                output_csv_path=tret_csv_filepath,
                field_stats=field_stats,
                id_key=id_key,
                out_dict=out_dict,
                waporact_files=True)

            print('Temporal variation in relative evapotranspiration field statistics calculated: tret (Reliability PAI)')
        
        else:
            tret_field_stats = None

        return ((temporal_relative_evapotranspiration_raster_path, temporal_relative_evapotranspiration_vrt_path), tret_field_stats)

    ##########################
    def calc_crop_water_deficit(
        self, 
        mask_raster_path: str,
        mask_folder: str,     
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = None,
        percentile: int = 95,
        output_nodata:float = -9999,
        fields_shapefile_path: str=None,
        field_stats: list = ['mean'],
        id_key: str= 'wpid',
        out_dict: bool=False):
        """
        Description:
            calculate the crop water deficit score per cell to test for adequacy 
            for the given period and area as defined by the class shapefile

            crop_water_deficit: Potential evapotranspiration - Sum of Evapotranspiration

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            mask_folder: name to use for the mask folder auto set to nomask if not provided
            percentile: percentile of evapotranspiration values to choose as the 
            potential evapotranspiration value
            output_nodata: nodata value to use on output
            fields_shapefile_path: if the path to the fields shapefile_path is provided
            then the field level statistics are also calculated           
            field_stats: list of statistics to carry out during the field level analysis, 
            also used in the column names 
            id_key: name of shapefile column/feature dictionary key providing the feature indices 
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later) 
            out_dict: if true outputs a dictionary instead of a shapefile and does not
            write to csv.

        Return:
            tuple: path to the crop_water_deficit raster,  (dataframe/dict, csv of field statistics)
        """        
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period

        # create standardised crop_water_deficit file name
        crop_water_deficit_raster_path = self.structure.generate_output_file_path(
            description='cwd',
            period_start=period_start,
            period_end=period_end,
            output_folder='results',
            mask_folder=mask_folder,
            ext='.tif',
            )
        
        if not os.path.exists(crop_water_deficit_raster_path):
            # retrieve and calculate sum of evapotranspiration for the given period
            evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                    period_start=period_start,
                    period_end=period_end,
                    datacomponent= 'AETI',
                    numpy_function=np.nansum,
                    mask_raster_path=mask_raster_path,
                    mask_folder=mask_folder,
                    statistic='sum',
                    return_period=return_period,
                    output_nodata=output_nodata)  

            # calculate potential evapotranspiration for the given period
            potential_evapotranspiration = self.calc_potential_raster(
                input_raster_path=evapotranspiration,
                percentile=percentile,
                mask_raster_path=mask_raster_path,
                output_nodata=output_nodata)

            
            # calculate crop_water_deficit for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics( 
                a=potential_evapotranspiration,
                b=evapotranspiration,
                calc_function=statistics.floor_minus,
                output_raster_path=crop_water_deficit_raster_path,
                template_raster_path=mask_raster_path,
                output_nodata=output_nodata,
                mask_to_template=True)
            
            print('Crop water deficit raster calculated: cwd (Adequacy PAI)')

        else:
            print('Previously created crop water deficit raster found: cwd (Adequacy PAI)')

        if fields_shapefile_path:
            print('Calculating cwd field statistics...')
            
            cwd_csv_filepath = self.structure.generate_output_file_path(
                description='cwd',
                period_start=period_start,
                period_end=period_end,
                output_folder='results',
                mask_folder=mask_folder,
                ext='.csv',
            )

            cwd_field_stats = statistics.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=[crop_water_deficit_raster_path],
                output_csv_path=cwd_csv_filepath,
                field_stats=field_stats,
                id_key=id_key,
                out_dict=out_dict,
                waporact_files=True)

            print('Crop water deficit field statistics calculated: cwd (Adequacy PAI)')
        
        else:
            cwd_field_stats = None

        return (crop_water_deficit_raster_path, cwd_field_stats)

    ##########################
    def calc_beneficial_fraction(
        self, 
        mask_raster_path: str,
        mask_folder: str,
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = None,
        output_nodata:float = -9999,
        fields_shapefile_path: str=None,
        field_stats: list = ['mean'],
        id_key: str= 'wpid',
        out_dict: bool=False,):
        """
        Description:
            calculate an beneficial fraction score per cell to test for effeciency 
            for the given period and area as defined by the class shapefile

            beneficial fraction: Sum of Transpiration  / Sum of Evapotranspiration

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            mask_folder: name to use for the mask folder auto set to nomask if not provided
            output_nodata: nodata value to use on output
            fields_shapefile_path: if the path to the fields shapefile_path is provided
            then the field level statistics are also calculated           
            field_stats: list of statistics to carry out during the field level analysis, 
            also used in the column names 
            id_key: name of shapefile column/feature dictionary key providing the feature indices 
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later) 
            out_dict: if true outputs a dictionary instead of a shapefile and does not
            write to csv.

        Return:
            tuple: path to the beneficial fraction raster, (dataframe/dict, csv of field statistics)
        """    
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period
        

        # create standardised beneficial fraction file name
        beneficial_fraction_raster_path = self.structure.generate_output_file_path(
            description='bf',
            period_start=period_start,
            period_end=period_end,
            output_folder='results',
            mask_folder=mask_folder,
            ext='.tif',
            )

        if not os.path.exists(beneficial_fraction_raster_path):
            # retrieve and calculate average of evapotranspiration for the given period
            sum_evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                    period_start=period_start,
                    period_end=period_end,
                    datacomponent= 'AETI',
                    numpy_function=np.nansum,
                    mask_raster_path=mask_raster_path,
                    mask_folder=mask_folder,
                    statistic='sum',
                    return_period=return_period,
                    output_nodata=output_nodata)  
             
            # retrieve and calculate average of evapotranspiration for the given period
            sum_transpiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                    period_start=period_start,
                    period_end=period_end,
                    datacomponent= 'T',
                    numpy_function=np.nansum,
                    mask_raster_path=mask_raster_path,
                    mask_folder=mask_folder,
                    statistic='sum',
                    return_period=return_period,
                    output_nodata=output_nodata)  
  
            # calculate beneficial_fraction for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics( 
                a=sum_transpiration,
                b=sum_evapotranspiration,
                calc_function=statistics.ceiling_divide,
                output_raster_path=beneficial_fraction_raster_path,
                template_raster_path=mask_raster_path,
                output_nodata=output_nodata,
                mask_to_template=True)
            
            print('Beneficial fraction raster calculated: bf (Effeciency PAI)')
            
        else:
            print('Previously created beneficial fraction raster found: bf (Effeciency PAI)')

        if fields_shapefile_path:
            print('Calculating Beneficial Fraction field statistics...')
            
            bf_csv_filepath = self.structure.generate_output_file_path(
                description='bf',
                period_start=period_start,
                period_end=period_end,
                output_folder='results',
                mask_folder=mask_folder,
                ext='.csv',
            )       
            bf_field_stats = statistics.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=[beneficial_fraction_raster_path],
                output_csv_path=bf_csv_filepath,
                field_stats=field_stats,
                id_key=id_key,
                out_dict=out_dict,
                waporact_files=True)

            print('Beneficial fraction field stats calculated: bf (Effeciency PAI)')

        else:
            bf_field_stats = None

        return (beneficial_fraction_raster_path, bf_field_stats)

    ##########################
    # CV is a field calculation not pixel based needs a new location 
    # pretty sure this one is wrong currently but the method safi describes is field based lets see if this temporal pixel based version has worth
    def calc_coefficient_of_variation(
        self, 
        mask_raster_path: str,
        mask_folder: str,
        fields_shapefile_path: str,
        field_stats: list = ['mean', 'stddev'],
        id_key: str= 'wpid',
        out_dict: bool=False,
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = None,
        output_nodata:float = -9999):
        """
        Description:
            calculate a coefficient of variation score per field for the given period
            testing for equity in the area as defined by the class shapefile

            cov is a special pefromance indicator in that there is no raster equivalent

            equity: standard deviation of summed Evapotranspiration per field / 
            mean of summed evapotranspiration per field

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            mask_folder: name to use for the mask folder auto set to nomask if not provided
            percentile: percentile of evapotranspiration values to choose as the 
            potential evapotranspiration value
            output_nodata: nodata value to use on output
            fields_shapefile_path: required as cov is a field based statistic, used
            to calculate the field level statistics are also calculated           
            field_stats: list of statistics to carry out during the field level analysis, 
            also used in the column names 
            id_key: name of shapefile column/feature dictionary key providing the feature indices 
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later) 
            out_dict: if true outputs a dictionary instead of a shapefile and does not
            write to csv.

        Return:
            tuple: None ,  (dataframe/dict, csv of field statistics)
        """    
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period

        # create standardised coeffecient of variation file name
        cov_csv_filepath = self.structure.generate_output_file_path(
            description='cov',
            period_start=period_start,
            period_end=period_end,
            output_folder='results',
            mask_folder=mask_folder,
            ext='.csv',
            )
      
        # retrieve and calculate average of evapotranspiration for the given period
        sum_evapotranspiration_raster_path = self.retrieve_and_analyse_period_of_wapor_rasters(
                period_start=period_start,
                period_end=period_end,
                datacomponent= 'AETI',
                numpy_function=np.nansum,
                mask_raster_path=mask_raster_path,
                mask_folder=mask_folder,
                statistic='sum',
                return_period=return_period,
                output_nodata=output_nodata)  

        print('Calculating cov field statistics...')

        for stat in ['mean', 'stddev']:
            if stat not in field_stats:
                field_stats.append(stat)

        cov_dict = statistics.calc_field_statistics(
            fields_shapefile_path=fields_shapefile_path,
            input_rasters=[sum_evapotranspiration_raster_path],
            field_stats=field_stats,
            id_key=id_key,
            out_dict=True,
            waporact_files=True)[0]

        # calculate Coefficient of Variation
        for key in cov_dict.keys():
            mean_keys = [key for key in list(cov_dict[key].keys()) if 'mean' in key]
            if len(mean_keys) > 1:
                raise AttributeError('should not be more than one mean calculated')
            else:
                mean_key = mean_keys[0]
            stddev_keys = [key for key in list(cov_dict[key].keys()) if 'stddev' in key]
            if len(stddev_keys) > 1:
                raise AttributeError('should not be more than one stddev calculated')
            else:
                stddev_key = stddev_keys[0]

            if cov_dict[key][mean_key]== 0:
                cov_dict[key]['cov'] = np.nan
            else:
                cov_dict[key]['cov'] = cov_dict[key][stddev_key] / cov_dict[key][mean_key]

        if not out_dict:
            cov_field_stats = statistics.dict_to_dataframe(in_dict=cov_dict)
            statistics.output_table(
                table=cov_field_stats, 
                output_file_path=cov_csv_filepath)

        else:
            cov_field_stats = cov_dict
            cov_csv_filepath = None
        
        print('Coefficient of Variation field stats calculated: cov (Equity PAI)')

        return (None, (cov_field_stats, cov_csv_filepath))

    ##########################
    def calc_wapor_performance_indicators(
        self, 
        mask_raster_path: str,
        fields_shapefile_path: str,
        mask_folder: str,
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = None,
        output_nodata:float = -9999,
        field_stats: list = ['mean'],
        id_key: str= 'wpid',
        out_dict: bool=False):
        """
        Description:
            calculate all available performance indicators per cell to test for adequacy, effeciency
            reliability and equity for the given period and area as defined by the class shapefile

            beneficial fraction: Sum of Transpiration  / Sum of Evapotranspiration (bf)
            coeffecient of variation: standard deviation of summed Evapotranspiration per field / 
            mean of summed evapotranspiration per field
            crop_water_deficit: Potential evapotranspiration - Sum of Evapotranspiration (cwd)
            relative evapotranspiration: Sum of Evapotranspiration / Potential evapotranspiration (ret)
            temporal relative evapotranspiration: per dekad Sum of Evapotranspiration / Potential evapotranspiration
            (tret)

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            mask_folder: name to use for the mask folder auto set to nomask if not provided
            output_nodata: nodata value to use on output
            fields_shapefile_path: if the path to the fields shapefile_path is provided
            then the field level statistics are also calculated           
            field_stats: list of statistics to carry out during the field level analysis, 
            also used in the column names 
            id_key: name of shapefile column/feature dictionary key providing the feature indices 
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later) 
            out_dict: if true outputs a dictionary instead of a shapefile and does not
            write to csv.
        
        Return:
            tuple: list of paths to the performance indicator rasters,  (dataframe/dict, csv of field statistics)
        """    
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period

        # create standardised performance indicator stats file name
        pai_csv_filepath = self.structure.generate_output_file_path(
            description='pai',
            period_start=period_start,
            period_end=period_end,
            output_folder='results',
            mask_folder=mask_folder,
            ext='.csv',
        )

        # create standardised performance indicator html file name
        pai_html_filepath = self.structure.generate_output_file_path(
            description='pai',
            period_start=period_start,
            period_end=period_end,
            output_folder='images',
            mask_folder=mask_folder,
            ext='.html',
        )
        
        pai_rasters = []

        # calculate beneficial fraction
        bf_outputs = self.calc_beneficial_fraction( 
            mask_raster_path=mask_raster_path,
            mask_folder=mask_folder,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            output_nodata=output_nodata)

        pai_rasters.append(bf_outputs[0])

        # calculate crop water deficit
        cwd_outputs = self.calc_crop_water_deficit( 
            mask_raster_path=mask_raster_path,
            mask_folder=mask_folder,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            output_nodata=output_nodata)

        pai_rasters.append(cwd_outputs[0])

        # calculate relative evapotranspiration
        ret_outputs = self.calc_relative_evapotranspiration( 
            mask_raster_path=mask_raster_path,
            mask_folder=mask_folder,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            output_nodata=output_nodata)

        pai_rasters.append(ret_outputs[0])

        # calculate temporal_variation_of_relative_evapotranspiration
        tret_outputs = self.calc_temporal_variation_of_relative_evapotranspiration( 
            mask_raster_path=mask_raster_path,
            mask_folder=mask_folder,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            output_nodata=output_nodata)

        pai_rasters.append(tret_outputs[0][0])

        print('All PAI rasters calculated')

        if fields_shapefile_path:
            # calculate coefficient of variation
            cov_df = self.calc_coefficient_of_variation( 
                mask_raster_path=mask_raster_path,
                fields_shapefile_path=fields_shapefile_path,
                mask_folder=mask_folder,
                field_stats=field_stats,
                period_start=period_start,
                period_end=period_end,
                return_period=return_period,
                output_nodata=output_nodata,
                out_dict=False)[1][0]

            print('Calculating all remaining PAI field statistics...')

            pai_rasters = [r for r in pai_rasters if r]
            
            pai_df = statistics.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=pai_rasters,
                field_stats=field_stats,
                id_key=id_key,
                out_dict=True,
                waporact_files=True)[0]

            # add cov to the pai_dict
            pai_df = pai_df.merge(cov_df,left_on=id_key, right_on=id_key)

            statistics.output_table(
                table=pai_df, 
                output_file_path=pai_csv_filepath)

            if out_dict:
                pai_field_stats = pai_df.to_dict()

            else:
                pai_field_stats = pai_df

            pai_field_outputs = (pai_field_stats, pai_csv_filepath)

            print('Performance indicator field stats calculated: PAI')

        else:
            pai_field_outputs = None

        secondary_hovertemplate_inputs= {
            'mean_L3_bf_20200305_20200405':'mean_L3_bf_20200305_20200405',
            'mean_L3_cwd_20200305_20200405':'mean_L3_cwd_20200305_20200405',
            'mean_L3_tret_20200305_20200405': 'mean_L3_tret_20200305_20200405'
        }

        # create chloropleth output map
        interactive_choropleth_map(
            input_shapefile_path=fields_shapefile_path,
            input_csv_path=pai_field_outputs[1],
            z_column='mean_L3_ret_20200305_20200405',
            z_label='mean_L3_ret_20200305_20200405',
            secondary_hovertemplate_inputs=secondary_hovertemplate_inputs,
            output_html_path=pai_html_filepath,
            union_key=id_key)
            
        print('Performance indicator field stats map made: PAI')

        outputs = (pai_rasters, pai_field_outputs, pai_html_filepath)

        return outputs

if __name__ == "__main__":
    start = default_timer()
    args = sys.argv




