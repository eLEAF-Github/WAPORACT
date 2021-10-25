"""
minbuza_waterpip project

analysis script
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
#from numpy.core.defchararray import index

# from rasterstats import zonal_stats

from waterpip.scripts.support import raster, statistics, vector
from waterpip.scripts.structure.wapor_structure import WaporStructure
from waterpip.scripts.retrieval.wapor_retrieval import WaporRetrieval

##########################
class WaporAnalysis(WaporStructure):
    """
    Description:
        Given rasters and a shapefile calculates standardised statistics
        and stores them in the specific shapefile according to to the structure
        given in WaporStructure

    Args:
        period_start: datetime object specifying the start of the period 
        period_end: datetime object specifying the end of the period 
        wapor_directory: directory to output downloaded and processed data too
        project_name: name of the location to store the retrieved data  
        shapefile_path: path to the shapefile to clip downloaded data too if given
        wapor_level: wapor wapor_level integer to download data for either 1,2, or 3
        return_period: return period code of the component to be donwloaded (D (Dekadal) etc.)

    return: 
        Statisitics calculated on the basis of the WAPOR rasters retrieved stored in
        a shapefile, other mediums to come)
    """
    def __init__(        
        self,
        waterpip_directory: str,
        shapefile_path: str,
        wapor_level: int,
        project_name: int = 'test',
        period_start: datetime = datetime.now() - timedelta(days=1),
        period_end: datetime = datetime.now(),
        return_period: str = 'D',
        api_token: str = None,
    ):
        self.waterpip_directory = waterpip_directory
        self.project_name = project_name
        self.shapefile_path = shapefile_path
        self.wapor_level = wapor_level
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period
        self.api_token = api_token

        WaporStructure.__init__(self,
            return_period=self.return_period,
            waterpip_directory=self.waterpip_directory,
            project_name=self.project_name,
            period_end=self.period_end,
            period_start=self.period_start,
            wapor_level=self.wapor_level
        )

    ########################################################
    # Main Vector functions
    ########################################################
    """
    def calculate_rasterstats(
        self,
        input_raster_path: str, 
        input_shapefile_path: str=None,
        output_shapefile_name: str = None,
        stats: list = ['min', 'max', 'mean', 'sum']):
        #""#"
        Description:
            quickly calculates statistics for the specified shapefile
            and raster given and for each band in that raster if 
            applicable and stores them in a copy of the shapefile stored 
            in the analysis folders.
        
        Args:
            input_raster_path: raster to analysis
            input_shapefile_path: shapefile providing the analysis zones
            (uses self.shapefile_path if not provided)
            output_shapefile_name: name of the output shapefile. The path is autoset.
            if no name is provided this is set to input name plus *_analysis*
            stats: stats to calculate, accepts the following
            in list format

            ['min', 'max', 'mean', 'count', 'sum', 'std', 'median', 'majority', 'minority',
            'unique', 'range', 'percentile']

            NOTE: Note that certain statistics (majority, minority, 
            and unique) require significantly more processing due 
            to expensive counting of unique occurences for each pixel value.
  
        Return:
            str: path to the new/updated analysis shapefile
        #"#""
        all_statistics = ['min', 'max', 'mean', 
        'count', 'sum', 'std', 'median', 'majority', 
        'minority', 'unique', 'range', 'percentile'] 

        if not all(stat in all_statistics for stat in stats):
            raise AttributeError('all stats given must exist in: {}'.format(all_statistics))
        
        print('calculating basic zonal field statistics ...')
        
        if not input_shapefile_path:
            input_shapefile_path = self.shapefile_path

        if not output_shapefile_name:
            output_shapefile_path = os.path.join(
                self.project['results'],
                '{}_basic_stats.shp'.format(os.path.splitext(os.path.basename(input_shapefile_path))[0]))
        else:
             output_shapefile_path = os.path.join(
                self.project['results'],output_shapefile_name)

        #retrieve shapefile
        records = vector.file_to_records(input_shapefile_path)

        # add in a filter later

        # add in a categorical analysis later

        #run zonalstats through bands
        num_rasters = raster.gdal_info(input_raster_path)['band_count']

        band_stats = []
        for num in range(1, num_rasters+1):
            stats = zonal_stats(
                input_shapefile_path, 
                input_raster_path,
                band_num=num,
                stats=stats)

            band_name = raster.gdal_info(input_raster_path, band_num=num)['band_name']

            band_stats.append((band_name,stats))

        for band_name, stats in band_stats:
            band_name = band_name.split('_')[1] + band_name.split('_')[2]
            for stat in stats:
                records['{}_{}'.format(band_name,stat)] = [geom_stat[stat] for geom_stat in stats]

        for stat in stats:
            columns = [col for col in records.columns if stat in col]
            records['mean_{}'.format(stat)] = records[columns].mean(axis=1)

        records.to_file(output_shapefile_path)

        print('calculated basic zonal field statistics')

        return output_shapefile_path
    """
    ##########################
    def calc_field_statistics(
        self, 
        fields_shapefile_path: str,
        input_rasters: list,
        template_raster_path: str,
        crop: str='crop',
        field_stats: list=['min', 'max', 'mean', 'sum', 'stddev'],
        id_key: str='wpid',
        analysis_name: str = None,
        out_dict: bool=False,
        period_start: datetime=None,
        period_end: datetime=None,
        **kwargs):
        """
        Description:
            calculate a potential raster by selecting the 95% percentile 
            value within the input raster and assigning it to 
            all cells. Requires a mask to identify which cells to include in the analysis

        Args:
            self: (see class for details)
            fields_shapefile_path: path to the shapefile defining the fields
            percentile: percentile to choose as the potential value
            template_raster_path: template raster path to which the field indices are mapped
            must match dimensions of all input rasters
            crop: crop being analysed used in the name
            field_stats: list of statistics to carry out, also used in the column names 
            id_key: name of shapefile column/feature dictionary key providing the feature indices 
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later) 
            out_dict: if true outputs a dictionary instead of a shapefile and does not
            write to csv.
            analysis_name: name of analysis carried out is included in the csv output file name, if not provided
            is auto generated
            period_start: start of the season in datetime
            period_end: end of the season in datetime
        s
        Return:
            tuple: dataframe/dict made , path to the output csv
        """
        numpy_dict = {'sum': np.nansum, 'mean': np.nanmean, 'count': np.count, 'stddev': np.nanstd, 
        'min': np.nanmin, 'max': np.nanmax, 'mdeian': np.nanmedian, 
        'percentile': np.nanpercentile, 'variance': np.nanvar, 'quantile': np.nanquantile, 
        'cumsum': np.nancumsum, 'product': np.nanprod,'cumproduct': np.nancumprod }

        available_analysis = numpy_dict.keys()

        if not all(stat in available_analysis for stat in field_stats):
            raise KeyError('one of the analysis provided is not avaialble must be exist in: {}'.format(available_analysis))
        else:
            if not analysis_name:
                analysis_name = field_stats.join('')

            analyses = [(stat, numpy_dict[stat]) for stat in field_stats]

        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end

        multiple_rasters = False

        # create standardised csv file name
        period_start_str = period_start.strftime('%Y%m%d')
        period_end_str = period_end.strftime('%Y%m%d')

        csv_filename = 'L{}_{}_{}_{}_{}.csv'.format(
            self.wapor_level, crop, analysis_name, period_start_str,period_end_str)

        csv_file_path = os.path.join(self.project['results'], csv_filename)

        if any('vrt' in os.path.splitext(raster)[1] for raster in input_rasters):
            multiple_rasters = True

        if len(input_rasters) > 1:
            multiple_rasters = True

        if multiple_rasters:
            print('attempting to calculate zonal stats for multiple rasters')
            stats = statistics.multiple_raster_zonal_stats(
                template_raster_path=template_raster_path,
                input_shapefile_path=fields_shapefile_path,
                raster_path_list=input_rasters,
                analyses=analyses,
                out_dict=out_dict,
                index=id_key,
                **kwargs,
                )
        else:
            print('attempting to claculate zonal stats for a single raster')
            stats = statistics.single_raster_zonal_stats(
                input_shapefile_path=fields_shapefile_path,
                input_raster_path=input_rasters[0],
                analyses=analyses,
                out_dict=out_dict,
                id_key=id_key
                )

        if not out_dict:
            stats.to_csv(csv_file_path, sep = ';')
        else:
            csv_file_path = None

        return stats, csv_file_path

    ########################################################
    # Sub Raster functions
    ########################################################
    def retrieve_and_analyse_period_of_wapor_rasters(
        self, 
        datacomponent: str,
        numpy_function: FunctionType,
        crop_mask_path: str,
        crop: str,
        statistic: str, 
        api_token: str=None,
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
            api_token: token used to retrieve the data
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            datacomponent: wapor datacomponent to retrieve and analyse
            numpy_function: numpy function being called/ used to analyse the 
            set of rasters retrieved
            statistic: statistics being calculated used in the  output name, 
            should be related to the numpy funciton being used
            crop: crop being analysed used in the name
            crop_mask_path: path to the crop mask defining the area for analysis
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
        if not api_token:
            api_token = self.api_token

        crop = crop.lower().replace(' ', '_')

        if not retrieval_list:
            # create standardised file name
            period_start_str = period_start.strftime('%Y%m%d')
            period_end_str = period_end.strftime('%Y%m%d')

            output_filename = 'L{}_{}_{}_{}_{}_{}.tif'.format(
                self.wapor_level, crop, datacomponent, statistic, period_start_str, period_end_str)

            output_raster_path = os.path.join(self.project['analysis'], output_filename)

        else:
            # create standardised file name from retrieval list                
            period_start_str = sorted([d['period_start'] for d in retrieval_list])[0].strftime("%Y%m%d")
            period_end_str = sorted([d['period_end'] for d in retrieval_list])[-1].strftime("%Y%m%d")

            output_filename = 'L{}_{}_{}_{}_{}_{}.tif'.format(
                self.wapor_level, crop, datacomponent, statistic, period_start_str,period_end_str)

            output_raster_path = os.path.join(self.project['analysis'], output_filename)

        if not os.path.exists(output_raster_path):
            # set up the retrieval class
            retrieve = WaporRetrieval(
            waterpip_directory=self.waterpip_directory,
            project_name=self.project_name,
            shapefile_path=self.shapefile_path,
            wapor_level=self.wapor_level,
            return_period=return_period,
            api_token=api_token,
            silent=True,
            )

            if not retrieval_list:
                print('retrieving {} data between {} and {} for the crop: {}'.format(
                    datacomponent, period_start_str, period_end_str, crop))

                # retrieve the download info
                retrieval_info = retrieve.retrieve_wapor_download_info(
                    period_start=period_start,
                    period_end=period_end,
                    datacomponents=[datacomponent])

            else:           
                retrieval_info = retrieval_list
            
            # download the rasters
            retrieved_data = retrieve.retrieve_wapor_rasters(
                wapor_list=retrieval_info,
                template_raster_path=crop_mask_path,
                mask_to_template=True)

            # calculate the aggregate raster using the specified numpy statistical function
            if len(retrieved_data[datacomponent]['raster_list']) > 1:
                statistics.calc_multiple_array_numpy_statistic(
                    input=retrieved_data[datacomponent]['vrt_path'],
                    numpy_function=numpy_function,
                    template_raster_path=crop_mask_path,
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
        crop_mask_path: str,
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
            crop_mask_path: path to the crop mask to use in 
            defining the area for analysis if provided
            crop: crop being analysed used in the name
            output_nodata: nodata value to use on output
        
        Return:
            str: path to the potential raster
        """
    	# create standardised filename
        potential_raster_path = os.path.splitext(input_raster_path)[0] + '_pot.tif'

        if not os.path.exists(potential_raster_path):
            # calculate the potet for the given date 
            statistics.calc_single_array_numpy_statistic(
                input=input_raster_path,
                numpy_function=np.nanpercentile,
                output_raster_path=potential_raster_path,
                template_raster_path=crop_mask_path,
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
        crop_mask_path: str,
        crop: str,  
        api_token: str=None,   
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
            api_token: token used to retrieve the data 
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            crop_mask_path: path to the crop mask defining the area for analysis
            crop: crop being analysed used in the name
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
            tuple: path to the relative evapotranspiration raster,  (dataframe/dict, csv of field statistics)
        """        
        if not period_start:
            period_start=self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period
        if not api_token:
            api_token = self.api_token

        # create standardised relative evapotranspiration file name
        period_start_str = period_start.strftime('%Y%m%d')
        period_end_str = period_end.strftime('%Y%m%d')

        relative_evapotranspiration_filename = 'L{}_{}_ret_{}_{}.tif'.format(
            self.wapor_level, crop, period_start_str,period_end_str)

        relative_evapotranspiration_raster_path = os.path.join(self.project['results'], relative_evapotranspiration_filename)

        if not os.path.exists(relative_evapotranspiration_raster_path):
            # retrieve and calculate sum of evapotranspiration for the given period
            evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                    api_token=api_token,
                    period_start=period_start,
                    period_end=period_end,
                    datacomponent= 'AETI',
                    numpy_function=np.nansum,
                    crop_mask_path=crop_mask_path,
                    crop=crop,
                    statistic='sum',
                    return_period=return_period,
                    output_nodata=output_nodata)  

            # calculate potential evapotranspiration for the given period
            potential_evapotranspiration = self.calc_potential_raster(
                input_raster_path=evapotranspiration,
                percentile=percentile,
                crop_mask_path=crop_mask_path,
                output_nodata=output_nodata)

            # calculate relative evapotranspiration for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics( 
                a=evapotranspiration,
                b=potential_evapotranspiration,
                calc_function=statistics.ceiling_divide,
                output_raster_path=relative_evapotranspiration_raster_path,
                template_raster_path=crop_mask_path,
                output_nodata=output_nodata,
                mask_to_template=True)

            print('Relative evapotranspiration raster calculated: ret (Adequacy PAI)')
        
        else:
            print('Previously created relative evapotranspiration raster found: ret (Adequacy PAI)')
            
        if fields_shapefile_path:
            print('Calculating ret field statistics...')

            ret_field_stats = self.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=[relative_evapotranspiration_raster_path],
                template_raster_path=crop_mask_path,
                crop=crop,
                field_stats=field_stats,
                analysis_name='ret',
                id_key=id_key,
                period_start= period_start,
                period_end=period_end,
                out_dict=out_dict)

            print('Relative evapotranspiration field statistics calculated: ret (Adequacy PAI)')

        else:
            ret_field_stats = None

        return (relative_evapotranspiration_raster_path, ret_field_stats)
    
    ########################################################
    def calc_temporal_variation_of_relative_evapotranspiration(
        self, 
        crop_mask_path: str,
        crop: str,
        api_token: str=None,     
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
            api_token: token used to retrieve the data 
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for
            autoset to dekad
            crop_mask_path: path to the crop mask defining the area for analysis
            crop: crop being analysed used in the name
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
        if not api_token:
            api_token = self.api_token

        # create standardised temporal relative evapotranspiration file name
        period_start_str = period_start.strftime('%Y%m%d')
        period_end_str = period_end.strftime('%Y%m%d')

        temporal_relative_evapotranspiration_raster_path = os.path.join(self.project['results'],'L{}_{}_tret_{}_{}.tif'.format(
            self.wapor_level, crop, period_start_str,period_end_str))

        temporal_relative_evapotranspiration_vrt_path = os.path.join(self.project['results'], 'L{}_{}_tret_{}_{}.vrt'.format(
            self.wapor_level, crop, period_start_str,period_end_str))

        if not any(os.path.exists(file) for file in [temporal_relative_evapotranspiration_raster_path, temporal_relative_evapotranspiration_vrt_path]):
            
            # retrieve all available dekadal data for the period

            # set up the retrieval class
            retrieve = WaporRetrieval(
                waterpip_directory=self.waterpip_directory,
                project_name=self.project_name,
                shapefile_path=self.shapefile_path,
                wapor_level=self.wapor_level,
                return_period=return_period,
                api_token=api_token,
                silent=True,
                )

            # retrieve the download info
            retrieval_info = retrieve.retrieve_wapor_download_info(
                period_start=period_start,
                period_end=period_end,
                datacomponents=['AETI'])

            print("calculating {} sets of relative evapotranspiration".format(len(retrieval_info)))
            output_rasters = []
            
            periodic_relative_evapotranspiration_raster_dir = os.path.join(self.project['results'],'tret_rasters_{}_{}_{}_{}_{}'.format(
                self.wapor_level, crop, return_period, period_start_str, period_end_str))

            if not os.path.exists(periodic_relative_evapotranspiration_raster_dir):
                os.makedirs(periodic_relative_evapotranspiration_raster_dir)

            for retrieval_dict in retrieval_info:
                # create standardised relative evapotranspiration file name
                relative_evapotranspiration_filename = 'L{}_{}_ret_{}.tif'.format(
                    self.wapor_level, crop, retrieval_dict['period_str'])

                relative_evapotranspiration_raster_path = os.path.join(periodic_relative_evapotranspiration_raster_dir,
                    relative_evapotranspiration_filename)

                # retrieve and calculate sum of evapotranspiration for the given period
                evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                        api_token=api_token,
                        datacomponent= 'AETI',
                        numpy_function=np.nansum,
                        crop_mask_path=crop_mask_path,
                        crop=crop,
                        statistic='sum',
                        retrieval_list=[retrieval_dict],
                        return_period=return_period,
                        output_nodata=output_nodata)  

                # calculate potential evapotranspiration for the given period
                potential_evapotranspiration = self.calc_potential_raster(
                    input_raster_path=evapotranspiration,
                    percentile=percentile,
                    crop_mask_path=crop_mask_path,
                    output_nodata=output_nodata)

                # calculate relative evapotranspiration for the given period (AETI/POTET)
                statistics.calc_dual_array_statistics( 
                    a=evapotranspiration,
                    b=potential_evapotranspiration,
                    calc_function=statistics.ceiling_divide,
                    output_raster_path=relative_evapotranspiration_raster_path,
                    template_raster_path=crop_mask_path,
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
                        template_raster_path=crop_mask_path,
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

            tret_field_stats = self.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=[temporal_relative_evapotranspiration_vrt_path, temporal_relative_evapotranspiration_raster_path],
                template_raster_path=crop_mask_path,
                crop=crop,
                field_stats=field_stats,
                analysis_name='tret',
                id_key=id_key,
                period_start= period_start,
                period_end=period_end,
                out_dict=out_dict)

            print('Temporal variation in relative evapotranspiration field statistics calculated: tret (Reliability PAI)')
        
        else:
            tret_field_stats = None

        return ((temporal_relative_evapotranspiration_raster_path, temporal_relative_evapotranspiration_vrt_path), tret_field_stats)

    ##########################
    def calc_crop_water_deficit(
        self, 
        crop_mask_path: str,
        crop: str,        
        api_token: str=None,
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
            api_token: token used to retrieve the data 
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            crop_mask_path: path to the crop mask defining the area for analysis
            crop: crop being analysed used in the name
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
        if not api_token:
            api_token = self.api_token

        crop = crop.lower().replace(' ', '_')

        # create standardised crop_water_deficit file name
        period_start_str = period_start.strftime('%Y%m%d')
        period_end_str = period_end.strftime('%Y%m%d')

        crop_water_deficit_filename = 'L{}_{}_cwd_{}_{}.tif'.format(
            self.wapor_level, crop, period_start_str,period_end_str)

        crop_water_deficit_raster_path = os.path.join(self.project['results'], crop_water_deficit_filename)

        if not os.path.exists(crop_water_deficit_raster_path):
            # retrieve and calculate sum of evapotranspiration for the given period
            evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                    api_token=api_token,
                    period_start=period_start,
                    period_end=period_end,
                    datacomponent= 'AETI',
                    numpy_function=np.nansum,
                    crop_mask_path=crop_mask_path,
                    crop=crop,
                    statistic='sum',
                    return_period=return_period,
                    output_nodata=output_nodata)  

            # calculate potential evapotranspiration for the given period
            potential_evapotranspiration = self.calc_potential_raster(
                input_raster_path=evapotranspiration,
                percentile=percentile,
                crop_mask_path=crop_mask_path,
                output_nodata=output_nodata)

            
            # calculate crop_water_deficit for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics( 
                a=potential_evapotranspiration,
                b=evapotranspiration,
                calc_function=statistics.floor_minus,
                output_raster_path=crop_water_deficit_raster_path,
                template_raster_path=crop_mask_path,
                output_nodata=output_nodata,
                mask_to_template=True)
            
            print('Crop water deficit raster calculated: cwd (Adequacy PAI)')

        else:
            print('Previously created crop water deficit raster found: cwd (Adequacy PAI)')

        if fields_shapefile_path:
            print('Calculating cwd field statistics...')

            cwd_field_stats = self.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=[crop_water_deficit_raster_path],
                template_raster_path=crop_mask_path,
                crop=crop,
                field_stats=field_stats,
                analysis_name='cwd',
                id_key=id_key,
                period_start= period_start,
                period_end=period_end,
                out_dict=out_dict)

            print('Crop water deficit field statistics calculated: cwd (Adequacy PAI)')
        
        else:
            cwd_field_stats = None

        return (crop_water_deficit_raster_path, cwd_field_stats)

    ##########################
    def calc_beneficial_fraction(
        self, 
        crop_mask_path: str,
        crop: str,
        api_token: str=None,
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
            api_token: token used to retrieve the data 
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            crop_mask_path: path to the crop mask defining the area for analysis
            crop: crop being analysed used in the name
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
        if not api_token:
            api_token = self.api_token
        
        crop = crop.lower().replace(' ', '_')

        # create standardised beneficial fraction file name
        period_start_str = period_start.strftime('%Y%m%d')
        period_end_str = period_end.strftime('%Y%m%d')

        beneficial_fraction_filename = 'L{}_{}_bf_{}_{}.tif'.format(
            self.wapor_level, crop, period_start_str,period_end_str)

        beneficial_fraction_raster_path = os.path.join(self.project['results'], beneficial_fraction_filename)

        if not os.path.exists(beneficial_fraction_raster_path):
            # retrieve and calculate average of evapotranspiration for the given period
            sum_evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                    api_token=api_token,
                    period_start=period_start,
                    period_end=period_end,
                    datacomponent= 'AETI',
                    numpy_function=np.nansum,
                    crop_mask_path=crop_mask_path,
                    crop=crop,
                    statistic='sum',
                    return_period=return_period,
                    output_nodata=output_nodata)  

            # retrieve and calculate average of evapotranspiration for the given period
            sum_transpiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                    api_token=api_token,
                    period_start=period_start,
                    period_end=period_end,
                    datacomponent= 'T',
                    numpy_function=np.nansum,
                    crop_mask_path=crop_mask_path,
                    crop=crop,
                    statistic='sum',
                    return_period=return_period,
                    output_nodata=output_nodata)  

            # calculate beneficial_fraction for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics( 
                a=sum_transpiration,
                b=sum_evapotranspiration,
                calc_function=statistics.ceiling_divide,
                output_raster_path=beneficial_fraction_raster_path,
                template_raster_path=crop_mask_path,
                output_nodata=output_nodata,
                mask_to_template=True)
            
            print('Beneficial fraction raster calculated: bf (Effeciency PAI)')
            
        else:
            print('Previously created beneficial fraction raster found: bf (Effeciency PAI)')

        if fields_shapefile_path:
            print('Calculating Beneficial Fraction field statistics...')
            
            bf_field_stats = self.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=[beneficial_fraction_raster_path],
                template_raster_path=crop_mask_path,
                crop=crop,
                field_stats=field_stats,
                analysis_name='bf',
                id_key=id_key,
                period_start= period_start,
                period_end=period_end,
                out_dict=out_dict)

            print('Beneficial fraction field stats calculated: bf (Effeciency PAI)')

        else:
            bf_field_stats = None

        return (beneficial_fraction_raster_path, bf_field_stats)

    ##########################
    # CV is a field calculation not pixel based needs a new location 
    # pretty sure this one is wrong currently but the method safi describes is field based lets see if this temporal pixel based version has worth
    def calc_coefficient_of_variation(
        self, 
        crop_mask_path: str,
        crop: str,
        fields_shapefile_path: str,
        field_stats: list = ['mean', 'stddev'],
        id_key: str= 'wpid',
        out_dict: bool=False,
        api_token: str=None,
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = None,
        output_nodata:float = -9999):
        """
        Description:
            calculate a coefficient of variation score per field for the given period
            testing for equity in the area as defined by the class shapefile

            cov is a special pefromancie indicator in that there is no raster equivalent

            equity: standard deviation of summed Evapotranspiration per field / 
            mean of summed evapotranspiration per field

        Args:
            self: (see class for details)
            api_token: token used to retrieve the data 
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            crop_mask_path: path to the crop mask defining the area for analysis
            crop: crop being analysed used in the name
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
        if not api_token:
            api_token = self.api_token

        crop = crop.lower().replace(' ', '_')

        # create standardised coeffecient of variation file name
        period_start_str = period_start.strftime('%Y%m%d')
        period_end_str = period_end.strftime('%Y%m%d')

        cov_dictname = 'L{}_{}_cov_{}_{}'.format(
            self.wapor_level, crop, period_start_str,period_end_str)

        cov_filename = 'L{}_{}_cov_{}_{}.csv'.format(
            self.wapor_level, crop, period_start_str,period_end_str)

        cov_csv_path = os.path.join(self.project['results'], cov_filename)

        # retrieve and calculate average of evapotranspiration for the given period
        sum_evapotranspiration_raster_path = self.retrieve_and_analyse_period_of_wapor_rasters(
                api_token=api_token,
                period_start=period_start,
                period_end=period_end,
                datacomponent= 'AETI',
                numpy_function=np.nansum,
                crop_mask_path=crop_mask_path,
                crop=crop,
                statistic='sum',
                return_period=return_period,
                output_nodata=output_nodata)  

        print('Calculating cov field statistics...')

        for stat in ['mean', 'stddev']:
            if stat not in field_stats:
                field_stats.append(stat)

        cov_dict, __ = self.calc_field_statistics(
            fields_shapefile_path=fields_shapefile_path,
            input_rasters=[sum_evapotranspiration_raster_path],
            template_raster_path=crop_mask_path,
            crop=crop,
            field_stats=field_stats,
            analysis_name='cov',
            id_key=id_key,
            period_start= period_start,
            period_end=period_end,
            out_dict=True)

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
            cov_field_stats = statistics.dict_to_dataframe(in_dict=cov_dict, orient='index')
            statistics.output_table(
                table=cov_field_stats, 
                output_file_path=cov_csv_path)

        else:
            cov_field_stats = cov_dict
            cov_csv_path = None
        
        print('Coefficient of Variation field stats calculated: cov (Equity PAI)')

        return (None, (cov_field_stats, cov_csv_path))

    ##########################
    def calc_wapor_performance_indicators(
        self, 
        crop_mask_path: str,
        fields_shapefile_path: str,
        crop: str,
        api_token: str=None,
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str = None,
        output_nodata:float = -9999,
        field_stats: list = ['mean'],
        id_key: str= 'wpid',
        out_dict: bool=False):
        """
        Description:
            calculate all available perfornamce indicators per cell to test for adequacy, effeciency
            reliability and equity for the given period and area as defined by the class shapefile

            beneficial fraction: Sum of Transpiration  / Sum of Evapotranspiration (bf)
            equity here: standard deviation of Evapotranspiration / Evapotranspiration Mean (cov)
            equity safi: standard deviation of summed Evapotranspiration per field / 
            mean of summed evapotranspiration per field (cov)
            crop_water_deficit: Potential evapotranspiration - Sum of Evapotranspiration (cwd)
            relative evapotranspiration: Sum of Evapotranspiration / Potential evapotranspiration (ret)
            temporal relative evapotranspiration: per dekad Sum of Evapotranspiration / Potential evapotranspiration
            (tret)

        Args:
            self: (see class for details)
            api_token: token used to retrieve the data 
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for, 
            auto set to monthly
            crop_mask_path: path to the crop mask defining the area for analysis
            crop: crop being analysed used in the name
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
        if not api_token:
            api_token = self.api_token

        crop = crop.lower().replace(' ', '_')

        # create standardised coeffecient of variation file name
        period_start_str = period_start.strftime('%Y%m%d')
        period_end_str = period_end.strftime('%Y%m%d')

        pai_filename = 'L{}_{}_pai_{}_{}.csv'.format(
            self.wapor_level, crop, period_start_str,period_end_str)

        pai_csv_path = os.path.join(self.project['results'], pai_filename)

        pai_rasters = []

        # calculate beneficial fraction
        bf_outputs = self.calc_beneficial_fraction( 
            api_token=api_token,
            crop_mask_path=crop_mask_path,
            crop=crop,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            output_nodata=output_nodata)

        pai_rasters.append(bf_outputs[0])

        # calculate crop water deficit
        cwd_outputs = self.calc_crop_water_deficit( 
            api_token=api_token,
            crop_mask_path=crop_mask_path,
            crop=crop,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            output_nodata=output_nodata)

        pai_rasters.append(cwd_outputs[0])

        # calculate relative evapotranspiration
        ret_outputs = self.calc_relative_evapotranspiration( 
            api_token=api_token,
            crop_mask_path=crop_mask_path,
            crop=crop,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            output_nodata=output_nodata)

        pai_rasters.append(ret_outputs[0])

        # calculate temporal_variation_of_relative_evapotranspiration
        tret_outputs = self.calc_temporal_variation_of_relative_evapotranspiration( 
            api_token=api_token,
            crop_mask_path=crop_mask_path,
            crop=crop,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            output_nodata=output_nodata)

        pai_rasters.append(tret_outputs[0][0])
        pai_rasters.append(tret_outputs[0][1])

        print('All PAI rasters calculated')

        if fields_shapefile_path:
            # calculate coefficient of variation
            cov_dict = self.calc_coefficient_of_variation( 
                api_token=api_token,
                crop_mask_path=crop_mask_path,
                fields_shapefile_path=fields_shapefile_path,
                crop=crop,
                field_stats=field_stats,
                period_start=period_start,
                period_end=period_end,
                return_period=return_period,
                output_nodata=output_nodata,
                out_dict=True)[1][0]

            print('Calculating all remaining PAI field statistics...')

            pai_rasters = [r for r in pai_rasters if r]
            
            pai_dict = self.calc_field_statistics(
                fields_shapefile_path=fields_shapefile_path,
                input_rasters=pai_rasters,
                template_raster_path=crop_mask_path,
                field_stats=field_stats,
                id_key=id_key,
                crop=crop,
                analysis_name='pai',
                period_start= period_start,
                period_end=period_end,
                out_dict=True)[0]

            # add cov to the pai_dict
            for key in pai_dict.keys():
                for stat in cov_dict[key].keys():
                    pai_dict[key][stat] = cov_dict[key][stat]

            if not out_dict:
                pai_field_stats = statistics.dict_to_dataframe(in_dict=pai_dict, orient='index')
                statistics.output_table(
                    table=pai_field_stats, 
                    output_file_path=pai_csv_path)

                pai_field_outputs = (pai_field_stats, pai_csv_path)

            else:
                pass

            print('Performance indicator field stats calculated: PAI')

        else:
            pai_field_outputs = None

        output_statistics = (pai_rasters, pai_field_outputs)

        return output_statistics



if __name__ == "__main__":
    print('main')
  