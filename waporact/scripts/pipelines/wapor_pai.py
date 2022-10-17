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


from waporact.scripts.retrieval.wapor_retrieval import WaporRetrieval
from waporact.scripts.tools import raster, statistics, vector
from waporact.scripts.tools.plots import (
    interactive_choropleth_map,
    shapeplot,
    rasterplot,
    scatterplot,
)

import logging

from waporact.scripts.tools.logger import format_root_logger

logger = logging.getLogger(__name__)

##########################
class WaporPAI(WaporRetrieval):
    def __init__(
        self,
        waporact_directory: str,
        vector_path: str,
        wapor_level: int,
        period_start: datetime,
        period_end: datetime,
        api_token: str,
        project_name: int = "test",
        return_period: str = "D",
        silent: bool = None,
        print_wait_bar: bool = True,
    ):
        """Class that provides access to functions that calculate wapor
        based  water use Performance Area Indicators (PAI).

        Parameters
        ----------
        waporact_directory : str
            directory to output downloaded and processed data too
        vector_path : str
            path to the vector file to clip downloaded data too if given
        wapor_level : int
            wapor wapor_level integer to download data for either 1,2, or 3
        period_start : datetime
            datetime object specifying the start of the period
        period_end : datetime
            datetime object specifying the end of the period
        api_token : str
            api token to use when downloading data
        country_code : str, optional
            _description_, by default "notyetset"
        return_period : str, optional
            return period code of the component to be downloaded (D (Dekadal) etc.), by default "D"
        project_name : str, optional
            name of the location to store the retrieved data, by default "test"
        wapor_version : int, optional
            _description_, by default 2
        silent : bool, optional
            if True the more general messages in the run are not printed, if false it shares those message, if none dose not change the set level, by default False
        print_wait_bar : bool, optional
            if true prints the wait bar when downloading, by default True
        """
        # set verbosity (feedback) parameter
        if silent == True:
            format_root_logger(logging_level=logging.WARNING)
        elif silent == False:
            format_root_logger(logging_level=logging.INFO)
        else:
            pass

        super().__init__(
            waporact_directory=waporact_directory,
            project_name=project_name,
            vector_path=vector_path,
            wapor_level=wapor_level,
            period_start=period_start,
            period_end=period_end,
            return_period=return_period,
            api_token=api_token,
            print_wait_bar=print_wait_bar,
        )

    ########################################################
    # Sub functions
    ########################################################
    def retrieve_and_analyse_period_of_wapor_rasters(
        self,
        datacomponent: str,
        numpy_function: FunctionType,
        mask_raster_path: str,
        aoi_name: str,
        statistic: str,
        period_start: datetime = None,
        period_end: datetime = None,
        return_period: str = None,
        output_nodata: float = -9999,
    ):
        """retrieve and analyse a set of rasters from the wapor database for a given period using
        a specific numpy statistic and if you want mask to an area.


        Parameters
        ----------
        datacomponent : str
            wapor datacomponent to retrieve and analyse
        numpy_function : FunctionType
            numpy function being called/ used to analyse the set of rasters retrieved
        mask_raster_path : str
            path to the crop mask defining the area for analysis
        aoi_name : str
            area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
        statistic : str
            statistics being calculated used in the  output name
        period_start : datetime, optional
            start of the season in datetime, by default None
        period_end : datetime, optional
            end of the season in datetime, by default None
        return_period : str, optional
            return period to retrieve data for auto set to monthly, by default None
        output_nodata : float, optional
            nodata value to use on output, by default -9999

        Returns
        -------
        str
            path to the outputted raster
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        output_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{datacomponent}-{statistic}",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="analysis",
            aoi_name=aoi_name,
            ext=".tif",
        )

        if not os.path.exists(output_raster_path):
            logger.info(
                f"retrieving {datacomponent} data between {self.period_start} and {self.period_end} for aoi (mask): {aoi_name}"
            )

            retrieved_data = self.download_wapor_rasters(
                datacomponents=[datacomponent],
                period_start=self.period_start,
                period_end=self.period_end,
                return_period=self.return_period,
                aoi_name=aoi_name,
                template_raster_path=mask_raster_path,
            )

            # calculate the aggregate raster using the specified numpy statistical function
            if len(retrieved_data[datacomponent]["raster_list"]) > 1:
                statistics.calc_multiple_array_numpy_statistic(
                    input=retrieved_data[datacomponent]["vrt_path"],
                    numpy_function=numpy_function,
                    template_raster_path=mask_raster_path,
                    output_raster_path=output_raster_path,
                    axis=0,
                    output_nodata=output_nodata,
                    mask_to_template=True,
                )

            else:
                logger.info(
                    "only one raster found for the given period, summing process skipped and file copied instead"
                )
                shutil.copy2(
                    src=retrieved_data[datacomponent]["raster_list"][0],
                    dst=output_raster_path,
                )

        else:
            logger.info(
                f"preexisting raster found skipping retrieval and analysis of: {os.path.basename(output_raster_path)}"
            )

        return output_raster_path

    ##########################
    def calc_potential_raster(
        self,
        input_raster_path: str,
        percentile: int,
        mask_raster_path: str,
        output_nodata: float = -9999,
    ):
        """calculate a potential raster by selecting the 95% percentile of the value with the array
        and assigning it to all cells

        Parameters
        ----------
        input_raster_path : str
            path to the evapotranspiration raster
        percentile : int
            percentile to choose as the potential value
        mask_raster_path : str
            path to the raster mask defining the area for analysis if provided
        output_nodata : float, optional
            nodata value to use on output, by default -9999

        Returns
        -------
        str
            path to the outputted potential raster
        """
        # create standardised filename
        file_parts = self.deconstruct_output_file_path(
            output_file_path=input_raster_path
        )

        potential_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{file_parts['description']}-pot",
            period_start=file_parts["period_start"],
            period_end=file_parts["period_end"],
            output_folder="analysis",
            aoi_name=file_parts["mask_folder"],
            ext=".tif",
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
                mask_to_template=True,
            )
        else:
            logger.info(
                f"preexisting raster found skipping analysis of: {os.path.basename(potential_raster_path)}"
            )

        return potential_raster_path

    ##########################
    def create_pai_csv_and_plots(
        self,
        input_raster_path: str,
        mask_raster_path: str,
        file_description: str,
        title: str,
        fields_vector_path: str = None,
        z_label: str = None,
        z_column: str = None,
        field_stats: list = ["mean"],
        period_start: datetime = None,
        period_end: datetime = None,
        aoi_name: str = None,
        id_key: str = "wpid",
        zmin: float = None,
        zmax: float = None,
        output_static_map: bool = True,
        output_interactive_map: bool = True,
        output_csv: bool = True,
    ):
        """
        Description:
            subfunction to create visualisations in a standard way for the different pai's

        Args:
            self: (see class for details)
            input_raster_path: path to the raster to plot
            and calculate field statistics from
            file_description: name for the files made, used in combo with standard parts
            (ret, cwd, mean ret etc)
            title: title of the plots made, used in combo with standard parts
            (relative evapotranspiration etc)
            z_label: label for the colour bar,
            z_column: name of the column in the csv to plot
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            output_nodata: nodata value to use on output
            field_stats: list of statistics to carry out during the field level analysis,
            also used in the column names
            fields_vector_path: if the path to the fields shapefile path is provided
            then the field level statistics are also calculated
            id_key: name of shapefile column/feature dictionary key providing the feature indices
            zmin: minimum value on the z axis, autoset based on available values if not provided
            zmax: minimum value on the z axis, autoset based on available values if not provided
            output_static_map: if true outputs a static map
            output_interactive_map: if true outputs a interactive map
            output_csv:if true outputs a csv and shape plot file to a standardised location

        Return:
            dict: field statistics calculated
        """
        if output_static_map:
            # if true create and output a static raster map and shape map
            rasterplot_filepath = self.generate_output_file_path(
                wapor_level=self.wapor_level,
                description=file_description,
                period_start=period_start,
                period_end=period_end,
                output_folder="images",
                aoi_name=aoi_name,
                ext=".png",
            )

            rasterplot(
                input_value_raster_path=input_raster_path,
                output_plot_path=rasterplot_filepath,
                input_mask_raster_path=mask_raster_path,
                zmin=zmin,
                zmax=zmax,
                title=title,
            )

            logger.info(f"{title} raster plot made: {rasterplot_filepath}")

        if fields_vector_path:
            logger.info(f"Calculating {file_description} field statistics...")

            field_stats_dict = statistics.calc_field_statistics(
                fields_shapefile_path=fields_vector_path,
                input_rasters=[input_raster_path],
                field_stats=field_stats,
                statistic_name=file_description,
                id_key=id_key,
                out_dict=True,
            )

            logger.info(f"{title} field statistics calculated")

            if output_interactive_map or output_csv:
                # if true output the results to a csv and use it to create an interactive map if applicable
                csv_filepath = self.generate_output_file_path(
                    wapor_level=self.wapor_level,
                    description=file_description,
                    period_start=period_start,
                    period_end=period_end,
                    output_folder="results",
                    aoi_name=aoi_name,
                    ext=".csv",
                )

                statistics.output_table(field_stats_dict, output_file_path=csv_filepath)

                logger.info(f"{title} csv made: {csv_filepath}")

                shapeplot_filepath = self.generate_output_file_path(
                    wapor_level=self.wapor_level,
                    description=f"{file_description}_fields",
                    period_start=period_start,
                    period_end=period_end,
                    output_folder="images",
                    aoi_name=aoi_name,
                    ext=".png",
                )

                shapeplot(
                    input_shape_path=fields_vector_path,
                    output_plot_path=shapeplot_filepath,
                    title=title,
                    z_column=z_column,
                    zmin=zmin,
                    zmax=zmax,
                    input_table=csv_filepath,
                    join_column=id_key,
                )

                logger.info(f"{title} field plot made: {shapeplot_filepath}")

                if output_interactive_map:
                    # if true create and output an interactive map
                    shapeplot_html = self.generate_output_file_path(
                        wapor_level=self.wapor_level,
                        description=file_description,
                        period_start=period_start,
                        period_end=period_end,
                        output_folder="images",
                        aoi_name=aoi_name,
                        ext=".html",
                    )

                    interactive_choropleth_map(
                        input_shapefile_path=fields_vector_path,
                        input_table=csv_filepath,
                        z_column=z_column,
                        z_label=z_label,
                        zmin=zmin,
                        zmax=zmax,
                        output_html_path=shapeplot_html,
                    )

                    logger.info(f"{title} interactive plot made: {shapeplot_html}")

        else:
            field_stats_dict = None
            logger.info(
                f"no field shapefile provided so no field level statistics, field plot, interactive plot or csv could be made for: {file_description}"
            )

        return field_stats_dict

    ########################################################
    # Performance Indicators functions
    ########################################################
    def calc_relative_evapotranspiration(
        self,
        mask_raster_path: str,
        aoi_name: str,
        period_start: datetime = None,
        period_end: datetime = None,
        return_period: str = None,
        percentile: int = 95,
        output_nodata: float = -9999,
        fields_vector_path: str = None,
        field_stats: list = ["mean"],
        id_key: str = "wpid",
        output_static_map: bool = True,
        output_interactive_map: bool = True,
        output_csv: bool = True,
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
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            percentile: percentile of evapotranspiration values to choose as the
            potential evapotranspiration value
            output_nodata: nodata value to use on output
            fields_vector_path: if the path to the fields shapefile path is provided
            then the field level statistics are also calculated
            field_stats: list of statistics to carry out during the field level analysis,
            also used in the column names
            id_key: name of shapefile column/feature dictionary key providing the feature indices
            wpid is a reliable autogenerated index provided while making the mask
            (note: also handy for joining tables and the mask shape/other shapes back later)
            output_static_map: if true outputs a static map
            output_interactive_map: if true outputs a interactive map
            output_csv:if true outputs a csv and shape plot file to a standardised location



        Return:
            tuple: path to the relative evapotranspiration raster, dict of field statistics
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        # create standardised relative evapotranspiration file name
        relative_evapotranspiration_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="ret",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="results",
            aoi_name=aoi_name,
            ext=".tif",
        )

        if not os.path.exists(relative_evapotranspiration_raster_path):
            # retrieve and calculate sum of evapotranspiration for the given period
            evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                period_start=self.period_start,
                period_end=self.period_end,
                datacomponent="AETI",
                numpy_function=np.nansum,
                mask_raster_path=mask_raster_path,
                aoi_name=aoi_name,
                statistic="sum",
                return_period=self.return_period,
                output_nodata=output_nodata,
            )

            # calculate potential evapotranspiration for the given period
            potential_evapotranspiration = self.calc_potential_raster(
                input_raster_path=evapotranspiration,
                percentile=percentile,
                mask_raster_path=mask_raster_path,
                output_nodata=output_nodata,
            )

            # calculate relative evapotranspiration for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics(
                a=evapotranspiration,
                b=potential_evapotranspiration,
                calc_function=statistics.ceiling_divide,
                output_raster_path=relative_evapotranspiration_raster_path,
                template_raster_path=mask_raster_path,
                output_nodata=output_nodata,
                mask_to_template=True,
            )

            logger.info(
                "Relative evapotranspiration raster calculated: ret (Adequacy PAI)"
            )

        else:
            logger.info(
                "Previously created relative evapotranspiration raster found: ret (Adequacy PAI)"
            )

        # create and output applicable visuaisations and the field stats
        ret_field_stats = self.create_pai_csv_and_plots(
            input_raster_path=relative_evapotranspiration_raster_path,
            mask_raster_path=mask_raster_path,
            file_description="ret",
            title="relative evapotranspiration (adequacy indicator)",
            fields_vector_path=fields_vector_path,
            z_label="mean_ret",
            z_column="mean_ret",
            field_stats=field_stats,
            aoi_name=aoi_name,
            id_key=id_key,
            zmin=0,
            zmax=1,
            period_start=self.period_start,
            period_end=self.period_end,
            output_static_map=output_static_map,
            output_interactive_map=output_interactive_map,
            output_csv=output_csv,
        )

        return (relative_evapotranspiration_raster_path, ret_field_stats)

    ########################################################
    def calc_temporal_variation_of_relative_evapotranspiration(
        self,
        mask_raster_path: str,
        fields_vector_path: str,
        aoi_name: str,
        period_start: datetime = None,
        period_end: datetime = None,
        return_period: str = None,
        percentile: int = 95,
        output_nodata: float = -9999,
        field_stats: list = ["mean"],
        id_key: str = "wpid",
    ):
        """
        Description:
            calculate the relative evapotranspiration score per dekad for the
            given period to test for reliability
            per cell for the given period and area as
            defined by the class shapefile

            temporal relative evapotranspiration:
                Sum of Evapotranspiration / Potential evapotranspiration per return period as a time series

            NOTE: Field shapefile is required
            NOTE: dekad is the recommended return period

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for
            autoset to dekad
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            percentile: percentile of evapotranspiration values to choose as the
            potential evapotranspiration value
            output_nodata: nodata value to use on output
            fields_vector_path: path to the fields shapefile used to calcuate field level statistics
            to make the time series graph
            field_stats: list of statistics to carry out during the field level analysis,
            also used in the column names
            id_key: name of shapefile column/feature dictionary key providing the feature indices
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later)

        Return:
            tuple: path to the temporal variation in relative evapotranspiration raster, dict of field statistics
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        # create standardised temporal relative evapotranspiration file name
        temporal_relative_evapotranspiration_vrt_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="tret",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="results",
            aoi_name=aoi_name,
            ext=".vrt",
        )

        # if true output the results to a csv and use it to create an interactive map if applicable
        tret_csv_filepath = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="tret",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="results",
            aoi_name=aoi_name,
            ext=".csv",
        )

        if not any(
            os.path.exists(file)
            for file in [
                temporal_relative_evapotranspiration_vrt_path,
                tret_csv_filepath,
            ]
        ):
            # retrieve all available dekadal data for the period

            # setup download to be carried out per year depending on the requested dates
            date_tuples = self.wapor_organise_request_dates_per_year(
                period_start=self.period_start,
                period_end=self.period_end,
                return_period=self.return_period,
            )

            # setup download variables
            num_of_downloads = len(date_tuples)
            current_download = 1

            logger.info(
                f"attempting to download raster data for {num_of_downloads} periods"
            )

            retrieval_info_list = []
            for dt in date_tuples:
                logger.info(
                    f"downloading raster download info for time period: {dt[0]} to {dt[1]}, period {current_download} out of {num_of_downloads}"
                )

                # retrieve the download info
                retrieval_info = self.retrieve_wapor_download_info(
                    datacomponents=["AETI"],
                    period_start=dt[0],
                    period_end=dt[1],
                    return_period=self.return_period,
                    aoi_name=aoi_name,
                )

                retrieval_info_list.extend(retrieval_info)

                current_download += 1

            logger.info(
                f"calculating {len(retrieval_info)} sets of relative evapotranspiration"
            )

            ret_rasters_list = []
            aeti_rasters_list = []
            field_stats_dicts = []
            date_list = []
            period_steps = 1

            for retrieval_dict in retrieval_info_list:
                # create standardised relative evapotranspiration file name
                relative_evapotranspiration_raster_path = (
                    self.generate_output_file_path(
                        wapor_level=self.wapor_level,
                        description="ret",
                        period_start=retrieval_dict["period_start"],
                        period_end=retrieval_dict["period_end"],
                        output_folder="analysis",
                        aoi_name=aoi_name,
                        ext=".tif",
                    )
                )

                # retrieve and calculate periodic evapotranspiration for the given period
                # download the rasters
                retrieved_data = self.retrieve_actual_wapor_rasters(
                    wapor_download_list=[retrieval_dict],
                    template_raster_path=mask_raster_path,
                    aoi_name=aoi_name,
                )

                aeti_rasters_list.append(retrieved_data["AETI"]["raster_list"][0])

                date_list.append(
                    str(retrieval_dict["period_start"].strftime("%Y-%m-%d"))
                )

            # calculate the max actual evapotranspiration scored throughout the period using the specified numpy statistical function
            highest_actual_evapotranspiration = self.generate_output_file_path(
                wapor_level=self.wapor_level,
                description="max_aeti",
                period_start=self.period_start,
                period_end=self.period_end,
                output_folder="analysis",
                aoi_name=aoi_name,
                ext=".tif",
            )

            statistics.calc_multiple_array_numpy_statistic(
                input=aeti_rasters_list,
                numpy_function=np.nanmax,
                template_raster_path=mask_raster_path,
                output_raster_path=highest_actual_evapotranspiration,
                axis=0,
                output_nodata=output_nodata,
                mask_to_template=True,
            )

            # calculate potential evapotranspiration for the given period
            potential_evapotranspiration = self.calc_potential_raster(
                input_raster_path=highest_actual_evapotranspiration,
                percentile=percentile,
                mask_raster_path=mask_raster_path,
                output_nodata=output_nodata,
            )

            for aeti_raster, date_string in zip(aeti_rasters_list, date_list):
                # calculate relative evapotranspiration for the given period (AETI/POTET)
                statistics.calc_dual_array_statistics(
                    a=aeti_raster,
                    b=potential_evapotranspiration,
                    calc_function=statistics.ceiling_divide,
                    output_raster_path=relative_evapotranspiration_raster_path,
                    template_raster_path=mask_raster_path,
                    output_nodata=output_nodata,
                    mask_to_template=True,
                )

                ret_rasters_list.append(relative_evapotranspiration_raster_path)

                temporary_field_stats_dict = statistics.calc_field_statistics(
                    fields_shapefile_path=fields_vector_path,
                    input_rasters=[relative_evapotranspiration_raster_path],
                    field_stats=field_stats,
                    statistic_name=f"ret_{date_string}",
                    id_key=id_key,
                    out_dict=True,
                )

                field_stats_dicts.append(temporary_field_stats_dict)

                period_steps += 1

            logger.info(
                "periodic relative evapotranspiration raster(s) calculated for measuring temporal variation"
            )

            raster.build_vrt(
                raster_list=ret_rasters_list,
                output_vrt_path=temporal_relative_evapotranspiration_vrt_path,
                action="time",
            )

            # merge dictionary of dictionaries using id_key as the join key
            combined_dict = field_stats_dicts[0].copy()
            for base_dict_entry in combined_dict.keys():
                for _dict in field_stats_dicts:
                    for dict_entry in _dict.keys():
                        if (
                            _dict[dict_entry][id_key]
                            == combined_dict[base_dict_entry][id_key]
                        ):
                            for key in list(_dict[dict_entry].keys()):
                                if key in combined_dict[base_dict_entry].keys():
                                    pass
                                else:
                                    combined_dict[base_dict_entry][key] = _dict[
                                        dict_entry
                                    ][key]

            # reformat for scatterplot with x y z variables
            plotting_dict = {}
            counter = 1
            for id_ in list(combined_dict.keys()):  # id keys
                for date_string in date_list:
                    plotting_dict[counter] = {
                        "wpid": str(id_),
                        "time_step": date_string,
                        "mean_ret": combined_dict[id_][f"mean_ret_{date_string}"],
                    }
                    counter += 1

            statistics.output_table(plotting_dict, output_file_path=tret_csv_filepath)

            logger.info(
                f"Temporal variation in relative evapotranspiration outputted to csv: {tret_csv_filepath}"
            )

        else:
            logger.info(
                "Previously created periodic relative evapotranspiration raster(s) found"
            )

        scatterplot_html = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="tret",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="images",
            aoi_name=aoi_name,
            ext=".html",
        )

        scatterplot_png = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="tret",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="images",
            aoi_name=aoi_name,
            ext=".png",
        )

        scatterplot(
            input_table=tret_csv_filepath,
            x="time_step",
            y="mean_ret",
            title=f"Temporal variation in relative evapotranspiration: {period_start.strftime('%Y-%m-%d')} - {period_end.strftime('%Y-%m-%d')}",
            x_label=f"time_steps: {self.return_period}",
            y_label="mean_ret",
            color="wpid",
            output_html_path=scatterplot_html,
            output_png_path=scatterplot_png,
        )

        logger.info(
            f"Temporal variation in relative scatterplot made and outputted to html: {scatterplot_html}"
        )
        logger.info(
            f"Temporal variation in relative scatterplot made and outputted to png: {scatterplot_png}"
        )

        return (temporal_relative_evapotranspiration_vrt_path, None)

    ##########################
    def calc_crop_water_deficit(
        self,
        mask_raster_path: str,
        aoi_name: str,
        period_start: datetime = None,
        period_end: datetime = None,
        return_period: str = None,
        percentile: int = 95,
        output_nodata: float = -9999,
        fields_vector_path: str = None,
        field_stats: list = ["mean"],
        id_key: str = "wpid",
        output_static_map: bool = True,
        output_interactive_map: bool = True,
        output_csv: bool = True,
    ):
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
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            percentile: percentile of evapotranspiration values to choose as the
            potential evapotranspiration value
            output_nodata: nodata value to use on output
            fields_vector_path: if the path to the fields vector_path is provided
            then the field level statistics are also calculated
            field_stats: list of statistics to carry out during the field level analysis,
            also used in the column names
            id_key: name of shapefile column/feature dictionary key providing the feature indices
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later)
            output_static_map: if true outputs a static map
            output_interactive_map: if true outputs a interactive map
            output_csv:if true outputs a csv and shape plot file to a standardised location

        Return:
            tuple: path to the crop_water_deficit raster,  dict of field stats
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        # create standardised crop_water_deficit file name
        crop_water_deficit_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="cwd",
            period_start=self.period_start,
            period_end=period_end,
            output_folder="results",
            aoi_name=aoi_name,
            ext=".tif",
        )

        if not os.path.exists(crop_water_deficit_raster_path):
            # retrieve and calculate sum of evapotranspiration for the given period
            evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                period_start=self.period_start,
                period_end=self.period_end,
                datacomponent="AETI",
                numpy_function=np.nansum,
                mask_raster_path=mask_raster_path,
                aoi_name=aoi_name,
                statistic="sum",
                return_period=self.return_period,
                output_nodata=output_nodata,
            )

            # calculate potential evapotranspiration for the given period
            potential_evapotranspiration = self.calc_potential_raster(
                input_raster_path=evapotranspiration,
                percentile=percentile,
                mask_raster_path=mask_raster_path,
                output_nodata=output_nodata,
            )

            # calculate crop_water_deficit for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics(
                a=potential_evapotranspiration,
                b=evapotranspiration,
                calc_function=statistics.floor_minus,
                output_raster_path=crop_water_deficit_raster_path,
                template_raster_path=mask_raster_path,
                output_nodata=output_nodata,
                mask_to_template=True,
            )

            logger.info("Crop water deficit raster calculated: cwd (Adequacy PAI)")

        else:
            logger.info(
                "Previously created crop water deficit raster found: cwd (Adequacy PAI)"
            )

        # create and output applicable visuaisations and the field stats
        cwd_field_stats = self.create_pai_csv_and_plots(
            input_raster_path=crop_water_deficit_raster_path,
            mask_raster_path=mask_raster_path,
            file_description="cwd",
            title="crop water deficit (adequacy indicator)",
            fields_vector_path=fields_vector_path,
            z_label="mean_cwd",
            z_column="mean_cwd",
            field_stats=field_stats,
            aoi_name=aoi_name,
            id_key=id_key,
            period_start=self.period_start,
            period_end=self.period_end,
            output_static_map=output_static_map,
            output_interactive_map=output_interactive_map,
            output_csv=output_csv,
        )

        return (crop_water_deficit_raster_path, cwd_field_stats)

    ##########################
    def calc_beneficial_fraction(
        self,
        mask_raster_path: str,
        aoi_name: str,
        period_start: datetime = None,
        period_end: datetime = None,
        return_period: str = None,
        output_nodata: float = -9999,
        fields_vector_path: str = None,
        field_stats: list = ["mean"],
        id_key: str = "wpid",
        output_static_map: bool = True,
        output_interactive_map: bool = True,
        output_csv: bool = True,
    ):
        """
        Description:
            calculate an beneficial fraction score per cell to test for effeciency
            for the given period and area as defined by the class shapefile

            beneficial fraction: Sum of Transpiration / Sum of Evapotranspiration

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for,
            auto set to monthly
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            output_nodata: nodata value to use on output
            fields_vector_path: if the path to the fields vector_path is provided
            then the field level statistics are also calculated
            field_stats: list of statistics to carry out during the field level analysis,
            also used in the column names
            id_key: name of shapefile column/feature dictionary key providing the feature indices
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later)
            output_static_map: if true outputs a static map
            output_interactive_map: if true outputs a interactive map
            output_csv:if true outputs a csv and shape plot file to a standardised location



        Return:
            tuple: path to the beneficial fraction raster, (dataframe/dict, csv of field statistics)
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        # create standardised beneficial fraction file name
        beneficial_fraction_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="bf",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="results",
            aoi_name=aoi_name,
            ext=".tif",
        )

        if not os.path.exists(beneficial_fraction_raster_path):
            # retrieve and calculate average of evapotranspiration for the given period
            sum_evapotranspiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                period_start=self.period_start,
                period_end=self.period_end,
                datacomponent="AETI",
                numpy_function=np.nansum,
                mask_raster_path=mask_raster_path,
                aoi_name=aoi_name,
                statistic="sum",
                return_period=self.return_period,
                output_nodata=output_nodata,
            )

            # retrieve and calculate average of evapotranspiration for the given period
            sum_transpiration = self.retrieve_and_analyse_period_of_wapor_rasters(
                period_start=self.period_start,
                period_end=self.period_end,
                datacomponent="T",
                numpy_function=np.nansum,
                mask_raster_path=mask_raster_path,
                aoi_name=aoi_name,
                statistic="sum",
                return_period=self.return_period,
                output_nodata=output_nodata,
            )

            # calculate beneficial_fraction for the given period (AETI/POTET)
            statistics.calc_dual_array_statistics(
                a=sum_transpiration,
                b=sum_evapotranspiration,
                calc_function=statistics.ceiling_divide,
                output_raster_path=beneficial_fraction_raster_path,
                template_raster_path=mask_raster_path,
                output_nodata=output_nodata,
                mask_to_template=True,
            )

            logger.info("Beneficial fraction raster calculated: bf (Effeciency PAI)")

        else:
            logger.info(
                "Previously created beneficial fraction raster found: bf (Effeciency PAI)"
            )

        # create and output applicable visuaisations and the field stats
        bf_field_stats = self.create_pai_csv_and_plots(
            input_raster_path=beneficial_fraction_raster_path,
            mask_raster_path=mask_raster_path,
            file_description="bf",
            title="beneficial fraction (effeciency indicator)",
            fields_vector_path=fields_vector_path,
            z_label="mean_bf",
            z_column="mean_bf",
            field_stats=field_stats,
            aoi_name=aoi_name,
            id_key=id_key,
            zmin=0,
            zmax=1,
            period_start=self.period_start,
            period_end=self.period_end,
            output_static_map=output_static_map,
            output_interactive_map=output_interactive_map,
            output_csv=output_csv,
        )

        return (beneficial_fraction_raster_path, bf_field_stats)

    ##########################
    def calc_coefficient_of_variation(
        self,
        mask_raster_path: str,
        aoi_name: str,
        fields_vector_path: str,
        field_stats: list = ["mean", "stddev"],
        id_key: str = "wpid",
        period_start: datetime = None,
        period_end: datetime = None,
        return_period: str = None,
        output_nodata: float = -9999,
        output_interactive_map: bool = True,
    ):
        """
        Description:
            calculate a coefficient of variation score per field for the given period
            testing for equity in the area as defined by the class shapefile

            NOTE: cov is a special perfromance indicator in that there is no raster equivalent

            equity: standard deviation of summed Evapotranspiration per field /
            mean of summed evapotranspiration per field

        Args:
            self: (see class for details)
            period_start: start of the season in datetime
            period_end: end of the season in datetime
            return_period: return period to retrieve data for,
            auto set to monthly
            mask_raster_path: path to the raster mask defining the area for analysis if provided
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            percentile: percentile of evapotranspiration values to choose as the
            potential evapotranspiration value
            output_nodata: nodata value to use on output
            fields_vector_path: required as cov is a field based statistic, used
            to calculate the field level statistics are also calculated
            field_stats: list of statistics to carry out during the field level analysis,
            also used in the column names
            id_key: name of shapefile column/feature dictionary key providing the feature indices
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later)
            output_interactive_map: if true outputs a interactive map

        Return:
            tuple: None ,  dict
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        # retrieve and calculate average of evapotranspiration for the given period
        sum_evapotranspiration_raster_path = (
            self.retrieve_and_analyse_period_of_wapor_rasters(
                period_start=self.period_start,
                period_end=self.period_end,
                datacomponent="AETI",
                numpy_function=np.nansum,
                mask_raster_path=mask_raster_path,
                aoi_name=aoi_name,
                statistic="sum",
                return_period=self.return_period,
                output_nodata=output_nodata,
            )
        )

        logger.info("Calculating cov field statistics...")

        for stat in ["mean", "stddev"]:
            if stat not in field_stats:
                field_stats.append(stat)

        cov_field_stats = statistics.calc_field_statistics(
            fields_shapefile_path=fields_vector_path,
            input_rasters=[sum_evapotranspiration_raster_path],
            field_stats=field_stats,
            statistic_name="cov",
            id_key=id_key,
            out_dict=True,
        )

        # calculate Coefficient of Variation
        for key in cov_field_stats.keys():
            mean_keys = [
                key for key in list(cov_field_stats[key].keys()) if "mean" in key
            ]
            if len(mean_keys) > 1:
                raise AttributeError("should not be more than one mean calculated")
            else:
                mean_key = mean_keys[0]
            stddev_keys = [
                key for key in list(cov_field_stats[key].keys()) if "stddev" in key
            ]
            if len(stddev_keys) > 1:
                raise AttributeError("should not be more than one stddev calculated")
            else:
                stddev_key = stddev_keys[0]

            if cov_field_stats[key][mean_key] == 0:
                cov_field_stats[key]["cov"] = np.nan
            else:
                cov_field_stats[key]["cov"] = (
                    cov_field_stats[key][stddev_key] / cov_field_stats[key][mean_key]
                )

        csv_filepath = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="cov",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="results",
            aoi_name=aoi_name,
            ext=".csv",
        )

        statistics.output_table(cov_field_stats, output_file_path=csv_filepath)

        logger.info(f"Coefficient of Variation csv made: {csv_filepath}")

        shapeplot_filepath = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="cov_fields",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="images",
            aoi_name=aoi_name,
            ext=".png",
        )

        shapeplot(
            input_shape_path=fields_vector_path,
            output_plot_path=shapeplot_filepath,
            title="Coefficient of Variation (Equity PAI)",
            z_column="cov",
            zmin=0,
            zmax=0.25,
            input_table=csv_filepath,
            join_column=id_key,
        )

        logger.info(
            f"Coefficient of Variation (Equity PAI) field plot made: {shapeplot_filepath}"
        )

        if output_interactive_map:
            # if true create and output an interactive map
            shapeplot_html = self.generate_output_file_path(
                wapor_level=self.wapor_level,
                description="cov_fields",
                period_start=self.period_start,
                period_end=self.period_end,
                output_folder="images",
                aoi_name=aoi_name,
                ext=".html",
            )

            interactive_choropleth_map(
                input_shapefile_path=fields_vector_path,
                input_table=csv_filepath,
                z_column="cov",
                z_label="cov",
                zmin=0,
                zmax=0.25,
                output_html_path=shapeplot_html,
            )

            logger.info(
                f"Coefficient of Variation (Equity PAI) interactive plot made: {shapeplot_html}"
            )

        return (None, cov_field_stats)

    ##########################
    def calc_wapor_performance_indicators(
        self,
        mask_raster_path: str,
        fields_vector_path: str,
        aoi_name: str,
        period_start: datetime = None,
        period_end: datetime = None,
        return_period: str = None,
        output_nodata: float = -9999,
        id_key: str = "wpid",
        output_static_map: bool = True,
        output_interactive_map: bool = True,
        output_csv: bool = True,
    ):
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
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            output_nodata: nodata value to use on output
            fields_vector_path: if the path to the fields vector_path is provided
            then the field level statistics are also calculated
            field_stats: list of statistics to carry out during the field level analysis,
            also used in the column names
            id_key: name of shapefile column/feature dictionary key providing the feature indices
            wpid is a reliable autogenerated index provided while making the crop mask
            (note: also handy for joining tables and the crop mask shape/other shapes back later)
            output_static_map: if true outputs a static map
            output_interactive_map: if true outputs a interactive map
            output_csv:if true outputs a csv and shape plot file to a standardised location

        Return:
            tuple: list of paths to the performance indicator rasters,  dict of combined pai field statistics
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        pai_rasters = []

        pai_dicts = []

        # calculate beneficial fraction
        bf_outputs = self.calc_beneficial_fraction(
            mask_raster_path=mask_raster_path,
            fields_vector_path=fields_vector_path,
            aoi_name=aoi_name,
            period_start=self.period_start,
            period_end=self.period_end,
            return_period=self.return_period,
            output_nodata=output_nodata,
            id_key=id_key,
            output_static_map=output_static_map,
            output_interactive_map=output_interactive_map,
            output_csv=output_csv,
        )

        pai_rasters.append(bf_outputs[0])
        pai_dicts.append(bf_outputs[1])

        # calculate crop water deficit
        cwd_outputs = self.calc_crop_water_deficit(
            mask_raster_path=mask_raster_path,
            fields_vector_path=fields_vector_path,
            aoi_name=aoi_name,
            period_start=self.period_start,
            period_end=self.period_end,
            return_period=self.return_period,
            output_nodata=output_nodata,
            id_key=id_key,
            output_static_map=output_static_map,
            output_interactive_map=output_interactive_map,
            output_csv=output_csv,
        )

        pai_rasters.append(cwd_outputs[0])
        pai_dicts.append(cwd_outputs[1])

        # calculate relative evapotranspiration
        ret_outputs = self.calc_relative_evapotranspiration(
            mask_raster_path=mask_raster_path,
            fields_vector_path=fields_vector_path,
            aoi_name=aoi_name,
            period_start=self.period_start,
            period_end=self.period_end,
            return_period=self.return_period,
            output_nodata=output_nodata,
            id_key=id_key,
            output_static_map=output_static_map,
            output_interactive_map=output_interactive_map,
            output_csv=output_csv,
        )

        pai_rasters.append(ret_outputs[0])
        pai_dicts.append(ret_outputs[1])

        # calculate temporal_variation_of_relative_evapotranspiration
        tret_outputs = self.calc_temporal_variation_of_relative_evapotranspiration(
            mask_raster_path=mask_raster_path,
            fields_vector_path=fields_vector_path,
            aoi_name=aoi_name,
            period_start=self.period_start,
            period_end=self.period_end,
            return_period=self.return_period,
            output_nodata=output_nodata,
            id_key=id_key,
        )

        pai_rasters.append(tret_outputs[0])

        logger.info("All raster based PAIs calculated")

        # calculate coefficient of variation
        cov_dict = self.calc_coefficient_of_variation(
            mask_raster_path=mask_raster_path,
            fields_vector_path=fields_vector_path,
            aoi_name=aoi_name,
            period_start=self.period_start,
            period_end=self.period_end,
            return_period=self.return_period,
            output_nodata=output_nodata,
            id_key=id_key,
        )[1]

        pai_dicts.append(cov_dict)

        logger.info(
            "combining and exporting all PAI field statistics to a single csv and shapefile..."
        )
        # create standardised performance indicator stats file name
        pai_csv_filepath = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="pai",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="results",
            aoi_name=aoi_name,
            ext=".csv",
        )

        # merge dictionary of dictionaries using id_key as the join key
        pai_dict = pai_dicts[0].copy()
        for base_dict_entry in pai_dict.keys():
            for _dict in pai_dicts:
                for dict_entry in _dict.keys():
                    if _dict[dict_entry][id_key] == pai_dict[base_dict_entry][id_key]:
                        for key in list(_dict[dict_entry].keys()):
                            if key in pai_dict[base_dict_entry].keys():
                                pass
                            else:
                                pai_dict[base_dict_entry][key] = _dict[dict_entry][key]

        if output_csv:
            statistics.output_table(pai_dict, output_file_path=pai_csv_filepath)

            logger.info(
                f"Performance indicator field stats calculated and outputted to csv: {pai_csv_filepath}"
            )

        pai_shapefile_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description="pai",
            period_start=self.period_start,
            period_end=self.period_end,
            output_folder="results",
            aoi_name=aoi_name,
            ext=".shp",
        )

        vector.records_to_vector(
            field_records=pai_dict,
            output_vector_path=pai_shapefile_path,
            fields_vector_path=fields_vector_path,
            union_key="wpid",
        )

        logger.info(
            f"Performance indicator field stats outputted to shapefile: {pai_shapefile_path}"
        )

        return pai_rasters, pai_dict
