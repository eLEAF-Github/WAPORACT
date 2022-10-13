"""
waporact package

retrieval class (stand alone/support class and functions)

script for the retrieval of WAPOR data utilising the class WAPORAPI from the package WAPOROCW made by ITC DELFT
"""
##########################
# import packages
import os
import sys
from datetime import datetime, timedelta
from timeit import default_timer
import time

from typing import Union

import numpy as np
import pandas as pd
from ast import literal_eval
import requests

from waporact.scripts.retrieval.wapor_api import WaporAPI
from waporact.scripts.structure.wapor_structure import WaporStructure
from waporact.scripts.retrieval.wapor_retrieval_support import (
    check_datacomponent_availability,
    generate_wapor_cube_code,
)
from waporact.scripts.retrieval.wapor_retrieval_masking import (
    check_bbox_overlaps_l3_location,
    create_raster_mask_from_shapefile_and_template_raster,
    create_raster_mask_from_wapor_landcover_rasters,
)

from waporact.scripts.tools import raster, vector, statistics

import logging

from waporact.scripts.tools.logger import format_root_logger


logger = logging.getLogger(__name__)

#################################
# stand alone functions
#################################
def printWaitBar(i, total, prefix="", suffix="", decimals=1, length=100, fill="█"):
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
    if os.name == "posix" and total == 0:
        total = 0.0001

    percent = ("{0:." + str(decimals) + "f}").format(100 * (i / float(total)))
    filled = int(length * i // total)
    bar = fill * filled + "-" * (length - filled)

    sys.stdout.write("\r%s |%s| %s%% %s " % (prefix, bar, percent, suffix))
    sys.stdout.flush()

    if i == total:
        print()


#################################
# retrieval class
#################################
class WaporRetrieval(WaporStructure):
    def __init__(
        self,
        waporact_directory: str,
        vector_path: str,
        wapor_level: int,
        period_start: datetime,
        period_end: datetime,
        api_token: str,
        country_code: str = "notyetset",
        return_period: str = "D",
        project_name: str = "test",
        wapor_version: int = 2,
        silent: bool = None,
        print_wait_bar: bool = True,
    ):
        """Retrieves rasters from the Wapor database given the appropriate inputs

         inherits/built on the WaporStructure class which is needed for setting
         class/ self parameters and the automated folder structure


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

         Raises
         ------
         vector_exception
             on failing to retrieve input vector file crs
         AttributeError
             return period given not foudn amongst available options
        """
        # set verbosity (feedback) parameter
        if silent == True:
            format_root_logger(logging_level=logging.WARNING)
        elif silent == False:
            format_root_logger(logging_level=logging.INFO)
        else:
            pass

        self.print_wait_bar = print_wait_bar

        self.api_token = api_token
        self.wapor_version = wapor_version

        # attach and setup the waporAPI class
        self.wapor_api = WaporAPI(version=self.wapor_version)

        # inherit and initialise the WaporStructure class
        super().__init__(
            waporact_directory=waporact_directory,
            project_name=project_name,
            wapor_level=wapor_level,
            period_end=period_end,
            period_start=period_start,
            return_period=return_period,
        )

        self.waporact_directory = waporact_directory
        self.project_name = project_name
        self.wapor_level = wapor_level

        self.catalogue = ""

        self.country_code = country_code

        # set vector file parameters
        bbox_geojson_name = (
            os.path.splitext(os.path.basename(vector_path))[0] + "_bbox.geojson"
        )
        bbox_geojson_path = os.path.join(self.project["reference"], bbox_geojson_name)

        # set and generate shapefile parameters
        self.bbox = vector.retrieve_vector_file_bbox(
            input_vector_path=vector_path,
            output_bbox_vector_file=bbox_geojson_path,
            output_crs=4326,  # coordinate system expected by api
        )

        try:
            logger.info(
                "retrieving the input vector file crs and setting it to the class instance: output_crs"
            )
            self.output_crs = vector.retrieve_vector_crs(vector_path)
        except Exception as vector_exception:
            logger.error(
                "failed to retrieve crs from the input vector, please check the error message and or your input vector and try again"
            )
            raise vector_exception

        # run basic data availability checks
        if self.return_period not in self.period_codes:
            raise AttributeError(
                f"given return period {self.return_period} not found amongst available options: {self.period_codes}"
            )

        if self.wapor_level == 3:
            l3_locations_vector_path = self.retrieve_level_3_availability_shapefile()

            # check if the bbox falls within an available level 3 area:
            self.country_code = check_bbox_overlaps_l3_location(
                bbox_vector_path=bbox_geojson_path,
                l3_locations_vector_path=l3_locations_vector_path,
            )

        logger.info("WaporRetrieval class initiated and ready for WaPOR data retrieval")

    #################################
    # properties
    #################################
    @property
    def wapor_level(self):
        return self._wapor_level

    @wapor_level.setter
    def wapor_level(self, value: int):
        """check if the wapor level is correct and resets the catalogue

        Parameters
        ----------
        value : int
            wapor level

        Raises
        ------
        AttributeError
            if the level is not one of available options
        """
        if value not in [1, 2, 3]:
            raise AttributeError("wapor_level (int) needs to be either 1, 2 or 3")

        self._wapor_level = value

        self.catalogue = ""

        self.project = WaporStructure.set_active_wapor_level_directory(
            waporact_directory=self.waporact_directory,
            project_directory=os.path.join(self.waporact_directory, self.project_name),
            level=value,
        )

    #################################
    @property
    def waporact_directory(self):
        return self._waporact_directory

    @waporact_directory.setter
    def waporact_directory(self, value: str):
        """sets the waporact main directory

        Parameters
        ----------
        value : str
            directory path

        Raises
        ------
        AttributeError
            if nothing is provided
        directory_exception
            if the direcotry fails to build
        AttributeError
            if value is not a str
        """
        if not value:
            raise AttributeError("please provide a waporact_directory")
        if isinstance(value, str):
            if not os.path.exists(value):
                try:
                    os.makedirs(value)
                except Exception as directory_exception:
                    logger.error(f"failed to make the waporact directory: {value}")
                    raise directory_exception

            self._waporact_directory = value

        else:
            raise AttributeError

    #################################
    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self, value: int):
        """
        Description
            sets the project name and checks it exists

        Args:
            value: project name

        Raise:
            AttributeError: If no name is provided or is not a string
        """
        if not isinstance(value, str):
            raise AttributeError("please provide a project name as str")
        else:
            if value == "test":
                logger.info("using standard project name test")

        self._project_name = value

    #################################
    @property
    def catalogue(self):
        return self._catalogue

    @catalogue.setter
    def catalogue(self, value: pd.DataFrame):
        """set the path to the catalogue and if needed retrieve the catalogue

        Parameters
        ----------
        value : pd.DataFrame
            path to the catalogue

        Raises
        ------
        AttributeError
            if wapor level is not set
        AttributeError
            if the waporact directory is not set
        """
        if isinstance(value, pd.DataFrame):
            self._catalogue = value

        else:
            if not hasattr(self, "wapor_level"):
                raise AttributeError(
                    "to set a cataloguae wapor level must first be set"
                )
            if not hasattr(self, "waporact_directory"):
                raise AttributeError(
                    "to set a catalogue waporact_directory must first be set"
                )

            catalogue_folder = os.path.join(self.waporact_directory, "metadata")
            if not os.path.exists(catalogue_folder):
                os.makedirs(catalogue_folder)

            catalogue_df = self.retrieve_and_store_catalogue(
                catalogue_output_folder=catalogue_folder,
                wapor_level=self.wapor_level,
                cube_info=True,
            )

            self._catalogue = catalogue_df

        self.components = tuple(set(list(catalogue_df["component_code"])))
        self.cube_codes = tuple(set(list(catalogue_df["code"])))
        self.period_codes = tuple(set(list(catalogue_df["period_code"])))

        if self.wapor_level == 3:
            self.country_codes = tuple(
                set(
                    [
                        catalogue_country_code
                        for catalogue_country_code, country_desc in zip(
                            catalogue_df["country_code"],
                            catalogue_df["country_desc"],
                        )
                    ]
                )
            )
        else:
            self.country_codes = ("notlevel3notused",)

    #################################
    # class functions
    #################################
    @classmethod
    def wapor_connection_attempts_dict(cls):
        """sets the dict for connection attempt variables

        Returns
        -------
        dict
            dict of variables for connecting
        """
        return {
            "connection_attempts": 0,
            "connection_sleep": 3,
            "connection_attempts_limit": 20,
        }

    #################################
    @classmethod
    def deconstruct_wapor_time_code(cls, time_code: str):
        """dsconstruct a wpaor time code

        Parameters
        ----------
        time_code : str
            wapor time code to decosntruct

        Returns
        -------
        dict
            deconstructed time code
        """
        wapor_time_dict = {}
        period_start_str, period_end_str = time_code.split(",")
        for i in ["[", "]", "(", ")", "-"]:
            period_start_str = period_start_str.replace(i, "")
            period_end_str = period_end_str.replace(i, "")

        wapor_time_dict["period_string"] = period_start_str + "_" + period_end_str
        wapor_time_dict["period_start"] = datetime.strptime(period_start_str, "%Y%m%d")
        wapor_time_dict["period_end"] = datetime.strptime(period_end_str, "%Y%m%d")

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
        delete_input: bool = False,
        output_nodata: float = -9999,
    ):
        """process raster retrieved from wapor url in a standardised way
        and store

        Parameters
        ----------
        input_raster_path : str
            url retrieved raster
        output_raster_path : str
            path to output the raster too
        wapor_multiplier : float
            wapor scaling value
        return_period : str
            return period of the raster
        period_start : datetime
             start of the period the raster covers
        period_end : datetime
             end of the period the raster covers
        delete_input : bool, optional
            delete the input raster/url object, by default False
        output_nodata : float, optional
            no data value to use for the output raster, by default -9999

        Returns
        -------
        int
            0
        """
        # process the downloaded raster and write to the processed folder
        if return_period == "dekadal":
            ### number of days
            ndays = (period_end.timestamp() - period_start.timestamp()) / 86400
        else:
            ndays = 1

        # correct raster with multiplier and number of days in dekad if applicable
        array = raster.raster_to_array(input_raster_path)
        array = np.where(array < 0, 0, array)  # mask out flagged value -9998
        corrected_array = array * wapor_multiplier * ndays

        raster.array_to_raster(
            output_raster_path=output_raster_path,
            metadata=input_raster_path,
            input_array=corrected_array,
            output_nodata=output_nodata,
        )

        raster.check_gdal_open(output_raster_path)
        if delete_input:
            os.remove(input_raster_path)

        return 0

    #################################
    # mask functions
    #################################
    def create_raster_mask_from_vector_file(
        self,
        mask_name: str,
        input_vector_path: str,
        template_raster_path: str,
    ):
        """create a raster mask from a vector file and a template raster
        that is retrieved from wapor

        Parameters
        ----------
        mask_name : str
            name for the output mask
        input_vector_path : str,
            vector file to rasterize
        template_raster_path : str,
            raster to take metadata from as template

        Returns
        -------
        tuple
            path of the mask raster outputted, path to mask vector file created
        """
        # store time parameters as temp variables used below
        save_return_period = self.return_period
        save_period_start = self.period_start
        save_period_end = self.period_end

        if not mask_name:
            base_name = os.path.splitext(os.path.basename(input_vector_path))[0]
            mask_name = base_name

        mask_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=mask_name,
            output_folder="reference",
            aoi_name=mask_name,
            ext=".tif",
        )

        mask_vector_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=mask_name,
            output_folder="reference",
            aoi_name=mask_name,
            ext=".geojson",
        )
        if not os.path.exists(mask_raster_path):
            (
                mask_raster_path,
                mask_vector_path,
            ) = create_raster_mask_from_shapefile_and_template_raster(
                input_vector_path=input_vector_path,
                template_raster_path=template_raster_path,
                mask_raster_path=mask_raster_path,
                mask_vector_path=mask_vector_path,
                output_crs=self.output_crs,
            )

        else:
            logger.info("preexisting raster mask found skipping step")
            raster.check_gdal_open(mask_raster_path)

        # restore time parameters
        self.return_period = save_return_period
        self.period_start = save_period_start
        self.period_end = save_period_end

        return mask_raster_path, mask_vector_path

    #################################
    def create_raster_mask_from_wapor_lcc(
        self,
        mask_name: str,
        lcc_categories: list,
        template_raster_path: str,
        period_start: datetime = None,
        period_end: datetime = None,
        area_threshold_multiplier: int = 1,
        output_nodata: float = -9999,
        output_crs: int = 4326,
    ):
        """create a raster mask from wapor landcover rasters

        Parameters
        ----------
        mask_name : str
            name for all the mask files
        lcc_categories: list
             crops/land classification categories to mask too
        template_raster_path : str
            tmeplate raster holding the metadata
        period_start : datetime, optional
            start of period to retrieve lcc for, by default None
        period_end : datetime, optional
            end of period to retrieve lcc for, by default None
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
        self.period_start = period_start
        self.period_end = period_end

        # store return period
        save_return_period = self.return_period

        # create the file paths (as the LLC retrieved can differ depending on date the mask is period specific)
        mask_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=mask_name,
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".tif",
        )

        mask_values_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{mask_name}-values",
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".tif",
        )

        mask_shape_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=mask_name,
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".shp",
        )

        raw_mask_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{mask_name}-raw",
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".tif",
        )

        raw_mask_values_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{mask_name}-raw-values",
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".tif",
        )

        raw_mask_shape_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{mask_name}-raw",
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".shp",
        )

        lcc_count_csv_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{mask_name}-lcc-count",
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".csv",
        )

        masked_lcc_count_csv_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{mask_name}-lcc-count-masked",
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".csv",
        )

        most_common_lcc_raster_path = self.generate_output_file_path(
            wapor_level=self.wapor_level,
            description=f"{mask_name}-lcc-median",
            output_folder="reference",
            period_start=period_start,
            period_end=period_end,
            aoi_name=mask_name,
            ext=".tif",
        )

        # check for and produce if needed the raw mask rasters
        if not os.path.exists(raw_mask_raster_path) or not os.path.exists(
            raw_mask_values_raster_path
        ):
            # check for and produce if needed the most common class raster
            # retrieve the lcc rasters and find the most common class per cell across the given period
            if not os.path.exists(most_common_lcc_raster_path):
                if self.wapor_level == 3:
                    rp = "D"
                else:
                    rp = "A"
                    if period_end - period_start < 365:
                        # adjust period start to make sure it retieves something
                        period_start = period_start - timedelta(days=365)

                # create the base mask based on the input shapefile first
                wapor_rasters = self.download_wapor_rasters(
                    datacomponents=["LCC"],
                    template_raster_path=template_raster_path,
                    return_period=rp,
                    period_start=self.period_start,
                    period_end=self.period_end,
                    output_nodata=output_nodata,
                    aoi_name=mask_name,
                )

                (
                    mask_raster_path,
                    mask_shape_path,
                ) = create_raster_mask_from_wapor_landcover_rasters(
                    wapor_level=self.wapor_level,
                    lcc_categories=lcc_categories,
                    wapor_landcover_rasters=wapor_rasters["LCC"]["raster_list"],
                    most_common_lcc_raster_path=most_common_lcc_raster_path,
                    lcc_count_csv_path=lcc_count_csv_path,
                    raw_mask_values_raster_path=raw_mask_values_raster_path,
                    raw_mask_raster_path=raw_mask_raster_path,
                    raw_mask_shape_path=raw_mask_shape_path,
                    mask_shape_path=mask_shape_path,
                    mask_raster_path=mask_raster_path,
                    mask_values_raster_path=mask_values_raster_path,
                    masked_lcc_count_csv_path=masked_lcc_count_csv_path,
                    output_crs=output_crs,
                    area_threshold_multiplier=area_threshold_multiplier,
                    output_nodata=output_nodata,
                )

        # restore return period
        self.return_period = save_return_period

        return mask_raster_path, mask_shape_path

    #################################
    # retrieval functions
    #################################
    def retrieve_and_store_catalogue(
        self, catalogue_output_folder: str, wapor_level: int, cube_info: bool = True
    ):
        """retrieve as a dataframe and store the wapor catalogue as a file

        Parameters
        ----------
        catalogue_output_folder : str
            locaiton to store the file
        wapor_level : int
            wapor level to retrieve for
        cube_info : bool, optional
            if true retrieve and store cube info, by default True

        Returns
        -------
        pandas.Dateframe
            catalogue as a dataframe

        Raises
        ------
        AttributeError
            if wpaor levle is not one fo avaialble options
        """
        if wapor_level not in [1, 2, 3]:
            raise AttributeError("wapor_level (int) needs to be either 1, 2 or 3")
        retrieve = False
        catalogue_csv = os.path.join(
            catalogue_output_folder, f"wapor_catalogue_L{wapor_level}.csv"
        )
        if not os.path.exists(catalogue_csv):
            retrieve = True
        else:
            st = os.stat(catalogue_csv)
            if (time.time() - st.st_mtime) >= 5184000:  # 60 days
                logger("found wapor level catalogue older than 60 days replacing")
                retrieve = True

        if retrieve:
            logger.info(
                f"No or Outdated WaPOR catalogue found for wapor_level: {wapor_level}, retrieving now this may take a min..."
            )

            catlogue_df = self.wapor_api.retrieve_catalogue_as_dataframe(
                wapor_level=wapor_level, cubeInfo=cube_info
            )

            statistics.output_table(
                table=catlogue_df, output_file_path=catalogue_csv, csv_seperator=";"
            )
            logger.info(
                f"outputted WaPOR level {wapor_level} catalogue to file for wapor_level: {catalogue_csv}"
            )

        else:
            catlogue_df = pd.read_csv(catalogue_csv, sep=";")
            catlogue_df["measure"] = catlogue_df["measure"].apply(
                lambda x: literal_eval(x)
            )
            catlogue_df["dimension"] = catlogue_df["dimension"].apply(
                lambda x: literal_eval(x)
            )

        logger.info(
            f"Loaded WaPOR catalogue for wapor_level {wapor_level} from {catalogue_csv}"
        )

        return catlogue_df

    #################################
    def retrieve_wapor_cube_info(
        self,
        cube_code: str,
    ):
        """retrieve the info related to a wapor cube

        Parameters
        ----------
        cube_code : str
            cube code to retrieve info for

        Returns
        -------
        dict
            dict of cube info
        """
        # reset connection attempts
        self.wapor_connection_attempts = WaporRetrieval.wapor_connection_attempts_dict()
        cube_info = None
        while cube_info is None:
            # attempt to retrieve cube_info
            try:
                cube_info = self.wapor_api.getCubeInfo(
                    cube_code=cube_code,
                    wapor_level=self.wapor_level,
                    catalogue=self.catalogue,
                )
            except Exception as cube_retrieval_error:
                self.wapor_retrieval_connection_error(
                    description="retrieving cube info", _exception=cube_retrieval_error
                )

        return cube_info

    #################################
    def retrieve_wapor_data_availability(self, cube_code: str, time_range: str):
        """wrapper for WaporAPI getAvailData that runs it
            multiple times in an attempt to force a connection
            and retrieve the cube info
            from the WAPOR database in a more robust fashion

        Parameters
        ----------
        cube_code : str
            cube code tor retrieve available data for
        time_range : str
            time range to check for data

        Returns
        -------
        pandas.DataFrame
            dataframe of data availibility
        """
        # reset connection attempts
        self.wapor_connection_attempts = WaporRetrieval.wapor_connection_attempts_dict()
        data_availability = None
        while data_availability is None:
            # attempt to retrieve cube_info
            try:
                data_availability = self.wapor_api.getAvailData(
                    cube_code=cube_code,
                    time_range=time_range,
                    wapor_level=self.wapor_level,
                    catalogue=self.catalogue,
                )

            except Exception as data_availability_retrieval_error:
                self.wapor_retrieval_connection_error(
                    description="retrieving data avialability info",
                    _exception=data_availability_retrieval_error,
                )

        return data_availability

    #################################
    def retrieve_level_3_availability_shapefile(self):
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
        logger.info(
            "creating wapor_level 3 locations shapefile, this may take a min ..."
        )
        # set temporary date variables
        api_per_start = datetime(2010, 1, 1).strftime("%Y-%m-%d")
        api_per_end = datetime.now().strftime("%Y-%m-%d")
        api_period = f"{api_per_start},{api_per_end}"

        l3_locations_vector_path = os.path.join(
            self.project["meta"], "wapor_L3_locations.geojson"
        )
        if not os.path.exists(l3_locations_vector_path):
            # retrieve country codes
            bboxes = []
            codes = []
            # loop through countries check data availability and retrieve the bbox
            for code in self.country_codes:
                cube_code = f"L3_{code}_T_D"

                df_avail = self.retrieve_wapor_data_availability(
                    cube_code=cube_code, time_range=api_period
                )

                bbox = df_avail.iloc[0]["bbox"][1]["value"]

                bboxes.append(
                    [
                        (bbox[0], bbox[3]),
                        (bbox[2], bbox[3]),
                        (bbox[2], bbox[1]),
                        (bbox[0], bbox[1]),
                    ]
                )
                codes.append(code)

            vector.write_vectors_to_file(
                vectors=bboxes,
                values=codes,
                output_vector_path=l3_locations_vector_path,
                crs=4326,
            )

            logger.info(f"wapor_level 3 location shapefile: {l3_locations_vector_path}")

        return l3_locations_vector_path

    #################################
    def retrieve_crop_raster_url(
        self,
        bbox: tuple,
        cube_code: str,
        time_code: str,
        raster_id: str,
        api_token: str,
    ):
        """wrapper for WaporAPI getCropRasterURL that runs it
            multiple times in an attempt to force a connection
            and retrieve the raster url from the WAPOR database
            in a more robust fashion

        Parameters
        ----------
        bbox : tuple
            bounding box of the corpped area as a string  [xmin,ymin,xmax,ymax]
        cube_code : str
            cube code defining the raster to retrieve
        wapor_level : int
            wapor level  defining the raster to retrieve
        time_code : str
            time code  defining the raster to retrieve
        rasterId : str
            raster id  defining the raster to retrieve
        APIToken : str
            api token use dto retrieve the access token used to retrieve the raster

        Returns
        -------
        str
            url to the cropped raster
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
                    catalogue=self.catalogue,
                )

            except Exception as retrieve_raster_url_error:
                self.wapor_retrieval_connection_error(
                    description="retrieving crop raster url",
                    _exception=retrieve_raster_url_error,
                )

        return url

    #################################
    def wapor_raster_request(self, wapor_url: str, output_file_path: str):
        """wrapper function that attempts to retrieve the raster
            from the wpaor database using a stanrd api request and stores it

        Parameters
        ----------
        wapor_url : str
            url to retrieve raster from
        output_file_path : str
            location to output th retrieved raster too

        Returns
        -------
        int
            0
        """
        # reset connection attempts
        self.wapor_connection_attempts = WaporRetrieval.wapor_connection_attempts_dict()
        # initiate retrieval variables
        wapor_raster_result = None
        while wapor_raster_result is None:
            try:
                ### attempt to retrieve the download url
                wapor_raster_result = requests.get(wapor_url)

            except Exception as raster_retrieval_error:
                self.wapor_retrieval_connection_error(
                    description="retrieving crop raster url",
                    _exception=raster_retrieval_error,
                )

        open(output_file_path, "wb").write(wapor_raster_result.content)

        return 0

    #################################
    def wapor_retrieval_connection_error(
        self,
        description: str,
        _exception: Exception,
    ):
        """used if a connection error occurs, keeps track of the amount of connection/request attempts
            and increases the count on running the function.
            and if the limit of attempts is reached while running
            raises an error

        Parameters
        ----------
        description : str
            description of the error
        _exception : Exception
            excpetion captured

        Returns
        -------
        int
            0

        Raises
        ------
        ConnectionError
            if a RequestException is provided and connection could not be made
        TimeoutError
            if an unknown exception is provided
        """
        time.sleep(self.wapor_connection_attempts["connection_sleep"])
        self.wapor_connection_attempts["connection_attempts"] += 1
        if self.wapor_connection_attempts["connection_attempts"] == abs(
            self.wapor_connection_attempts["connection_attempts_limit"] / 2
        ):
            logger.warning(
                f"{description} failed,  {self.wapor_connection_attempts['connection_attempts']} connection errors noted, will continue connection attempts"
            )

        if (
            self.wapor_connection_attempts["connection_attempts"]
            >= self.wapor_connection_attempts["connection_attempts_limit"]
        ):
            if isinstance(_exception, requests.exceptions.RequestException):
                error_statement = (
                    f"{description} from WAPOR attempted to request data {self.wapor_connection_attempts['connection_attempts']} "
                    f"times every {self.wapor_connection_attempts['connection_sleep']} sec and failed due to request/connection error, "
                    "adjust the self.wapor_connection_attempts or sleep time to try for longer, there may also be no data available "
                    "for your combination of return period, period_start, period_end, and datacomponent"
                )
                raise ConnectionError(error_statement)

            else:
                error_statement = (
                    f"{description} from WAPOR attempted to request data {self.wapor_connection_attempts['connection_attempts']} "
                    f"times every {self.wapor_connection_attempts['connection_sleep']} sec and failed due to unknown error, "
                    "adjust the self.wapor_connection_attempts or sleep time to try for longer, there may also be no data available "
                    "for your combination of return period, period_start, period_end, and datacomponent"
                )

                raise TimeoutError(error_statement)

        return 0

    #################################
    def retrieve_wapor_download_info(
        self,
        datacomponents: list,
        period_start: datetime,
        period_end: datetime,
        return_period: str,
        aoi_name: str = "nomask",
    ):
        """WAPOR download works in two phases. the retrieval and setup of the download info and
        retrieval of the crop raster url and the actual download of that raster and preprocessing of it.
        this subfunction carries out the setup of the download info.


            NOTE: This is actually the longest part of the download process as generating the download url
            is a slow process

            NOTE: works in combination with retrieve_actual_wapor_rasters

            NOTE: aoi_name (mask name) if supplied should match that in retrieve_actual_wapor_rasters

            NOTE: does not use clas sinstance period_start, period_end or return period as it assumes
            these inputs are provided by download_wapor_rasters

        Parameters
        ----------
        datacomponents : list
            list of datacomponents to retrieve for
        period_start : datetime
            start of period to retrieve for
        period_end : datetime
            end of period to retrieve for
        return_period : str
            return period to retrieve for
        aoi_name : str, optional
            area of interest (aoi) name to use for the mask folder auto set to nomask if not provided, by default "nomask"

        Returns
        -------
        list
            list of wapor download dicts one per raster

        Raises
        ------
        TypeError
            if datacomponents is not a list
        AttributeError
            if country code not provided for wapor level 3
        AttributeError
            if given country code does not exist among existing ones
        """
        self.return_period = return_period

        assert isinstance(
            period_start, datetime
        ), "period_start must be a datetime object"
        assert isinstance(period_end, datetime), "period_end must be a datetime object"

        if not isinstance(datacomponents, list):
            raise TypeError(
                "datacomponents provided should be formatted as a list of strings"
            )

        if self.wapor_level == 3:
            if not self.country_code:
                raise AttributeError("country code needed if retrieving level 3 data")
            else:
                if self.country_code not in self.country_codes:
                    raise AttributeError(
                        "given country code does not found among existing ones"
                    )

        # generate dates for filenames
        dates_dict = WaporStructure.generate_dates_dict(
            period_start=period_start,
            period_end=period_end,
            return_period=self.return_period,
        )

        # setup output list
        wapor_download_list = []

        # check if the datacomponents are available
        datacomponents = check_datacomponent_availability(
            datacomponents=datacomponents,
            return_period=self.return_period,
            all_datacomponents=self.components,
            wapor_level=self.wapor_level,
            cube_codes=self.cube_codes,
            country_code=self.country_code,
        )

        # retrieve download info per available datacomponent
        for component in datacomponents:
            if self.wapor_level == 3:
                logger.info(
                    f"retrieving download info for wapor_level 3 region: {self.country_code}"
                )
            logger.info(f"retrieving download info for component: {component}")

            wapor_dict = dict()

            # construct the wapor_cube_code
            cube_code = generate_wapor_cube_code(
                datacomponent=component,
                return_period=self.return_period,
                wapor_level=self.wapor_level,
                cube_codes=self.cube_codes,
                country_code=self.country_code,
            )

            # attempt to retrieve cube code info
            cube_info = self.retrieve_wapor_cube_info(cube_code=cube_code)
            multiplier = cube_info["measure"]["multiplier"]

            # attempt to download data availability from WAPOR
            df_avail = self.retrieve_wapor_data_availability(
                cube_code=cube_code, time_range=dates_dict["api_period"]
            )

            logger.info(
                f"attempting to retrieve download info for {len(df_avail)} rasters from wapor"
            )

            # set up waitbar
            count = 0
            total_count = len(df_avail)

            if self.print_wait_bar:
                printWaitBar(
                    i=count,
                    total=total_count,
                    prefix="Retrieving Raster Urls:",
                    suffix=f"Complete: {count} out of {total_count}",
                    length=50,
                )

            # retrieve data
            for __, row in df_avail.iterrows():
                wapor_time_dict = WaporRetrieval.deconstruct_wapor_time_code(
                    time_code=row["time_code"]
                )

                # construct  wapor download dict
                wapor_dict = {}
                wapor_dict["component"] = component
                wapor_dict["cube_code"] = cube_code
                wapor_dict["period_string"] = wapor_time_dict["period_string"]
                wapor_dict["period_start"] = wapor_time_dict["period_start"]
                wapor_dict["period_end"] = wapor_time_dict["period_end"]
                wapor_dict["return_period"] = self.return_period
                wapor_dict["raster_id"] = row["raster_id"]
                wapor_dict["multiplier"] = multiplier

                # construct input file paths
                for folder_key in ["temp", "download"]:
                    # create and attach input paths including intermediaries
                    wapor_dict[folder_key] = self.generate_input_file_path(
                        component=component,
                        wapor_level=self.wapor_level,
                        raster_id=row["raster_id"],
                        return_period=self.return_period,
                        input_folder=folder_key,
                        ext=".tif",
                    )

                # create masked folder entry
                wapor_dict["masked"] = self.generate_output_file_path(
                    description=component,
                    wapor_level=self.wapor_level,
                    output_folder="masked",
                    period_start=wapor_dict["period_start"],
                    period_end=wapor_dict["period_end"],
                    aoi_name=aoi_name,
                    ext=".tif",
                )

                wapor_dict["url"] = None

                # check if files in a downward direction exist and if not note for downloading and processing as needed
                if os.path.exists(wapor_dict["masked"]):
                    wapor_dict["processing_steps"] = 0
                    logger.info("masked file found skipping raster url download")

                elif os.path.exists(wapor_dict["download"]):
                    wapor_dict["processing_steps"] = 1
                    logger.info("downloaded file found skipping raster url download")

                elif os.path.exists(wapor_dict["temp"]):
                    wapor_dict["processing_steps"] = 2
                    logger.info("temp file found skipping raster url download")

                else:
                    ### attempt to retrieve the download url
                    wapor_dict["url"] = self.retrieve_crop_raster_url(
                        bbox=self.bbox,
                        cube_code=cube_code,
                        time_code=row["time_code"],
                        raster_id=row["raster_id"],
                        api_token=self.api_token,
                    )

                    wapor_dict["processing_steps"] = 3

                wapor_download_list.append(wapor_dict)

                count += 1
                if self.print_wait_bar:
                    printWaitBar(
                        count,
                        total_count,
                        prefix="Retrieving Raster Urls:",
                        suffix=f"Complete: {count} out of {total_count}",
                        length=50,
                    )

        return wapor_download_list

    #################################
    def retrieve_actual_wapor_rasters(
        self,
        wapor_download_list: list,
        template_raster_path: str = None,
        aoi_name: str = "nomask",
        output_nodata: float = -9999,
    ) -> dict:
        """WAPOR download works in two phases. the retrieval and setup of the download info and
        retrieval of the crop raster url and the actual download of that raster and preprocessing of it.
        this subfunction carries out the actual raster download and preprocessing.

            NOTE: works in combination with retrieve_wapor_download_info

            NOTE: aoi_name (mask name) if supplied should match that in retrieve_wapor_download_info

        Parameters
        ----------
        wapor_download_list : list
            list of wapor downlaod dicts used to retrieve and preprocess rasters
        template_raster_path : str, optional
            if provided uses the template as the source for the metadata and matches rasters too
            it and masks them too match it too by default None
        aoi_name : str, optional
            name for the mask subfolder if not provided writes too nomask folder and possibly, by default "nomask"
        output_nodata : float, optional
            nodata value to use for the retrieved data, by default -9999

        Returns
        -------
        dict
            dictionary of dictionaries ordered by datacomponent each containing a list
            of rasters retrieved and the path to the compiled vrt
        """
        assert isinstance(
            wapor_download_list, list
        ), "please provide a list constructed using retrieve_wapor_download_info"
        assert isinstance(
            wapor_download_list[0], dict
        ), "please provide a list constructed using retrieve_wapor_download_info"

        # start retrieving data using the wapor dicts
        logger.info(
            f"attempting to retrieve {len(wapor_download_list)} rasters from wapor"
        )
        # set up waitbar
        total_count = len(wapor_download_list)
        count = 0
        if self.print_wait_bar:
            printWaitBar(
                count,
                total_count,
                prefix="Download/Process Raster Progress:",
                suffix=f"Complete: {count} out of {total_count} ",
                length=50,
            )

        # retrieve and process data per wapor download dict as needed
        for wapor_dict in wapor_download_list:
            if wapor_dict["processing_steps"] >= 3:
                # retrieve the raster and write to the download folder
                self.wapor_raster_request(
                    wapor_url=wapor_dict["url"], output_file_path=wapor_dict["temp"]
                )

            if wapor_dict["processing_steps"] >= 2:
                # Process the downloaded raster and write to the process folder
                WaporRetrieval.wapor_raster_processing(
                    input_raster_path=wapor_dict["temp"],
                    output_raster_path=wapor_dict["download"],
                    wapor_multiplier=wapor_dict["multiplier"],
                    return_period=wapor_dict["return_period"],
                    period_start=wapor_dict["period_start"],
                    period_end=wapor_dict["period_end"],
                    delete_input=True,
                    output_nodata=output_nodata,
                )

            count += 1
            if self.print_wait_bar:
                printWaitBar(
                    count,
                    total_count,
                    prefix="Download/Process Raster Progress:",
                    suffix=f"Complete: {count} out of {total_count}",
                    length=50,
                )

        # prepare the template for processing all rasters to match
        if not isinstance(template_raster_path, str):
            template_raster_path = wapor_download_list[0]["download"]
            mask_raster_path = None
        else:
            template_raster_path = template_raster_path
            mask_raster_path = template_raster_path

        # reset count and initiate processing and masking
        count = 0
        if self.print_wait_bar:
            printWaitBar(
                count,
                total_count,
                prefix="Process and Mask Raster Progress:",
                suffix=f"Complete: {count} out of {total_count}",
                length=50,
            )

        for wapor_dict in wapor_download_list:
            if wapor_dict["processing_steps"] >= 1:
                # process the processed rasters to match their proj, dimensions and mask them as needed and write to the masked folder
                # if no mask is provided this step is always carried out
                raster.match_raster(
                    match_raster_path=template_raster_path,
                    input_raster_path=wapor_dict["download"],
                    output_raster_path=wapor_dict["masked"],
                    output_crs=self.output_crs,
                    mask_raster_path=mask_raster_path,
                    output_nodata=output_nodata,
                )

                count += 1
                if self.print_wait_bar:
                    printWaitBar(
                        count,
                        total_count,
                        prefix="Process/Mask Raster Progress:",
                        suffix=f"Complete: {count} out of {total_count} ",
                        length=50,
                    )

        # set output dictionary
        retrieved_rasters_dict = {}
        # reorganise the files per datacomponent and create a vrt as needed
        datacomponent_list = list(set([d["component"] for d in wapor_download_list]))

        for comp in datacomponent_list:
            # find the all processed rasters of a certain datacomponent
            masked_raster_list = [
                d["masked"] for d in wapor_download_list if d["component"] == comp
            ]

            # retrieve the total temporal range they cover
            period_start = sorted(
                [
                    d["period_start"]
                    for d in wapor_download_list
                    if d["component"] == comp
                ]
            )[0]
            period_end = sorted(
                [d["period_end"] for d in wapor_download_list if d["component"] == comp]
            )[-1]

            # generate the vrt file name
            vrt_path = self.generate_output_file_path(
                description=comp,
                wapor_level=self.wapor_level,
                period_start=period_start,
                period_end=period_end,
                output_folder="masked",
                aoi_name=aoi_name,
                ext=".vrt",
            )

            # if the vrt does not already exist create it
            if not os.path.exists(vrt_path):
                raster.build_vrt(
                    raster_list=masked_raster_list,
                    output_vrt_path=vrt_path,
                    action="time",
                )

            out_files = {"raster_list": masked_raster_list, "vrt_path": vrt_path}

            retrieved_rasters_dict[comp] = out_files

        return retrieved_rasters_dict

    #################################
    def download_wapor_rasters(
        self,
        datacomponents: list,
        period_start: datetime = None,
        period_end: datetime = None,
        return_period: str = None,
        template_raster_path: str = None,
        aoi_name: str = "nomask",
        output_nodata: float = -9999,
    ):
        """wrapper function for retrieve_wapor_download_info and retrieve_actual_wapor_rasters
        uaed to download and preprocess wapor rasters

        Parameters
        ----------
        datacomponents : list
            datacomponents to retrieve
        period_start : datetime, optional
            start of period to retrieve data for, by default None
        period_end : datetime, optional
            end of period to retrieve data for, by default None
        return_period : str, optional
            return period to retrieve for, by default None
        template_raster_path : str, optional
            if provided uses the template as the source for the metadata and matches rasters too
            it and masks them too match it too, by default None
        aoi_name : str, optional
            area of interest (aoi) name to use for the mask folder auto set to nomask if not provided, by default "nomask"
        output_nodata : float, optional
            nodata value to use for the retrieved data, by default -9999

        Returns
        -------
        dict
            dictionary of dictionaries ordered by datacomponent each containing a list of rasters
            downloaded, a list of yearly vrts and the path to the full period vrt

        Raises
        ------
        TypeError
            if datacomponents is not a list
        """
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period

        if not isinstance(datacomponents, list):
            raise TypeError(
                "datacomponents provided should be formatted as a list of strings"
            )

        # setup download to be carried out per year depending on the requested dates
        date_tuples = WaporStructure.wapor_organise_request_dates_per_year(
            period_start=self.period_start,
            period_end=self.period_end,
            return_period=self.return_period,
        )

        # setup download variables
        num_of_downloads = len(date_tuples)
        current_download = 1
        download_dict = dict()

        logger.info(
            f"attempting to download raster data for {num_of_downloads} periods"
        )

        for dt in date_tuples:
            logger.info(
                f"downloading rasters for time period: {dt[0]} to {dt[1]}, period {current_download} out of {num_of_downloads}"
            )
            # retrieve the download info
            retrieval_info = self.retrieve_wapor_download_info(
                datacomponents=datacomponents,
                period_start=dt[0],
                period_end=dt[1],
                return_period=self.return_period,
                aoi_name=aoi_name,
            )

            # retrieve and process the rasters
            retrieved_rasters_dict = self.retrieve_actual_wapor_rasters(
                wapor_download_list=retrieval_info,
                template_raster_path=template_raster_path,
                aoi_name=aoi_name,
                output_nodata=output_nodata,
            )

            current_download += 1

            # update download_dict with the yearly retrieved rasters dict
            for datacomponent in retrieved_rasters_dict:
                if not datacomponent in download_dict:
                    download_dict[datacomponent] = {}
                    download_dict[datacomponent]["raster_list"] = []
                    download_dict[datacomponent]["vrt_list"] = []

                download_dict[datacomponent]["raster_list"].extend(
                    retrieved_rasters_dict[datacomponent]["raster_list"]
                )
                download_dict[datacomponent]["vrt_list"].append(
                    retrieved_rasters_dict[datacomponent]["vrt_path"]
                )

        # generate whole period vrts
        for datacomponent in retrieved_rasters_dict:
            # generate the vrt file name
            complete_vrt_path = self.generate_output_file_path(
                wapor_level=self.wapor_level,
                description=datacomponent,
                period_start=self.period_start,
                period_end=self.period_end,
                output_folder="masked",
                aoi_name=aoi_name,
                ext=".vrt",
            )

            raster.build_vrt(
                raster_list=download_dict[datacomponent]["raster_list"],
                output_vrt_path=complete_vrt_path,
                action="time",
            )

            download_dict[datacomponent]["vrt_path"] = complete_vrt_path

        return download_dict
