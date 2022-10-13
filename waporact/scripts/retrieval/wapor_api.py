# -*- coding: utf-8 -*-
"""
Original Authors: Bich Tran
         IHE Delft 2019
Contact: b.tran@un-ihe.org

Edited: Roeland de Koning
    eLEAF 2021 - 2022

Script used to retrieve data from the wapor api
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

from osgeo import ogr

from typing import Union


import logging

logger = logging.getLogger(__name__)


class WaporAPI(object):
    def __init__(
        self,
        path_catalog: str = r"https://io.apps.fao.org/gismgr/api/v1/catalog/workspaces/",
        path_sign_in: str = r"https://io.apps.fao.org/gismgr/api/v1/iam/sign-in/",
        path_refresh: str = r"https://io.apps.fao.org/gismgr/api/v1/iam/token",
        path_download: str = r"https://io.apps.fao.org/gismgr/api/v1/download/",
        path_query: str = r"https://io.apps.fao.org/gismgr/api/v1/query/",
        path_jobs: str = r"https://io.apps.fao.org/gismgr/api/v1/catalog/workspaces/WAPOR/jobs/",
        version: int = 2,
    ):
        """provides access to the FAO run WAPOR API and the data hosted on
        the portal. This script was originally developed by Bich Tran
        and IHE Delft and adjusted for use within waporact.

        Parameters
        ----------
        path_catalog : _type_, optional
            api url to the catalog, by default r'https://io.apps.fao.org/gismgr/api/v1/catalog/workspaces/'
        path_sign_in : _type_, optional
            api url to sign in , by default r'https://io.apps.fao.org/gismgr/api/v1/iam/sign-in/'
        path_refresh : _type_, optional
            api url to refresh your token, by default r'https://io.apps.fao.org/gismgr/api/v1/iam/token'
        path_download : _type_, optional
            api url to downloads, by default r'https://io.apps.fao.org/gismgr/api/v1/download/'
        path_query : _type_, optional
            api url to send a query, by default r'https://io.apps.fao.org/gismgr/api/v1/query/'
        path_jobs : _type_, optional
            api url to jobs, by default r'https://io.apps.fao.org/gismgr/api/v1/catalog/workspaces/WAPOR/jobs/'
        version : int, optional
            version of the portal api to use, by default 2
        """

        self.request_start = datetime.now() - timedelta(days=1)
        self.request_end = datetime.now()
        self.path_catalog = path_catalog
        self.path_sign_in = path_sign_in
        self.path_refresh = path_refresh
        self.path_download = path_download
        self.path_query = path_query
        self.path_jobs = path_jobs
        self.version = version

    #################################
    # properties
    #################################
    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value: Union[int, str]):
        """set and check the version parameter

        Parameters
        ----------
        value : Union[int, str]
            version to set/check

        Raises
        ------
        AttributeError
            if version is not an int of 1 or 2
        """
        if value in ["WAPOR_1", "WAPOR_2"]:
            self._version = value
        else:
            if value in [1, 2]:
                self._version = f"WAPOR_{value}"
            else:
                raise AttributeError("version needs to be an int and of of: [1,2]")

    #################################
    # functions
    #################################
    def retrieve_catalogue_as_dataframe(self, wapor_level: int, cubeInfo=True):
        """retrieves the wapor catalogue from the WaPOR database
        and formats it as a dataframe

        NOTE: based on and replaces the WaporAPI
        original class function getCatalog


        Parameters
        ----------
        wapor_level : int
            wapor level to retrieve catalogue for
        cubeInfo : bool, optional
            if true also retrieves and formats the cube
            info from the catalogue into
            the output dataframe, by default True

        Returns
        -------
        pandas.DataFrame
            retrieved catalogue as a datafrmae

        Raises
        ------
        AttributeError
            if wapor level is not one of 1,2 or 3
        Exception
            unknown error stopping the retrieval with additional message
        """
        if wapor_level not in [1, 2, 3]:
            raise AttributeError("wapor_level (int) needs to be either 1, 2 or 3")
        try:
            df = self._query_catalog(wapor_level)
        except Exception as query_catalog_error:
            logger.error(
                f"data of the specified wapor_level could not be retrieved or there was a connection error (wapor_level: {wapor_level})"
            )
            raise query_catalog_error

        if cubeInfo:
            cubes_measure = []
            cubes_dimension = []
            for cube_code in df["code"].values:
                cubes_measure.append(self._query_cubeMeasures(cube_code))
                cubes_dimension.append(self._query_cubeDimensions(cube_code))
            df["measure"] = cubes_measure
            df["dimension"] = cubes_dimension

        df["period_code"] = df["code"].str.split("_").str[-1]
        df["component_code"] = df["code"].str.split("_").str[-2]
        df["component_desc"] = df["caption"].str.split("(").str[0]

        df.loc[df["period_code"] == "LT", "period_desc"] = "Long Term"
        df.loc[df["period_code"] == "A", "period_desc"] = "Annual"
        df.loc[df["period_code"] == "S", "period_desc"] = "Seasonal"
        df.loc[df["period_code"] == "M", "period_desc"] = "Monthly"
        df.loc[df["period_code"] == "D", "period_desc"] = "Dekadal"
        df.loc[df["period_code"] == "E", "period_desc"] = "Daily"

        if wapor_level == 3:
            df["country_code"] = df["code"].str.split("_").str[1]
            df["country_desc"] = (
                df["caption"].str.split("\(").str[-1].str.split("-").str[0]
            )

        df.loc[df["code"].str.contains("QUAL"), "component_code"] = (
            "QUAL_" + df["component_code"]
        )

        df = df.fillna("NaN")

        logger.info(
            f"wapor catalogue retrieved and formatted for use, wapor level: {wapor_level}"
        )

        return df

    #################################
    def _query_catalog(self, level: int):
        """query the catalog

        Parameters
        ----------
        level : level to query
            wapor level to retrieve catalog for

        Returns
        -------
        pandas.DataFrame
            catalogue as a dataframe
        """
        if level == None:
            request_url = (
                rf"{self.path_catalog}{self.version}/cubes?overview=false&paged=false"
            )

        else:
            request_url = rf"{self.path_catalog}{self.version}/cubes?overview=false&paged=false&tags=L{level}"
        resp = requests.get(request_url, timeout=30)
        meta_data_items = resp.json()

        try:
            response = meta_data_items["response"]
            df = pd.DataFrame.from_dict(response, orient="columns")
        except Exception as response_read_error:
            logger.error("No response whey reading the catalog response")
            df = None
            raise response_read_error

        return df

    #################################
    def getCubeInfo(
        self, cube_code: str, wapor_level: int, catalogue: pd.DataFrame = None
    ):
        """check the WaPOR level catalogue for a cube code
            and related info.

        Parameters
        ----------
        cube_code : str
            cube code to check/ query for
        wapor_level : int
            wapor level to check for
        catalogue : pd.DataFrame, optional
            catalogue to check if it is already in memory, by default None

        Returns
        -------
        dict
            dictionary containing retrieval data (dimensions, measures)
            related to the cube code given

        Raises
        ------
        AttributeError
            if wapor_level not in [1,2,3]
        Exception
            unknown error and an extra message suggesting that the user
            check their cube code against the catalogue
        """
        if wapor_level not in [1, 2, 3]:
            raise AttributeError("wapor_level (int) needs to be either 1, 2 or 3")

        if not isinstance(catalogue, pd.DataFrame):
            catalogue = self.retrieve_catalogue_as_dataframe(
                wapor_level=wapor_level, cubeInfo=True
            )

        elif "measure" not in catalogue.columns:
            catalogue = self.retrieve_catalogue_as_dataframe(
                wapor_level=wapor_level, cubeInfo=True
            )

        else:
            pass

        try:
            cube_info = catalogue.loc[catalogue["code"] == cube_code].to_dict(
                "records"
            )[0]
        except Exception as e:
            logger.error(
                f"Data for your given cube code: {cube_code} and wapor version: {self.version} could not be found in the catalogue, "
                "please check the catalogue and or your inputs"
            )
            raise e

        return cube_info

    #################################
    def _query_cubeMeasures(self, cube_code: str):
        """query for cube measurement info

        Parameters
        ----------
        cube_code : str
            cube code to query measurements for

        Returns
        -------
        dict
            cube measurements
        """
        request_url = rf"{self.path_catalog}{self.version}/cubes/{cube_code}/measures?overview=false&paged=false"
        resp = requests.get(request_url)
        cube_measures = resp.json()["response"][0]
        return cube_measures

    #################################
    def _query_cubeDimensions(self, cube_code: str):
        """query for cube dimension info

        Parameters
        ----------
        cube_code : str
            cube code to query dimensions for

        Returns
        -------
        dict
            cube dimensions
        """
        request_url = rf"{self.path_catalog}{self.version}/cubes/{cube_code}/dimensions?overview=false&paged=false"
        resp = requests.get(request_url)
        cube_dimensions = resp.json()["response"]
        return cube_dimensions

    #################################
    def _query_accessToken(self, APIToken: str):
        """query for a wapor access token

        Parameters
        ----------
        APIToken : str
            api token used to retrieve the access token

        Returns
        -------
        str
            access token
        """
        resp_vp = requests.post(
            self.path_sign_in, headers={"X-GISMGR-API-KEY": APIToken}
        )
        resp_vp = resp_vp.json()
        try:
            self.AccessToken = resp_vp["response"]["accessToken"]
            self.RefreshToken = resp_vp["response"]["refreshToken"]
            self.time_expire = resp_vp["response"]["expiresIn"]
        except KeyError:
            logger.error(
                f"token could not be found possibly no longer valid: {resp_vp}"
            )
            raise KeyError
        return self.AccessToken

    #################################
    def _query_refreshToken(self, RefreshToken):
        """query to refresh the access token

        Parameters
        ----------
        RefreshToken : str
            api token used to refresh the access token

        Returns
        -------
        str
            new/refreshed access token
        """
        resp_vp = requests.post(
            self.path_refresh,
            params={"grandType": "refresh_token", "refreshToken": RefreshToken},
        )
        resp_vp = resp_vp.json()
        self.AccessToken = resp_vp["response"]["accessToken"]
        return self.AccessToken

    #################################
    def get_access_token(self, APIToken: str):
        """query for an access token

        Parameters
        ----------
        APIToken : str
            api token used to retrieve the access token

        Returns
        -------
        str
            access token used to retrieve data
        """
        # Get AccessToken
        self.request_end = datetime.now().timestamp()
        try:
            AccessToken = self.AccessToken
            if self.request_end - self.request_start > self.time_expire:
                AccessToken = self._query_refreshToken(self.RefreshToken)
                self.request_start = self.request_end
        except:
            AccessToken = self._query_accessToken(APIToken)

        return AccessToken

    #################################
    def getAvailData(
        self,
        cube_code: str,
        wapor_level: int,
        time_range="2009-01-01,2018-12-31",
        location=[],
        season=[],
        stage=[],
        catalogue: pd.DataFrame = None,
    ):
        """retrieve dict of available data on the wapor portal
        according to the input info

        Parameters
        ----------
        cube_code : str
            cube code used to find specific data  ex. 'L2_CTY_PHE_S'
        wapor_level : int
            wapor level to retrieve data for
        time_range : str, optional
            time range data is wanted for, by default '2009-01-01,2018-12-31'
        location : list of strings, optional
            locations data is wanted for, by default [] , return all available locations ex. ['ETH']
        season : list of strings, optional
            seasons data is wanted for, by default [] , return all available seasons ex. ['S1']
        stage : list of strings, optional
            stages data is wanted for, by default [], return all available stages ex. ['EOS','SOS']
        catalogue : pd.DataFrame, optional
            catalogue dataframe to get cube info from if it already exists in memory, by default None

        Returns
        -------
        pandas.DataFrame
            dataframe of available data

        Raises
        ------
        Exception
            raise Unknown error and pass message that cube info could not be retrieved
        Exception
            raise unknown error and pass message that no available data could be retrieved
        """
        try:
            # get cube info
            cube_info = self.getCubeInfo(
                cube_code=cube_code, wapor_level=wapor_level, catalogue=catalogue
            )
            # get measures
            measure_code = cube_info["measure"]["code"]
            # get dimension
            dimensions = cube_info["dimension"]
        except Exception as e:
            logger.error("cannont get cube info")
            raise e

        dims_ls = []
        columns_codes = ["MEASURES"]
        rows_codes = []
        try:
            for dims in dimensions:
                if dims["type"] == "TIME":  # get time dims
                    time_dims_code = dims["code"]
                    df_time = self._query_dimensionsMembers(cube_code, time_dims_code)
                    time_dims = {
                        "code": time_dims_code,
                        "range": f"[{time_range})",
                    }
                    dims_ls.append(time_dims)
                    rows_codes.append(time_dims_code)
                if dims["type"] == "WHAT":
                    dims_code = dims["code"]
                    df_dims = self._query_dimensionsMembers(cube_code, dims_code)
                    members_ls = [row["code"] for i, row in df_dims.iterrows()]
                    if dims_code == "COUNTRY" or dims_code == "BASIN":
                        if location:
                            members_ls = location
                    if dims_code == "SEASON":
                        if season:
                            members_ls = season
                    if dims_code == "STAGE":
                        if stage:
                            members_ls = stage

                    what_dims = {"code": dims["code"], "values": members_ls}
                    dims_ls.append(what_dims)
                    rows_codes.append(dims["code"])

            df = self._query_availData(
                cube_code, measure_code, dims_ls, columns_codes, rows_codes
            )
        except Exception as e:
            logger.error("Cannot get list of available data")

        # sorted df
        keys = rows_codes + ["raster_id", "bbox", "time_code"]
        df_dict = {i: [] for i in keys}
        for irow, row in df.iterrows():
            for i in range(len(row)):
                if row[i]["type"] == "ROW_HEADER":
                    key_info = row[i]["value"]
                    df_dict[keys[i]].append(key_info)
                    if keys[i] == time_dims_code:
                        time_info = df_time.loc[df_time["caption"] == key_info].to_dict(
                            orient="records"
                        )
                        df_dict["time_code"].append(time_info[0]["code"])
                if row[i]["type"] == "DATA_CELL":
                    raster_info = row[i]["metadata"]["raster"]
            df_dict["raster_id"].append(raster_info["id"])
            df_dict["bbox"].append(raster_info["bbox"])
        df_sorted = pd.DataFrame.from_dict(df_dict)
        return df_sorted

    #################################
    def _query_availData(
        self,
        cube_code: str,
        measure_code: str,
        dims_ls: list,
        columns_codes: list,
        rows_codes: list,
    ):  # check these input tyes
        """query for available data

        Parameters
        ----------
        cube_code : str
            cube code to query for
        measure_code : str
            measure code to query for
        dims_ls : list
            list of dimensions to query for
        columns_codes : list
            list of columns to query for
        rows_codes : list
            list of rows to query for

        Returns
        -------
        pandas.DataFrame
            dataframe of pandas data
        """
        query_load = {
            "type": "MDAQuery_Table",
            "params": {
                "properties": {
                    "metadata": True,
                    "paged": False,
                },
                "cube": {
                    "workspaceCode": self.version,
                    "code": cube_code,
                    "language": "en",
                },
                "dimensions": dims_ls,
                "measures": [measure_code],
                "projection": {"columns": columns_codes, "rows": rows_codes},
            },
        }

        resp = requests.post(self.path_query, json=query_load, timeout=30)
        resp_vp = resp.json()
        if resp_vp["message"] == "OK":
            try:
                results = resp_vp["response"]["items"]
                results_df = pd.DataFrame(results)
            except Exception as e:
                logger.error("Cannot get list of available data")

        else:
            logger.info(resp_vp["message"])

        return results_df

    #################################
    def _query_dimensionsMembers(self, cube_code: str, dims_code: str):
        """query for dimension members

        Parameters
        ----------
        cube_code : str
            cube code to query for
        dims_code : str
            dminesions code to query for

        Returns
        -------
        pandas.DataFrame
            dataframe of dimensions
        """
        base_url = "{0}{1}/cubes/{2}/dimensions/{3}/members?overview=false&paged=false"
        request_url = base_url.format(
            self.path_catalog, self.version, cube_code, dims_code
        )
        resp = requests.get(request_url, timeout=30)
        resp_vp = resp.json()
        if resp_vp["message"] == "OK":
            try:
                avail_items = resp_vp["response"]
                df = pd.DataFrame.from_dict(avail_items, orient="columns")
            except:
                logger.error("Cannot get dimensions Members")
                df = None
        else:
            logger.error(resp_vp["message"])
        return df

    #################################
    def getLocations(self, level: int = None):
        """get wapor locations

        Parameters
        ----------
        level : int, optional
            wapor level to retrieve locatiosn for, by default None

        Returns
        -------
        pandas.DataFrame
            locations dataframe
        """

        try:
            df_loc = self.locationsTable
        except:
            df_loc = self._query_locations()
            df_loc = self.locationsTable
        if level is not None:
            df_loc = df_loc.loc[df_loc[f"l{level}"] == True]
        return df_loc

    #################################
    def _query_locations(self):
        """query for locations

        Returns
        -------
        pandas.DataFrame
            locatiosn dataframe
        """
        query_location = {
            "type": "TableQuery_GetList_1",
            "params": {
                "table": {"workspaceCode": self.version, "code": "LOCATION"},
                "properties": {"paged": False},
                "sort": [{"columnName": "name"}],
            },
        }
        resp = requests.post(self.path_query, json=query_location, timeout=30)
        resp_vp = resp.json()
        if resp_vp["message"] == "OK":
            avail_items = resp_vp["response"]
            df_loc = pd.DataFrame.from_dict(avail_items, orient="columns")
            self.locationsTable = df_loc
            df_CTY = df_loc.loc[(df_loc["l2"] == True) & (df_loc["type"] == "COUNTRY")]
            df_BAS = df_loc.loc[(df_loc["l2"] == True) & (df_loc["type"] == "BASIN")]
            self.list_countries = [rows["code"] for index, rows in df_CTY.iterrows()]
            self.list_basins = [rows["code"] for index, rows in df_BAS.iterrows()]

        else:
            logger.info(resp_vp["message"])
            df_loc = None

        return df_loc

    #################################
    def getRasterUrl(self, cube_code: str, rasterId: str, APIToken: str):
        """get the url generated by wapor and used to download the raster

        Parameters
        ----------
        cube_code : str
            cuve code used ot find the raster
        rasterId : str
            id of the raster
        APIToken : str
            api token used to retrieve the access token used to retrieve the raster

        Returns
        -------
        str
            url of the raster
        """
        AccessToken = self.get_access_token(APIToken)
        download_url = self._query_rasterUrl(cube_code, rasterId, AccessToken)
        return download_url

    #################################
    def _query_rasterUrl(self, cube_code: str, rasterId: str, AccessToken: str):
        """get the url generated by wapor and used to download the raster

        Parameters
        ----------
        cube_code : str
            cuve code used ot find the raster
        rasterId : str
            id of the raster
        AccessToken : str
            access token used to retrieve the raster

        Returns
        -------
        str
            url of the raster
        """
        base_url = f"{self.path_download}{self.version}"

        headers_val = {"Authorization": "Bearer " + AccessToken}
        params_val = {
            "language": "en",
            "requestType": "mapset_raster",
            "cubeCode": cube_code,
            "rasterId": rasterId,
        }

        resp_vp = requests.get(
            base_url, headers=headers_val, params=params_val, tiemout=30
        )
        resp_vp = resp_vp.json()
        try:
            resp = resp_vp["response"]
            expiry_date = datetime.now() + timedelta(seconds=int(resp["expiresIn"]))
            download_url = {"url": resp["downloadUrl"], "expiry_datetime": expiry_date}

        except:
            logger.error("Cannot get Raster URL")
            download_url = None

        return download_url

    #################################
    def _query_jobOutput(self, job_url: str):
        """queyr for the raster/job output

        Parameters
        ----------
        job_url : str
            url used to retrieve the job output

        Returns
        -------
        dict/dataframe
            output of the job
        """
        _continue = True
        while _continue:
            resp = requests.get(job_url, timeout=30)
            resp = resp.json()
            jobType = resp["response"]["type"]
            if resp["response"]["status"] == "COMPLETED":
                _continue = False
                if jobType == "CROP RASTER":
                    output = resp["response"]["output"]["downloadUrl"]
                elif jobType == "AREA STATS":
                    results = resp["response"]["output"]
                    output = pd.DataFrame(results["items"], columns=results["header"])
                else:
                    logger.error("Invalid jobType")
                    output = None
            if resp["response"]["status"] == "COMPLETED WITH ERRORS":
                _continue = False
                logger.error(resp["response"]["log"])
                output = None

        return output

    #################################
    def getCropRasterURL(
        self,
        bbox: tuple,
        cube_code: str,
        wapor_level: int,
        time_code: str,
        rasterId: str,
        APIToken: str,
        season: str = None,
        stage: str = None,
        print_job: bool = False,
        catalogue: pd.DataFrame = None,
    ):
        """get the url of a cropped raster from wapor

        Parameters
        ----------
        bbox : tuple
            bounding box of the cropped area  [xmin,ymin,xmax,ymax]
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
        season : str, optional
            season  defining the raster to retrieve, by default None
        stage : str, optional
            stage  defining the raster to retrieve, by default None
        print_job : bool, optional
            if true print the job, by default False
        catalogue : pd.DataFrame, optional
            catalogue to get cube info from if available, by default None

        Returns
        -------
        str
            download url for the cropped raster
        """
        AccessToken = self.get_access_token(APIToken)
        # Create Polygon
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        Polygon = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
        # Get measure_code and dimension_code
        cube_info = self.getCubeInfo(
            cube_code=cube_code, wapor_level=wapor_level, catalogue=catalogue
        )

        cube_measure_code = cube_info["measure"]["code"]
        cube_dimensions = cube_info["dimension"]

        dimension_params = []

        for cube_dimension in cube_dimensions:
            if cube_dimension["type"] == "TIME":
                cube_dimension_code = cube_dimension["code"]
                dimension_params.append(
                    {"code": cube_dimension_code, "values": [time_code]}
                )
            if cube_dimension["code"] == "SEASON":
                dimension_params.append({"code": "SEASON", "values": [season]})
            if cube_dimension["code"] == "STAGE":
                dimension_params.append({"code": "STAGE", "values": [stage]})
        # print(dimension_params)

        # Query payload
        query_crop_raster = {
            "type": "CropRaster",
            "params": {
                "properties": {
                    "outputFileName": "{0}.tif".format(rasterId),
                    "cutline": True,
                    "tiled": True,
                    "compressed": True,
                    "overviews": True,
                },
                "cube": {
                    "code": cube_code,
                    "workspaceCode": self.version,
                    "language": "en",
                },
                "dimensions": dimension_params,
                "measures": [cube_measure_code],
                "shape": {
                    "type": "Polygon",
                    "properties": {"name": "epsg:4326"},  # latlon projection
                    "coordinates": [Polygon],
                },
            },
        }
        resp_vp = requests.post(
            self.path_query,
            headers={"Authorization": "Bearer {0}".format(AccessToken)},
            json=query_crop_raster,
            timeout=30,
        )
        resp_vp = resp_vp.json()
        try:
            job_url = resp_vp["response"]["links"][0]["href"]
            if print_job:
                logger.info(f"Getting download url from: {job_url}")
            download_url = self._query_jobOutput(job_url)
        except:
            logger.error("Cannot get cropped raster URL")
            download_url = None

        return download_url

    #################################
    def getAreaTimeseries(
        self,
        shapefile_fh: str,
        cube_code: str,
        APIToken: str,
        wapor_level: int,
        catalogue: pd.DataFrame = None,
        time_range: str = "2009-01-01,2018-12-31",
    ):
        """get a time series from wapor

        Parameters
        ----------
        shapefile_fh : str
            path to the shapefile
        cube_code : str
            cube code defining the raster to retrieve
        wapor_level : int
            wapor level  defining the raster to retrieve
        APIToken : str
            api token use dto retrieve the access token used to retrieve the raster
        catalogue : pd.DataFrame, optional
            catalogue to get cube info from if available, by default None
        time_range : str, optional
            time range to retrieve time series for, by default "2009-01-01,2018-12-31"

        Returns
        -------
        output
            time series
        """
        AccessToken = self.get_access_token(APIToken)
        # get shapefile info
        dts = ogr.Open(shapefile_fh)
        layer = dts.GetLayer()
        epsg_code = layer.GetSpatialRef().GetAuthorityCode(None)
        shape = layer.GetFeature(0).ExportToJson(as_object=True)["geometry"]
        shape["properties"] = {"name": "EPSG:{0}".format(epsg_code)}

        # get cube info
        cube_info = self.getCubeInfo(
            cube_code=cube_code, wapor_level=wapor_level, catalogue=catalogue
        )

        cube_measure_code = cube_info["measure"]["code"]
        for dims in cube_info["dimension"]:
            if dims["type"] == "TIME":
                cube_dimension_code = dims["code"]

        # query load
        query_areatimeseries = {
            "type": "AreaStatsTimeSeries",
            "params": {
                "cube": {
                    "code": cube_code,
                    "workspaceCode": self.version,
                    "language": "en",
                },
                "dimensions": [
                    {"code": cube_dimension_code, "range": "[{0})".format(time_range)}
                ],
                "measures": [cube_measure_code],
                "shape": shape,
            },
        }

        resp_query = requests.post(
            self.path_query,
            headers={"Authorization": "Bearer {0}".format(AccessToken)},
            json=query_areatimeseries,
            timeout=30,
        )
        resp_query = resp_query.json()
        try:
            job_url = resp_query["response"]["links"][0]["href"]
        except:
            logger.error("Cannot get server response")
            job_url = None
            output = None

        if job_url is not None:
            try:
                logger.info(f"Getting result from: {job_url}")
                output = self._query_jobOutput(job_url)
            except:
                logger.error("Cannot get job output")
                output = None

        return output

    #################################
    def getPixelTimeseries(
        self,
        pixelCoordinates: list,
        cube_code: str,
        wapor_level: int,
        time_range="2009-01-01,2018-12-31",
        catalogue: pd.DataFrame = None,
    ):
        """get a pixel timeseries from wapor

        Parameters
        ----------
        pixelCoordinates : list
            list of two pixel coordinates [37.95883206252312, 7.89534]
        cube_code : str
            cube code defining the raster to retrieve pixel values from
        wapor_level : int
            wapor level  defining the raster to retrieve pixel values from
        catalogue : pd.DataFrame, optional
            catalogue to get cube info from if available, by default None
        time_range : str, optional
            time range to retrieve time series for, by default "2009-01-01,2018-12-31"

        Returns
        -------
        pandas.DataFrame
            datafrmae of the time series
        """
        # get cube info
        cube_info = self.getCubeInfo(
            cube_code=cube_code, wapor_level=wapor_level, catalogue=catalogue
        )

        cube_measure_code = cube_info["measure"]["code"]
        for dims in cube_info["dimension"]:
            if dims["type"] == "TIME":
                cube_dimension_code = dims["code"]

        # query load
        query_pixeltimeseries = {
            "type": "PixelTimeSeries",
            "params": {
                "cube": {
                    "code": cube_code,
                    "workspaceCode": self.version,
                    "language": "en",
                },
                "dimensions": [
                    {"code": cube_dimension_code, "range": "[{0})".format(time_range)}
                ],
                "measures": [cube_measure_code],
                "point": {
                    "crs": "EPSG:4326",  # latlon projection
                    "x": pixelCoordinates[0],
                    "y": pixelCoordinates[1],
                },
            },
        }

        # requests
        resp_query = requests.post(
            self.path_query, json=query_pixeltimeseries, timeout=30
        )
        resp_vp = resp_query.json()
        if resp_vp["message"] == "OK":
            try:
                results = resp_vp["response"]
                df = pd.DataFrame(results["items"], columns=results["header"])
            except:
                logger.error("Server response is empty")
                df = None
        else:
            logger.error(resp_vp["message"])
            df = None

        return df
