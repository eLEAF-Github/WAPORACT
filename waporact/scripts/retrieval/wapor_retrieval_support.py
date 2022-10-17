"""
waporact package

retrieval support functions (support functions)
"""
from typing import Union

import logging

logger = logging.getLogger(__name__)


#################################
# check functions
#################################
def check_datacomponent_availability(
    datacomponents: list,
    all_datacomponents: tuple,
    return_period: str,
    wapor_level: int,
    cube_codes: tuple,
    country_code: str = "notyetset",
):
    """check if the given datacomponent is available

    Parameters
    ----------
    datacomponents : list
        datacomponents to check
    all_datacomponents : tuple
        all available datacomponents at level
    return_period : str
        return period to check
    wapor_level : int
        level to check at
    cube_codes : tuple
        all available cube codes at the given level
    country_code : str, optional
        country code to check, by default "notyetset"

    Returns
    -------
    list
        list of available datacomponents

    Raises
    ------
    AttributeError
        if their are missing datacomponents among those requested
    AttributeError
        if no datacomponents are found
    """
    skip_missing_components = False
    if datacomponents[0] == "ALL":
        skip_missing_components = True
        datacomponents = list(all_datacomponents)
    if wapor_level == 3:
        potential_cube_codes = [
            (comp, f"L{wapor_level}_{country_code}_{comp}_{return_period}")
            for comp in datacomponents
        ]
    else:
        potential_cube_codes = [
            (comp, f"L{wapor_level}_{comp}_{return_period}") for comp in datacomponents
        ]

    available_comps = [
        cube_code[0] for cube_code in potential_cube_codes if cube_code[1] in cube_codes
    ]

    missing_comps = [
        cube_code[0]
        for cube_code in potential_cube_codes
        if cube_code[1] not in cube_codes
    ]

    if not available_comps:
        raise AttributeError(
            f"at wapor level: {wapor_level} and return period: {return_period} , no datacomponents could be \
        found, check your requested datacomponent code against: {all_datacomponents}"
        )
    elif not skip_missing_components:
        if missing_comps:
            logger.info(f"Available datacomponents found: {available_comps}")
            raise AttributeError(
                f"at wapor level: {wapor_level} and return period: {return_period} , the following datacomponents could not be \
            found and may not exist as cube codes (at level 3 region may also affect availibility): {missing_comps}"
            )

    return available_comps


#################################
# retrieval functions
#################################
def generate_wapor_cube_code(
    datacomponent: str,
    return_period: str,
    wapor_level: str,
    cube_codes: tuple,
    country_code: str = "notyetset",
):
    """list generate a wapor cube code and check it exists

    Parameters
    ----------
    datacomponent : str
        datacomponent of the code
    return_period : str
        return period of the code
    wapor_level : str
        level of the code
    cube_codes : tuple
        all availabe cube codes at that level
    country_code : str, optional
        country code of the code if level 3, by default "notyetset"

    Returns
    -------
    str
        cube code

    Raises
    ------
    AttributeError
        if the cube code is not found among the available options
    """
    if wapor_level == 3:
        datacomponent = f"{country_code}_{datacomponent}"
    cube_code = f"L{wapor_level}_{datacomponent}_{return_period}"

    if cube_code not in cube_codes:
        raise AttributeError(
            f"cube code generated: {cube_code}, not found in available list: {cube_codes}"
        )

    return cube_code


#################################
def check_categories_exist_in_count_dict(categories_list: list, count_dict: dict):
    """check lcc category exists in counted categories dict
    and if found provide some feedback

    Parameters
    ----------
    categories_list : list
        list to check
    count_dict : dict
        dict to check keys

    Returns
    -------
    int
        0

    Raises
    ------
    KeyError
        if category not found
    """
    if any(cat in count_dict.keys() for cat in categories_list):
        for cat in categories_list:
            if cat in count_dict.keys():
                logger.info(
                    f"percentage of unmasked occurrence of your category: {cat} in the raster is: {count_dict[cat]['percentage']}"
                )
            else:
                logger.warning(f"the category: {cat} was not found in the raster")
    else:
        logger.info(
            "Provided below is a dataframe of the categories that are found in the raster:"
        )
        logger.info(count_dict)
        raise KeyError(
            f"all categories provided were not found in raster: {categories_list}"
        )

    return 0


#################################
def check_categories_exist_in_categories_dict(
    categories_list: list, categories_dict: dict
):
    """check key exists in dict

    Parameters
    ----------
    categories_list : list
        list of keys to check
    categories_dict : dict
        dict to check keys

    Returns
    -------
    int
        0

    Raises
    ------
    KeyError
        if category not found
    """
    # check that the categories provided exist
    all_categories = list(categories_dict.keys())
    if not all(cat in all_categories for cat in categories_list):
        raise KeyError(
            f"categories given: {categories_list} must exist in the list: {all_categories}"
        )

    return 0


#################################
def dissagregate_categories(categories_list: list, categories_dict: dict):
    """seperate list wapor cateogries into seperated dict keys

    Parameters
    ----------
    categories_list : list
        list to seperate
    categories_dict : dict
        dict ot seperate them into

    Returns
    -------
    dict
        dict with categoriy lists disaggregated into seperate key, value pairs
    """
    # create aggregateless reversed categories dict
    aggregateless_dict = {}
    for key in categories_dict.keys():
        if not isinstance(categories_dict[key], list):
            aggregateless_dict[key] = categories_dict[key]
    categories_dict_reversed = {
        aggregateless_dict[key]: key for key in aggregateless_dict.keys()
    }

    # disaggregate aggregate codes
    disaggregated_categories = []
    remove_categories = []
    for cat in categories_list:
        if isinstance(categories_dict[cat], list):
            value_list = categories_dict[cat]
            for value in value_list:
                disaggregated_categories.append(categories_dict_reversed[value])
            remove_categories.append(cat)

    categories_list.extend(disaggregated_categories)
    for cat in remove_categories:
        categories_list.remove(cat)

    return categories_list


#################################
def wapor_lcc(wapor_level: int):
    """dict of wapor land cover classification codes and names

    Parameters
    ----------
    wapor_level : int
        wapor levle to retrieve codes for

    Returns
    -------
    dict
        dict of wapor land cover classification codes

    Raises
    ------
    AttributeError
        if wrong level is provided
    """
    if wapor_level in (1, 2):
        lcc_dict = {
            "Unknown": 0,
            "Shrubland": 20,
            "Grassland": 30,
            "Cropland": [41, 42, 43],
            "Cropland_Rainfed": 41,
            "Cropland_Irrigated": 42,
            "Cropland_fallow ": 43,
            "Built": 50,
            "Bare": 60,
            "Permanent_Snow": 70,
            "Water_Bodies": 80,
            "Temporary_Water_Bodies": 81,
            "Shrub": 90,
            "Tree_Closed_Evergreen_Needleleaved": 111,
            "Tree_Closed_Evergreen_Broadleaved": 112,
            "Tree_Closed_Decidious_Broadleaved": 114,
            "Tree_Closed_Mixed": 115,
            "Tree_Closed_Unknown": 116,
            "Tree_Open_Evergreen_Needleleaved": 121,
            "Tree_Open_Evergreen_Broadleaved": 122,
            "Tree_Open_Decidious_Needleleaved": 123,
            "Tree_Open_Decidious_Broadleaved": 124,
            "Tree_Open_Mixed": 125,
            "Tree_Open_Unknown": 126,
            "Sea": 200,
            "Nodata": 255,
        }

    elif wapor_level == 3:
        lcc_dict = {
            "Non Irrigated Tree Crops": [1, 2, 13, 14, 15, 16],
            "Irrigated Tree Crops": [113, 114, 115, 116],
            "Non Irrigated Crops": [
                8,
                9,
                10,
                11,
                20,
                21,
                22,
                23,
                24,
                25,
                25,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                50,
            ],
            "Irrigated Crops": [
                108,
                109,
                110,
                111,
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
                131,
                132,
                133,
                134,
                135,
                136,
                137,
                138,
                139,
                141,
                142,
                143,
                144,
                150,
            ],
            "Non Crop Cover": [5, 7, 12, 17, 19],
            "Grass and Shrubland": [4, 18],
            "Tree cover (closed)": 1,
            "Tree cover (open)": 2,
            "Grassland": 4,
            "Bare soil": 5,
            "Urban/Artificial": 7,
            "Wheat": 8,
            "Maize": 9,
            "Potato": 10,
            "Vegetables": 11,
            "Fallow": 12,
            "Orchard (closed)": 13,
            "Olive": 14,
            "Grapes": 15,
            "Orchard (open)": 16,
            "Wetland": 17,
            "Shrubland": 18,
            "Water": 19,
            "Rice": 20,
            "Other crop": 21,
            "Sugarcane": 22,
            "Teff": 23,
            "Cotton": 24,
            "Clover": 25,
            "Onions": 26,
            "Carrots": 27,
            "Eggplant": 28,
            "Flax": 29,
            "Non vegetation (reclass)": 30,
            "Sugar beet": 31,
            "Cassava": 32,
            "Sorghum": 33,
            "Millet": 34,
            "Groundnut": 35,
            "Pigeon Pea": 36,
            "Chickpea": 37,
            "Okra": 38,
            "Tomatoes": 39,
            "Cropland (reclass)": 40,
            "Bananas": 41,
            "Cowpea": 42,
            "Sesame": 43,
            "Soyabean": 44,
            "Other perennial": 50,
            "Irrigated wheat": 108,
            "Irrigated Maize": 109,
            "Irrigated Potato": 110,
            "Irrigated vegetables": 111,
            "Irrigated orchard (closed)": 113,
            "Irrigated olive": 114,
            "Irrigated grapes": 115,
            "Irrigated orchard (open)": 116,
            "Irrigated rice": 120,
            "Irrigated other crop": 121,
            "Irrigated sugar cane": 122,
            "Irrigated Teff": 123,
            "Irrigated cotton": 124,
            "Irrigated clover": 125,
            "Irrigated onions": 126,
            "Irrigated Carrots": 127,
            "Irrigated Eggplant": 128,
            "Irrigated Flax": 129,
            "Irrigated Sugar beet": 131,
            "Irrigated Cassava": 132,
            "Irrigated Sorghum": 133,
            "Irrigated Millet": 134,
            "Irrigated Groundnut": 135,
            "Irrigated Pigeon Pea": 136,
            "Irrigated Chickpea": 137,
            "Irrigated Okra": 138,
            "Irrigated Tomatoes": 139,
            "Irrigated Bananas": 141,
            "Irrigated Cowpea": 142,
            "Irrigated Sesame": 143,
            "Irrigated Soyabean": 144,
            "Irrigated other perennial": 150,
        }

    else:
        raise AttributeError("level needs to be either 1,2 or 3")

    # above the keys are as they are posted in wapor but we will work with them in lower case to make them case insensitive
    lcc_codes = list(lcc_dict.keys())

    for code in lcc_codes:
        lcc_dict[code.lower()] = lcc_dict.pop(code)
    return lcc_dict
