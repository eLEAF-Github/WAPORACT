
"""
minbuza_waterpip project

land cover classification classes
"""

# make it case insensitive
# add filter for exsting crops per location


def WaporLCC(wapor_level: int):
    """
    Description:
        provides the wapor land cover classification
        tables per wapor level for use in
        the WaporRetrieval class

    Args:
        level: WAPOR level to retrieve the table for as a dict

    Return:
        dict: wapor LCC dictionary 
    """
    if wapor_level in (1,2):
        lcc_dict = {
            'Unknown': 0,
            'Shrubland': 20,
            'Grassland': 30,
            'Cropland_Rainfed': 41,
            'Cropland_Irrigated': 42,
            'Cropland_fallow ': 43,
            'Built': 50,
            'Bare': 60,
            'Permanent_Snow': 70,
            'Water_Bodies': 80,
            'Temporary_Water_Bodies': 81,
            'Shrub': 90,
            'Tree_Closed_Evergreen_Needleleaved': 111,
            'Tree_Closed_Evergreen_Broadleaved': 112,
            'Tree_Closed_Decidious_Broadleaved': 114,
            'Tree_Closed_Mixed': 115,
            'Tree_Closed_Unknown': 116,
            'Tree_Open_Evergreen_Needleleaved': 121,
            'Tree_Open_Evergreen_Broadleaved': 122,
            'Tree_Open_Decidious_Needleleaved': 123,
            'Tree_Open_Decidious_Broadleaved': 124,
            'Tree_Open_Mixed': 125,
            'Tree_Open_Unknown': 126,
            'Sea': 200,
            'Nodata': 255,
        }

    elif wapor_level == 3:
        lcc_dict = {
            'Tree_Cover_Closed': 1,
            'Tree_Cover_Open': 2,
            'Grassland': 4,
            'Bare_soil': 5,
            'Urban': 7,
            'Wheat': 8,
            'Maize': 9,
            'Potato': 10,
            'Vegetables': 11,
            'Fallow': 12,
            'Orchard_Closed': 13,
            'Olive': 14,
            'Grapes': 15,
            'Orchard_Open': 16,
            'Wetland': 17,
            'Shrubland': 18,
            'Water': 19,
            'Rice': 20,
            'Other_Crop': 21,
            'Sugarcane': 22,
            'Teff': 23,
            'Cotton': 24,
            'Clover': 25,
            'Onions': 26,
            'Carrots': 27,
            'Eggplant': 28,
            'Flax': 29,
            'Non_Vegetation': 30,
            'Sugar_Beet': 31,
            'Cassava': 32,
            'Sorghum': 33,
            'Millet': 34,
            'Groundnut': 35,
            'Pigeonpea': 36,
            'Chickpea': 37,
            'Okra': 38,
            'Tomatoes': 39,
            'Cropland': 40,
            'Bananas': 41,
            'Cowpea': 42,
            'Sesame': 43,
            'Soyabean': 44,
            'Other_Perennial': 50,
            'Wheat_Irrigated': 108,
            'Maize_Irrigated': 109,
            'Potato_Irrigated': 110,
            'Vegetables_Irrigated': 111,
            'Orchard_Closed_Irrigated': 113,
            'Olive_Irrigated': 114,
            'Grapes_Irrigated': 115,
            'Orchard_Open_Irrigated': 116,
            'Rice_Irrigated': 120,
            'Mixed_Crop_Irrigated': 121,
            'Sugarcane_Irrigated': 122,
            'Teff_Irrigated': 123,
            'Cotton_Irrigated': 124,
            'Clover_Irrigated': 125,
            'Onions_Irrigated': 126,
            'Carrots_Irrigated': 127,
            'Eggplant_Irrigated': 128,
            'Flax_Irrigated': 129,
            'Sugar_Beet_Irrigated': 131,
            'Cassava_Irrigated': 132,
            'Sorghum_Irrigated': 133,
            'Millet_Irrigated': 134,
            'Groundnut_Irrigated': 135,
            'Pigeonpea_Irrigated': 136,
            'Chickpea_Irrigated': 137,
            'Okra_Irrigated': 138,
            'Tomatoes_Irrigated': 139,
            'Cropland_Irrigated': 140,
            'Bananas_Irrigated': 141,
            'Cowpea_Irrigated': 142,
            'Sesame_Irrigated': 143,
            'Soyabean_Irrigated': 144,
            'Other_Perrenial_Irrigated': 150,
        }

    else:
        raise AttributeError('level needs to be either 1,2 or 3')

    # above the keys are as they are posted in wapor but we will work with them in lower case to make them case insensitive
    lcc_codes = list(lcc_dict.keys())

    for code in lcc_codes:
        lcc_dict[code.lower()] = lcc_dict.pop(code)
    return lcc_dict


   
