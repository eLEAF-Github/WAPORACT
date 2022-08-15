"""
waporact package

land classification related support functions (support functions)
"""
#################################
def check_categories_exist_in_count_dict(
    categories_list: list,
    count_dict: dict
    ):
    """
    Description:
        check if at least one category can be found in the count_dict
        and thus in the raster, also provides some feedback

        NOTE: works best in combo with raster.calculate_raster_count_statistics

    Args:
        categories_list: list of categories to check
        count_dict: dict to check

    Return:
        int: 0

    Raise:
        KeyError: if no categories found
    """
    if any(cat in count_dict.keys() for cat in categories_list):
        for cat in categories_list:
            if cat in count_dict.keys():
                print('percentage of not nan occurrence of your category: {} in the raster'
                    ' is : {}'.format(cat, count_dict[cat]['percentage']))
            else:
                print('the category: {} was not found in the raster'.format(cat))
    else:
        print('Provided below is a dataframe of the categories that are found in the raster:')
        print(count_dict)
        raise KeyError('all categories provided were not found in raster: {}'.format(categories_list))

    return 0

#################################
def check_categories_exist_in_categories_dict(
    categories_list: list,
    categories_dict: dict
    ):
    """
    Description:
    check if all categories can be found in the cateogires_dict

    Args:
        categories_list: list of categories to check
        categories_dict: dict to check

    Return:
        int: 0

    Raise: 
        KeyError: if cateogry not found
    """  
    # check that the categories provided exist
    all_categories = list(categories_dict.keys())
    if not all(cat in all_categories for cat in categories_list):
        raise KeyError('categories given: {} must exist in'
        ' the list: {}'.format(categories_list, all_categories))

    return 0

#################################
def dissagregate_categories(
    categories_list: list,
    categories_dict: dict
    ):
    """
    Description:
    simple function that takes a list of categories and replaces aggregate
    categories with the ones they represent in the list

    NOTE: assumes structure of the categories dict give is the same as the
        wapor_lcc dict created below

    Args:
        categories_list: list of categories to dissagregate
        categories_dict: dict to compare against/use to dissagregate the list

    Return:
        list: disaggregated list
    """  
    # create aggregateless reversed categories dict
    aggregateless_dict = {}
    for key in categories_dict.keys():
        if not isinstance(categories_dict[key], list):
            aggregateless_dict[key] = categories_dict[key]
    categories_dict_reversed = {aggregateless_dict[key] : key for key in aggregateless_dict.keys()}

    # disaggregate aggregate codes
    disaggregated_categories =[]
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

    #if remove_categories or disaggregated_categories:
    print('categories list processed, aggregates removed: {} and '
        'disaggregates added: {}'.format(remove_categories, disaggregated_categories))

    return categories_list

#################################
def wapor_lcc(wapor_level: int):
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
            'Cropland': [41,42,43],
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
            'Non Irrigated Tree Crops':  [1,2,13,14,15,16],
            'Irrigated Tree Crops':  [113,114,115,116],
            'Non Irrigated Crops':  [8,9,10,11,20,21,22,23,24,25,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,50],
            'Irrigated Crops':  [108,109,110,111,120,121,122,123,124,125,126,127,128,129,131,132,133,134,135,136,137,138,139,141,142,143,144,150],
            'Non Crop Cover':  [5,7,12,17,19],
            'Grass and Shrubland':  [4,18],
            'Tree cover (closed)':1,
            'Tree cover (open)':2,
            'Grassland':4,
            'Bare soil':5,
            'Urban/Artificial':7,
            'Wheat':8,
            'Maize':9,
            'Potato':10,
            'Vegetables':11,
            'Fallow':12,
            'Orchard (closed)':13,
            'Olive':14,
            'Grapes':15,
            'Orchard (open)':16,
            'Wetland':17,
            'Shrubland':18,
            'Water':19,
            'Rice':20,
            'Other crop':21,
            'Sugarcane':22,
            'Teff':23,
            'Cotton':24,
            'Clover':25,
            'Onions':26,
            'Carrots':27,
            'Eggplant':28,
            'Flax':29,
            'Non vegetation (reclass)':30,
            'Sugar beet':31,
            'Cassava':32,
            'Sorghum':33,
            'Millet':34,
            'Groundnut':35,
            'Pigeon Pea':36,
            'Chickpea':37,
            'Okra':38,
            'Tomatoes':39,
            'Cropland (reclass)':40,
            'Bananas':41,
            'Cowpea':42,
            'Sesame':43,
            'Soyabean':44,
            'Other perennial':50,
            'Irrigated wheat':108,
            'Irrigated Maize':109,
            'Irrigated Potato':110,
            'Irrigated vegetables':111,
            'Irrigated orchard (closed)':113,
            'Irrigated olive':114,
            'Irrigated grapes':115,
            'Irrigated orchard (open)':116,
            'Irrigated rice':120,
            'Irrigated other crop':121,
            'Irrigated sugar cane':122,
            'Irrigated Teff':123,
            'Irrigated cotton':124,
            'Irrigated clover':125,
            'Irrigated onions':126,
            'Irrigated Carrots':127,
            'Irrigated Eggplant':128,
            'Irrigated Flax':129,
            'Irrigated Sugar beet':131,
            'Irrigated Cassava':132,
            'Irrigated Sorghum':133,
            'Irrigated Millet':134,
            'Irrigated Groundnut':135,
            'Irrigated Pigeon Pea':136,
            'Irrigated Chickpea':137,
            'Irrigated Okra':138,
            'Irrigated Tomatoes':139,
            'Irrigated Bananas':141,
            'Irrigated Cowpea':142,
            'Irrigated Sesame':143,
            'Irrigated Soyabean':144,
            'Irrigated other perennial':150,         
        }

    else:
        raise AttributeError('level needs to be either 1,2 or 3')

    # above the keys are as they are posted in wapor but we will work with them in lower case to make them case insensitive
    lcc_codes = list(lcc_dict.keys())

    for code in lcc_codes:
        lcc_dict[code.lower()] = lcc_dict.pop(code)
    return lcc_dict


   
