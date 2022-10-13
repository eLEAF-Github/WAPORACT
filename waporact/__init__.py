########################################################################################################################
#############################                       tools                                #############################
########################################################################################################################

from waporact.scripts.tools.raster import (
    reproject_coordinates,
    reproject_geotransform,
    area_of_latlon_pixel,
    check_gdal_open,
    set_band_descriptions,
    gdal_info,
    check_dimensions,
    raster_to_array,
    count_raster_values,
    array_to_raster,
    retrieve_raster_crs,
    rasterize_vector,
    create_polygon_index_dict,
    create_values_specific_mask,
    mask_raster,
    match_raster,
    build_vrt,
)

from waporact.scripts.tools.vector import (
    file_to_records,
    records_to_vector,
    geodataframe_to_vector_file,
    retrieve_geodataframe_bbox,
    retrieve_geodataframe_central_coords,
    get_plotting_zoom_level_and_central_coords_from_gdf,
    retrieve_vector_crs,
    compare_raster_vector_crs,
    vector_reprojection,
    create_bbox_polygon,
    create_polygon,
    delete_shapefile,
    copy_shapefile,
    check_add_wpid_to_shapefile,
    add_matched_values_to_shapefile,
    check_column_exists,
    create_spatial_index,
    calc_lat_lon_polygon_area,
    polygon_area_drop,
    fill_small_polygon_holes,
    check_for_overlap,
    overlap_among_features,
    union_and_drop,
    polygonize_cleanup,
    raster_to_polygon,
)

from waporact.scripts.tools.statistics import (
    dict_to_dataframe,
    output_table,
    latlon_dist,
    ceiling_divide,
    floor_minus,
    generate_zonal_stats_column_and_function,
    calc_field_statistics,
    multiple_raster_zonal_stats,
    equal_dimensions_zonal_stats,
    single_raster_zonal_stats,
    raster_count_statistics,
    calc_dual_array_statistics,
    calc_single_array_numpy_statistic,
    calc_multiple_array_numpy_statistic,
    mostcommonzaxis,
)

from waporact.scripts.tools.plots import (
    roundup_max,
    calculate_max_and_min,
    generate_hover_template_chloro,
    spiderplot,
    bargraph,
    scatterplot,
    violinplot,
    piechart,
    interactive_categorical_map,
    interactive_choropleth_map,
    rasterplot,
    shapeplot,
)

from waporact.scripts.tools.logger import format_root_logger

########################################################################################################################
#############################                       structure                              #############################
########################################################################################################################

from waporact.scripts.structure.wapor_structure import (
    WaporStructure,
    check_windows_file_length,
)

########################################################################################################################
#############################                       retrieval                              #############################
########################################################################################################################

from waporact.scripts.retrieval.wapor_retrieval_support import (
    wapor_lcc,
    dissagregate_categories,
    check_categories_exist_in_count_dict,
    check_categories_exist_in_categories_dict,
    check_datacomponent_availability,
    generate_wapor_cube_code,
)

from waporact.scripts.retrieval.wapor_retrieval_masking import (
    create_raster_mask_from_wapor_landcover_rasters,
    check_bbox_overlaps_l3_location,
    create_raster_mask_from_shapefile_and_template_raster,
)

from waporact.scripts.retrieval.wapor_api import WaporAPI

from waporact.scripts.retrieval.wapor_retrieval import WaporRetrieval, printWaitBar

########################################################################################################################
#############################                       pipeline                               #############################
########################################################################################################################

from waporact.scripts.pipelines.wapor_pai import WaporPAI
