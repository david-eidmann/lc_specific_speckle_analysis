#!/usr/bin/env python3
"""Debug coordinate alignment between AOIs, polygons, and satellite images."""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import geopandas as gpd
import rasterio
from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.patch_yielder import PatchYielder
from lc_speckle_analysis.config import logger


def main():
    """Debug coordinate alignment issues."""
    logger.info("Starting coordinate alignment debug")
    
    # Load configuration
    config = TrainingDataConfig.from_file(Path("data/config.conf"))
    
    # Create patch yielder
    yielder = PatchYielder(config, seed=42)
    
    # Get the first image tuple and its AOI
    first_tuple = yielder.image_tuples[0]
    first_aoi = yielder.tuple_aois[first_tuple.date]
    
    logger.info(f"Analyzing tuple: {first_tuple.date}")
    logger.info(f"VV file: {first_tuple.vv_path}")
    logger.info(f"VH file: {first_tuple.vh_path}")
    
    # Check satellite image properties
    with rasterio.open(first_tuple.vv_path) as src:
        sat_bounds = src.bounds
        sat_crs = src.crs
        sat_transform = src.transform
        sat_width = src.width
        sat_height = src.height
        
    logger.info(f"Satellite image properties:")
    logger.info(f"  CRS: {sat_crs}")
    logger.info(f"  Bounds: {sat_bounds}")
    logger.info(f"  Width x Height: {sat_width} x {sat_height}")
    logger.info(f"  Transform: {sat_transform}")
    
    # Check AOI properties
    aoi_bounds = first_aoi.total_bounds
    aoi_crs = first_aoi.crs
    
    logger.info(f"AOI properties:")
    logger.info(f"  CRS: {aoi_crs}")
    logger.info(f"  Bounds: {aoi_bounds}")
    logger.info(f"  Area: {first_aoi.geometry.iloc[0].area / 1e6:.1f} km²")
    
    # Check training data properties
    training_bounds = yielder.gdf.total_bounds
    training_crs = yielder.gdf.crs
    
    logger.info(f"Training data properties:")
    logger.info(f"  CRS: {training_crs}")
    logger.info(f"  Bounds: {training_bounds}")
    logger.info(f"  Total polygons: {len(yielder.gdf)}")
    
    # Check intersecting polygons with AOI
    intersecting_mask = yielder.gdf.intersects(first_aoi.geometry.iloc[0])
    intersecting_polygons = yielder.gdf[intersecting_mask]
    
    logger.info(f"Polygons intersecting with AOI: {len(intersecting_polygons)}")
    
    if len(intersecting_polygons) > 0:
        # Get first intersecting polygon
        first_polygon = intersecting_polygons.iloc[0]
        polygon_bounds = first_polygon.geometry.bounds
        
        logger.info(f"First intersecting polygon:")
        logger.info(f"  Bounds: {polygon_bounds}")
        logger.info(f"  Class: {first_polygon[config.column_id]}")
        
        # Convert satellite bounds to target CRS for comparison
        if sat_crs != yielder.target_epsg:
            from rasterio.warp import transform_bounds
            sat_bounds_target_crs = transform_bounds(sat_crs, yielder.target_epsg, *sat_bounds)
            logger.info(f"Satellite bounds in {yielder.target_epsg}: {sat_bounds_target_crs}")
        else:
            sat_bounds_target_crs = sat_bounds
            
        # Check overlaps
        aoi_sat_overlap = (
            aoi_bounds[0] < sat_bounds_target_crs[2] and  # aoi_minx < sat_maxx
            aoi_bounds[2] > sat_bounds_target_crs[0] and  # aoi_maxx > sat_minx
            aoi_bounds[1] < sat_bounds_target_crs[3] and  # aoi_miny < sat_maxy
            aoi_bounds[3] > sat_bounds_target_crs[1]      # aoi_maxy > sat_miny
        )
        
        polygon_sat_overlap = (
            polygon_bounds[0] < sat_bounds_target_crs[2] and
            polygon_bounds[2] > sat_bounds_target_crs[0] and
            polygon_bounds[1] < sat_bounds_target_crs[3] and
            polygon_bounds[3] > sat_bounds_target_crs[1]
        )
        
        logger.info(f"Overlap analysis:")
        logger.info(f"  AOI overlaps with satellite: {aoi_sat_overlap}")
        logger.info(f"  Polygon overlaps with satellite: {polygon_sat_overlap}")
        
        # Try actual masking to see the exact error
        try:
            from rasterio.mask import mask
            logger.info("Attempting to mask polygon with satellite image...")
            
            with rasterio.open(first_tuple.vv_path) as src:
                masked_data, mask_transform = mask(src, [first_polygon.geometry], crop=True, filled=False)
                logger.info(f"SUCCESS: Masking worked! Masked shape: {masked_data.shape}")
                
        except Exception as e:
            logger.error(f"MASKING FAILED: {e}")
            
        # Check if polygon needs reprojection for masking
        if first_polygon.geometry.__geo_interface__['coordinates']:
            coords = first_polygon.geometry.__geo_interface__['coordinates']
            if isinstance(coords[0], list) and isinstance(coords[0][0], list):
                sample_coord = coords[0][0]
            else:
                sample_coord = coords[0]
            logger.info(f"Sample polygon coordinate: {sample_coord}")
            
    else:
        logger.error("No polygons intersect with AOI - this explains the masking failure!")
        
    # Check spatial relationship between AOI and training polygons
    logger.info("Detailed spatial analysis:")
    logger.info(f"AOI X range: {aoi_bounds[0]:.0f} to {aoi_bounds[2]:.0f}")
    logger.info(f"AOI Y range: {aoi_bounds[1]:.0f} to {aoi_bounds[3]:.0f}")
    logger.info(f"Training data X range: {training_bounds[0]:.0f} to {training_bounds[2]:.0f}")
    logger.info(f"Training data Y range: {training_bounds[1]:.0f} to {training_bounds[3]:.0f}")
    
    if sat_crs != yielder.target_epsg:
        logger.info(f"Satellite X range (in {yielder.target_epsg}): {sat_bounds_target_crs[0]:.0f} to {sat_bounds_target_crs[2]:.0f}")
        logger.info(f"Satellite Y range (in {yielder.target_epsg}): {sat_bounds_target_crs[1]:.0f} to {sat_bounds_target_crs[3]:.0f}")


if __name__ == "__main__":
    main()
