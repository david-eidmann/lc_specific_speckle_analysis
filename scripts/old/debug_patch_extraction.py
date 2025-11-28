#!/usr/bin/env python3
"""Diagnostic script for patch extraction coordinate issues."""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from lc_speckle_analysis.config import logger
from lc_speckle_analysis.data_config import TrainingDataConfig

def main():
    """Debug patch extraction coordinate issues."""
    logger.info("=== Patch Extraction Diagnostic ===")
    
    config_path = Path(__file__).parent.parent / "data" / "config.conf"
    config = TrainingDataConfig.from_file(config_path)
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    
    # Load a buffered polygon and a specific satellite image for 20220611
    global_files = list(cache_dir.glob("global_buffered_polygons*.gpkg"))
    if not global_files:
        logger.error("No global buffered polygons found")
        return
        
    logger.info(f"Loading buffered polygons: {global_files[0].name}")
    polygons = gpd.read_file(global_files[0])
    logger.info(f"Loaded {len(polygons)} polygons")
    
    # Get a sample polygon
    sample_polygon = polygons.iloc[0]
    logger.info(f"Sample polygon CRS: {polygons.crs}")
    logger.info(f"Sample polygon bounds: {sample_polygon.geometry.bounds}")
    
    # Find a 20220611 satellite file
    sat_pattern = "/mnt/cephfs/data/CorDAu/S1/download/preproc_1/data/UTMZ_32N/2022/D139/S1A_IW_GRDH_1SDV_20220611T054106_20220611T054131_043611_0534F5_0A4F_VV.tif"
    
    logger.info(f"Testing with satellite file: {Path(sat_pattern).name}")
    
    try:
        with rasterio.open(sat_pattern) as src:
            logger.info(f"Satellite CRS: {src.crs}")
            logger.info(f"Satellite bounds: {src.bounds}")
            logger.info(f"Satellite transform: {src.transform}")
            
            # Check if polygon bounds overlap with raster bounds
            poly_bounds = sample_polygon.geometry.bounds  # (minx, miny, maxx, maxy)
            raster_bounds = src.bounds  # BoundingBox(left, bottom, right, top)
            
            bounds_overlap = (
                poly_bounds[0] < raster_bounds.right and  # poly_minx < raster_right
                poly_bounds[2] > raster_bounds.left and   # poly_maxx > raster_left
                poly_bounds[1] < raster_bounds.top and    # poly_miny < raster_top
                poly_bounds[3] > raster_bounds.bottom     # poly_maxy > raster_bottom
            )
            
            logger.info(f"Bounds overlap: {bounds_overlap}")
            logger.info(f"Polygon bounds: {poly_bounds}")
            logger.info(f"Raster bounds: {raster_bounds}")
            
            if bounds_overlap:
                # Try to mask the polygon
                logger.info("Attempting to mask polygon...")
                
                # Ensure polygon is in same CRS as raster
                if polygons.crs != src.crs:
                    logger.info(f"Reprojecting polygon from {polygons.crs} to {src.crs}")
                    sample_geom_reprojected = sample_polygon.geometry
                    # For this test, assume they're already in the same CRS
                else:
                    sample_geom_reprojected = sample_polygon.geometry
                
                try:
                    # Attempt masking
                    masked_data, masked_transform = mask(src, [sample_geom_reprojected], crop=True, filled=False)
                    logger.info(f"✅ Masking successful!")
                    logger.info(f"Masked data shape: {masked_data.shape}")
                    logger.info(f"Masked transform: {masked_transform}")
                    
                except Exception as mask_error:
                    logger.error(f"❌ Masking failed: {mask_error}")
                    
                    # Additional diagnostics
                    logger.info("=== Additional Diagnostics ===")
                    
                    # Check if polygon is valid
                    logger.info(f"Polygon is valid: {sample_polygon.geometry.is_valid}")
                    logger.info(f"Polygon area: {sample_polygon.geometry.area:.2f} m²")
                    
                    # Check polygon type
                    logger.info(f"Polygon type: {type(sample_polygon.geometry)}")
                    
                    # Try to get pixel coordinates of polygon center
                    center_x, center_y = sample_polygon.geometry.centroid.x, sample_polygon.geometry.centroid.y
                    logger.info(f"Polygon centroid: ({center_x:.2f}, {center_y:.2f})")
                    
                    # Convert to pixel coordinates
                    try:
                        pixel_x, pixel_y = ~src.transform * (center_x, center_y)
                        logger.info(f"Centroid in pixel coords: ({pixel_x:.2f}, {pixel_y:.2f})")
                        logger.info(f"Raster shape: {src.shape}")
                        
                        # Check if pixel is within raster bounds
                        within_raster = (0 <= pixel_x < src.shape[1] and 0 <= pixel_y < src.shape[0])
                        logger.info(f"Centroid within raster: {within_raster}")
                        
                    except Exception as coord_error:
                        logger.error(f"Coordinate conversion failed: {coord_error}")
                        
            else:
                logger.warning("No bounds overlap - this explains the masking failure")
                
    except Exception as e:
        logger.error(f"Error opening satellite file: {e}")

if __name__ == "__main__":
    main()
