#!/usr/bin/env python3
"""Check which satellite files are actually being used by the system."""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import geopandas as gpd
from lc_speckle_analysis.config import logger
from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.patch_yielder import PatchYielder

def main():
    """Check actual satellite files being used."""
    logger.info("=== Checking Actual Satellite Files ===")
    
    config_path = Path(__file__).parent.parent / "data" / "config.conf"
    config = TrainingDataConfig.from_file(config_path)
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    
    # Load filtered polygons that are actually being used
    global_files = list(cache_dir.glob("global_buffered_polygons*.gpkg"))
    if global_files:
        polygons = gpd.read_file(global_files[0])
        # Filter to the ones that intersect with AOIs (like the system does)
        
        # Load an AOI to see which polygons are actually selected
        aoi_files = list((cache_dir / "valid_aoi").glob("valid_aoi_D139_S1A_IW_GRDH_*.gpkg"))
        # Filter out intermediate files
        final_aoi_files = [f for f in aoi_files if not any(suffix in f.name for suffix in ['raw_polygon', 'reprojected'])]
        
        if final_aoi_files:
            logger.info(f"Checking AOI: {final_aoi_files[0].name}")
            aoi = gpd.read_file(final_aoi_files[0])
            logger.info(f"AOI bounds: {aoi.total_bounds}")
            
            # Find polygons that intersect with this AOI
            intersecting_mask = polygons.intersects(aoi.geometry[0])
            intersecting_polygons = polygons[intersecting_mask]
            
            if len(intersecting_polygons) > 0:
                logger.info(f"Found {len(intersecting_polygons)} intersecting polygons")
                sample_intersecting = intersecting_polygons.iloc[0]
                logger.info(f"Sample intersecting polygon bounds: {sample_intersecting.geometry.bounds}")
            else:
                logger.warning("No intersecting polygons found!")
                
    # Now check what satellite files the PatchYielder is actually using
    logger.info("\n=== Checking PatchYielder's actual satellite files ===")
    
    try:
        # Create a minimal PatchYielder to see what it loads
        patch_yielder = PatchYielder(config, seed=42)
        
        logger.info(f"Number of image tuples: {len(patch_yielder.image_tuples)}")
        
        for i, image_tuple in enumerate(patch_yielder.image_tuples):
            logger.info(f"Image tuple {i+1}:")
            logger.info(f"  VV file: {image_tuple.vv_path.name}")
            logger.info(f"  VH file: {image_tuple.vh_path.name}")
            logger.info(f"  Date: {image_tuple.date}")
            
            # Check bounds of this specific file
            import rasterio
            try:
                with rasterio.open(image_tuple.vv_path) as src:
                    logger.info(f"  Bounds: {src.bounds}")
            except Exception as e:
                logger.error(f"  Error reading bounds: {e}")
                
    except Exception as e:
        logger.error(f"Error creating PatchYielder: {e}")

if __name__ == "__main__":
    main()
