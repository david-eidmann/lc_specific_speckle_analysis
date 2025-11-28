#!/usr/bin/env python3
"""Diagnostic script to understand AOI vs polygon mismatch."""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import geopandas as gpd
from lc_speckle_analysis.config import logger
from lc_speckle_analysis.data_config import TrainingDataConfig

def main():
    """Diagnose AOI polygon mismatch."""
    logger.info("=== AOI vs Polygon Diagnostic ===")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "data" / "config.conf"
    config = TrainingDataConfig.from_file(config_path)
    
    # Check what we have in cache
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    
    # 1. Check globally buffered polygons
    global_files = list(cache_dir.glob("global_buffered_polygons*.gpkg"))
    if global_files:
        logger.info(f"Found global buffered polygons: {global_files[0].name}")
        global_polygons = gpd.read_file(global_files[0])
        logger.info(f"Global polygons: {len(global_polygons)} polygons")
        logger.info(f"Global polygons CRS: {global_polygons.crs}")
        global_bounds = global_polygons.total_bounds
        logger.info(f"Global polygons bounds: {global_bounds}")
        logger.info(f"Global polygons area: {global_polygons.geometry.area.sum() / 1e6:.1f} km²")
        
    # 2. Check valid AOIs
    valid_aoi_dir = cache_dir / "valid_aoi"
    if valid_aoi_dir.exists():
        aoi_files = list(valid_aoi_dir.glob("valid_aoi_D139_S1A_IW_GRDH_*.gpkg"))
        # Filter out intermediate files (raw_polygon, reprojected)
        final_aoi_files = [f for f in aoi_files if not any(suffix in f.name for suffix in ['raw_polygon', 'reprojected'])]
        
        logger.info(f"Found {len(final_aoi_files)} final AOI files")
        
        for i, aoi_file in enumerate(final_aoi_files[:3]):  # Check first 3
            logger.info(f"\nAOI {i+1}: {aoi_file.name}")
            aoi = gpd.read_file(aoi_file)
            logger.info(f"  CRS: {aoi.crs}")
            aoi_bounds = aoi.total_bounds
            logger.info(f"  Bounds: {aoi_bounds}")
            logger.info(f"  Area: {aoi.geometry.area.sum() / 1e6:.1f} km²")
            
            # Check if global polygons and AOI even overlap in bounds
            if global_files:
                bounds_overlap = (
                    global_bounds[0] < aoi_bounds[2] and global_bounds[2] > aoi_bounds[0] and
                    global_bounds[1] < aoi_bounds[3] and global_bounds[3] > aoi_bounds[1]
                )
                logger.info(f"  Bounds overlap with global polygons: {bounds_overlap}")
                
                if bounds_overlap:
                    # Check actual geometric intersection
                    logger.info("  Computing geometric intersections...")
                    intersections = global_polygons.intersects(aoi.geometry[0])
                    intersecting_count = intersections.sum()
                    logger.info(f"  Geometrically intersecting polygons: {intersecting_count}")
                    
                    if intersecting_count > 0:
                        # Show some details about intersecting polygons
                        intersecting_polys = global_polygons[intersections]
                        logger.info(f"  Sample intersecting polygon bounds: {intersecting_polys.iloc[0].geometry.bounds}")
    
    # 3. Check combined AOI
    combined_files = list(cache_dir.glob("combined_aoi*.gpkg"))
    if combined_files:
        logger.info(f"\nFound combined AOI: {combined_files[0].name}")
        combined_aoi = gpd.read_file(combined_files[0])
        logger.info(f"Combined AOI area: {combined_aoi.geometry.area.sum() / 1e6:.1f} km²")
        combined_bounds = combined_aoi.total_bounds
        logger.info(f"Combined AOI bounds: {combined_bounds}")
        
        if global_files:
            # Check intersection with combined AOI
            intersections = global_polygons.intersects(combined_aoi.geometry[0])
            intersecting_count = intersections.sum()
            logger.info(f"Polygons intersecting with combined AOI: {intersecting_count}")

if __name__ == "__main__":
    main()
