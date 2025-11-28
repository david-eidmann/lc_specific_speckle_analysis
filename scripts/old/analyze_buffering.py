#!/usr/bin/env python3
"""Analyze polygon buffering effects."""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import geopandas as gpd
from lc_speckle_analysis.config import logger
from lc_speckle_analysis.data_config import TrainingDataConfig
import numpy as np

def main():
    """Analyze buffering effects on polygons."""
    logger.info("=== Polygon Buffering Analysis ===")
    
    config_path = Path(__file__).parent.parent / "data" / "config.conf"
    config = TrainingDataConfig.from_file(config_path)
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    
    # Load original training data
    logger.info(f"Loading original training data from: {config.training_data_path}")
    original_gdf = gpd.read_file(config.training_data_path)
    
    # Filter by classes and reproject
    class_filtered = original_gdf[original_gdf[config.column_id].isin(config.classes)]
    logger.info(f"Original polygons (class filtered): {len(class_filtered)}")
    
    # Reproject to EPSG:32632
    reprojected = class_filtered.to_crs("EPSG:32632")
    logger.info(f"Original area (sum): {reprojected.geometry.area.sum() / 1e6:.1f} km²")
    logger.info(f"Original area statistics:")
    areas = reprojected.geometry.area
    logger.info(f"  Mean: {areas.mean():.1f} m²")
    logger.info(f"  Median: {areas.median():.1f} m²")
    logger.info(f"  Min: {areas.min():.1f} m²")
    logger.info(f"  Max: {areas.max():.1f} m²")
    
    # Check buffering effect
    patch_size = config.patch_size
    buffer_distance = -(np.sqrt(2) * 10 * patch_size)  # Negative for inward buffer
    logger.info(f"Buffer distance: {buffer_distance:.1f} m (inward)")
    
    logger.info("Analyzing buffering effects...")
    
    # Check how many polygons survive buffering
    buffered_geoms = []
    survived_count = 0
    too_small_count = 0
    
    for idx, geom in enumerate(reprojected.geometry):
        try:
            buffered = geom.buffer(buffer_distance)
            if buffered.is_empty or buffered.area < 1:  # Less than 1 m²
                too_small_count += 1
            else:
                buffered_geoms.append(buffered)
                survived_count += 1
        except Exception as e:
            logger.warning(f"Buffering failed for polygon {idx}: {e}")
            too_small_count += 1
    
    logger.info(f"Polygons surviving buffering: {survived_count}")
    logger.info(f"Polygons too small after buffering: {too_small_count}")
    logger.info(f"Survival rate: {survived_count / len(reprojected) * 100:.1f}%")
    
    if buffered_geoms:
        buffered_areas = [geom.area for geom in buffered_geoms]
        total_buffered_area = sum(buffered_areas)
        logger.info(f"Total buffered area: {total_buffered_area / 1e6:.1f} km²")
        logger.info(f"Buffered area statistics:")
        logger.info(f"  Mean: {np.mean(buffered_areas):.1f} m²")
        logger.info(f"  Median: {np.median(buffered_areas):.1f} m²")
        logger.info(f"  Min: {min(buffered_areas):.1f} m²")
        logger.info(f"  Max: {max(buffered_areas):.1f} m²")
    
    # Load cached global buffered polygons
    global_files = list(cache_dir.glob("global_buffered_polygons*.gpkg"))
    if global_files:
        logger.info(f"\nCached global buffered polygons: {len(global_files[0])}")
        cached_global = gpd.read_file(global_files[0])
        logger.info(f"Cached polygons: {len(cached_global)}")
        logger.info(f"Cached total area: {cached_global.geometry.area.sum() / 1e6:.1f} km²")

if __name__ == "__main__":
    main()
