#!/usr/bin/env python3
"""Find satellite images that actually overlap with training data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import rasterio
from rasterio.warp import transform_bounds
from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.config import logger


def main():
    """Find overlapping satellite images."""
    logger.info("Searching for satellite images that overlap with training data")
    
    # Load configuration
    config = TrainingDataConfig.from_file(Path("data/config.conf"))
    
    # Load training data to get bounds
    import geopandas as gpd
    gdfs = []
    for path in config.train_data_paths:
        gdf = gpd.read_file(path)
        gdfs.append(gdf)
    
    combined_gdf = gpd.pd.concat(gdfs, ignore_index=True)
    # Filter by classes
    combined_gdf = combined_gdf[combined_gdf[config.column_id].isin(config.classes)]
    # Reproject to target CRS
    target_epsg = 'EPSG:32632'
    if combined_gdf.crs != target_epsg:
        combined_gdf = combined_gdf.to_crs(target_epsg)
    
    training_bounds = combined_gdf.total_bounds
    logger.info(f"Training data bounds in {target_epsg}: {training_bounds}")
    
    # Get all satellite files
    sat_files = config.get_file_paths()
    logger.info(f"Checking {len(sat_files)} satellite files for overlap")
    
    overlapping_files = []
    non_overlapping_sample = []
    
    for i, sat_file in enumerate(sat_files):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(sat_files)} files checked")
            
        try:
            with rasterio.open(sat_file) as src:
                sat_bounds = src.bounds
                sat_crs = src.crs
                
                # Convert satellite bounds to target CRS
                if sat_crs != target_epsg:
                    sat_bounds_target = transform_bounds(sat_crs, target_epsg, *sat_bounds)
                else:
                    sat_bounds_target = sat_bounds
                
                # Check overlap
                overlap = (
                    sat_bounds_target[0] < training_bounds[2] and  # sat_minx < train_maxx
                    sat_bounds_target[2] > training_bounds[0] and  # sat_maxx > train_minx
                    sat_bounds_target[1] < training_bounds[3] and  # sat_miny < train_maxy
                    sat_bounds_target[3] > training_bounds[1]      # sat_maxy > train_miny
                )
                
                if overlap:
                    overlapping_files.append({
                        'file': sat_file,
                        'bounds_original': sat_bounds,
                        'bounds_target': sat_bounds_target,
                        'crs': sat_crs
                    })
                else:
                    if len(non_overlapping_sample) < 3:  # Keep sample for comparison
                        non_overlapping_sample.append({
                            'file': sat_file,
                            'bounds_target': sat_bounds_target,
                            'crs': sat_crs
                        })
                        
        except Exception as e:
            logger.warning(f"Error reading {sat_file}: {e}")
    
    logger.info(f"\nResults:")
    logger.info(f"Overlapping files: {len(overlapping_files)}")
    logger.info(f"Non-overlapping files: {len(sat_files) - len(overlapping_files)}")
    
    if overlapping_files:
        logger.info(f"\nOverlapping satellite files:")
        for i, item in enumerate(overlapping_files[:10]):  # Show first 10
            logger.info(f"  {i+1}. {Path(item['file']).name}")
            logger.info(f"     Bounds: {item['bounds_target']}")
            if i >= 9 and len(overlapping_files) > 10:
                logger.info(f"     ... and {len(overlapping_files) - 10} more")
                break
    else:
        logger.warning(f"\nNO OVERLAPPING SATELLITE FILES FOUND!")
        logger.info(f"\nSample non-overlapping files:")
        for i, item in enumerate(non_overlapping_sample):
            logger.info(f"  {i+1}. {Path(item['file']).name}")
            logger.info(f"     Bounds: {item['bounds_target']}")
        
        logger.info(f"\nTraining data bounds: {training_bounds}")
        logger.info(f"Gap analysis:")
        if non_overlapping_sample:
            sample_bounds = non_overlapping_sample[0]['bounds_target']
            logger.info(f"  Sample sat Y-range: {sample_bounds[1]:.0f} to {sample_bounds[3]:.0f}")
            logger.info(f"  Training Y-range:   {training_bounds[1]:.0f} to {training_bounds[3]:.0f}")
            logger.info(f"  Y-gap: {training_bounds[1] - sample_bounds[3]:.0f} meters")


if __name__ == "__main__":
    main()
