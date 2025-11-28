#!/usr/bin/env python3
"""Create GPKG files with geometries of all cached patches for train/test/validation."""

import sys
from pathlib import Path
import logging
import geopandas as gpd
import pandas as pd
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lc_speckle_analysis.config import logger


def collect_patch_geometries(cache_mode_dir: Path) -> gpd.GeoDataFrame:
    """Collect all patch geometries from a cache mode directory.
    
    Args:
        cache_mode_dir: Path to cache directory (e.g., data/cache/patches/train)
        
    Returns:
        Combined GeoDataFrame with all patch geometries
    """
    logger.info(f"Collecting patch geometries from: {cache_mode_dir}")
    
    all_gdfs = []
    hash_dirs = [d for d in cache_mode_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(hash_dirs)} cache hash directories")
    
    for hash_dir in hash_dirs:
        # Find all GPKG files in this hash directory
        gpkg_files = list(hash_dir.glob("patches_*.gpkg"))
        
        for gpkg_file in gpkg_files:
            try:
                logger.debug(f"Reading: {gpkg_file.name}")
                gdf = gpd.read_file(gpkg_file)
                
                # Add cache information
                gdf['cache_hash'] = hash_dir.name
                gdf['cache_file'] = gpkg_file.name
                
                all_gdfs.append(gdf)
                
            except Exception as e:
                logger.warning(f"Could not read {gpkg_file}: {e}")
    
    if not all_gdfs:
        logger.warning(f"No patch geometries found in {cache_mode_dir}")
        return gpd.GeoDataFrame()
    
    # Combine all GeoDataFrames
    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    logger.info(f"Combined {len(combined_gdf)} patch geometries")
    
    return combined_gdf


def create_summary_stats(gdf: gpd.GeoDataFrame, mode: str) -> Dict:
    """Create summary statistics for patch geometries.
    
    Args:
        gdf: GeoDataFrame with patch geometries
        mode: Data mode (train/validation/test)
        
    Returns:
        Dictionary with summary statistics
    """
    if len(gdf) == 0:
        return {"mode": mode, "total_patches": 0}
    
    stats = {
        "mode": mode,
        "total_patches": len(gdf),
        "unique_cache_dirs": len(gdf['cache_hash'].unique()) if 'cache_hash' in gdf.columns else 0,
        "class_distribution": {str(k): int(v) for k, v in gdf['class'].value_counts().sort_index().items()} if 'class' in gdf.columns else {},
        "date_distribution": {str(k): int(v) for k, v in gdf['date'].value_counts().sort_index().items()} if 'date' in gdf.columns else {},
        "orbit_distribution": {str(k): int(v) for k, v in gdf['orbit'].value_counts().items()} if 'orbit' in gdf.columns else {},
        "bounds": {
            "minx": float(gdf.total_bounds[0]),
            "miny": float(gdf.total_bounds[1]),
            "maxx": float(gdf.total_bounds[2]),
            "maxy": float(gdf.total_bounds[3])
        },
        "total_area_km2": float(gdf.geometry.area.sum() / 1e6)  # Convert to km²
    }
    
    return stats


def main():
    """Main function to create patch geometry GPKG files."""
    logger.info("Starting patch geometry GPKG creation")
    
    # Set up paths
    cache_dir = Path("data/cache/patches")
    output_dir = Path("data/patch_geometries")
    output_dir.mkdir(exist_ok=True)
    
    modes = ["train", "validation", "test"]
    all_stats = {}
    
    for mode in modes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {mode.upper()} patches")
        logger.info(f"{'='*50}")
        
        mode_dir = cache_dir / mode
        
        if not mode_dir.exists():
            logger.warning(f"Cache directory not found: {mode_dir}")
            continue
        
        # Collect patch geometries
        gdf = collect_patch_geometries(mode_dir)
        
        if len(gdf) == 0:
            logger.warning(f"No geometries found for {mode}")
            continue
        
        # Create output GPKG
        output_file = output_dir / f"{mode}_patch_geometries.gpkg"
        
        # Add metadata columns
        gdf['data_mode'] = mode
        gdf['created_date'] = pd.Timestamp.now().isoformat()
        
        # Reorder columns for better readability
        column_order = ['class', 'date', 'orbit', 'data_mode', 'src_vv', 'src_vh', 
                       'cache_hash', 'cache_file', 'created_date', 'geometry']
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in gdf.columns]
        remaining_columns = [col for col in gdf.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        gdf = gdf[final_columns]
        
        # Save to GPKG
        logger.info(f"Saving {len(gdf)} geometries to: {output_file}")
        gdf.to_file(output_file, driver='GPKG')
        
        # Create summary statistics
        stats = create_summary_stats(gdf, mode)
        all_stats[mode] = stats
        
        # Log statistics
        logger.info(f"Statistics for {mode}:")
        logger.info(f"  Total patches: {stats['total_patches']}")
        logger.info(f"  Cache directories: {stats.get('unique_cache_dirs', 0)}")
        logger.info(f"  Classes: {stats.get('class_distribution', {})}")
        logger.info(f"  Total area: {stats.get('total_area_km2', 0):.1f} km²")
        
        if stats.get('bounds'):
            bounds = stats['bounds']
            logger.info(f"  Bounds: ({bounds['minx']:.1f}, {bounds['miny']:.1f}) to ({bounds['maxx']:.1f}, {bounds['maxy']:.1f})")
    
    # Save combined statistics
    stats_file = output_dir / "patch_statistics.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*50}")
    
    total_patches = sum(stats.get('total_patches', 0) for stats in all_stats.values())
    logger.info(f"Total patches across all modes: {total_patches}")
    
    for mode, stats in all_stats.items():
        logger.info(f"{mode}: {stats.get('total_patches', 0)} patches")
    
    logger.info(f"\nOutput files created in: {output_dir}")
    logger.info(f"Statistics saved to: {stats_file}")
    
    # List created files
    created_files = list(output_dir.glob("*.gpkg")) + [stats_file]
    logger.info(f"Created files:")
    for file in created_files:
        logger.info(f"  {file}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
