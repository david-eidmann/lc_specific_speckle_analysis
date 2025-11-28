#!/usr/bin/env python3
"""
Create a GPKG file from cached train/validation/test patches.
This script reads the pickle cache files containing full Patch objects and creates 
geometries with real spatial coordinates for each patch.
"""
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import pickle
from datetime import datetime
import rasterio.transform

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent

def read_patch_cache_file(pkl_file):
    """Read a single pickle cache file and extract Patch objects."""
    try:
        with open(pkl_file, 'rb') as f:
            patches = pickle.load(f)
        
        logger.info(f"Loaded {len(patches)} patches from {pkl_file.name}")
        return patches
    except Exception as e:
        logger.error(f"Error reading {pkl_file}: {e}")
        return None

def create_patch_geometries(patches, mode, cache_filename):
    """Create point and bounds geometries for patches using their real spatial information."""
    patch_records = []
    
    for i, patch in enumerate(patches):
        try:
            # Extract patch metadata
            patch_size_pixels = patch.data.shape[0]  # Assuming square patches
            
            # Calculate center coordinates from transform
            # Transform gives us the top-left corner, we need center
            center_x, center_y = rasterio.transform.xy(
                patch.transform, 
                patch_size_pixels // 2, 
                patch_size_pixels // 2
            )
            
            # Create point geometry for patch center
            center_point = Point(center_x, center_y)
            
            # Create bounding box geometry
            bounds_box = box(*patch.bounds)
            
            # Calculate patch area and pixel size
            pixel_size = abs(patch.transform[0])  # Pixel width in CRS units
            patch_area = bounds_box.area
            
            # Handle src_files which might be strings or Path objects
            vv_file = 'unknown'
            vh_file = 'unknown'
            if len(patch.src_files) > 0:
                vv_src = patch.src_files[0]
                vv_file = vv_src.name if hasattr(vv_src, 'name') else Path(str(vv_src)).name
            if len(patch.src_files) > 1:
                vh_src = patch.src_files[1]
                vh_file = vh_src.name if hasattr(vh_src, 'name') else Path(str(vh_src)).name
            
            record = {
                'patch_id': i,
                'class_id': int(patch.class_id),
                'data_mode': patch.data_mode.value if hasattr(patch.data_mode, 'value') else str(patch.data_mode),
                'date': patch.date,
                'orbit': patch.orbit,
                'vv_file': vv_file,
                'vh_file': vh_file,
                'patch_size_pixels': patch_size_pixels,
                'pixel_size_meters': pixel_size,
                'patch_area_sqm': patch_area,
                'bounds_minx': patch.bounds[0],
                'bounds_miny': patch.bounds[1],
                'bounds_maxx': patch.bounds[2],
                'bounds_maxy': patch.bounds[3],
                'center_x': center_x,
                'center_y': center_y,
                'crs': patch.crs,
                'cache_file': cache_filename,
                'geometry_center': center_point,
                'geometry_bounds': bounds_box
            }
            
            patch_records.append(record)
            
        except Exception as e:
            logger.warning(f"Error processing patch {i}: {e}")
            continue
    
    return patch_records

def create_patches_gpkg():
    """Create a GPKG file from all cached patch files."""
    cache_dir = PROJECT_ROOT / "data" / "cache" / "patches"
    
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return
    
    all_patch_records = []
    
    # Process each mode (train, validation, test)
    for mode_dir in cache_dir.iterdir():
        if mode_dir.is_dir():
            mode = mode_dir.name
            logger.info(f"Processing {mode} patches...")
            
            pkl_files = list(mode_dir.glob("*.pkl"))
            logger.info(f"Found {len(pkl_files)} cache files for {mode}")
            
            for pkl_file in pkl_files:
                patches = read_patch_cache_file(pkl_file)
                
                if patches is not None:
                    # Create records for this cache file
                    records = create_patch_geometries(patches, mode, pkl_file.name)
                    
                    all_patch_records.extend(records)
                    logger.info(f"Added {len(records)} patch records from {pkl_file.name}")
    
    if not all_patch_records:
        logger.error("No patch records found!")
        return
    
    logger.info(f"Total patch records: {len(all_patch_records)}")
    
    # Group records by data mode
    records_by_mode = {}
    for record in all_patch_records:
        mode = record['data_mode']
        if mode not in records_by_mode:
            records_by_mode[mode] = []
        records_by_mode[mode].append(record)
    
    # Create output directory
    output_dir = PROJECT_ROOT / "data" / "output"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    created_files = []
    
    # Create separate GPKG files for each mode
    for mode, mode_records in records_by_mode.items():
        logger.info(f"Creating GPKG files for {mode} mode ({len(mode_records)} patches)")
        
        # Separate center and bounds records for this mode
        center_records = []
        bounds_records = []
        
        for record in mode_records:
            # Center point record
            center_rec = record.copy()
            center_rec['geometry'] = center_rec.pop('geometry_center')
            center_rec.pop('geometry_bounds')
            center_records.append(center_rec)
            
            # Bounds record
            bounds_rec = record.copy()
            bounds_rec['geometry'] = bounds_rec.pop('geometry_bounds')
            bounds_rec.pop('geometry_center')
            bounds_records.append(bounds_rec)
        
        # Create GeoDataFrames for this mode
        first_crs = mode_records[0]['crs']
        center_gdf = gpd.GeoDataFrame(center_records, crs=first_crs)
        bounds_gdf = gpd.GeoDataFrame(bounds_records, crs=first_crs)
        
        # Save to GPKG files
        center_file = output_dir / f"patches_{mode}_centers_{timestamp}.gpkg"
        bounds_file = output_dir / f"patches_{mode}_bounds_{timestamp}.gpkg"
        
        logger.info(f"Saving {mode} patch centers to: {center_file.name}")
        center_gdf.to_file(center_file, driver='GPKG')
        
        logger.info(f"Saving {mode} patch bounds to: {bounds_file.name}")
        bounds_gdf.to_file(bounds_file, driver='GPKG')
        
        created_files.extend([center_file, bounds_file])
        
        # Print mode-specific statistics
        logger.info(f"=== {mode.upper()} MODE SUMMARY ===")
        logger.info(f"Total {mode} patches: {len(center_gdf)}")
        
        class_counts = center_gdf['class_id'].value_counts().sort_index()
        logger.info(f"{mode} class distribution:")
        for class_id, count in class_counts.items():
            logger.info(f"  Class {class_id}: {count} patches")
    
    # Overall summary
    logger.info("=== OVERALL SUMMARY ===")
    total_patches = sum(len(records) for records in records_by_mode.values())
    logger.info(f"Total patches across all modes: {total_patches}")
    
    for mode, records in records_by_mode.items():
        logger.info(f"  {mode}: {len(records)} patches")
    
    logger.info("Created files:")
    for file_path in created_files:
        logger.info(f"  {file_path}")
    
    return created_files

if __name__ == "__main__":
    logger.info("Starting GPKG creation from cached patches...")
    create_patches_gpkg()
    logger.info("GPKG creation completed!")
