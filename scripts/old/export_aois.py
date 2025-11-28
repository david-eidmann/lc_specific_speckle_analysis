#!/usr/bin/env python3
"""Export AOIs from 20220611 to GPKG for visualization."""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import geopandas as gpd
import pandas as pd
from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.patch_yielder import PatchYielder
from lc_speckle_analysis.config import logger


def main():
    """Export AOIs to GPKG."""
    logger.info("Exporting AOIs from 20220611 to GPKG")
    
    # Load configuration
    config = TrainingDataConfig.from_file(Path("data/config.conf"))
    logger.info(f"Config loaded - date: {config.dates}")
    
    # Create patch yielder to get AOIs
    yielder = PatchYielder(config, seed=42)
    
    logger.info(f"Found {len(yielder.image_tuples)} image tuples")
    logger.info(f"Found {len(yielder.tuple_aois)} AOIs")
    
    # Prepare data for GPKG export
    aoi_records = []
    
    for i, (unique_key, aoi_gdf) in enumerate(yielder.tuple_aois.items()):
        logger.info(f"Processing AOI {i+1}: {unique_key}")
        
        # Parse the unique key to get date and index
        if '_' in unique_key:
            date_part, index_part = unique_key.rsplit('_', 1)
            try:
                tuple_index = int(index_part)
            except ValueError:
                tuple_index = i  # Fallback
        else:
            date_part = unique_key
            tuple_index = i
        
        # Get corresponding image tuple info
        if tuple_index < len(yielder.image_tuples):
            tuple_obj = yielder.image_tuples[tuple_index]
            
            # Extract meaningful info from file paths
            vv_name = tuple_obj.vv_path.name
            vh_name = tuple_obj.vh_path.name
            
            # Extract time info from filename (e.g., T054221)
            import re
            time_match = re.search(r'T(\d{6})', vv_name)
            time_str = time_match.group(1) if time_match else "unknown"
        else:
            vv_name = "unknown_vv"
            vh_name = "unknown_vh"  
            time_str = "unknown"
        
        # Create record for each AOI
        record = {
            'aoi_id': unique_key,
            'date': date_part,
            'time': time_str,
            'vv_file': vv_name,
            'vh_file': vh_name,
            'area_km2': aoi_gdf.geometry.iloc[0].area / 1e6,
            'geometry': aoi_gdf.geometry.iloc[0]
        }
        aoi_records.append(record)
        
        logger.info(f"  AOI {unique_key}: {time_str}, area: {record['area_km2']:.1f} km²")
    
    # Create GeoDataFrame
    if aoi_records:
        aois_gdf = gpd.GeoDataFrame(aoi_records, crs=yielder.target_epsg)
        
        # Add training data bounds for reference
        training_bounds = yielder.gdf.total_bounds
        from shapely.geometry import box
        training_bbox = box(training_bounds[0], training_bounds[1], training_bounds[2], training_bounds[3])
        
        # Add training bounds as a separate feature
        training_record = {
            'aoi_id': 'training_bounds',
            'date': 'reference',
            'time': 'reference',
            'vv_file': 'training_data_bounds',
            'vh_file': 'training_data_bounds',
            'area_km2': training_bbox.area / 1e6,
            'geometry': training_bbox
        }
        
        # Combine AOIs and training bounds
        all_records = aoi_records + [training_record]
        combined_gdf = gpd.GeoDataFrame(all_records, crs=yielder.target_epsg)
        
        # Export to GPKG
        output_file = Path("data") / "aois_20220611_export.gpkg"
        combined_gdf.to_file(output_file, driver='GPKG')
        
        logger.info(f"Exported {len(aoi_records)} AOIs + training bounds to: {output_file}")
        logger.info(f"Total area covered by AOIs: {sum(r['area_km2'] for r in aoi_records):.1f} km²")
        logger.info(f"Training data area: {training_record['area_km2']:.1f} km²")
        
        # Also export combined AOI (union of all AOIs)
        if hasattr(yielder, 'combined_aoi') and yielder.combined_aoi is not None:
            combined_aoi_record = {
                'aoi_id': 'combined_aoi',
                'date': '20220611',
                'time': 'combined',
                'vv_file': 'union_of_all_aois',
                'vh_file': 'union_of_all_aois',
                'area_km2': yielder.combined_aoi.geometry.iloc[0].area / 1e6,
                'geometry': yielder.combined_aoi.geometry.iloc[0]
            }
            
            combined_only_gdf = gpd.GeoDataFrame([combined_aoi_record], crs=yielder.target_epsg)
            combined_output = Path("data") / "combined_aoi_20220611.gpkg"
            combined_only_gdf.to_file(combined_output, driver='GPKG')
            
            logger.info(f"Exported combined AOI to: {combined_output}")
            logger.info(f"Combined AOI area: {combined_aoi_record['area_km2']:.1f} km²")
        
        # Export training polygons that intersect with AOIs for reference
        intersecting_polygons = yielder.gdf.copy()
        training_output = Path("data") / "training_polygons_filtered_20220611.gpkg"
        intersecting_polygons.to_file(training_output, driver='GPKG')
        
        logger.info(f"Exported {len(intersecting_polygons)} filtered training polygons to: {training_output}")
        
        # Summary
        logger.info(f"\nExported files:")
        logger.info(f"  1. {output_file} - Individual AOIs + training bounds")
        logger.info(f"  2. {combined_output} - Combined AOI union")
        logger.info(f"  3. {training_output} - Filtered training polygons")
        logger.info(f"\nUse QGIS or similar GIS software to visualize these files.")
        
    else:
        logger.warning("No AOI records found to export")


if __name__ == "__main__":
    main()
