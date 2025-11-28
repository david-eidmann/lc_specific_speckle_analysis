#!/usr/bin/env python3
"""
RAW DEBUG SCRIPT - Removes all exception handling from patch extraction!
This will show the exact rasterio call that fails with the original stack trace.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lc_speckle_analysis import get_training_config
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
import logging
import rasterio
from rasterio.mask import mask
import geopandas as gpd

# Enable full debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

def debug_extract_patches_raw(yielder, polygon_row, vv_src, vh_src, image_tuple, mode):
    """
    Raw version of _extract_patches_to_objects with NO exception handling!
    This will crash exactly where the rasterio operation fails.
    """
    
    geometry = polygon_row.geometry
    class_id = polygon_row[yielder.config.column_id]
    patch_size = yielder.config.neural_network.patch_size
    
    print(f"\n🔬 RAW DEBUG EXTRACTION:")
    print(f"  Polygon bounds: {geometry.bounds}")
    print(f"  Raster bounds: {vv_src.bounds}")
    print(f"  Class ID: {class_id}")
    print(f"  VV path: {vv_src.name}")
    print(f"  VH path: {vh_src.name}")
    
    # Check basic overlap manually
    poly_minx, poly_miny, poly_maxx, poly_maxy = geometry.bounds
    raster_minx, raster_miny, raster_maxx, raster_maxy = vv_src.bounds
    
    overlap = (poly_minx < raster_maxx and poly_maxx > raster_minx and 
              poly_miny < raster_maxy and poly_maxy > raster_miny)
    
    print(f"  Manual overlap check: {overlap}")
    if not overlap:
        print(f"  ❌ NO OVERLAP DETECTED!")
        print(f"     Polygon X: {poly_minx:.0f} to {poly_maxx:.0f}")
        print(f"     Raster X:  {raster_minx:.0f} to {raster_maxx:.0f}")
        print(f"     Polygon Y: {poly_miny:.0f} to {poly_maxy:.0f}")
        print(f"     Raster Y:  {raster_miny:.0f} to {raster_maxy:.0f}")
    
    print(f"\n💥 ABOUT TO CALL rasterio.mask.mask() - NO EXCEPTION HANDLING!")
    print(f"   This will crash with the original rasterio error...")
    
    # THE CRITICAL CALL - NO TRY/CATCH!
    # This will fail with the raw rasterio exception showing exactly what's wrong
    masked_vv, mask_transform = mask(vv_src, [geometry], crop=True, filled=False)
    masked_vh, _ = mask(vh_src, [geometry], crop=True, filled=False)
    
    print(f"✓ mask() succeeded - this shouldn't happen if there's an overlap issue!")
    return []

def main():
    """Debug the exact failure point with no exception handling!"""
    
    print("🐛 RAW RASTERIO DEBUG - HUNT FOR THE FAILING POLYGON!")
    print("=" * 60)
    print("This will test polygons until it finds one that crashes!\n")
    
    # Initialize
    config = get_training_config()
    yielder = PatchYielder(config, seed=42)
    
    print(f"✓ Initialized PatchYielder")
    
    # Test ALL image tuples and their polygons until we find the failing one
    for img_idx, image_tuple in enumerate(yielder.image_tuples):
        print(f"\n🎯 TESTING IMAGE TUPLE {img_idx + 1}/{len(yielder.image_tuples)}")
        print(f"   VV: {image_tuple.vv_path.name}")
        
        # Open raster files
        vv_src = rasterio.open(image_tuple.vv_path)
        vh_src = rasterio.open(image_tuple.vh_path)
        
        print(f"   Raster bounds: {vv_src.bounds}")
        
        # Get unique key for intersection mapping
        unique_key = yielder._generate_stable_unique_key(image_tuple)
        intersecting_positions = yielder.polygon_image_intersections.get(unique_key, set())
        
        print(f"   Found {len(intersecting_positions)} polygons for AOI {unique_key}")
        
        # Get train split data
        train_data = yielder.data_splits[DataMode.TRAIN]
        data_indices = set(train_data.index)
        valid_indices = intersecting_positions & data_indices
        
        if not valid_indices:
            print(f"   ⚠️ No valid polygons in train split for this image")
            vv_src.close()
            vh_src.close()
            continue
        
        # Test ALL valid polygons for this image tuple
        valid_polygons = train_data.loc[list(valid_indices)]
        
        for poly_idx, (idx, polygon) in enumerate(valid_polygons.iterrows()):
            print(f"\n📍 TESTING POLYGON {poly_idx + 1}/{len(valid_polygons)} (index {idx})")
            print(f"   Geometry bounds: {polygon.geometry.bounds}")
            print(f"   Class: {polygon[yielder.config.column_id]}")
            
            # Check manual overlap
            poly_minx, poly_miny, poly_maxx, poly_maxy = polygon.geometry.bounds
            raster_minx, raster_miny, raster_maxx, raster_maxy = vv_src.bounds
            
            overlap = (poly_minx < raster_maxx and poly_maxx > raster_minx and 
                      poly_miny < raster_maxy and poly_maxy > raster_miny)
            
            print(f"   Manual overlap check: {overlap}")
            
            if not overlap:
                print(f"   ❌ NO OVERLAP! This should crash rasterio.mask()!")
                print(f"      Polygon X: {poly_minx:.0f} to {poly_maxx:.0f}")
                print(f"      Raster X:  {raster_minx:.0f} to {raster_maxx:.0f}")
                print(f"      Polygon Y: {poly_miny:.0f} to {poly_maxy:.0f}")
                print(f"      Raster Y:  {raster_miny:.0f} to {raster_maxy:.0f}")
                print(f"\n💥 CALLING rasterio.mask() ON NON-OVERLAPPING POLYGON - SHOULD CRASH!")
                
                # THE CRITICAL CALL THAT SHOULD FAIL - NO TRY/CATCH!
                masked_vv, mask_transform = mask(vv_src, [polygon.geometry], crop=True, filled=False)
                
                print(f"   😱 WTF?? mask() succeeded when it shouldn't have!")
            else:
                print(f"   ✓ Overlap detected - this polygon should work")
        
        vv_src.close()
        vh_src.close()
    
    print(f"\n🤔 No failing polygons found - all mask() calls succeeded!")

if __name__ == "__main__":
    main()
