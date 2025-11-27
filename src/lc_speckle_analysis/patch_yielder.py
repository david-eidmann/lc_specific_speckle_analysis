"""PatchYielder for LC speckle analysis."""

import logging
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Optional
import random
from dataclasses import dataclass

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from rasterio.features import shapes
from sklearn.model_selection import train_test_split
import pandas as pd
from shapely.geometry import box, Polygon, shape
from shapely.ops import unary_union
import pickle
import hashlib

from .config import logger, PROJECT_ROOT
from .data_config import TrainingDataConfig

class DataMode(Enum):
    """Data mode enumeration."""
    TRAIN = "train"
    VALIDATION = "validation" 
    TEST = "test"


@dataclass
class ImageTuple:
    """Container for VV/VH image pair."""
    vv_path: Path
    vh_path: Path
    date: str
    
    def __post_init__(self):
        if not self.vv_path.exists():
            raise FileNotFoundError(f"VV image not found: {self.vv_path}")
        if not self.vh_path.exists():
            raise FileNotFoundError(f"VH image not found: {self.vh_path}")


class PatchYielder:
    """Yields patches from satellite data for training/validation/testing."""
    
    def __init__(self, config: TrainingDataConfig, seed: int = 42):
        """Initialize PatchYielder.
        
        Args:
            config: Training data configuration
            seed: Random seed for reproducible splits
        """
        self.config = config
        self.seed = seed
        self.target_epsg = 'EPSG:32632'  # UTM Zone 32N
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(f"PatchYielder initialized with patch size: {config.neural_network.patch_size}")
        logger.info(f"Target EPSG: {self.target_epsg}")
        
        # Load and prepare data
        self._load_training_data()
        self._find_image_tuples()
        self._compute_aois_for_all_tuples()
        self._filter_polygons_by_combined_aoi()
        self._split_data()
    
    def _load_training_data(self) -> None:
        """Load training data and reproject to target EPSG."""
        logger.info(f"Loading training data from: {self.config.train_data_path}")
        
        self.gdf = gpd.read_file(self.config.train_data_path)
        original_crs = self.gdf.crs
        logger.info(f"Original CRS: {original_crs}, Total polygons: {len(self.gdf)}")
        
        # Reproject to target EPSG
        if self.gdf.crs != self.target_epsg:
            logger.info(f"Reprojecting from {original_crs} to {self.target_epsg}")
            self.gdf = self.gdf.to_crs(self.target_epsg)
        
        # Filter by configured classes
        class_col = self.config.column_id
        if class_col not in self.gdf.columns:
            raise ValueError(f"Column '{class_col}' not found in training data")
        
        before_filter = len(self.gdf)
        self.gdf = self.gdf[self.gdf[class_col].isin(self.config.classes)]
        after_filter = len(self.gdf)
        
        logger.info(f"Filtered to {after_filter} polygons from {before_filter} (classes: {self.config.classes})")
        
        # Log class distribution
        class_counts = self.gdf[class_col].value_counts().sort_index()
        logger.info(f"Class distribution: {dict(class_counts)}")
    
    def _split_data(self) -> None:
        """Split polygons into train/validation/test sets."""
        logger.info("Splitting data into train/validation/test sets")
        
        # Stratify by class to maintain distribution
        class_col = self.config.column_id
        
        # First split: 80% train+val, 20% test
        train_val_gdf, test_gdf = train_test_split(
            self.gdf, 
            test_size=0.2, 
            stratify=self.gdf[class_col],
            random_state=self.seed
        )
        
        # Second split: 75% train, 25% validation (of the 80%)
        train_gdf, val_gdf = train_test_split(
            train_val_gdf,
            test_size=0.25,  # 0.25 * 0.8 = 0.2 of total = 20% validation
            stratify=train_val_gdf[class_col],
            random_state=self.seed
        )
        
        self.data_splits = {
            DataMode.TRAIN: train_gdf,
            DataMode.VALIDATION: val_gdf,
            DataMode.TEST: test_gdf
        }
        
        for mode, data in self.data_splits.items():
            logger.info(f"{mode.value}: {len(data)} polygons")
            class_dist = dict(data[class_col].value_counts().sort_index())
            logger.info(f"  Class distribution: {class_dist}")
    
    def _find_image_tuples(self) -> None:
        """Find VV/VH image pairs from satellite data."""
        logger.info("Finding VV/VH image pairs")
        
        sat_files = self.config.get_file_paths()
        logger.info(f"Found {len(sat_files)} satellite files")
        
        # Group files by date and polarization
        file_groups: Dict[str, Dict[str, Path]] = {}
        
        for file_path in sat_files:
            filename = file_path.name
            
            # Extract date and polarization from filename
            # Format: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_..._VV.tif
            parts = filename.split('_')
            if len(parts) >= 6:
                date_part = parts[4][:8]  # Extract YYYYMMDD
                pol_part = parts[-1].split('.')[0]  # Extract VV or VH
                
                if date_part not in file_groups:
                    file_groups[date_part] = {}
                
                file_groups[date_part][pol_part] = file_path
        
        # Create image tuples for dates with both VV and VH
        self.image_tuples = []
        for date, polarizations in file_groups.items():
            if 'VV' in polarizations and 'VH' in polarizations:
                tuple_obj = ImageTuple(
                    vv_path=polarizations['VV'],
                    vh_path=polarizations['VH'],
                    date=date
                )
                self.image_tuples.append(tuple_obj)
        
        logger.info(f"Created {len(self.image_tuples)} VV/VH image pairs")
        
        if not self.image_tuples:
            raise ValueError("No matching VV/VH image pairs found")
        
        # Filter by configured dates if specified
        if self.config.dates:
            filtered_tuples = []
            for target_date in self.config.dates:
                matching = [t for t in self.image_tuples if t.date == target_date]
                if matching:
                    filtered_tuples.extend(matching)
                else:
                    logger.warning(f"No image tuple found for date {target_date}")
            
            if filtered_tuples:
                self.image_tuples = filtered_tuples
                logger.info(f"Filtered to {len(self.image_tuples)} image tuples for specified dates")
            else:
                logger.warning("No tuples found for specified dates, using all available")
        
        logger.info(f"Using {len(self.image_tuples)} image tuples for processing")
    
    def _compute_aois_for_all_tuples(self) -> None:
        """Compute valid AOI for each image tuple and cache results."""
        logger.info(f"Computing valid AOIs for {len(self.image_tuples)} image tuples")
        
        cache_dir = PROJECT_ROOT / "data" / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        self.tuple_aois = {}  # Store AOI for each tuple
        aoi_polygons = []
        
        for i, image_tuple in enumerate(self.image_tuples):
            logger.info(f"Processing tuple {i+1}/{len(self.image_tuples)}: {image_tuple.date}")
            
            # Create hash from image paths for unique cache key
            image_hash = hashlib.md5(
                f"{image_tuple.vv_path}_{image_tuple.vh_path}".encode()
            ).hexdigest()
            cache_file = cache_dir / f"valid_aoi_{image_hash}.pkl"
            
            # Try to load from cache first
            if cache_file.exists():
                logger.info(f"Loading AOI from cache for {image_tuple.date}")
                with open(cache_file, 'rb') as f:
                    aoi = pickle.load(f)
            else:
                # Compute AOI from scratch
                logger.info(f"Computing AOI for {image_tuple.date} (this may take a moment)")
                aoi = self._compute_single_aoi(image_tuple, cache_file)
            
            self.tuple_aois[image_tuple.date] = aoi
            aoi_polygons.append(aoi.geometry[0])
            
            # Log individual AOI info
            aoi_bounds = aoi.total_bounds
            logger.info(f"AOI for {image_tuple.date}: area={aoi.geometry[0].area / 1e6:.1f} km²")
        
        # Combine all AOIs into one (union of all valid areas)
        if len(aoi_polygons) > 1:
            combined_polygon = unary_union(aoi_polygons)
            self.combined_aoi = gpd.GeoDataFrame([1], geometry=[combined_polygon], crs=self.target_epsg)
            logger.info(f"Combined AOI area: {combined_polygon.area / 1e6:.1f} km²")
        else:
            self.combined_aoi = list(self.tuple_aois.values())[0]
            logger.info("Using single AOI as combined AOI")
    
    def _compute_single_aoi(self, image_tuple: ImageTuple, cache_file: Path) -> gpd.GeoDataFrame:
        """Compute AOI for a single image tuple using proper polygon extraction."""
        
        with rasterio.open(image_tuple.vv_path) as src:
            # Read data with coarser resolution (factor 10x10) for efficiency
            scale_factor = 10
            vv_data = src.read(1, out_shape=(src.height // scale_factor, src.width // scale_factor))
            transform = src.transform * src.transform.scale(scale_factor, scale_factor)
            bounds = src.bounds
            crs = src.crs
            prf=src.profile
        
        # Create mask of valid (non-NaN) data
        valid_mask = ~np.isnan(vv_data) & (vv_data != prf["nodata"])
        
        if not np.any(valid_mask):
            raise ValueError(f"No valid data found in satellite image: {image_tuple.vv_path}")
        
        coverage = np.sum(valid_mask) / valid_mask.size * 100
        logger.info(f"Valid data coverage for {image_tuple.date}: {coverage:.1f}%")
        
        # Convert valid mask to uint8 for rasterio.features.shapes
        valid_uint8 = valid_mask.astype(np.uint8)
        
        # Extract polygons from valid pixels using rasterio.features.shapes
        logger.info(f"Extracting polygons from valid pixels for {image_tuple.date}")
        polygon_shapes = []
        
        for geom, value in shapes(valid_uint8, mask=valid_uint8, transform=transform):
            if value == 1:  # Only valid pixels
                polygon_shapes.append(shape(geom))
        
        if not polygon_shapes:
            raise ValueError(f"No valid polygons extracted from image: {image_tuple.vv_path}")
        
        logger.info(f"Extracted {len(polygon_shapes)} polygon fragments")
        
        # Combine all polygons into one using union
        if len(polygon_shapes) > 1:
            combined_polygon = unary_union(polygon_shapes)
        else:
            combined_polygon = polygon_shapes[0]
        
        # Create GeoDataFrame in original CRS
        aoi = gpd.GeoDataFrame({'id': [1]}, geometry=[combined_polygon], crs=crs)
        
        # Reproject to target EPSG if necessary for buffering (needs metric units)
        if aoi.crs != self.target_epsg:
            aoi = aoi.to_crs(self.target_epsg)
        
        # Buffer outwards by 2000m to fill holes
        logger.info(f"Buffering polygon outwards by 2000m to fill holes")
        aoi_buffered_out = aoi.copy()
        aoi_buffered_out.geometry = aoi.geometry.buffer(2000)
        
        # Buffer inwards by 5000m to be safe (net result: -3000m from original)
        logger.info(f"Buffering polygon inwards by 5000m for safety margin")
        aoi_final = aoi_buffered_out.copy()
        aoi_final.geometry = aoi_buffered_out.geometry.buffer(-5000)
        
        # Check if polygon still exists after negative buffer
        if aoi_final.geometry.iloc[0].is_empty:
            logger.warning("Polygon became empty after safety buffer, using smaller buffer")
            # Try with smaller buffer
            aoi_final.geometry = aoi_buffered_out.geometry.buffer(-2000)
            
            if aoi_final.geometry.iloc[0].is_empty:
                logger.warning("Still empty, using original buffered polygon")
                aoi_final = aoi_buffered_out
        
        # Simplify polygon (tolerance in meters, appropriate for UTM)
        logger.info(f"Simplifying polygon with 100m tolerance")
        aoi_simplified = aoi_final.copy()
        aoi_simplified.geometry = aoi_final.geometry.simplify(tolerance=100)
        
        # Log final AOI stats
        final_area = aoi_simplified.geometry.iloc[0].area / 1e6  # Convert to km²
        logger.info(f"Final AOI for {image_tuple.date}: area={final_area:.1f} km²")
        
        # Save as GPKG for human inspection
        gpkg_file = cache_file.with_suffix('.gpkg')
        aoi_simplified.to_file(gpkg_file, driver='GPKG')
        logger.info(f"Saved AOI as GPKG for inspection: {gpkg_file}")
        
        # Cache the result as pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(aoi_simplified, f)
        
        return aoi_simplified
    
    def _filter_polygons_by_combined_aoi(self) -> None:
        """Filter training polygons to only those within valid combined AOI."""
        logger.info("Filtering polygons by combined valid AOI")
        
        original_count = len(self.gdf)
        
        # Find polygons that intersect with the combined AOI
        aoi_geometry = self.combined_aoi.geometry[0]
        
        # Use spatial index for efficient intersection
        intersects = self.gdf.intersects(aoi_geometry)
        self.gdf = self.gdf[intersects].copy()
        
        filtered_count = len(self.gdf)
        logger.info(f"Filtered from {original_count} to {filtered_count} polygons "
                   f"({filtered_count/original_count*100:.1f}% remaining)")
        
        if filtered_count == 0:
            raise ValueError("No training polygons found within valid AOI")
        
        # Log class distribution after filtering
        class_col = self.config.column_id
        class_counts = self.gdf[class_col].value_counts().sort_index()
        logger.info(f"Class distribution after AOI filtering: {dict(class_counts)}")
    
    def _extract_patches_from_polygon(self, polygon_row: pd.Series, vv_src: rasterio.DatasetReader, 
                                    vh_src: rasterio.DatasetReader) -> List[np.ndarray]:
        """Extract patches from a polygon area.
        
        Args:
            polygon_row: Row from GeoDataFrame containing polygon
            vv_src: VV polarization raster dataset reader
            vh_src: VH polarization raster dataset reader
            
        Returns:
            List of patches as numpy arrays
        """
        geometry = polygon_row.geometry
        patch_size = self.config.neural_network.patch_size
        
        try:
            # Mask the data with polygon (crop to polygon bounds)
            masked_vv, mask_transform = mask(vv_src, [geometry], crop=True, filled=False)
            masked_vh, _ = mask(vh_src, [geometry], crop=True, filled=False)
            
        except Exception as e:
            logger.warning(f"Failed to mask polygon: {e}")
            return []
        
        # Extract patches from masked area
        patches = []
        masked_vv = masked_vv[0]  # Remove band dimension
        masked_vh = masked_vh[0]
        
        # Find valid (non-masked, non-NaN) areas
        valid_mask = ~masked_vv.mask & ~masked_vh.mask & ~np.isnan(masked_vv) & ~np.isnan(masked_vh)
        valid_mask = valid_mask & (masked_vv != 0) & (masked_vh != 0)  # Also exclude zeros
        
        if not np.any(valid_mask):
            return []
        
        # Get coordinates of valid pixels
        valid_coords = np.where(valid_mask)
        
        if len(valid_coords[0]) == 0:
            return []
        
        # Randomly sample patch centers from valid pixels
        max_patches = min(10, len(valid_coords[0]))  # Reduced limit for faster processing
        if max_patches == 0:
            return []
        
        selected_indices = np.random.choice(len(valid_coords[0]), 
                                          size=min(max_patches, len(valid_coords[0])), 
                                          replace=False)
        
        half_patch = patch_size // 2
        
        for idx in selected_indices:
            center_y = valid_coords[0][idx] 
            center_x = valid_coords[1][idx]
            
            # Extract patch around center
            y_start = max(0, center_y - half_patch)
            y_end = min(masked_vv.shape[0], center_y + half_patch + 1)
            x_start = max(0, center_x - half_patch) 
            x_end = min(masked_vv.shape[1], center_x + half_patch + 1)
            
            # Skip if patch is too small
            if (y_end - y_start) < patch_size or (x_end - x_start) < patch_size:
                continue
            
            # Crop to exact patch size if needed
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            if y_end > masked_vv.shape[0] or x_end > masked_vv.shape[1]:
                continue
            
            patch_vv = masked_vv[y_start:y_end, x_start:x_end]
            patch_vh = masked_vh[y_start:y_end, x_start:x_end]
            
            # Check if patch has valid data
            patch_valid = ~patch_vv.mask & ~patch_vh.mask
            if np.sum(patch_valid) < (patch_size * patch_size * 0.8):  # At least 80% valid
                continue
            
            # Convert to regular arrays and fill masked values with 0
            patch_vv_filled = np.where(patch_vv.mask, 0, patch_vv.data)
            patch_vh_filled = np.where(patch_vh.mask, 0, patch_vh.data)
            
            # Ensure patch is exactly the right size
            if patch_vv_filled.shape == (patch_size, patch_size) and patch_vh_filled.shape == (patch_size, patch_size):
                # Stack VV and VH as channels
                patch = np.stack([patch_vv_filled, patch_vh_filled], axis=-1)
                patches.append(patch)
        
        return patches
    
    def yield_batch(self, mode: DataMode) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield batches of patches for specified mode.
        
        Args:
            mode: Data mode (train/validation/test)
            
        Yields:
            Tuple of (patches, labels) as numpy arrays
        """
        if mode not in self.data_splits:
            raise ValueError(f"Invalid mode: {mode}")
        
        data = self.data_splits[mode]
        batch_size = self.config.neural_network.batch_size
        class_col = self.config.column_id
        
        logger.info(f"Starting batch yielding for {mode.value} mode")
        logger.info(f"Available image tuples: {[t.date for t in self.image_tuples]}")
        
        # Keep track of opened datasets for cleanup
        opened_datasets = {}
        
        def get_open_datasets(image_tuple: ImageTuple) -> Tuple[rasterio.DatasetReader, rasterio.DatasetReader]:
            """Get or open datasets for an image tuple."""
            tuple_key = image_tuple.date
            if tuple_key not in opened_datasets:
                vv_src = rasterio.open(image_tuple.vv_path)
                vh_src = rasterio.open(image_tuple.vh_path)
                opened_datasets[tuple_key] = (vv_src, vh_src)
            return opened_datasets[tuple_key]
        
        try:
            # Group by class for balanced sampling
            class_groups = data.groupby(class_col)
            
            while True:  # Infinite generator
                batch_patches = []
                batch_labels = []
                
                # Sample from each class to create balanced batch
                for class_id in self.config.classes:
                    if class_id not in class_groups.groups:
                        continue
                        
                    class_data = class_groups.get_group(class_id)
                    patches_needed = batch_size // len(self.config.classes)
                    
                    patches_collected = 0
                    attempts = 0
                    max_attempts = len(class_data) * 2
                    
                    while patches_collected < patches_needed and attempts < max_attempts:
                        # Random polygon from this class
                        polygon_row = class_data.sample(n=1, random_state=None).iloc[0]
                        
                        # Randomly select an image tuple for this polygon
                        # Filter tuples to those that intersect with polygon
                        valid_tuples = []
                        for tuple_obj in self.image_tuples:
                            aoi = self.tuple_aois[tuple_obj.date]
                            if polygon_row.geometry.intersects(aoi.geometry[0]):
                                valid_tuples.append(tuple_obj)
                        
                        if not valid_tuples:
                            attempts += 1
                            continue
                        
                        # Select random valid tuple
                        selected_tuple = random.choice(valid_tuples)
                        vv_src, vh_src = get_open_datasets(selected_tuple)
                        
                        # Extract patches from polygon
                        patches = self._extract_patches_from_polygon(
                            polygon_row, vv_src, vh_src
                        )
                        
                        for patch in patches:
                            if patches_collected >= patches_needed:
                                break
                            batch_patches.append(patch)
                            batch_labels.append(class_id)
                            patches_collected += 1
                        
                        attempts += 1
                
                if len(batch_patches) > 0:
                    # Convert to numpy arrays
                    patches_array = np.array(batch_patches)
                    labels_array = np.array(batch_labels)
                    
                    # Shuffle batch
                    indices = np.random.permutation(len(patches_array))
                    patches_array = patches_array[indices]
                    labels_array = labels_array[indices]
                    
                    logger.debug(f"Yielding batch: {patches_array.shape}, labels: {len(labels_array)}")
                    yield patches_array, labels_array
                else:
                    logger.warning(f"No patches generated for {mode.value} mode")
                    break
        
        finally:
            # Ensure resources are cleaned up
            for vv_src, vh_src in opened_datasets.values():
                try:
                    vv_src.close()
                    vh_src.close()
                except:
                    pass
