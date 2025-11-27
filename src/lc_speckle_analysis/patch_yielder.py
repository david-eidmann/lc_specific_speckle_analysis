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
import rasterio.transform
from sklearn.model_selection import train_test_split
import pandas as pd
from shapely.geometry import box, Polygon, shape
from shapely.ops import unary_union
import pickle
import hashlib

from .config import logger, PROJECT_ROOT
from .data_config import TrainingDataConfig

@dataclass
class Patch:
    """Container for a single patch with metadata."""
    data: np.ndarray  # Shape: (height, width, channels)
    transform: rasterio.Affine  # Geospatial transform
    bounds: tuple  # (minx, miny, maxx, maxy)
    crs: str  # Coordinate reference system
    orbit: str  # Satellite orbit
    src_files: tuple  # (vv_path, vh_path)
    date: str  # YYYYMMDD
    class_id: int  # Class label

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
        self._buffer_polygons_globally()
        self._find_image_tuples()
        self._compute_aois_for_all_tuples()
        self._filter_polygons_by_individual_aois()
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
        
        # Check class sizes and filter out classes with too few samples
        class_counts = self.gdf[class_col].value_counts()
        min_samples_required = 2  # Minimum for stratified split
        valid_classes = class_counts[class_counts >= min_samples_required].index.tolist()
        
        if len(valid_classes) < len(self.config.classes):
            removed_classes = [c for c in self.config.classes if c not in valid_classes]
            logger.warning(f"Classes {removed_classes} have too few samples and will be excluded from splits")
            
            # Filter data to only include valid classes
            self.gdf = self.gdf[self.gdf[class_col].isin(valid_classes)].copy()
            logger.info(f"Filtered to {len(self.gdf)} polygons after removing small classes")
        
        if len(self.gdf) == 0:
            raise ValueError("No polygons remaining after filtering small classes")
        
        # Update config classes to only include valid ones
        original_classes = self.config.classes.copy()
        self.config.classes = [c for c in self.config.classes if c in valid_classes]
        
        if len(valid_classes) < 2:
            logger.warning("Only 1 class remaining - using simple random split without stratification")
            # Simple random split without stratification
            train_val_gdf, test_gdf = train_test_split(
                self.gdf, 
                test_size=0.2, 
                random_state=self.seed
            )
            
            train_gdf, val_gdf = train_test_split(
                train_val_gdf,
                test_size=0.25,
                random_state=self.seed
            )
        else:
            # Stratified split
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
        
        logger.info(f"Active classes for training: {self.config.classes}")
        if len(self.config.classes) != len(original_classes):
            logger.warning(f"Note: Reduced from {len(original_classes)} to {len(self.config.classes)} classes due to insufficient samples")
    
    def _find_image_tuples(self) -> None:
        """Find VV/VH image pairs from satellite data."""
        logger.info("Finding VV/VH image pairs")
        
        sat_files = self.config.get_file_paths()
        logger.info(f"Found {len(sat_files)} satellite files")
        
        # Group files by base name (everything except polarization)
        file_groups: Dict[str, Dict[str, Path]] = {}
        
        for file_path in sat_files:
            filename = file_path.name
            
            # Extract base name and polarization from filename
            # Format: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_..._VV.tif
            # Base name is everything before the last underscore, polarization is VV or VH
            parts = filename.split('_')
            if len(parts) >= 2:
                pol_part = parts[-1].split('.')[0]  # Extract VV or VH (remove .tif)
                
                if pol_part in ['VV', 'VH']:
                    # Base name is everything except the polarization part
                    base_name = '_'.join(parts[:-1])
                    
                    if base_name not in file_groups:
                        file_groups[base_name] = {}
                    
                    file_groups[base_name][pol_part] = file_path
        
        # Create image tuples for base names with both VV and VH
        self.image_tuples = []
        for base_name, polarizations in file_groups.items():
            if 'VV' in polarizations and 'VH' in polarizations:
                # Extract date from base name for tuple identification
                # Format: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_...
                parts = base_name.split('_')
                if len(parts) >= 5:
                    date_part = parts[4][:8]  # Extract YYYYMMDD from timestamp
                else:
                    date_part = base_name  # Fallback to base_name if parsing fails
                
                tuple_obj = ImageTuple(
                    vv_path=polarizations['VV'],
                    vh_path=polarizations['VH'],
                    date=date_part
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
        
        # Create valid_aoi subfolder
        valid_aoi_cache_dir = cache_dir / "valid_aoi"
        valid_aoi_cache_dir.mkdir(exist_ok=True)
        
        self.tuple_aois = {}  # Store AOI for each tuple
        aoi_polygons = []
        
        # Create parameters hash for cache key (to avoid conflicts when params change)
        params_str = f"scale_factor_10_buffer_out_2000_buffer_in_5000_simplify_100_epsg_{self.target_epsg}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        for i, image_tuple in enumerate(self.image_tuples):
            logger.info(f"Processing tuple {i+1}/{len(self.image_tuples)}: {image_tuple.date}")
            
            # Extract orbit and file basename for cache naming
            orbit = self.config.orbits[0] if self.config.orbits else "unknown"
            file_basename = image_tuple.vv_path.stem.split('_')[0:3]  # Get S1A_IW_GRDH from filename
            file_basename_str = '_'.join(file_basename)
            
            # Create hash from image paths and parameters for unique cache key
            image_hash = hashlib.md5(
                f"{image_tuple.vv_path}_{image_tuple.vh_path}".encode()
            ).hexdigest()[:6]  # Use only first 6 chars of hash
            cache_key = f"valid_aoi_{orbit}_{file_basename_str}_{image_hash}"
            gpkg_file = valid_aoi_cache_dir / f"{cache_key}.gpkg"
            
            # Try to load from cache first
            if gpkg_file.exists():
                logger.info(f"Loading AOI from cache for {image_tuple.date}: {gpkg_file.name}")
                aoi = gpd.read_file(gpkg_file)
                # Ensure correct CRS
                if aoi.crs != self.target_epsg:
                    aoi = aoi.to_crs(self.target_epsg)
            else:
                # Compute AOI from scratch
                logger.info(f"Computing AOI for {image_tuple.date} (this may take a moment)")
                aoi = self._compute_single_aoi(image_tuple, cache_key, valid_aoi_cache_dir)
            
            self.tuple_aois[image_tuple.date] = aoi
            aoi_polygons.append(aoi.geometry[0])
            
            # Log individual AOI info
            logger.info(f"AOI for {image_tuple.date}: area={aoi.geometry[0].area / 1e6:.1f} km²")
        
        # Combine all AOIs into one (union of all valid areas)
        if len(aoi_polygons) > 1:
            combined_polygon = unary_union(aoi_polygons)
            self.combined_aoi = gpd.GeoDataFrame({'id': [1]}, geometry=[combined_polygon], crs=self.target_epsg)
            logger.info(f"Combined AOI area: {combined_polygon.area / 1e6:.1f} km²")
            
            # Cache combined AOI as well
            combined_dates = "_".join([t.date for t in self.image_tuples])
            combined_hash = hashlib.md5(combined_dates.encode()).hexdigest()[:8]
            combined_cache_key = f"combined_aoi_{params_hash}_{combined_hash}"
            combined_gpkg = cache_dir / f"{combined_cache_key}.gpkg"
            
            if not combined_gpkg.exists():
                self.combined_aoi.to_file(combined_gpkg, driver='GPKG')
                logger.info(f"Cached combined AOI: {combined_gpkg.name}")
        else:
            self.combined_aoi = list(self.tuple_aois.values())[0]
            logger.info("Using single AOI as combined AOI")
    
    def _compute_single_aoi(self, image_tuple: ImageTuple, cache_key: str, cache_dir: Path) -> gpd.GeoDataFrame:
        """Compute AOI for a single image tuple using proper polygon extraction."""
        
        # Cache intermediate steps
        raw_polygon_cache = cache_dir / f"{cache_key}_raw_polygon.gpkg"
        reprojected_cache = cache_dir / f"{cache_key}_reprojected.gpkg" 
        final_aoi_cache = cache_dir / f"{cache_key}.gpkg"
        
        # Step 1: Extract raw polygon from raster (cache in original CRS)
        if raw_polygon_cache.exists():
            logger.info(f"Loading raw polygon from cache: {raw_polygon_cache.name}")
            aoi_raw = gpd.read_file(raw_polygon_cache)
        else:
            logger.info(f"Extracting raw polygon for {image_tuple.date}")
            aoi_raw = self._extract_raw_polygon(image_tuple)
            aoi_raw.to_file(raw_polygon_cache, driver='GPKG')
            logger.info(f"Cached raw polygon: {raw_polygon_cache.name}")
        
        # Step 2: Reproject to target EPSG (cache reprojected version)
        if reprojected_cache.exists():
            logger.info(f"Loading reprojected polygon from cache: {reprojected_cache.name}")
            aoi_reprojected = gpd.read_file(reprojected_cache)
        else:
            logger.info(f"Reprojecting to {self.target_epsg}")
            if aoi_raw.crs != self.target_epsg:
                aoi_reprojected = aoi_raw.to_crs(self.target_epsg)
            else:
                aoi_reprojected = aoi_raw.copy()
            aoi_reprojected.to_file(reprojected_cache, driver='GPKG')
            logger.info(f"Cached reprojected polygon: {reprojected_cache.name}")
        
        # Step 3: Apply buffering and simplification (final result)
        if final_aoi_cache.exists():
            logger.info(f"Loading final AOI from cache: {final_aoi_cache.name}")
            aoi_final = gpd.read_file(final_aoi_cache)
        else:
            logger.info(f"Applying buffering and simplification for {image_tuple.date}")
            # Apply buffering to reprojected polygon first
            aoi_buffered = self._buffer_reprojected_polygon(aoi_reprojected)
            aoi_final = self._process_aoi_polygon(aoi_buffered, image_tuple.date)
            aoi_final.to_file(final_aoi_cache, driver='GPKG')
            logger.info(f"Cached final AOI: {final_aoi_cache.name}")
        
        return aoi_final
    
    def _extract_raw_polygon(self, image_tuple: ImageTuple) -> gpd.GeoDataFrame:
        """Extract raw polygon from raster data."""
        with rasterio.open(image_tuple.vv_path) as src:
            # Read data with coarser resolution (factor 10x10) for efficiency
            scale_factor = 10
            vv_data = src.read(1, out_shape=(src.height // scale_factor, src.width // scale_factor))
            transform = src.transform * src.transform.scale(scale_factor, scale_factor)
            crs = src.crs
            profile = src.profile
        
        # Create mask of valid (non-NaN) data
        nodata_val = profile.get("nodata", 0)
        valid_mask = ~np.isnan(vv_data) & (vv_data != nodata_val)
        
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
        return gpd.GeoDataFrame({'id': [1]}, geometry=[combined_polygon], crs=crs)
    
    def _process_aoi_polygon(self, aoi: gpd.GeoDataFrame, date: str) -> gpd.GeoDataFrame:
        """Apply buffering and simplification to AOI polygon."""
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
        logger.info(f"Final AOI for {date}: area={final_area:.1f} km²")
        
        return aoi_simplified
    
    def _buffer_reprojected_polygon(self, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Apply initial buffering to reprojected polygon."""
        logger.info("Applying initial buffer to reprojected polygon")
        
        # Apply small outward buffer to clean up geometry
        buffered_aoi = aoi.copy()
        buffered_aoi.geometry = aoi.geometry.buffer(50)  # 50m outward buffer
        
        return buffered_aoi
    
    def _filter_polygons_by_individual_aois(self) -> None:
        """Filter training polygons by individual AOIs and track which polygons work with which images."""
        logger.info("Filtering polygons by individual valid AOIs")
        
        original_count = len(self.gdf)
        
        # Create a mapping of which polygons intersect with which image AOIs
        self.polygon_image_intersections = {}
        all_valid_polygon_indices = set()
        
        for image_tuple in self.image_tuples:
            aoi = self.tuple_aois[image_tuple.date]
            aoi_geometry = aoi.geometry[0]
            
            # Find polygons that intersect with this specific AOI
            intersects = self.gdf.intersects(aoi_geometry)
            valid_indices = self.gdf.index[intersects].tolist()
            
            logger.info(f"AOI {image_tuple.date}: {len(valid_indices)} intersecting polygons")
            
            # Store the mapping for this image
            self.polygon_image_intersections[image_tuple.date] = set(valid_indices)
            all_valid_polygon_indices.update(valid_indices)
        
        # Filter to only polygons that intersect with at least one AOI
        if all_valid_polygon_indices:
            self.gdf = self.gdf.loc[list(all_valid_polygon_indices)].copy()
        else:
            self.gdf = self.gdf.iloc[0:0].copy()  # Empty dataframe with same structure
        
        filtered_count = len(self.gdf)
        logger.info(f"Filtered from {original_count} to {filtered_count} polygons "
                   f"({filtered_count/original_count*100:.1f}% remaining)")
        
        if filtered_count == 0:
            raise ValueError("No training polygons found within any valid AOI")
        
        # Log class distribution after filtering
        class_col = self.config.column_id
        class_counts = self.gdf[class_col].value_counts().sort_index()
        logger.info(f"Class distribution after AOI filtering: {dict(class_counts)}")
        
        # Log intersection statistics
        logger.info("Per-image intersection statistics:")
        for image_date, polygon_indices in self.polygon_image_intersections.items():
            logger.info(f"  {image_date}: {len(polygon_indices)} polygons")
    
    def _buffer_polygons_globally(self) -> None:
        """Buffer all reprojected polygons globally at the beginning."""
        logger.info("Buffering all polygons globally for minimum patch size requirements")
        
        cache_dir = PROJECT_ROOT / "data" / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        patch_size = self.config.neural_network.patch_size
        # Buffer inwards by sqrt(2) * 10m * patch_size to ensure minimum area
        buffer_distance = -(patch_size * 10 * (2 ** 0.5))  # Negative for inward buffer
        
        logger.info(f"Global buffer: inward {abs(buffer_distance):.1f}m for minimum viable polygon size (patch size {patch_size})")
        
        # Create cache key based on parameters and data hash
        original_count = len(self.gdf)
        data_hash = hashlib.md5(str(self.gdf.geometry.bounds.values.tobytes()).encode()).hexdigest()[:8]
        cache_key = f"global_buffered_polygons_patch{patch_size}_epsg{self.target_epsg.split(':')[1]}_{data_hash}"
        cache_file = cache_dir / f"{cache_key}.gpkg"
        
        # Try to load from cache first
        if cache_file.exists():
            logger.info(f"Loading globally buffered polygons from cache: {cache_file.name}")
            self.gdf = gpd.read_file(cache_file)
            # Ensure correct CRS
            if self.gdf.crs != self.target_epsg:
                self.gdf = self.gdf.to_crs(self.target_epsg)
        else:
            # Buffer polygons and filter out empty ones
            logger.info("Computing globally buffered polygons")
            
            # Create buffered version
            buffered_gdf = self.gdf.copy()
            buffered_gdf['geometry_buffered'] = self.gdf.geometry.buffer(buffer_distance)
            
            # Filter out empty geometries in both original and buffered
            original_valid = ~self.gdf.geometry.is_empty
            buffered_valid = ~buffered_gdf.geometry_buffered.is_empty
            valid_mask = original_valid & buffered_valid
            
            # Keep only valid polygons and use original geometry (buffered was just for filtering)
            self.gdf = self.gdf[valid_mask].copy()
            
            filtered_count = len(self.gdf)
            logger.info(f"After global buffering filter: {filtered_count} polygons from {original_count} "
                       f"({filtered_count/original_count*100:.1f}% remaining)")
            
            if filtered_count == 0:
                raise ValueError("No polygons remaining after global buffering filter - patch size too large")
            
            # Log class distribution after buffering
            class_col = self.config.column_id
            class_counts = self.gdf[class_col].value_counts().sort_index()
            logger.info(f"Global class distribution after buffering: {dict(class_counts)}")
            
            # Cache the filtered polygons
            self.gdf.to_file(cache_file, driver='GPKG')
            logger.info(f"Cached globally buffered polygons: {cache_file.name}")
        
        logger.info(f"Final polygon count for global processing: {len(self.gdf)}")
    
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
    
    def _cache_patches(self, patches: List[Patch], mode: DataMode) -> None:
        """Cache a list of patches with metadata.
        
        Args:
            patches: List of Patch objects to cache
            mode: Data mode (train/validation/test)
        """
        if not patches:
            return
        
        # Create patches cache directory structure
        cache_dir = PROJECT_ROOT / "data" / "cache" / "patches" / mode.value
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create hash from patch metadata for cache key
        patch_info = []
        for patch in patches:
            info = f"{patch.date}_{patch.orbit}_{patch.class_id}_{patch.bounds}"
            patch_info.append(info)
        
        combined_info = "_".join(sorted(patch_info))
        cache_hash = hashlib.md5(combined_info.encode()).hexdigest()[:8]
        hash_dir = cache_dir / cache_hash
        hash_dir.mkdir(exist_ok=True)
        
        # Determine chunk size and create files
        chunk_size = 1000  # patches per file
        
        for i in range(0, len(patches), chunk_size):
            chunk = patches[i:i + chunk_size]
            n_start = i
            n_end = min(i + chunk_size - 1, len(patches) - 1)
            
            # Cache patch data as pickle
            patch_file = hash_dir / f"patches_{n_start:06d}-{n_end:06d}.pkl"
            with open(patch_file, 'wb') as f:
                pickle.dump(chunk, f)
            
            # Create metadata GeoDataFrame for patches
            geometries = []
            metadata = []
            
            for patch in chunk:
                # Create polygon from patch bounds
                minx, miny, maxx, maxy = patch.bounds
                geom = box(minx, miny, maxx, maxy)
                geometries.append(geom)
                
                metadata.append({
                    'class': patch.class_id,
                    'date': patch.date,
                    'orbit': patch.orbit,
                    'src_vv': str(patch.src_files[0]),
                    'src_vh': str(patch.src_files[1])
                })
            
            # Create GeoDataFrame and save as GPKG
            gdf = gpd.GeoDataFrame(metadata, geometry=geometries, crs=patches[0].crs)
            gpkg_file = hash_dir / f"patches_{n_start:06d}-{n_end:06d}.gpkg"
            gdf.to_file(gpkg_file, driver='GPKG')
        
        logger.info(f"Cached {len(patches)} patches in {hash_dir}")
    
    def _extract_patches_to_objects(self, polygon_row: pd.Series, vv_src: rasterio.DatasetReader, 
                                   vh_src: rasterio.DatasetReader, image_tuple: ImageTuple) -> List[Patch]:
        """Extract patches as Patch objects with full metadata.
        
        Args:
            polygon_row: Row from GeoDataFrame containing polygon
            vv_src: VV polarization raster dataset reader
            vh_src: VH polarization raster dataset reader
            image_tuple: ImageTuple containing metadata
            
        Returns:
            List of Patch objects
        """
        geometry = polygon_row.geometry
        patch_size = self.config.neural_network.patch_size
        class_id = polygon_row[self.config.column_id]
        
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
        orbit = self.config.orbits[0] if self.config.orbits else "unknown"
        
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
                patch_data = np.stack([patch_vv_filled, patch_vh_filled], axis=-1)
                
                # Calculate patch transform and bounds
                patch_transform = mask_transform * rasterio.Affine.translation(x_start, y_start)
                
                # Calculate bounds in geographic coordinates
                patch_bounds = rasterio.transform.array_bounds(
                    patch_size, patch_size, patch_transform
                )
                
                # Create Patch object
                patch = Patch(
                    data=patch_data,
                    transform=patch_transform,
                    bounds=patch_bounds,
                    crs=str(vv_src.crs),
                    orbit=orbit,
                    src_files=(image_tuple.vv_path, image_tuple.vh_path),
                    date=image_tuple.date,
                    class_id=class_id
                )
                
                patches.append(patch)
        
        return patches
    
    def yield_batch(self, mode: DataMode, n_samples_per_polygon: int = 1) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield batches of patches for specified mode with image exhaustion strategy.
        
        Args:
            mode: Data mode (train/validation/test)
            n_samples_per_polygon: Number of samples per polygon (default=1)
            
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
        logger.info(f"Samples per polygon: {n_samples_per_polygon}")
        
        # Group by class and calculate exhaustion threshold
        class_groups = data.groupby(class_col)
        min_polygons_per_class = min(len(group) for group in class_groups.groups.values())
        exhaustion_threshold = n_samples_per_polygon * min_polygons_per_class
        
        logger.info(f"Image exhaustion threshold: {exhaustion_threshold} samples per class")
        logger.info(f"Minimum polygons per class: {min_polygons_per_class}")
        
        # Keep track of opened datasets and image usage
        opened_datasets = {}
        image_usage_counter = {tuple_obj.date: {class_id: 0 for class_id in self.config.classes} 
                              for tuple_obj in self.image_tuples}
        current_image_idx = 0
        
        def get_open_datasets(image_tuple: ImageTuple) -> Tuple[rasterio.DatasetReader, rasterio.DatasetReader]:
            """Get or open datasets for an image tuple."""
            tuple_key = image_tuple.date
            if tuple_key not in opened_datasets:
                vv_src = rasterio.open(image_tuple.vv_path)
                vh_src = rasterio.open(image_tuple.vh_path)
                opened_datasets[tuple_key] = (vv_src, vh_src)
            return opened_datasets[tuple_key]
        
        def is_image_exhausted(image_date: str) -> bool:
            """Check if current image is exhausted for all classes."""
            return all(count >= exhaustion_threshold 
                      for count in image_usage_counter[image_date].values())
        
        def get_next_available_image() -> Optional[ImageTuple]:
            """Get next available (non-exhausted) image tuple."""
            for i, tuple_obj in enumerate(self.image_tuples):
                if not is_image_exhausted(tuple_obj.date):
                    return tuple_obj
            return None
        
        try:
            while True:  # Infinite generator
                # Get current image tuple
                current_tuple = get_next_available_image()
                if current_tuple is None:
                    logger.info("All images exhausted, cycling back to start")
                    # Reset counters and start over
                    image_usage_counter = {tuple_obj.date: {class_id: 0 for class_id in self.config.classes} 
                                          for tuple_obj in self.image_tuples}
                    current_tuple = self.image_tuples[0]
                
                logger.debug(f"Using image tuple: {current_tuple.date}")
                vv_src, vh_src = get_open_datasets(current_tuple)
                
                batch_patches = []
                batch_labels = []
                
                # Sample from each class to create balanced batch
                for class_id in self.config.classes:
                    if class_id not in class_groups.groups:
                        continue
                    
                    # Skip if this class is exhausted for current image
                    if image_usage_counter[current_tuple.date][class_id] >= exhaustion_threshold:
                        continue
                        
                    class_data = class_groups.get_group(class_id)
                    patches_needed = batch_size // len(self.config.classes)
                    
                    patches_collected = 0
                    attempts = 0
                    max_attempts = len(class_data) * 2
                    
                    while patches_collected < patches_needed and attempts < max_attempts:
                        # Check if we've exhausted this class for current image
                        if image_usage_counter[current_tuple.date][class_id] >= exhaustion_threshold:
                            break
                        
                        # Use pre-computed intersection mapping for efficiency
                        intersecting_indices = self.polygon_image_intersections.get(current_tuple.date, set())
                        class_indices = set(class_data.index)
                        valid_indices = intersecting_indices & class_indices
                        
                        if not valid_indices:
                            attempts += 1
                            continue
                        
                        valid_polygons = class_data.loc[list(valid_indices)]
                        
                        polygon_row = valid_polygons.sample(n=1, random_state=None).iloc[0]
                        
                        # Extract patches from polygon
                        patch_objects = self._extract_patches_to_objects(
                            polygon_row, vv_src, vh_src, current_tuple
                        )
                        patches = [p.data for p in patch_objects]  # Extract data for backward compatibility
                        
                        if patches:
                            # Cache the patch objects
                            if patch_objects:
                                self._cache_patches(patch_objects, mode)
                            
                            # Take up to n_samples_per_polygon patches
                            samples_to_take = min(len(patches), n_samples_per_polygon)
                            selected_patches = random.sample(patches, samples_to_take)
                            
                            for patch in selected_patches:
                                if patches_collected >= patches_needed:
                                    break
                                batch_patches.append(patch)
                                batch_labels.append(class_id)
                                patches_collected += 1
                                
                                # Increment usage counter
                                image_usage_counter[current_tuple.date][class_id] += 1
                        
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
