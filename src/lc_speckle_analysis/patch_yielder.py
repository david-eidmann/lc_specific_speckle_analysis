"""PatchYielder for LC speckle analysis."""

import logging
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Optional
import random
from dataclasses import dataclass
import json
import datetime
import re

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
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion

from .config import logger, PROJECT_ROOT
from .data_config import TrainingDataConfig


class DataMode(Enum):
    """Data mode enumeration."""
    TRAIN = "train"
    VALIDATION = "validation" 
    TEST = "test"

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
    data_mode: DataMode  # Data mode (train/validation/test)

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
        self._filter_image_tuples_by_bounds_and_classes()
        self._compute_aois_for_all_tuples()
        self._filter_polygons_by_individual_aois()  # Filter by AOIs first
        self._split_data()  # Do data splitting after AOI filtering to ensure consistency
    
    def _generate_stable_unique_key(self, image_tuple) -> str:
        """Generate a stable unique key for an image tuple that matches AOI file naming."""
        import re
        import hashlib
        time_match = re.search(r'T(\d{6})', image_tuple.vv_path.name)
        time_str = time_match.group(1) if time_match else "unknown"
        # Use same hash generation as AOI file naming (both VV and VH paths)
        path_hash = hashlib.md5(f"{image_tuple.vv_path}_{image_tuple.vh_path}".encode()).hexdigest()[:6]
        return f"{image_tuple.date}_{time_str}_{path_hash}"
    
    def _load_training_data(self) -> None:
        """Load training data from multiple sources and reproject to target EPSG."""
        # Create cache for the processed training data
        cache_dir = PROJECT_ROOT / "data" / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Create cache key based on input paths, target EPSG, and classes
        paths_hash = hashlib.md5('|'.join(sorted(self.config.train_data_paths)).encode()).hexdigest()[:8]
        classes_hash = hashlib.md5('|'.join(map(str, sorted(self.config.classes))).encode()).hexdigest()[:4]
        cache_key = f"processed_training_data_{paths_hash}_{classes_hash}_epsg{self.target_epsg.split(':')[1]}"
        cache_file = cache_dir / f"{cache_key}.gpkg"
        
        # Try to load from cache first
        if cache_file.exists():
            logger.info(f"Loading processed training data from cache: {cache_file.name}")
            self.gdf = gpd.read_file(cache_file)
            logger.info(f"Loaded {len(self.gdf)} polygons from cache")
            return
        
        logger.info(f"Loading training data from {len(self.config.train_data_paths)} sources")
        
        # Load and combine multiple training datasets
        gdfs = []
        total_polygons = 0
        
        for i, path in enumerate(self.config.train_data_paths):
            logger.info(f"Loading dataset {i+1}/{len(self.config.train_data_paths)}: {path}")
            gdf = gpd.read_file(path)
            gdfs.append(gdf)
            total_polygons += len(gdf)
            logger.info(f"  Loaded {len(gdf)} polygons from {Path(path).name}")
        
        # Combine all datasets
        self.gdf = gpd.pd.concat(gdfs, ignore_index=True)
        original_crs = self.gdf.crs
        logger.info(f"Combined datasets: {total_polygons} total polygons")
        logger.info(f"Original CRS: {original_crs}, Final combined polygons: {len(self.gdf)}")
        
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
        
        # Save processed data to cache
        logger.info(f"Caching processed training data to {cache_file.name}")
        self.gdf.to_file(cache_file, driver='GPKG')
    
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
            self.gdf = self.gdf[self.gdf[class_col].isin(valid_classes)].reset_index(drop=True)
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
        
        # No complex position mapping needed since split happens after AOI filtering
        # The intersection mapping positions should match the DataFrame indices directly
        
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
    
    def _filter_image_tuples_by_bounds_and_classes(self) -> None:
        """Filter image tuples to keep only those with bounds intersecting training data and all required classes."""
        logger.info("Filtering image tuples by bounds intersection and class availability")
        
        # Get training data bounds
        training_bounds = self.gdf.total_bounds  # [minx, miny, maxx, maxy]
        logger.info(f"Training data bounds: {training_bounds}")
        
        # Cache directory for discarded images list
        cache_dir = PROJECT_ROOT / "data" / "cache"
        cache_key = f"discarded_images_{hashlib.md5(str(training_bounds).encode()).hexdigest()[:8]}"
        discarded_cache_file = cache_dir / f"{cache_key}.json"
        
        # Check if we have cached discarded images list
        discarded_images = set()
        if discarded_cache_file.exists():
            logger.info(f"Loading discarded images from cache: {discarded_cache_file.name}")
            import json
            with open(discarded_cache_file, 'r') as f:
                discarded_data = json.load(f)
                discarded_images = set(discarded_data.get('discarded_images', []))
        
        filtered_tuples = []
        newly_discarded = []
        
        for image_tuple in self.image_tuples:
            # Create unique identifier for this image tuple
            image_id = f"{image_tuple.vv_path.stem}_{image_tuple.vh_path.stem}"
            
            # Skip if already known to be discarded
            if image_id in discarded_images:
                logger.debug(f"Skipping known discarded image: {image_tuple.date}")
                continue
            
            # Check bounds intersection by reading raster metadata
            try:
                with rasterio.open(image_tuple.vv_path) as src:
                    # Get image bounds in target CRS
                    image_bounds = src.bounds
                    image_crs = src.crs
                    
                    # Convert to target CRS if needed
                    if image_crs != self.target_epsg:
                        from rasterio.warp import transform_bounds
                        image_bounds = transform_bounds(image_crs, self.target_epsg, *image_bounds)
                    
                    # Check if bounds intersect
                    bounds_intersect = (
                        image_bounds[0] < training_bounds[2] and  # image_minx < training_maxx
                        image_bounds[2] > training_bounds[0] and  # image_maxx > training_minx
                        image_bounds[1] < training_bounds[3] and  # image_miny < training_maxy
                        image_bounds[3] > training_bounds[1]      # image_maxy > training_miny
                    )
                    
                    if not bounds_intersect:
                        logger.info(f"Discarding {image_tuple.date}: no bounds intersection")
                        newly_discarded.append(image_id)
                        discarded_images.add(image_id)
                        continue
                    
                    logger.debug(f"Image {image_tuple.date} bounds intersect with training data")
                    filtered_tuples.append(image_tuple)
                    
            except Exception as e:
                logger.warning(f"Error checking bounds for {image_tuple.date}: {e}")
                newly_discarded.append(image_id)
                discarded_images.add(image_id)
        
        # Update cached discarded images if we found new ones
        if newly_discarded:
            import json
            cache_data = {
                'discarded_images': list(discarded_images),
                'training_bounds': training_bounds.tolist(),
                'timestamp': datetime.datetime.now().isoformat()
            }
            with open(discarded_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached {len(newly_discarded)} newly discarded images: {discarded_cache_file.name}")
        
        # Update image tuples
        original_count = len(self.image_tuples)
        self.image_tuples = filtered_tuples
        logger.info(f"Filtered image tuples: {original_count} → {len(self.image_tuples)} (discarded {original_count - len(self.image_tuples)})")
        
        if not self.image_tuples:
            raise ValueError("No image tuples remaining after bounds filtering")
    
    def _filter_aois_by_class_availability(self) -> None:
        """Filter AOIs to keep only those that have sufficient class diversity (at least 80% of required classes)."""
        logger.info("Filtering AOIs by class availability")
        
        required_classes = set(self.config.classes)
        min_required_classes = max(1, int(len(required_classes) * 0.8))  # At least 80% of classes
        valid_dates = []
        
        for date, aoi in self.tuple_aois.items():
            # Check which polygons intersect with this AOI
            intersecting_mask = self.gdf.intersects(aoi.geometry[0])
            intersecting_polygons = self.gdf[intersecting_mask]
            
            if len(intersecting_polygons) == 0:
                logger.info(f"Discarding AOI {date}: no intersecting polygons")
                continue
            
            # Check which classes are available in intersecting polygons
            available_classes = set(intersecting_polygons[self.config.column_id].unique())
            missing_classes = required_classes - available_classes
            
            if len(available_classes) >= min_required_classes:
                valid_dates.append(date)
                if missing_classes:
                    logger.info(f"AOI {date} accepted with {len(available_classes)}/{len(required_classes)} classes (missing: {missing_classes})")
                else:
                    logger.info(f"AOI {date} has all required classes: {available_classes}")
            else:
                logger.info(f"Discarding AOI {date}: insufficient class diversity ({len(available_classes)}/{len(required_classes)} < {min_required_classes})")
                logger.info(f"  Available classes: {available_classes}")
                logger.info(f"  Missing classes: {missing_classes}")
        
        # Filter image tuples and AOIs to keep only valid ones
        original_count = len(self.image_tuples)
        # Keep image tuples whose corresponding unique keys are valid
        valid_image_indices = []
        for i, t in enumerate(self.image_tuples):
            unique_key = self._generate_stable_unique_key(t)
            if unique_key in valid_dates:
                valid_image_indices.append(i)
        
        self.image_tuples = [self.image_tuples[i] for i in valid_image_indices]
        self.tuple_aois = {date: aoi for date, aoi in self.tuple_aois.items() if date in valid_dates}
        
        logger.info(f"Filtered AOIs by class availability: {original_count} → {len(self.image_tuples)} (discarded {original_count - len(self.image_tuples)})")
        
        if not self.image_tuples:
            # If strict filtering fails, be more lenient - keep AOIs with any classes
            logger.warning("No AOIs with 80% class coverage found, falling back to any class coverage")
            valid_dates = []
            for date, aoi in list(self.tuple_aois.items()):
                intersecting_mask = self.gdf.intersects(aoi.geometry[0])
                intersecting_polygons = self.gdf[intersecting_mask]
                
                if len(intersecting_polygons) > 0:
                    available_classes = set(intersecting_polygons[self.config.column_id].unique())
                    valid_dates.append(date)
                    logger.info(f"Fallback: keeping AOI {date} with classes: {available_classes}")
            
            if valid_dates:
                # Restore from the full list using the same logic
                valid_image_indices = []
                for i in range(len(self.image_tuples)):  # Use original length before filtering
                    import re
                    unique_key = self._generate_stable_unique_key(self.image_tuples[i])
                    if unique_key in valid_dates:
                        valid_image_indices.append(i)
                
                self.image_tuples = [self.image_tuples[i] for i in valid_image_indices]
                self.tuple_aois = {date: aoi for date, aoi in self.tuple_aois.items() if date in valid_dates}
                logger.info(f"Fallback filtering: kept {len(self.image_tuples)} AOIs with any class coverage")
            else:
                raise ValueError("No AOIs remaining after class filtering")
    
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
            
            # Create unique key using date, index and filename to avoid overwrites
            # Extract time info from filename for better identification
            import re
            unique_key = self._generate_stable_unique_key(image_tuple)
            self.tuple_aois[unique_key] = aoi
            aoi_polygons.append(aoi.geometry[0])
            
            # Log individual AOI info
            logger.info(f"AOI for {unique_key}: area={aoi.geometry[0].area / 1e6:.1f} km²")
        
        # Filter AOIs by class availability
        self._filter_aois_by_class_availability()
        
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
            aoi_final = self._process_aoi_polygon(aoi_buffered, image_tuple)
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
    
    def _process_aoi_polygon(self, aoi: gpd.GeoDataFrame, image_tuple: ImageTuple) -> gpd.GeoDataFrame:
        """Apply buffering and simplification to AOI polygon."""
        # Buffer outwards by 2000m to fill holes
        logger.info(f"Buffering polygon outwards by 2000m to fill holes")
        aoi_buffered_out = aoi.copy()
        aoi_buffered_out.geometry = aoi.geometry.buffer(2000)
        
        # Buffer inwards by 5000m to be safe (net result: -3000m from original)
        logger.info(f"Buffering polygon inwards by 5000m for safety margin")
        aoi_final = aoi_buffered_out.copy()
        aoi_final.geometry = aoi_buffered_out.geometry.buffer(-5000)
        
        # Intersect with actual raster bounds to ensure AOI matches raster coverage
        logger.info(f"Intersecting AOI with actual raster bounds")
        from shapely.geometry import box
        
        # Get the actual raster bounds by opening the raster file
        import rasterio
        with rasterio.open(image_tuple.vv_path) as src:
            raster_bounds = src.bounds
            raster_crs = src.crs
        
        # Create bbox from raster bounds and reproject to target EPSG if needed
        raster_bbox = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
        raster_bbox_gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[raster_bbox], crs=raster_crs)
        
        # Reproject raster bbox to target EPSG if needed
        if raster_crs != self.target_epsg:
            raster_bbox_gdf = raster_bbox_gdf.to_crs(self.target_epsg)
        
        # Intersect AOI with actual raster bounds
        aoi_intersected = gpd.overlay(aoi_final, raster_bbox_gdf, how='intersection')
        
        if len(aoi_intersected) == 0 or aoi_intersected.geometry.iloc[0].is_empty:
            logger.warning(f"AOI for {image_tuple.date} does not intersect with raster bounds - skipping")
            # Return empty AOI that will be filtered out later
            empty_geom = gpd.GeoDataFrame({'id': [1]}, geometry=[raster_bbox_gdf.geometry.iloc[0].buffer(0)], crs=self.target_epsg)
            empty_geom.geometry = empty_geom.geometry.buffer(-1)  # Make it empty
            return empty_geom
        else:
            # Use the intersected AOI
            aoi_final = aoi_intersected.copy()
            # Ensure we have the right structure
            aoi_final = gpd.GeoDataFrame({'id': [1]}, geometry=[aoi_final.geometry.iloc[0]], crs=self.target_epsg)
        
        # Check if AOI contains at least 50 polygons
        intersecting_polygons = self.gdf.intersects(aoi_final.geometry.iloc[0])
        n_intersecting = intersecting_polygons.sum()
        
        if n_intersecting < 50:
            logger.warning(f"AOI for {image_tuple.date} contains only {n_intersecting} polygons (minimum 50 required) - skipping")
            # Return empty AOI that will be filtered out later  
            empty_geom = gpd.GeoDataFrame({'id': [1]}, geometry=[raster_bbox_gdf.geometry.iloc[0].buffer(0)], crs=self.target_epsg)
            empty_geom.geometry = empty_geom.geometry.buffer(-1)  # Make it empty
            return empty_geom
        else:
            logger.info(f"AOI for {image_tuple.date} contains {n_intersecting} polygons (meets minimum requirement)")
        
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
        # Use position-based indexing to avoid DataFrame index issues
        self.polygon_image_intersections = {}
        all_valid_positions = set()
        
        for i, image_tuple in enumerate(self.image_tuples):
            # Use a stable key based on file path hash instead of list index
            unique_key = self._generate_stable_unique_key(image_tuple)
            
            if unique_key not in self.tuple_aois:
                logger.warning(f"No AOI found for image tuple {unique_key}")
                continue
                
            aoi = self.tuple_aois[unique_key]
            aoi_geometry = aoi.geometry[0]
            
            # Find polygons that intersect with this specific AOI using position-based indexing
            intersects = self.gdf.intersects(aoi_geometry)
            valid_positions = []
            for pos, intersects_flag in enumerate(intersects):
                if intersects_flag:
                    valid_positions.append(pos)
            
            logger.info(f"AOI {unique_key}: {len(valid_positions)} intersecting polygons")
            
            # Store the mapping using position-based indices (0, 1, 2, ...)
            self.polygon_image_intersections[unique_key] = set(valid_positions)
            all_valid_positions.update(valid_positions)
        
        # Filter to only polygons that intersect with at least one AOI using position-based indexing
        if all_valid_positions:
            # Convert positions to boolean mask and filter
            mask = [i in all_valid_positions for i in range(len(self.gdf))]
            self.gdf = self.gdf[mask].reset_index(drop=True)
            
            # Update intersection mapping to reflect the new positions after filtering
            # Create reverse mapping from old position to new position
            old_to_new_pos = {}
            new_pos = 0
            for old_pos in range(len(mask)):
                if mask[old_pos]:
                    old_to_new_pos[old_pos] = new_pos
                    new_pos += 1
            
            # Update intersection mappings with new positions
            for unique_key in self.polygon_image_intersections:
                old_positions = self.polygon_image_intersections[unique_key]
                new_positions = set()
                for old_pos in old_positions:
                    if old_pos in old_to_new_pos:
                        new_positions.add(old_to_new_pos[old_pos])
                self.polygon_image_intersections[unique_key] = new_positions
        else:
            self.gdf = self.gdf.iloc[0:0].reset_index(drop=True)  # Empty dataframe with same structure
        
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
        
        # Create descriptive cache key with GPKG basenames and parameters
        original_count = len(self.gdf)
        data_hash = hashlib.md5(str(self.gdf.geometry.bounds.values.tobytes()).encode()).hexdigest()[:8]
        
        # Extract basenames from training data paths for verbose naming
        gpkg_basenames = []
        for path_str in self.config.train_data_paths:
            path = Path(path_str)
            # Extract meaningful part: e.g., "Niedersachsen_2022" from "Niedersachsen_2022_InvekosDataset_c823cf0c.gpkg"
            basename = path.stem
            if '_InvekosDataset_' in basename:
                meaningful_name = basename.split('_InvekosDataset_')[0]
            else:
                meaningful_name = basename
            gpkg_basenames.append(meaningful_name)
        
        # Create verbose cache key
        gpkg_names_str = "_".join(sorted(set(gpkg_basenames)))  # Remove duplicates and sort
        cache_key = f"global_filtered_polygons_{gpkg_names_str}_patch{patch_size}_epsg{self.target_epsg.split(':')[1]}_{data_hash}"
        cache_file = cache_dir / f"{cache_key}.gpkg"
        
        # Try to load from cache first
        if cache_file.exists():
            logger.info(f"Loading globally filtered polygons from cache: {cache_file.name}")
            self.gdf = gpd.read_file(cache_file)
            # Ensure correct CRS
            if self.gdf.crs != self.target_epsg:
                self.gdf = self.gdf.to_crs(self.target_epsg)
        else:
            # Buffer polygons to check viability but keep original geometry
            logger.info("Computing globally filtered polygons (buffering for size check only)")
            
            # Create buffered version for filtering
            buffered_gdf = self.gdf.copy()
            buffered_gdf['geometry_buffered'] = self.gdf.geometry.buffer(buffer_distance)
            
            # Filter out empty geometries in both original and buffered
            original_valid = ~self.gdf.geometry.is_empty
            buffered_valid = ~buffered_gdf.geometry_buffered.is_empty
            valid_mask = original_valid & buffered_valid
            
            # Keep only valid polygons but use ORIGINAL geometry (buffering was just for filtering)
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
            
            # Cache the filtered polygons (original geometry, filtered by buffer viability)
            self.gdf.to_file(cache_file, driver='GPKG')
            logger.info(f"Cached globally filtered polygons: {cache_file.name}")
        
        logger.info(f"Final polygon count for global processing: {len(self.gdf)}")
    
    def _extract_patches_from_polygon(self, polygon_row: pd.Series, vv_src: rasterio.DatasetReader, 
                                    vh_src: rasterio.DatasetReader) -> List[np.ndarray]:
        """Extract patches from a polygon area using proper inward buffering workflow.
        
        Workflow:
        1. Load masked array on UNBUFFERED polygon (array_1)
        2. Create copy and buffer nan-values inward by patch_width/2 pixels (array_2)
        3. Valid pixels in array_2 are potential centroids (ensures 100% valid patches)
        4. Sample centroids using farthest-point sampling
        5. Extract patches from array_1 using these centroids
        
        Args:
            polygon_row: Row from GeoDataFrame containing polygon
            vv_src: VV polarization raster dataset reader
            vh_src: VH polarization raster dataset reader
            
        Returns:
            List of patches as numpy arrays
        """
        geometry = polygon_row.geometry
        patch_size = self.config.neural_network.patch_size
        half_patch = patch_size // 2
        
        try:
            # Step 1: Load masked array on UNBUFFERED polygon (array_1)
            masked_vv, mask_transform = mask(vv_src, [geometry], crop=True, filled=False)
            masked_vh, _ = mask(vh_src, [geometry], crop=True, filled=False)
            
        except Exception as e:
            logger.warning(f"Failed to mask polygon: {e}")
            return []
        
        # Remove band dimension
        array_1_vv = masked_vv[0]  
        array_1_vh = masked_vh[0]
        
        # Step 2: Create array_2 - buffer nan-values inward by patch_width/2 pixels
        # Find valid (non-masked, non-NaN, non-zero) areas in array_1
        valid_mask_1 = (~array_1_vv.mask & ~array_1_vh.mask & 
                       ~np.isnan(array_1_vv) & ~np.isnan(array_1_vh) &
                       (array_1_vv != 0) & (array_1_vh != 0))
        
        if not np.any(valid_mask_1):
            return []
        
        # Create structuring element for erosion (disk-like for patch_width/2)
        # Use binary erosion to shrink valid area inward by half_patch pixels
        structure = np.ones((2 * half_patch + 1, 2 * half_patch + 1))
        valid_mask_2 = binary_erosion(valid_mask_1, structure=structure)
        
        if not np.any(valid_mask_2):
            return []
        
        # Step 3: Get potential centroid coordinates from array_2
        potential_centroids = np.column_stack(np.where(valid_mask_2))
        
        if len(potential_centroids) == 0:
            return []
        
        # Step 4: Calculate how many patches we need
        max_patches = self._calculate_max_patches_for_polygon(geometry, patch_size)
        max_patches = min(max_patches, len(potential_centroids))
        
        if max_patches == 0:
            return []
        
        # Step 5: Sample centroids using farthest-point sampling
        if max_patches >= len(potential_centroids):
            selected_centroids = potential_centroids
        else:
            selected_centroids = self._farthest_point_sampling(potential_centroids, max_patches)
        
        # Step 6: Extract patches from array_1 using selected centroids
        patches = []
        
        for centroid in selected_centroids:
            center_y, center_x = centroid
            
            # Extract patch from array_1 around this centroid
            y_start = center_y - half_patch
            y_end = center_y + half_patch + 1
            x_start = center_x - half_patch
            x_end = center_x + half_patch + 1
            
            # Safety check (should not happen with proper erosion)
            if (y_start < 0 or y_end > array_1_vv.shape[0] or 
                x_start < 0 or x_end > array_1_vv.shape[1]):
                continue
            
            # Extract patch
            patch_vv = array_1_vv[y_start:y_end, x_start:x_end]
            patch_vh = array_1_vh[y_start:y_end, x_start:x_end]
            
            # Ensure exact size
            if patch_vv.shape != (patch_size, patch_size) or patch_vh.shape != (patch_size, patch_size):
                continue
            
            # Convert masked arrays to regular arrays
            # Since we used proper erosion, all pixels should be valid
            patch_vv_data = np.ma.filled(patch_vv, fill_value=0)
            patch_vh_data = np.ma.filled(patch_vh, fill_value=0)
            
            # Final validity check (should be 100% valid due to erosion)
            if np.any(patch_vv.mask) or np.any(patch_vh.mask):
                logger.warning("Found masked pixels in patch despite erosion - this should not happen")
                continue
            
            # Stack VV and VH as channels
            patch = np.stack([patch_vv_data, patch_vh_data], axis=-1)
            patches.append(patch)
        
        return patches
    
    def _farthest_point_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample points using farthest-point sampling algorithm.
        
        Args:
            points: Array of shape (n_points, 2) with y,x coordinates
            n_samples: Number of points to sample
            
        Returns:
            Array of shape (n_samples, 2) with selected points
        """
        if n_samples >= len(points):
            return points
        
        # Start with a random corner point (could be first point)
        selected = [0]  # Start with first point
        remaining = list(range(1, len(points)))
        
        for _ in range(n_samples - 1):
            if not remaining:
                break
            
            # Calculate distances from all remaining points to all selected points
            distances = cdist(points[remaining], points[selected])
            
            # For each remaining point, find distance to closest selected point
            min_distances = np.min(distances, axis=1)
            
            # Select the point that is farthest from all selected points
            farthest_idx = np.argmax(min_distances)
            selected.append(remaining[farthest_idx])
            remaining.pop(farthest_idx)
        
        return points[selected]
    
    def _calculate_max_patches_for_polygon(self, geometry, patch_size: int) -> int:
        """Calculate maximum number of patches to extract from a polygon.
        
        Args:
            geometry: Shapely polygon geometry
            patch_size: Size of patch in pixels
            
        Returns:
            Maximum number of patches to extract
        """
        # Convert pixel size to square meters (assuming 10m resolution for Sentinel-1)
        pixel_area_sqm = 10 * 10  # 100 square meters per pixel
        patch_area_sqm = (patch_size * patch_size) * pixel_area_sqm
        
        # Get polygon area in square meters
        polygon_area_sqm = geometry.area
        
        # Calculate patches based on area ratio
        area_based_patches = int((polygon_area_sqm / patch_area_sqm) * self.config.n_patches_per_area)
        
        # Apply both limits
        max_patches = min(
            self.config.n_patches_per_feature,  # Traditional per-feature limit
            max(1, area_based_patches)  # Area-based limit (minimum 1)
        )
        
        return max_patches
    
    def _cache_patches_for_file(self, patches: List[Patch], mode: DataMode, image_tuple: ImageTuple) -> None:
        """Cache all patches for a specific image file at once.
        
        Args:
            patches: List of Patch objects to cache
            mode: Data mode (train/validation/test) 
            image_tuple: ImageTuple containing the source files
        """
        if not patches:
            return
        
        # Create cache directory
        cache_dir = PROJECT_ROOT / "data" / "cache" / "patches" / mode.value
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create hash from filenames and relevant parameters
        cache_info = {
            "mode": mode.value,
            "vv_file": image_tuple.vv_path.name,
            "vh_file": image_tuple.vh_path.name,
            "patch_size": self.config.neural_network.patch_size,
            "classes": sorted(self.config.classes),
            "n_patches_per_feature": self.config.n_patches_per_feature,
            "n_patches_per_area": self.config.n_patches_per_area,
            "epsg": self.target_epsg,
            "seed": self.seed
        }
        
        # Create stable hash from file info and parameters
        combined_info = json.dumps(cache_info, sort_keys=True)
        file_hash = hashlib.md5(combined_info.encode()).hexdigest()[:12]
        
        # Simple filename: mode_filehash.pkl
        cache_filename = f"{mode.value}_{file_hash}.pkl"
        cache_file = cache_dir / cache_filename
        
        # Save full Patch objects list using pickle to preserve all spatial metadata
        with open(cache_file, 'wb') as f:
            pickle.dump(patches, f)
        
        logger.info(f"Cached {len(patches)} patches for {image_tuple.date} in {cache_filename}")
        
    def _load_cached_patches(self, mode: DataMode, image_tuple: ImageTuple) -> List[Patch]:
        """Load cached patches for a specific image file.
        
        Args:
            mode: Data mode (train/validation/test)
            image_tuple: ImageTuple containing the source files
            
        Returns:
            List of Patch objects or None if not cached
        """
        cache_dir = PROJECT_ROOT / "data" / "cache" / "patches" / mode.value
        
        # Create the same hash as used for caching
        cache_info = {
            "mode": mode.value,
            "vv_file": image_tuple.vv_path.name,
            "vh_file": image_tuple.vh_path.name,
            "patch_size": self.config.neural_network.patch_size,
            "classes": sorted(self.config.classes),
            "n_patches_per_feature": self.config.n_patches_per_feature,
            "n_patches_per_area": self.config.n_patches_per_area,
            "epsg": self.target_epsg,
            "seed": self.seed
        }
        
        combined_info = json.dumps(cache_info, sort_keys=True)
        file_hash = hashlib.md5(combined_info.encode()).hexdigest()[:12]
        cache_filename = f"{mode.value}_{file_hash}.pkl"
        cache_file = cache_dir / cache_filename
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    patches = pickle.load(f)
                
                logger.info(f"Loaded {len(patches)} cached patches for {image_tuple.date} from {cache_filename}")
                return patches
            except Exception as e:
                logger.warning(f"Failed to load cached patches from {cache_filename}: {e}")
                return None
        else:
            return None
    

                
    
    def cache_patches_for_file(self, mode: DataMode, image_tuple: ImageTuple) -> None:
        """
        Generate and cache patches for a specific image file. This is the heavy computation.
        
        Args:
            mode: Data mode (train/validation/test)
            image_tuple: ImageTuple containing the source files
        """
        logger.info(f"Generating patches for {image_tuple.date} (this may take a while)")
        
        # Get the split data for this mode
        split_data = self.data_splits[mode]
        
        # Load the valid AOI for this image
        unique_key = self._generate_stable_unique_key(image_tuple)
        
        if unique_key not in self.tuple_aois:
            logger.warning(f"No valid AOI found for image {unique_key}")
            return
            
        valid_aoi = self.tuple_aois[unique_key]
        
        # Validate the AOI
        if valid_aoi.geometry.iloc[0].is_empty or valid_aoi.geometry.iloc[0].area == 0:
            logger.warning(f"AOI for {unique_key} is empty or has zero area")
            return
            
        # Get polygons that intersect with this AOI and are in the current split
        intersecting_positions = self.polygon_image_intersections.get(unique_key, set())
        split_indices = set(split_data.index)
        valid_indices = intersecting_positions & split_indices
        
        if not valid_indices:
            logger.debug(f"No valid polygons found for {unique_key} in {mode.value} split")
            return
            
        # Get the subset of polygons for this image
        image_polygons = split_data.loc[list(valid_indices)]
        logger.info(f"Processing {len(image_polygons)} polygons for image {unique_key}")
        
        all_patches_for_file = []
        
        try:
            with rasterio.open(image_tuple.vv_path) as vv_src, \
                 rasterio.open(image_tuple.vh_path) as vh_src:
                
                logger.debug(f"Opened rasters: VV {vv_src.bounds}, VH {vh_src.bounds}")
                
                # Iterate over polygons and extract patches
                for poly_idx, (idx, polygon_row) in enumerate(image_polygons.iterrows()):
                    if (poly_idx + 1) % 50 == 0:
                        logger.info(f"  Processing polygon {poly_idx + 1}/{len(image_polygons)}")
                    
                    try:
                        patch_objects = self._extract_patches_following_workflow(
                            polygon_row, vv_src, vh_src, image_tuple, mode
                        )
                        
                        if patch_objects:
                            all_patches_for_file.extend(patch_objects)
                            
                    except Exception as e:
                        logger.error(f"Error processing polygon {idx}: {e}")
                        continue
                
                # Cache all patches for this file
                if all_patches_for_file:
                    self._cache_patches_for_file(all_patches_for_file, mode, image_tuple)
                    logger.info(f"Cached {len(all_patches_for_file)} patches for {unique_key}")
                else:
                    logger.warning(f"No patches generated for {unique_key}")
                        
        except Exception as e:
            logger.error(f"Error opening raster files for {unique_key}: {e}")

    def yield_patch(self, mode: DataMode) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield individual patches from cache. Only loads cached patches and shuffles them.
        
        Args:
            mode: Data mode (train/validation/test)
            
        Yields:
            Tuples of (patch_data, class_id) for individual patches
        """
        logger.info(f"Loading cached patches for {mode.value} mode")
        
        # Sort image tuples for consistent order
        self.image_tuples.sort(key=lambda x: (x.date, x.vh_path, x.vv_path))
        
        for img_idx, image_tuple in enumerate(self.image_tuples):
            # Check if cache exists
            cached_patches = self._load_cached_patches(mode, image_tuple)
            
            if cached_patches is None:
                # Cache doesn't exist, generate it
                logger.warning(f"Cache not found for {image_tuple.date}, generating patches...")
                self.cache_patches_for_file(mode, image_tuple)
                
                # Try loading again after caching
                cached_patches = self._load_cached_patches(mode, image_tuple)
                if cached_patches is not None:
                    logger.debug(f"Loaded {len(cached_patches)} newly cached patches for {image_tuple.date}")
                else:
                    logger.error(f"Failed to cache or load patches for {image_tuple.date}")
                    continue

            indices = np.random.permutation(len(cached_patches))

            # Yield shuffled patches
            for idx in indices:
                patch = cached_patches[idx]
                yield patch.data, patch.class_id

    def yield_batch(self, mode: DataMode) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate batches of patches using the yield_patch method.
        
        This method collects individual patches from yield_patch into batches
        and handles shuffling and batch management.
        
        Args:
            mode: Data mode (train/validation/test)
            
        Yields:
            Tuples of (patches_array, labels_array) for batches
        """
        logger.info(f"Starting batch generation for {mode.value} mode")
        
        batch_size = self.config.neural_network.batch_size
        batch_patches = []
        batch_labels = []
        
        # Use yield_patch to get individual patches with caching
        patch_generator = self.yield_patch(mode)
        
        for patch_data, class_id in patch_generator:
            batch_patches.append(patch_data)
            batch_labels.append(class_id)
            
            # Yield batch when full
            if len(batch_patches) >= batch_size:
                # Convert to arrays
                patches_array = np.array(batch_patches)
                labels_array = np.array(batch_labels)
                
                # Shuffle batch for training
                indices = np.random.permutation(len(patches_array))
                patches_array = patches_array[indices]
                labels_array = labels_array[indices]
                
                logger.debug(f"Yielding batch: {patches_array.shape}")
                yield patches_array, labels_array
                
                # Reset batch
                batch_patches = []
                batch_labels = []
        
        # Yield final partial batch if any patches remain
        if batch_patches:
            logger.info(f"Final batch: {len(batch_patches)} patches for {mode.value} mode")
            patches_array = np.array(batch_patches)
            labels_array = np.array(batch_labels)
            
            # Shuffle final batch
            indices = np.random.permutation(len(patches_array))
            patches_array = patches_array[indices]
            labels_array = labels_array[indices]
            
            yield patches_array, labels_array
        
    def _extract_patches_following_workflow(self, polygon_row: pd.Series, vv_src: rasterio.DatasetReader,
                                          vh_src: rasterio.DatasetReader, image_tuple: ImageTuple, 
                                          mode: DataMode) -> List[Patch]:
        """
        Extract patches following your specified workflow:
        1. Load masked array on UNBUFFERED polygon
        2. Copy array
        3. Buffer nan-values outside valid area into valid area by n=patch_width/2 pixels
        4. Valid pixels of buffered array are potential centroids (ensures 100% inside polygons)
        5. Take n samples using farthest-point sampling starting from one corner
        6. Create patches from array_1 using these centroids
        """
        geometry = polygon_row.geometry
        class_id = polygon_row[self.config.column_id]
        patch_size = self.config.neural_network.patch_size
        
        # Calculate number of patches to extract using the config parameters
        n_samples = self._calculate_max_patches_for_polygon(geometry, patch_size)
        
        logger.debug(f"Extracting patches following workflow for class {class_id}")
        logger.debug(f"Polygon bounds: {geometry.bounds}")
        logger.debug(f"Raster bounds: {vv_src.bounds}")
        logger.debug(f"Max patches for this polygon: {n_samples}")
        
        try:
            # Step 1: Load masked array on UNBUFFERED polygon
            from rasterio.mask import mask
            masked_vv, mask_transform = mask(vv_src, [geometry], crop=True, filled=False)
            masked_vh, _ = mask(vh_src, [geometry], crop=True, filled=False)
            
            # Remove band dimension
            array_1_vv = masked_vv[0]  # Original unbuffered data
            array_1_vh = masked_vh[0]
            
            logger.debug(f"Loaded masked arrays: VV shape {array_1_vv.shape}, VH shape {array_1_vh.shape}")
            
            # Step 2: Copy array
            array_2_vv = array_1_vv.copy()
            array_2_vh = array_1_vh.copy()
            
            # Step 3: Buffer nan-values outside valid area into valid area by n=patch_width/2 pixels
            buffer_pixels = patch_size // 2
            
            # Create binary mask of valid pixels (not NaN and not masked)
            valid_mask_vv = ~np.ma.getmaskarray(array_2_vv) & ~np.isnan(array_2_vv)
            valid_mask_vh = ~np.ma.getmaskarray(array_2_vh) & ~np.isnan(array_2_vh)
            valid_mask = valid_mask_vv & valid_mask_vh
            
            # Apply binary erosion to create inward buffer
            from scipy.ndimage import binary_erosion
            structure = np.ones((buffer_pixels * 2 + 1, buffer_pixels * 2 + 1))
            buffered_valid_mask = binary_erosion(valid_mask, structure=structure)
            
            logger.debug(f"Valid pixels before buffering: {valid_mask.sum()}")
            logger.debug(f"Valid pixels after buffering: {buffered_valid_mask.sum()}")
            
            if buffered_valid_mask.sum() == 0:
                logger.debug("No valid pixels remaining after buffering")
                return []
            
            # Step 4: Valid pixels of array_2 are potential centroids
            potential_centroids = np.column_stack(np.where(buffered_valid_mask))
            
            if len(potential_centroids) == 0:
                logger.debug("No potential centroids found")
                return []
            
            # Step 5: Take n samples using farthest-point sampling
            n_samples_actual = min(n_samples, len(potential_centroids))
            
            if n_samples_actual == 1:
                # Just take one point (could be random or corner)
                selected_centroids = potential_centroids[:1]
            else:
                # Use farthest-point sampling
                selected_centroids = self._farthest_point_sampling(potential_centroids, n_samples_actual)
            
            logger.debug(f"Selected {len(selected_centroids)} centroids from {len(potential_centroids)} candidates")
            
            # Step 6: Create patches from array_1 using selected centroids
            patches = []
            half_patch = patch_size // 2
            
            for centroid in selected_centroids:
                row, col = centroid
                
                # Extract patch bounds
                row_start = row - half_patch
                row_end = row + half_patch
                col_start = col - half_patch  
                col_end = col + half_patch
                
                # Check bounds
                if (row_start >= 0 and row_end <= array_1_vv.shape[0] and
                    col_start >= 0 and col_end <= array_1_vv.shape[1]):
                    
                    # Extract patch from original unbuffered data (array_1)
                    patch_vv = array_1_vv[row_start:row_end, col_start:col_end]
                    patch_vh = array_1_vh[row_start:row_end, col_start:col_end]
                    
                    # Combine VV and VH channels
                    patch_data = np.stack([patch_vv, patch_vh], axis=-1)
                    
                    # Calculate patch bounds in world coordinates
                    patch_transform = mask_transform * rasterio.Affine.translation(col_start, row_start)
                    patch_bounds = rasterio.transform.array_bounds(
                        patch_size, patch_size, patch_transform
                    )
                    
                    # Create patch object
                    patch_obj = Patch(
                        data=patch_data,
                        transform=patch_transform,
                        bounds=patch_bounds,
                        crs=str(vv_src.crs),
                        orbit=image_tuple.date,  # Using date as orbit identifier
                        src_files=(str(image_tuple.vv_path), str(image_tuple.vh_path)),
                        date=image_tuple.date,
                        class_id=class_id,
                        data_mode=mode
                    )
                    
                    patches.append(patch_obj)
                    
                else:
                    logger.debug(f"Centroid {centroid} would create out-of-bounds patch")
            
            logger.debug(f"Successfully created {len(patches)} patches")
            return patches
            
        except Exception as e:
            logger.error(f"Error in patch extraction workflow: {e}")
            return []
        pass