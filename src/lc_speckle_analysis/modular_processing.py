"""Modular data processing pipeline for patch transformations.

This module implements the new modular processing system where transformations
are applied in a specific order: shuffle -> zero_mean -> normalize -> quantiles -> aggregation.
"""

import logging
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


def apply_spatial_shuffle(patch_data: np.ndarray, seed: int = 42) -> np.ndarray:
    """Apply spatial shuffling to patch data.
    
    Shuffles pixels spatially while preserving spectral relationships.
    Both channels are shuffled in the same order.
    
    Args:
        patch_data: Input patch data with shape (channels, height, width)
        seed: Random seed for reproducible shuffling
        
    Returns:
        Spatially shuffled patch data
    """
    np.random.seed(seed)
    processed_data = patch_data.copy()
    channels, height, width = processed_data.shape
    
    # Create consistent permutation for all channels
    n_pixels = height * width
    perm_indices = np.random.permutation(n_pixels)
    
    # Apply same shuffling to all channels
    for channel_idx in range(channels):
        channel_data = processed_data[channel_idx, :, :]
        flat_data = channel_data.flatten()
        shuffled_data = flat_data[perm_indices]
        processed_data[channel_idx, :, :] = shuffled_data.reshape(height, width)
    
    return processed_data


def apply_zero_mean(patch_data: np.ndarray) -> np.ndarray:
    """Apply zero-mean normalization to patch data.
    
    Subtracts the mean per patch and channel to center each patch at zero.
    
    Args:
        patch_data: Input patch data with shape (channels, height, width)
        
    Returns:
        Zero-mean patch data
    """
    processed_data = patch_data.copy()
    
    for channel in range(processed_data.shape[0]):
        channel_data = processed_data[channel, :, :]
        channel_mean = np.mean(channel_data)
        processed_data[channel, :, :] = channel_data - channel_mean
    
    logger.debug(f"Applied zero-mean normalization to patch with shape {patch_data.shape}")
    return processed_data


def apply_normalization(patch_data: np.ndarray) -> np.ndarray:
    """Normalize patch data to std=1.
    
    Applies per-channel standardization to achieve std=1.
    
    Args:
        patch_data: Input patch data with shape (channels, height, width)
        
    Returns:
        Normalized patch data with std=1 per channel
    """
    processed_data = patch_data.copy()
    
    for channel in range(processed_data.shape[0]):
        channel_data = processed_data[channel, :, :]
        channel_std = np.std(channel_data)
        
        if channel_std > 0:  # Avoid division by zero
            processed_data[channel, :, :] = channel_data / channel_std
        else:
            logger.warning(f"Channel {channel} has zero standard deviation, skipping normalization")
    
    return processed_data


def apply_quantile_transformation(patch_data: np.ndarray) -> np.ndarray:
    """Apply quantile transformation to patch data.
    
    Transforms pixel values to uniform quantiles per channel.
    
    Args:
        patch_data: Input patch data with shape (channels, height, width)
        
    Returns:
        Quantile-transformed patch data
    """
    processed_data = patch_data.copy()
    channels, height, width = processed_data.shape
    
    for channel in range(channels):
        channel_data = processed_data[channel, :, :]
        flat_data = channel_data.flatten().reshape(-1, 1)
        
        # Use quantile transformer with appropriate number of quantiles
        # Set n_quantiles to min(1000, n_samples) to avoid sklearn warning
        n_samples = flat_data.shape[0]
        n_quantiles = min(1000, n_samples)
        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles, random_state=42)
        transformed_flat = qt.fit_transform(flat_data)
        processed_data[channel, :, :] = transformed_flat.reshape(height, width)
    
    return processed_data


def apply_aggregation(patch_data: np.ndarray, aggregation_type: str) -> np.ndarray:
    """Apply statistical aggregation to patch data.
    
    Computes statistical features from patch data.
    
    Args:
        patch_data: Input patch data with shape (channels, height, width)
        aggregation_type: Type of aggregation ('std', 'mean', or 'stdandmean')
        
    Returns:
        Aggregated features as 1D array
    """
    if aggregation_type not in ['std', 'mean', 'stdandmean']:
        raise ValueError(f"Invalid aggregation type: {aggregation_type}")
    
    vv_channel = patch_data[0, :, :]
    vh_channel = patch_data[1, :, :]
    
    if aggregation_type == 'mean':
        return np.array([np.mean(vv_channel), np.mean(vh_channel)])
    elif aggregation_type == 'std':
        return np.array([np.std(vv_channel), np.std(vh_channel)])
    elif aggregation_type == 'stdandmean':
        return np.array([
            np.mean(vv_channel), np.std(vv_channel),
            np.mean(vh_channel), np.std(vh_channel)
        ])


def process_patch(patch_data: np.ndarray, 
                  shuffled: bool = False,
                  zero_mean: bool = False,
                  normalized: bool = False, 
                  quantiles: bool = False,
                  aggregation: Optional[str] = None,
                  seed: int = 42) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Apply modular processing pipeline to patch data.
    
    Applies transformations in the specified order:
    1. Spatial shuffling (if enabled)
    2. Zero-mean normalization (if enabled)
    3. Normalization to std=1 (if enabled)  
    4. Quantile transformation (if enabled)
    5. Statistical aggregation (if specified)
    
    Args:
        patch_data: Input patch data with shape (channels, height, width)
        shuffled: Whether to apply spatial shuffling
        zero_mean: Whether to subtract mean per patch and channel
        normalized: Whether to normalize to std=1
        quantiles: Whether to apply quantile transformation
        aggregation: Statistical aggregation type ('std', 'mean', 'stdandmean', or None)
        seed: Random seed for reproducible operations
        
    Returns:
        Processed patch data. Shape depends on processing configuration.
        3D array (channels, height, width) if no aggregation,
        1D feature vector if aggregation is applied.
    """
    processed_data = patch_data.copy()
    
    # Step 1: Spatial shuffling
    if shuffled:
        processed_data = apply_spatial_shuffle(processed_data, seed=seed)
    
    # Step 2: Zero-mean normalization
    if zero_mean:
        processed_data = apply_zero_mean(processed_data)
    
    # Step 3: Normalization to std=1
    if normalized:
        processed_data = apply_normalization(processed_data)
    
    # Step 4: Quantile transformation
    if quantiles:
        processed_data = apply_quantile_transformation(processed_data)
    
    # Step 5: Aggregation
    if aggregation is not None:
        processed_data = apply_aggregation(processed_data, aggregation)
    
    return processed_data


def determine_network_architecture(aggregation: Optional[str]) -> str:
    """Determine appropriate network architecture based on aggregation setting.
    
    Args:
        aggregation: Aggregation type or None
        
    Returns:
        Network architecture identifier
    """
    if aggregation is not None:
        # Statistical features -> use LinearStatsNet
        return 'linear_stats_net'
    else:
        # Spatial data -> use Conv2D_N2
        return 'test_conv2d_n2'


def calculate_input_size(aggregation: Optional[str], patch_size: int = 10) -> int:
    """Calculate input size for network based on processing configuration.
    
    Args:
        aggregation: Aggregation type or None
        patch_size: Size of spatial patches (only relevant if no aggregation)
        
    Returns:
        Input size for the network
    """
    if aggregation == 'mean' or aggregation == 'std':
        return 2  # 2 features (VV, VH)
    elif aggregation == 'stdandmean':
        return 4  # 4 features (VV_mean, VV_std, VH_mean, VH_std)
    else:
        return 2 * patch_size * patch_size  # Spatial data: channels * height * width


def get_expected_input_size(patch_size: int, aggregation: Optional[str]) -> int:
    """Get expected input size for network based on processing configuration.
    
    Args:
        patch_size: Size of spatial patches (only relevant if no aggregation)
        aggregation: Aggregation type or None
        
    Returns:
        Input size for the network
    """
    return calculate_input_size(aggregation, patch_size)
