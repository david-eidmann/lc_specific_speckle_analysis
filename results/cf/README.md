# Test Confusion Matrices - All Configuration Combinations

This folder contains test set confusion matrices for all 8 architecture and configuration combinations.

## Files Overview

### TestConv2D (93,700 parameters)
| File | Architecture | Data Processing | Temporal Coverage | OA | Macro F1 |
|------|-------------|-----------------|------------------|-------|----------|
| `test_cf_conv2d_raw_single_20220611.png` | TestConv2D | Raw data | Single (20220611) | 0.701 | 0.695 |
| `test_cf_conv2d_raw_multi_20220611_20220623.png` | TestConv2D | Raw data | Multi (20220611,20220623) | 0.711 | 0.710 |
| `test_cf_conv2d_zeromean_single_20220611.png` | TestConv2D | Zero-mean | Single (20220611) | 0.495 | 0.473 |
| `test_cf_conv2d_zeromean_multi_20220611_20220623.png` | TestConv2D | Zero-mean | Multi (20220611,20220623) | 0.480 | 0.459 |

### TestConv2D_N2 (1,436 parameters)
| File | Architecture | Data Processing | Temporal Coverage | OA | Macro F1 |
|------|-------------|-----------------|------------------|-------|----------|
| `test_cf_conv2d_n2_raw_single_20220611.png` | **TestConv2D_N2** | **Raw data** | **Single (20220611)** | **0.731** ⭐ | **0.729** ⭐ |
| `test_cf_conv2d_n2_raw_multi_20220611_20220623.png` | TestConv2D_N2 | Raw data | Multi (20220611,20220623) | 0.672 | 0.653 |
| `test_cf_conv2d_n2_zeromean_single_20220611.png` | TestConv2D_N2 | Zero-mean | Single (20220611) | 0.521 | 0.498 |
| `test_cf_conv2d_n2_zeromean_multi_20220611_20220623.png` | TestConv2D_N2 | Zero-mean | Multi (20220611,20220623) | 0.485 | 0.457 |

## Performance Ranking (by Overall Accuracy)

1. **🥇 TestConv2D_N2 + Raw + Single**: 73.1% OA (`test_cf_conv2d_n2_raw_single_20220611.png`)
2. **🥈 TestConv2D + Raw + Multi**: 71.1% OA (`test_cf_conv2d_raw_multi_20220611_20220623.png`)
3. **🥉 TestConv2D + Raw + Single**: 70.1% OA (`test_cf_conv2d_raw_single_20220611.png`)
4. TestConv2D_N2 + Raw + Multi: 67.2% OA (`test_cf_conv2d_n2_raw_multi_20220611_20220623.png`)
5. TestConv2D_N2 + Zero-mean + Single: 52.1% OA (`test_cf_conv2d_n2_zeromean_single_20220611.png`)
6. TestConv2D + Zero-mean + Single: 49.5% OA (`test_cf_conv2d_zeromean_single_20220611.png`)
7. TestConv2D_N2 + Zero-mean + Multi: 48.5% OA (`test_cf_conv2d_n2_zeromean_multi_20220611_20220623.png`)
8. TestConv2D + Zero-mean + Multi: 48.0% OA (`test_cf_conv2d_zeromean_multi_20220611_20220623.png`)

## Key Insights from Confusion Matrices

### Classes
- **Class 1**: First land cover type
- **Class 4**: Fourth land cover type  
- **Class 6**: Sixth land cover type
- **Class 12**: Twelfth land cover type

### Pattern Analysis
- **Raw data configurations** show much clearer class separation and higher diagonal values
- **Zero-mean configurations** exhibit more confusion between classes and lower overall accuracy
- **TestConv2D_N2** achieves the best classification performance despite having 65x fewer parameters

### Best Performing Model
The **TestConv2D_N2 with raw data and single date** configuration shows:
- Highest diagonal values (correct predictions)
- Minimal off-diagonal confusion
- Balanced performance across all 4 classes
- 73.1% overall accuracy with only 1,436 parameters

**Date**: December 1, 2025  
**Total Files**: 8 test confusion matrices  
**Best Model**: TestConv2D_N2 + Raw + Single (20220611)
