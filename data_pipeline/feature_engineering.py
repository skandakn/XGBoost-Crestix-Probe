"""
feature_engineering.py
======================
Comprehensive feature extraction and tabular transformation pipeline for the
LarNO UKEA 32x32 urban flood benchmark.

Extracts scientifically meaningful topographic, meteorological, hydraulic lag,
and spatio-temporal features into high-performance tabular datasets compatible
with XGBoost, LightGBM, and SHAP explainability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from data_pipeline.load_data import FloodEventData


def compute_topographic_features(
    dem: np.ndarray,
    cell_size_m: float = 8.0,
) -> Dict[str, np.ndarray]:
    """
    Compute static topographic features from Digital Elevation Model (DEM).

    Parameters
    ----------
    dem : np.ndarray
        Elevation grid of shape (H, W), float32 in meters.
    cell_size_m : float
        Grid resolution (8.0 meters for UKEA benchmark).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of 2D feature arrays of shape (H, W).
    """
    dem_arr = np.asarray(dem, dtype=np.float32)
    H, W = dem_arr.shape

    # Detect building or nodata mask (NaNs or extreme sentinels)
    building_mask = np.isnan(dem_arr) | (dem_arr < -100.0) | (dem_arr > 10000.0)
    clean_dem = dem_arr.copy()
    if np.any(building_mask):
        valid_median = np.nanmedian(clean_dem)
        clean_dem[building_mask] = valid_median

    # 1. Slope & First Spatial Derivatives (central differences)
    grad_y, grad_x = np.gradient(clean_dem, cell_size_m, cell_size_m)
    slope_mag = np.sqrt(grad_x**2 + grad_y**2)  # Dimensionless gradient [m/m]

    # 2. Aspect (Direction of steepest slope)
    # Decompose into continuous cosine and sine components to avoid angular discontinuity at 0/2pi
    aspect_rad = np.arctan2(-grad_y, -grad_x)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    # 3. Second Spatial Derivatives & Curvature
    grad_xx = np.gradient(grad_x, cell_size_m, axis=1)
    grad_yy = np.gradient(grad_y, cell_size_m, axis=0)
    grad_xy = np.gradient(grad_x, cell_size_m, axis=0)

    # Laplacian: ∇²z (positive for local depressions/valleys, negative for ridges/peaks)
    laplacian = grad_xx + grad_yy

    # Profile & Planform Curvatures
    denom = slope_mag**2 + 1e-6
    profile_curv = (grad_xx * grad_x**2 + 2 * grad_xy * grad_x * grad_y + grad_yy * grad_y**2) / denom
    plan_curv = (grad_xx * grad_y**2 - 2 * grad_xy * grad_x * grad_y + grad_yy * grad_x**2) / (slope_mag**3 + 1e-6)

    # 4. Local Neighborhood Variations (3x3 and 5x5 kernels)
    mean_3x3 = uniform_filter(clean_dem, size=3, mode="reflect")
    sq_mean_3x3 = uniform_filter(clean_dem**2, size=3, mode="reflect")
    std_3x3 = np.sqrt(np.maximum(0.0, sq_mean_3x3 - mean_3x3**2))

    mean_5x5 = uniform_filter(clean_dem, size=5, mode="reflect")
    sq_mean_5x5 = uniform_filter(clean_dem**2, size=5, mode="reflect")
    std_5x5 = np.sqrt(np.maximum(0.0, sq_mean_5x5 - mean_5x5**2))

    # Topographic Position Index (TPI): relative elevation vs neighborhood
    tpi_3x3 = clean_dem - mean_3x3
    tpi_5x5 = clean_dem - mean_5x5

    # 5. 8-Connected Neighbour Elevation Differences
    # Pad DEM with border reflection
    padded = np.pad(clean_dem, pad_width=1, mode="reflect")
    # Extract 8 neighbors
    neighbors = [
        padded[0:H, 0:W], padded[0:H, 1:W+1], padded[0:H, 2:W+2],
        padded[1:H+1, 0:W],                     padded[1:H+1, 2:W+2],
        padded[2:H+2, 0:W], padded[2:H+2, 1:W+1], padded[2:H+2, 2:W+2],
    ]
    diffs = [n - clean_dem for n in neighbors]
    diff_stack = np.stack(diffs, axis=0)  # Shape: (8, H, W)

    diff_mean = np.mean(diff_stack, axis=0)
    diff_min = np.min(diff_stack, axis=0)  # Most negative indicates primary drainage path
    diff_max = np.max(diff_stack, axis=0)

    # 6. Proxy Flow Accumulation (upslope drainage proxy based on negative slope divergence)
    flow_acc_proxy = np.maximum(0.0, -laplacian) + np.maximum(0.0, -tpi_5x5)

    return {
        "elevation": clean_dem,
        "slope_magnitude": slope_mag.astype(np.float32),
        "slope_x": grad_x.astype(np.float32),
        "slope_y": grad_y.astype(np.float32),
        "aspect_sin": aspect_sin.astype(np.float32),
        "aspect_cos": aspect_cos.astype(np.float32),
        "laplacian_elevation": laplacian.astype(np.float32),
        "profile_curvature": profile_curv.astype(np.float32),
        "plan_curvature": plan_curv.astype(np.float32),
        "local_elevation_std_3x3": std_3x3.astype(np.float32),
        "local_elevation_std_5x5": std_5x5.astype(np.float32),
        "tpi_3x3": tpi_3x3.astype(np.float32),
        "tpi_5x5": tpi_5x5.astype(np.float32),
        "neighbour_elevation_diff_mean": diff_mean.astype(np.float32),
        "neighbour_elevation_diff_min": diff_min.astype(np.float32),
        "neighbour_elevation_diff_max": diff_max.astype(np.float32),
        "flow_accumulation_proxy": flow_acc_proxy.astype(np.float32),
        "is_building_mask": building_mask.astype(np.float32),
    }


def compute_rainfall_features(
    rainfall: np.ndarray,
    dt_min: float = 5.0,
    window_steps: Tuple[int, ...] = (3, 6, 12),
) -> Dict[str, np.ndarray]:
    """
    Compute dynamic precipitation forcing features across all time steps.

    Parameters
    ----------
    rainfall : np.ndarray
        Rainfall grid array of shape (T, H, W) or (T,), float32 in mm/5min.
    dt_min : float
        Time step resolution in minutes (5.0 min for LarNO UKEA).
    window_steps : Tuple[int, ...]
        Steps for historical rolling accumulation (e.g. 3=15m, 6=30m, 12=60m).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of dynamic 3D feature arrays of shape (T, H, W).
    """
    r_arr = np.asarray(rainfall, dtype=np.float32)
    if r_arr.ndim == 1:
        raise ValueError("Rainfall array must be broadcasted to 3D (T, H, W) before feature extraction.")

    T, H, W = r_arr.shape

    # 1. Instantaneous Intensity
    rain_current = r_arr.copy()

    # 2. Cumulative Storm Rainfall (strictly causal: sum up to time t)
    rain_cumulative = np.cumsum(r_arr, axis=0)

    # 3. Rolling Window Precipitation Sums & Means
    rolling_features: Dict[str, np.ndarray] = {}
    for w in window_steps:
        duration_min = int(w * dt_min)
        w_sum = np.zeros_like(r_arr)
        for t in range(T):
            t_start = max(0, t - w + 1)
            w_sum[t] = np.sum(r_arr[t_start : t + 1], axis=0)

        rolling_features[f"rainfall_window_{duration_min}m"] = w_sum.astype(np.float32)
        rolling_features[f"rainfall_mean_{duration_min}m"] = (w_sum / w).astype(np.float32)

    # 4. Maximum Intensity Experienced So Far
    rain_max_so_far = np.maximum.accumulate(r_arr, axis=0)

    # 5. Rate of Change (Storm Acceleration)
    rate_of_change = np.zeros_like(r_arr)
    rate_of_change[1:] = (r_arr[1:] - r_arr[:-1]) / dt_min

    features = {
        "rainfall_intensity_current": rain_current,
        "rainfall_cumulative": rain_cumulative.astype(np.float32),
        "rainfall_max_so_far": rain_max_so_far.astype(np.float32),
        "rainfall_rate_of_change": rate_of_change.astype(np.float32),
        **rolling_features,
    }
    return features


def compute_hydraulic_lag_features(
    h: np.ndarray,
    dem: np.ndarray,
    dt_min: float = 5.0,
    lag_steps: Tuple[int, ...] = (1, 2, 3),
) -> Dict[str, np.ndarray]:
    """
    Compute dynamic hydraulic state, antecedent lags, and fluid gradients.

    Parameters
    ----------
    h : np.ndarray
        Water depth array of shape (T, H, W), float32 in meters.
    dem : np.ndarray
        Terrain elevation grid of shape (H, W), float32 in meters.
    dt_min : float
        Time step in minutes (5.0 min).
    lag_steps : Tuple[int, ...]
        Historical observation lags in timesteps (e.g. 1=5m, 2=10m, 3=15m).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of 3D feature arrays of shape (T, H, W).
    """
    h_arr = np.asarray(h, dtype=np.float32)
    dem_arr = np.asarray(dem, dtype=np.float32)
    T, H, W = h_arr.shape

    # 1. Current Water Depth (h_t)
    current_depth = h_arr.copy()

    # 2. Historical Lags (strictly causal: h at t - lag)
    lag_features: Dict[str, np.ndarray] = {}
    for lag in lag_steps:
        lag_min = int(lag * dt_min)
        lag_arr = np.zeros_like(h_arr)
        if lag < T:
            # Shift back by lag steps; pad initial timesteps with 0.0 (antecedent dry condition)
            lag_arr[lag:] = h_arr[:-lag]
        lag_features[f"water_depth_lag_{lag_min}m"] = lag_arr.astype(np.float32)

    # 3. Rate of Rise (dh / dt) in m / min
    rate_of_rise = np.zeros_like(h_arr)
    rate_of_rise[1:] = (h_arr[1:] - h_arr[:-1]) / dt_min

    # 4. Acceleration of Rise (d²h / dt²) in m / min²
    acceleration = np.zeros_like(h_arr)
    if T >= 3:
        acceleration[2:] = (h_arr[2:] - 2 * h_arr[1:-1] + h_arr[:-2]) / (dt_min**2)

    # 5. Spatial Neighborhood Depths (3x3 mean and max surrounding cells)
    neigh_depth_mean = np.zeros_like(h_arr)
    neigh_depth_max = np.zeros_like(h_arr)
    for t in range(T):
        neigh_depth_mean[t] = uniform_filter(h_arr[t], size=3, mode="reflect")
        # Extract max in 3x3 window using pad
        padded_h = np.pad(h_arr[t], pad_width=1, mode="reflect")
        slices = [
            padded_h[0:H, 0:W], padded_h[0:H, 1:W+1], padded_h[0:H, 2:W+2],
            padded_h[1:H+1, 0:W], padded_h[1:H+1, 1:W+1], padded_h[1:H+1, 2:W+2],
            padded_h[2:H+2, 0:W], padded_h[2:H+2, 1:W+1], padded_h[2:H+2, 2:W+2],
        ]
        neigh_depth_max[t] = np.max(np.stack(slices, axis=0), axis=0)

    # 6. Hydraulic Head (Piezometric Water Surface Elevation: H = z + h)
    hydraulic_head = dem_arr[np.newaxis, :, :] + h_arr

    # 7. Hydraulic Gradient Magnitude ||∇(z + h)|| (momentum driver)
    head_grad_mag = np.zeros_like(h_arr)
    for t in range(T):
        gy, gx = np.gradient(hydraulic_head[t])
        head_grad_mag[t] = np.sqrt(gx**2 + gy**2)

    features = {
        "current_water_depth": current_depth,
        "water_depth_rate_of_rise": rate_of_rise.astype(np.float32),
        "water_depth_acceleration": acceleration.astype(np.float32),
        "neighbour_water_depth_mean_3x3": neigh_depth_mean.astype(np.float32),
        "neighbour_water_depth_max_3x3": neigh_depth_max.astype(np.float32),
        "hydraulic_head": hydraulic_head.astype(np.float32),
        "hydraulic_head_grad_mag": head_grad_mag.astype(np.float32),
        **lag_features,
    }
    return features


def compute_spatial_temporal_features(
    T: int,
    grid_shape: Tuple[int, int] = (32, 32),
    dt_min: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Compute static spatial coordinates and temporal progress features.

    Parameters
    ----------
    T : int
        Number of event timesteps.
    grid_shape : Tuple[int, int]
        Grid dimensions (H, W).
    dt_min : float
        Timestep in minutes (5.0).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of 3D feature arrays of shape (T, H, W).
    """
    H, W = grid_shape
    # Spatial coordinates normalized to [0, 1]
    y_idx, x_idx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    norm_x = x_idx / (W - 1)
    norm_y = y_idx / (H - 1)

    # Normalized distance from domain center
    center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
    dist_center = np.sqrt((y_idx - center_y) ** 2 + (x_idx - center_x) ** 2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    norm_dist = dist_center / max_dist

    # Broadcast spatial coordinates to (T, H, W)
    coord_x_3d = np.broadcast_to(norm_x[np.newaxis, :, :], (T, H, W)).copy()
    coord_y_3d = np.broadcast_to(norm_y[np.newaxis, :, :], (T, H, W)).copy()
    dist_center_3d = np.broadcast_to(norm_dist[np.newaxis, :, :], (T, H, W)).copy()

    # Temporal features
    t_elapsed = np.arange(T, dtype=np.float32) * dt_min
    t_fraction = np.arange(T, dtype=np.float32) / max(1, T - 1)

    t_elapsed_3d = np.broadcast_to(t_elapsed[:, np.newaxis, np.newaxis], (T, H, W)).copy()
    t_fraction_3d = np.broadcast_to(t_fraction[:, np.newaxis, np.newaxis], (T, H, W)).copy()

    return {
        "norm_coord_x": coord_x_3d.astype(np.float32),
        "norm_coord_y": coord_y_3d.astype(np.float32),
        "dist_to_center": dist_center_3d.astype(np.float32),
        "time_since_onset_min": t_elapsed_3d.astype(np.float32),
        "time_fraction": t_fraction_3d.astype(np.float32),
    }


def build_event_tabular_dataframe(
    event: FloodEventData,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Extract all features for a single event and flatten into a tabular DataFrame.

    Each row represents a specific grid cell (row_idx, col_idx) at a specific
    time step (timestep_idx) during the flood event.

    Parameters
    ----------
    event : FloodEventData
        The loaded flood event.
    config : Optional[Dict[str, Any]]
        Optional feature toggle configuration.

    Returns
    -------
    pd.DataFrame
        Tabular dataset ready for target attachment and modeling.
    """
    T = event.timesteps
    H, W = event.grid_shape

    # 1. Extract feature sets
    topo_dict = compute_topographic_features(event.dem, cell_size_m=event.spatial_resolution_m)
    rain_dict = compute_rainfall_features(event.rainfall, dt_min=event.temporal_resolution_min)
    hydr_dict = compute_hydraulic_lag_features(event.h, event.dem, dt_min=event.temporal_resolution_min)
    spat_dict = compute_spatial_temporal_features(T, grid_shape=event.grid_shape, dt_min=event.temporal_resolution_min)

    # 2. Flatten metadata coordinates
    t_indices, r_indices, c_indices = np.meshgrid(
        np.arange(T, dtype=np.int32),
        np.arange(H, dtype=np.int32),
        np.arange(W, dtype=np.int32),
        indexing="ij",
    )

    data_columns: Dict[str, np.ndarray] = {
        "event_id": np.full(T * H * W, event.event_id, dtype=object),
        "timestep_idx": t_indices.ravel(),
        "row_idx": r_indices.ravel(),
        "col_idx": c_indices.ravel(),
    }

    # 3. Flatten Topographic Features (broadcast 2D -> 3D -> 1D)
    for feat_name, feat_2d in topo_dict.items():
        feat_3d = np.broadcast_to(feat_2d[np.newaxis, :, :], (T, H, W))
        data_columns[feat_name] = feat_3d.ravel()

    # 4. Flatten Rainfall Features (3D -> 1D)
    for feat_name, feat_3d in rain_dict.items():
        data_columns[feat_name] = feat_3d.ravel()

    # 5. Flatten Hydraulic Features (3D -> 1D)
    for feat_name, feat_3d in hydr_dict.items():
        data_columns[feat_name] = feat_3d.ravel()

    # 6. Flatten Spatial-Temporal Features (3D -> 1D)
    for feat_name, feat_3d in spat_dict.items():
        data_columns[feat_name] = feat_3d.ravel()

    df = pd.DataFrame(data_columns)
    return df


def get_feature_column_names(df: pd.DataFrame) -> List[str]:
    """
    Return all feature columns in the dataframe, excluding metadata and target columns.
    """
    metadata_cols = {"event_id", "timestep_idx", "row_idx", "col_idx"}
    feature_cols = [
        col for col in df.columns
        if col not in metadata_cols and not col.startswith("target_")
    ]
    return feature_cols

