"""
target_generation.py
====================
Target variable generation and labeling for the LarNO UKEA 32x32 benchmark.

Defines continuous multi-horizon future water depth targets, peak flood depth,
and discretized municipal flood hazard categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_pipeline.load_data import FloodEventData


@dataclass
class TargetMetadata:
    """Documented specification for a single prediction target."""

    name: str
    target_type: str  # 'regression' or 'classification'
    units: str  # 'meters', 'dimensionless', 'hazard_class'
    forecast_horizon_min: Optional[float]
    lead_steps: Optional[int]
    description: str
    min_physical_bound: float = 0.0
    max_physical_bound: Optional[float] = None
    classification_thresholds_m: Optional[List[float]] = None


# Official target registry for Crestix Probe XGBoost modeling
TARGET_DEFINITIONS: Dict[str, TargetMetadata] = {
    "target_depth_lead_5m": TargetMetadata(
        name="target_depth_lead_5m",
        target_type="regression",
        units="meters [m]",
        forecast_horizon_min=5.0,
        lead_steps=1,
        description="Continuous water depth at grid cell (i, j) 5 minutes in the future (t + 1).",
    ),
    "target_depth_lead_15m": TargetMetadata(
        name="target_depth_lead_15m",
        target_type="regression",
        units="meters [m]",
        forecast_horizon_min=15.0,
        lead_steps=3,
        description="Continuous water depth at grid cell (i, j) 15 minutes in the future (t + 3). Rapid response horizon.",
    ),
    "target_depth_lead_30m": TargetMetadata(
        name="target_depth_lead_30m",
        target_type="regression",
        units="meters [m]",
        forecast_horizon_min=30.0,
        lead_steps=6,
        description="PRIMARY TARGET: Continuous water depth at grid cell (i, j) 30 minutes in the future (t + 6). Standard municipal emergency planning lead time.",
    ),
    "target_depth_lead_60m": TargetMetadata(
        name="target_depth_lead_60m",
        target_type="regression",
        units="meters [m]",
        forecast_horizon_min=60.0,
        lead_steps=12,
        description="Continuous water depth at grid cell (i, j) 60 minutes in the future (t + 12). Strategic evacuation lead time.",
    ),
    "target_peak_remaining_depth": TargetMetadata(
        name="target_peak_remaining_depth",
        target_type="regression",
        units="meters [m]",
        forecast_horizon_min=None,
        lead_steps=None,
        description="Maximum water depth attained at cell (i, j) from time t through the remainder of the storm event.",
    ),
    "target_event_peak_depth": TargetMetadata(
        name="target_event_peak_depth",
        target_type="regression",
        units="meters [m]",
        forecast_horizon_min=None,
        lead_steps=None,
        description="Overall peak water depth reached at cell (i, j) across the entire flood event duration.",
    ),
    "target_hazard_class_lead_30m": TargetMetadata(
        name="target_hazard_class_lead_30m",
        target_type="classification",
        units="hazard_class [0-3]",
        forecast_horizon_min=30.0,
        lead_steps=6,
        description="Discretized flood hazard category at t + 30m: 0 (Dry <3cm), 1 (Shallow 3-15cm), 2 (Moderate 15-30cm), 3 (Critical >=30cm).",
        classification_thresholds_m=[0.03, 0.15, 0.30],
    ),
    "target_is_flooded_lead_30m": TargetMetadata(
        name="target_is_flooded_lead_30m",
        target_type="classification",
        units="binary [0 or 1]",
        forecast_horizon_min=30.0,
        lead_steps=6,
        description="Binary indicator whether cell (i, j) is flooded (depth >= 0.03 m / 3 cm) at t + 30m.",
    ),
}


def generate_event_targets(
    h: np.ndarray,
    dt_min: float = 5.0,
    lead_steps: Tuple[int, ...] = (1, 3, 6, 12),
    hazard_thresholds_m: Tuple[float, ...] = (0.03, 0.15, 0.30),
) -> Dict[str, np.ndarray]:
    """
    Generate target arrays for all timesteps and grid cells of a flood event.

    Parameters
    ----------
    h : np.ndarray
        Water depth array of shape (T, H, W), float32 in meters.
    dt_min : float
        Time step in minutes (5.0 min).
    lead_steps : Tuple[int, ...]
        Future forecast leads in steps (1=5m, 3=15m, 6=30m, 12=60m).
    hazard_thresholds_m : Tuple[float, ...]
        Boundaries for flood hazard classification (in meters).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of target arrays of shape (T, H, W).
        Out-of-bounds future steps are marked with np.nan.
    """
    h_arr = np.asarray(h, dtype=np.float32)
    T, H, W = h_arr.shape

    targets: Dict[str, np.ndarray] = {}

    # 1. Multi-horizon lead-time water depth targets
    for lead in lead_steps:
        lead_min = int(lead * dt_min)
        target_arr = np.full_like(h_arr, fill_value=np.nan, dtype=np.float32)
        if lead < T:
            # target at time t is actual depth observed at t + lead
            target_arr[:-lead] = h_arr[lead:]
        targets[f"target_depth_lead_{lead_min}m"] = target_arr

    # 2. Peak remaining water depth from current time t to end of event: max_{tau >= t} h(tau)
    peak_remaining = np.zeros_like(h_arr)
    # Reverse cumulative maximum
    running_max = np.zeros((H, W), dtype=np.float32)
    for t in reversed(range(T)):
        running_max = np.maximum(running_max, h_arr[t])
        peak_remaining[t] = running_max.copy()
    targets["target_peak_remaining_depth"] = peak_remaining

    # 3. Overall event peak water depth: max_{0 <= tau < T} h(tau)
    event_peak = np.max(h_arr, axis=0)  # Shape (H, W)
    event_peak_3d = np.broadcast_to(event_peak[np.newaxis, :, :], (T, H, W)).copy()
    targets["target_event_peak_depth"] = event_peak_3d.astype(np.float32)

    # 4. Discretized flood hazard category at primary lead horizon (+30 min)
    lead_30m_depth = targets.get("target_depth_lead_30m")
    if lead_30m_depth is not None:
        th1, th2, th3 = hazard_thresholds_m
        hazard_class = np.full_like(lead_30m_depth, fill_value=np.nan, dtype=np.float32)
        valid_mask = ~np.isnan(lead_30m_depth)

        # 0: Dry (< 3cm)
        class_0 = valid_mask & (lead_30m_depth < th1)
        # 1: Shallow (3cm - 15cm)
        class_1 = valid_mask & (lead_30m_depth >= th1) & (lead_30m_depth < th2)
        # 2: Moderate (15cm - 30cm)
        class_2 = valid_mask & (lead_30m_depth >= th2) & (lead_30m_depth < th3)
        # 3: Critical (>= 30cm)
        class_3 = valid_mask & (lead_30m_depth >= th3)

        hazard_class[class_0] = 0.0
        hazard_class[class_1] = 1.0
        hazard_class[class_2] = 2.0
        hazard_class[class_3] = 3.0
        targets["target_hazard_class_lead_30m"] = hazard_class

        # Binary flood indicator (depth >= 3cm)
        is_flooded = np.full_like(lead_30m_depth, fill_value=np.nan, dtype=np.float32)
        is_flooded[valid_mask] = (lead_30m_depth[valid_mask] >= th1).astype(np.float32)
        targets["target_is_flooded_lead_30m"] = is_flooded

    return targets


def attach_targets_to_dataframe(
    df: pd.DataFrame,
    event: FloodEventData,
    drop_nan_targets: bool = True,
    primary_target_col: str = "target_depth_lead_30m",
) -> pd.DataFrame:
    """
    Generate target columns for an event and attach them to its tabular DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Tabular feature dataframe produced by build_event_tabular_dataframe.
    event : FloodEventData
        The flood event object.
    drop_nan_targets : bool
        If True, drops trailing rows where the primary forecast horizon exceeds
        the remaining length of the event (e.g. final 6 timesteps for 30m lead).
    primary_target_col : str
        Target column used to determine valid target rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns and target columns attached.
    """
    targets_dict = generate_event_targets(
        event.h, dt_min=event.temporal_resolution_min
    )

    df_copy = df.copy()
    for target_name, target_3d in targets_dict.items():
        df_copy[target_name] = target_3d.ravel()

    if drop_nan_targets and primary_target_col in df_copy.columns:
        # Filter out rows that lack valid future observations for the primary target
        valid_rows = ~df_copy[primary_target_col].isna()
        df_copy = df_copy[valid_rows].reset_index(drop=True)

    return df_copy


def get_target_metadata(target_name: str) -> TargetMetadata:
    """Retrieve documented metadata for a given target name."""
    if target_name not in TARGET_DEFINITIONS:
        raise KeyError(
            f"Target '{target_name}' not recognized. Available targets: {list(TARGET_DEFINITIONS.keys())}"
        )
    return TARGET_DEFINITIONS[target_name]

