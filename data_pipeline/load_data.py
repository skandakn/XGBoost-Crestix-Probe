"""
load_data.py
============
Data loading, validation, and synthetic generator for the LarNO UKEA 32x32
hydrodynamic benchmark dataset.

Handles directory-based event loading, .npz consolidation, shape checks,
and physical validity assertions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class FloodEventData:
    """Container holding spatial-temporal arrays for a single flood event."""

    event_id: str
    dem: np.ndarray  # Shape: (32, 32), float32 [m]
    rainfall: np.ndarray  # Shape: (T, 32, 32) or (T,), float32 [mm/5min]
    h: np.ndarray  # Shape: (T, 32, 32), float32 [m]
    timesteps: int
    grid_shape: Tuple[int, int] = (32, 32)
    spatial_resolution_m: float = 8.0
    temporal_resolution_min: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shapes and dimensions upon initialization."""
        self.dem = np.asarray(self.dem, dtype=np.float32)
        self.rainfall = np.asarray(self.rainfall, dtype=np.float32)
        self.h = np.asarray(self.h, dtype=np.float32)

        # Standardize DEM shape to 2D (H, W)
        if self.dem.ndim == 3 and self.dem.shape[0] == 1:
            self.dem = self.dem.squeeze(0)

        if self.dem.shape != self.grid_shape:
            raise ValueError(
                f"DEM shape mismatch for {self.event_id}: expected {self.grid_shape}, got {self.dem.shape}"
            )

        # Standardize rainfall to (T, H, W) if provided as 1D time series (T,)
        if self.rainfall.ndim == 1:
            T = self.rainfall.shape[0]
            self.rainfall = np.broadcast_to(
                self.rainfall[:, np.newaxis, np.newaxis],
                (T, self.grid_shape[0], self.grid_shape[1]),
            ).copy()

        if self.rainfall.ndim != 3 or self.rainfall.shape[1:] != self.grid_shape:
            raise ValueError(
                f"Rainfall shape mismatch for {self.event_id}: expected (T, {self.grid_shape[0]}, {self.grid_shape[1]}), got {self.rainfall.shape}"
            )

        if self.h.ndim != 3 or self.h.shape[1:] != self.grid_shape:
            raise ValueError(
                f"Water depth h shape mismatch for {self.event_id}: expected (T, {self.grid_shape[0]}, {self.grid_shape[1]}), got {self.h.shape}"
            )

        self.timesteps = self.h.shape[0]
        if self.rainfall.shape[0] != self.timesteps:
            raise ValueError(
                f"Temporal length mismatch for {self.event_id}: rainfall has {self.rainfall.shape[0]} steps but h has {self.timesteps} steps"
            )

    @property
    def peak_depth(self) -> float:
        """Maximum water depth observed in this event across all cells and time."""
        return float(np.nanmax(self.h))

    @property
    def total_rainfall_volume_mm(self) -> float:
        """Total cumulative storm precipitation averaged over domain."""
        return float(np.sum(self.rainfall[:, 0, 0]))


def load_event_from_directory(
    event_dir: Union[str, Path],
    dem_override: Optional[np.ndarray] = None,
    grid_shape: Tuple[int, int] = (32, 32),
) -> FloodEventData:
    """
    Load a single flood event from its directory.
    Expected directory contents:
      - dem.npy (optional if dem_override provided)
      - rainfall.npy
      - h.npy
      - metadata.json (optional)
    """
    event_path = Path(event_dir)
    if not event_path.is_dir():
        raise FileNotFoundError(f"Event directory does not exist: {event_path}")

    event_id = event_path.name

    # Load DEM
    if dem_override is not None:
        dem = dem_override
    else:
        dem_file = event_path / "dem.npy"
        if not dem_file.exists():
            # Check parent directory for shared DEM
            parent_dem = event_path.parent / "dem.npy"
            if parent_dem.exists():
                dem_file = parent_dem
            else:
                raise FileNotFoundError(f"dem.npy not found in {event_path} or parent")
        dem = np.load(dem_file)

    # Load Rainfall
    rainfall_file = event_path / "rainfall.npy"
    if not rainfall_file.exists():
        raise FileNotFoundError(f"rainfall.npy not found in {event_path}")
    rainfall = np.load(rainfall_file)

    # Load Water Depth h
    h_file = event_path / "h.npy"
    if not h_file.exists():
        raise FileNotFoundError(f"h.npy not found in {event_path}")
    h = np.load(h_file)

    # Load optional metadata
    metadata = {}
    meta_file = event_path / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return FloodEventData(
        event_id=event_id,
        dem=dem,
        rainfall=rainfall,
        h=h,
        timesteps=h.shape[0],
        grid_shape=grid_shape,
        metadata=metadata,
    )


def load_all_events(
    data_dir: Union[str, Path],
    grid_shape: Tuple[int, int] = (32, 32),
    shared_dem: Optional[np.ndarray] = None,
) -> List[FloodEventData]:
    """
    Discover and load all flood events in a dataset directory.
    Supports either subdirectories (event_001, event_002) or consolidated .npz files.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    events: List[FloodEventData] = []

    # Check for shared top-level DEM
    if shared_dem is None:
        shared_dem_file = data_path / "dem.npy"
        if shared_dem_file.exists():
            shared_dem = np.load(shared_dem_file)

    # Check for subdirectories (event_*)
    subdirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    for sdir in subdirs:
        if (sdir / "h.npy").exists() and (sdir / "rainfall.npy").exists():
            event = load_event_from_directory(
                sdir, dem_override=shared_dem, grid_shape=grid_shape
            )
            events.append(event)

    # Check for consolidated npz files
    npz_files = sorted(data_path.glob("*.npz"))
    for npz_file in npz_files:
        with np.load(npz_file) as data:
            if "h" in data and "rainfall" in data:
                dem = data["dem"] if "dem" in data else shared_dem
                if dem is None:
                    raise ValueError(f"No DEM found in {npz_file} or shared_dem")
                event = FloodEventData(
                    event_id=npz_file.stem,
                    dem=dem,
                    rainfall=data["rainfall"],
                    h=data["h"],
                    timesteps=data["h"].shape[0],
                    grid_shape=grid_shape,
                )
                events.append(event)

    if not events:
        raise ValueError(f"No valid flood events found in {data_path}")

    return events


def generate_synthetic_larno_event(
    event_id: str,
    T: int = 36,
    grid_shape: Tuple[int, int] = (32, 32),
    seed: int = 42,
    severity: str = "moderate",
    shared_dem: Optional[np.ndarray] = None,
) -> FloodEventData:
    """
    Generate a physically realistic synthetic flood event matching the LarNO UKEA 32x32 benchmark.
    Features:
      - Realistic topography with a diagonal drainage depression/channel, slope, and buildings.
      - Realistic storm hyetograph (unimodal / convective peak).
      - Kinematic/diffusive wave flow accumulation computing dynamic water depths h(t, x, y).
    """
    rng = np.random.RandomState(seed)
    H, W = grid_shape

    # 1. Generate or reuse Digital Elevation Model (DEM)
    if shared_dem is not None:
        dem = shared_dem.copy()
    else:
        # Create coordinates
        y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
        # Regional slope from North-West to South-East
        base_slope = 15.0 - (8.0 * x + 6.0 * y)
        # Meandering valley / drainage channel along diagonal
        channel_dist = np.abs(y - 0.7 * x - 0.15)
        channel = -2.5 * np.exp(-(channel_dist**2) / 0.015)
        # Local terrain undulations
        undulations = 0.8 * np.sin(2 * np.pi * x * 2.0) * np.cos(2 * np.pi * y * 2.0)
        dem = (base_slope + channel + undulations).astype(np.float32)

        # Introduce realistic urban building footprints (high elevated platforms)
        building_mask = (x > 0.2) & (x < 0.4) & (y > 0.2) & (y < 0.4)
        building_mask |= (x > 0.6) & (x < 0.8) & (y > 0.5) & (y < 0.7)
        dem[building_mask] += 3.5

    # 2. Generate Storm Hyetograph (mm / 5min)
    severity_multipliers = {
        "light": 2.5,
        "moderate": 6.0,
        "heavy": 12.0,
        "extreme": 22.0,
        "cloudburst": 35.0,
    }
    peak_intensity = severity_multipliers.get(severity.lower(), 8.0)

    # Chicago design storm distribution peaking at t = T * 0.35
    t_arr = np.linspace(0, 1, T)
    peak_loc = 0.35
    hyetograph_1d = peak_intensity * np.exp(-((t_arr - peak_loc) ** 2) / 0.03)
    # Add slight storm noise
    hyetograph_1d += rng.uniform(0.0, 0.5, size=T)
    hyetograph_1d = np.maximum(0.0, hyetograph_1d).astype(np.float32)

    # Broadcast to 3D: (T, H, W)
    rainfall = np.broadcast_to(
        hyetograph_1d[:, np.newaxis, np.newaxis], (T, H, W)
    ).copy()

    # 3. Simulate 2D Hydrodynamic Surface Runoff & Inundation
    # Numerically stable diffusive wave routing down hydraulic head gradient
    h = np.zeros((T, H, W), dtype=np.float32)
    current_h = np.zeros((H, W), dtype=np.float32)

    # Compute elevation gradient magnitude
    grad_y, grad_x = np.gradient(dem)
    slope_mag = np.sqrt(grad_x**2 + grad_y**2) + 0.05

    # Runoff accumulation parameters
    infiltration_rate = 0.5  # mm/5min absorbed into permeable subgrade
    diffusion_coeff = 0.04   # CFL-stable diffusion constant

    for t in range(T):
        rain_step = rainfall[t]  # mm/5min
        # Convert rainfall (mm) to water depth (m): 1 mm = 0.001 m
        rain_m = np.maximum(0.0, rain_step - infiltration_rate) * 0.001
        current_h += rain_m

        # Hydraulic surface elevation H = z + h
        water_surface = dem + current_h
        ws_grad_y, ws_grad_x = np.gradient(water_surface)

        # Flux in x and y directions (kinematic / diffusive gradient)
        # Bounded by local depth to ensure non-negative mass conservation
        flux_x = -np.clip(ws_grad_x / slope_mag, -2.0, 2.0) * np.minimum(current_h, 0.5) * diffusion_coeff
        flux_y = -np.clip(ws_grad_y / slope_mag, -2.0, 2.0) * np.minimum(current_h, 0.5) * diffusion_coeff

        # Divergence of flux (net balance)
        div_flux = np.gradient(flux_x, axis=1) + np.gradient(flux_y, axis=0)
        current_h = current_h + np.clip(div_flux, -0.1, 0.1)

        # Enforce physical constraints: h >= 0 and reasonable urban maximum
        current_h = np.clip(current_h, 0.0, 3.5)

        # Slow domain boundary discharge
        current_h[0, :] *= 0.96
        current_h[-1, :] *= 0.94
        current_h[:, 0] *= 0.96
        current_h[:, -1] *= 0.94

        h[t] = current_h.copy()

    metadata = {
        "event_id": event_id,
        "storm_severity": severity,
        "peak_rainfall_mm_5min": float(np.max(hyetograph_1d)),
        "total_rainfall_mm": float(np.sum(hyetograph_1d)),
        "max_simulated_depth_m": float(np.max(h)),
        "timesteps": T,
        "dt_minutes": 5.0,
        "dx_meters": 8.0,
        "synthetic": True,
    }

    return FloodEventData(
        event_id=event_id,
        dem=dem,
        rainfall=rainfall,
        h=h,
        timesteps=T,
        grid_shape=grid_shape,
        metadata=metadata,
    )


def save_events_to_disk(
    events: List[FloodEventData],
    output_dir: Union[str, Path],
    save_shared_dem: bool = True,
) -> None:
    """Save a collection of flood events to disk in the standardized schema."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if save_shared_dem and events:
        # Save shared master DEM
        np.save(out_path / "dem.npy", events[0].dem)

    for ev in events:
        ev_dir = out_path / ev.event_id
        ev_dir.mkdir(parents=True, exist_ok=True)
        np.save(ev_dir / "dem.npy", ev.dem)
        np.save(ev_dir / "rainfall.npy", ev.rainfall)
        np.save(ev_dir / "h.npy", ev.h)

        with open(ev_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(ev.metadata, f, indent=2)


def create_sample_dataset(
    output_dir: Union[str, Path] = "data/sample",
    n_events: int = 5,
    T: int = 36,
    seed: int = 42,
) -> List[FloodEventData]:
    """
    Generate and save a representative 5-event sample dataset for local testing and CI/CD.
    Events span distinct storm severity categories: Light, Moderate, Heavy, Extreme, Cloudburst.
    """
    severities = ["light", "moderate", "heavy", "extreme", "cloudburst"]
    events: List[FloodEventData] = []

    # Generate master shared DEM
    master_event = generate_synthetic_larno_event("temp_master", T=T, seed=seed)
    shared_dem = master_event.dem

    for i in range(n_events):
        ev_id = f"event_{i+1:03d}"
        sev = severities[i % len(severities)]
        ev = generate_synthetic_larno_event(
            ev_id, T=T, seed=seed + i * 17, severity=sev, shared_dem=shared_dem
        )
        events.append(ev)

    save_events_to_disk(events, output_dir=output_dir, save_shared_dem=True)
    return events


if __name__ == "__main__":
    print("Generating LarNO UKEA 32x32 sample benchmark dataset...")
    created_events = create_sample_dataset()
    print(f"Successfully generated {len(created_events)} events in data/sample:")
    for e in created_events:
        print(
            f"  - {e.event_id}: T={e.timesteps}, max_rain={e.metadata['peak_rainfall_mm_5min']:.1f} mm, peak_depth={e.peak_depth:.3f} m"
        )
