# Crestix Probe — Data Schema & Feature Engineering Specification
**Target Catchment:** LarNO UKEA Urban Flood Benchmark ($32 \times 32$ Grid)  
**System:** XGBoost Tabular Machine Learning Pipeline  
**Version:** 1.0.0-LTS  

---

## 1. Domain & Spatio-Temporal Parameters

The Crestix Probe XGBoost pipeline operates on the **LarNO UKEA (UK Environment Agency)** urban flood benchmark dataset. The benchmark models 2D hydrodynamic flood propagation (governed by the 2D Saint-Venant shallow water equations) over complex urban topography under varying dynamic storm hyetographs.

| Parameter | Specification | Physical Value |
| :--- | :--- | :--- |
| **Grid Dimensions ($H \times W$)** | $32 \times 32$ grid cells | 1,024 spatial cells |
| **Spatial Resolution ($\Delta x, \Delta y$)** | 8.0 meters per cell | Domain size: $256\text{ m} \times 256\text{ m}$ ($65,536\text{ m}^2$) |
| **Temporal Resolution ($\Delta t$)** | 5.0 minutes per frame (300 seconds) | Standard hydrodynamic output cadence |
| **Event Duration ($T$)** | 12 to 72 timesteps | 1.0 to 6.0 hours total storm duration |
| **Coordinate System** | Normalized Cartesian $[0, 1] \times [0, 1]$ | Grid indices $y \in [0, 31]$, $x \in [0, 31]$ |

---

## 2. Raw Input Data Schema (Prompt 1 Contract)

Each flood event represents an independent numerical simulation run with distinct hyetograph forcing and resultant hydrodynamic wave inundation.

```text
data/raw/
├── dem.npy                     # Shared master elevation model: Shape (32, 32), float32 [m]
├── event_001/
│   ├── dem.npy                 # Event DEM (or inherits shared master): Shape (32, 32), float32 [m]
│   ├── rainfall.npy            # Dynamic hyetograph: Shape (T, 32, 32) or (T,), float32 [mm/5min]
│   ├── h.npy                   # Hydrodynamic water depth ground truth: Shape (T, 32, 32), float32 [m]
│   └── metadata.json           # Storm severity, peak intensity, total volume, duration metadata
├── event_002/
...
```

### Raw Array Specifications
1. **`dem.npy`** (Shape: `(32, 32)`, dtype: `float32`, Unit: **meters [m]**):
   - Digital Elevation Model representing bare-earth elevation above ordnance datum.
   - Elevated structures / buildings are indicated by high platform elevations or `NaN` / sentinel values.
2. **`rainfall.npy`** (Shape: `(T, 32, 32)` or `(T,)`, dtype: `float32`, Unit: **$\text{mm} / 5\text{min}$**):
   - Dynamic precipitation rate for each 5-minute timestep. Spatially uniform rainfalls of shape `(T,)` are automatically broadcast to `(T, 32, 32)` by `load_data.py`.
3. **`h.npy`** (Shape: `(T, 32, 32)`, dtype: `float32`, Unit: **meters [m]**):
   - Simulated water depth above ground surface at cell $(i, j)$ and time $t$. Enforces physical constraint $h \ge 0.0$.
4. **`metadata.json`**:
   - Stores `event_id`, `storm_severity` (`light`, `moderate`, `heavy`, `extreme`, `cloudburst`), peak intensity, and duration.

---

## 3. Tabular Representation & Feature Definitions

Each row in the tabular dataset represents a single grid cell $(i, j)$ at time step $t$ in flood event $e$.

$$\text{Index: } (\text{event\_id}, t, i, j)$$

### Feature Dictionary & Physical Units

| Feature Name | Category | Units | Mathematical Formulation / Description | Physical Rationale in Flood Dynamics |
| :--- | :--- | :--- | :--- | :--- |
| `elevation` | Topographic | $\text{m}$ | $z(i, j)$ | Water flows from higher to lower hydraulic potential. |
| `slope_magnitude` | Topographic | $\text{m/m}$ | $\|\nabla z\| = \sqrt{(\partial z/\partial x)^2 + (\partial z/\partial y)^2}$ | Controls gravitational overland flow velocity via Manning's formula. |
| `slope_x` | Topographic | $\text{m/m}$ | $\partial z / \partial x$ (central difference) | Directional bed slope momentum component in X. |
| `slope_y` | Topographic | $\text{m/m}$ | $\partial z / \partial y$ (central difference) | Directional bed slope momentum component in Y. |
| `aspect_sin` | Topographic | dimensionless | $\sin(\theta)$, where $\theta = \operatorname{atan2}(-\partial z/\partial y, -\partial z/\partial x)$ | Direction of steepest surface descent (sinusoidal). |
| `aspect_cos` | Topographic | dimensionless | $\cos(\theta)$ | Direction of steepest surface descent (cosinusoidal). |
| `laplacian_elevation` | Topographic | $\text{m}^{-1}$ | $\nabla^2 z = \partial^2 z/\partial x^2 + \partial^2 z/\partial y^2$ | Positive values indicate terrain hollows/depressions; negative indicates ridges. |
| `profile_curvature` | Topographic | $\text{m}^{-1}$ | Rate of slope change along flow path | Distinguishes accelerating flow on convex slopes vs decelerating/ponding flow on concave slopes. |
| `plan_curvature` | Topographic | $\text{m}^{-1}$ | Curvature perpendicular to flow path | Indicates lateral flow convergence into valleys vs divergence on ridges. |
| `local_elevation_std_3x3` | Topographic | $\text{m}$ | $\sigma_{3\times 3}(z)$ | Micro-scale urban terrain roughness / obstacle density. |
| `local_elevation_std_5x5` | Topographic | $\text{m}$ | $\sigma_{5\times 5}(z)$ | Meso-scale neighborhood terrain variation. |
| `tpi_3x3` | Topographic | $\text{m}$ | $z - \bar{z}_{3\times 3}$ | Topographic Position Index: relative depression depth below immediate neighbors. |
| `tpi_5x5` | Topographic | $\text{m}$ | $z - \bar{z}_{5\times 5}$ | Meso-scale Topographic Position Index. |
| `neighbour_elevation_diff_mean` | Topographic | $\text{m}$ | $\frac{1}{8}\sum_{k=1}^8 (z_k - z_0)$ | Mean elevation difference relative to 8 surrounding cells. |
| `neighbour_elevation_diff_min` | Topographic | $\text{m}$ | $\min_{k=1..8} (z_k - z_0)$ | Minimum difference; negative values reveal primary gravity escape route. |
| `neighbour_elevation_diff_max` | Topographic | $\text{m}$ | $\max_{k=1..8} (z_k - z_0)$ | Maximum surrounding barrier height. |
| `flow_accumulation_proxy` | Topographic | $\text{m}^2$ | $\max(0, -\nabla^2 z) + \max(0, -\text{TPI}_{5\times 5})$ | Proxy upstream catchment area funneling runoff into current cell. |
| `is_building_mask` | Topographic | binary | $1.0$ if building / masked, $0.0$ otherwise | Identifies impermeable building boundaries and flow diversions. |
| `rainfall_intensity_current` | Meteorological | $\text{mm}/5\text{min}$ | $R(t)$ | Instantaneous storm influx into the overland system. |
| `rainfall_cumulative` | Meteorological | $\text{mm}$ | $\sum_{\tau=0}^t R(\tau)$ | Total cumulative storm water volume introduced into catchment. |
| `rainfall_window_15m` | Meteorological | $\text{mm}$ | $\sum_{\tau=t-2}^t R(\tau)$ (3 timesteps) | Short-term flash rainfall pulse volume (15 min). |
| `rainfall_window_30m` | Meteorological | $\text{mm}$ | $\sum_{\tau=t-5}^t R(\tau)$ (6 timesteps) | Medium-term storm accumulation (30 min). |
| `rainfall_window_60m` | Meteorological | $\text{mm}$ | $\sum_{\tau=t-11}^t R(\tau)$ (12 timesteps) | Sustained hourly precipitation accumulation (60 min). |
| `rainfall_mean_15m` | Meteorological | $\text{mm}/5\text{min}$ | $\frac{1}{3}\text{rainfall\_window\_15m}$ | Average intensity over past 15 minutes. |
| `rainfall_mean_30m` | Meteorological | $\text{mm}/5\text{min}$ | $\frac{1}{6}\text{rainfall\_window\_30m}$ | Average intensity over past 30 minutes. |
| `rainfall_mean_60m` | Meteorological | $\text{mm}/5\text{min}$ | $\frac{1}{12}\text{rainfall\_window\_60m}$ | Average intensity over past 60 minutes. |
| `rainfall_max_so_far` | Meteorological | $\text{mm}/5\text{min}$ | $\max_{\tau \le t} R(\tau)$ | Peak storm intensity experienced up to time $t$. |
| `rainfall_rate_of_change` | Meteorological | $\text{mm}/5\text{min}^2$ | $(R_t - R_{t-1}) / \Delta t$ | Precipitation acceleration: detects cloudburst surges. |
| `current_water_depth` | Hydraulic State | $\text{m}$ | $h(t, i, j)$ | Initial condition for forward flood wave propagation. |
| `water_depth_lag_5m` | Hydraulic State | $\text{m}$ | $h(t-1, i, j)$ | Antecedent water depth 5 minutes ago (padded with 0.0 at $t=0$). |
| `water_depth_lag_10m` | Hydraulic State | $\text{m}$ | $h(t-2, i, j)$ | Antecedent water depth 10 minutes ago. |
| `water_depth_lag_15m` | Hydraulic State | $\text{m}$ | $h(t-3, i, j)$ | Antecedent water depth 15 minutes ago. |
| `water_depth_rate_of_rise` | Hydraulic State | $\text{m/min}$ | $(h_t - h_{t-1}) / \Delta t$ | Flood surge velocity / filling rate. |
| `water_depth_acceleration` | Hydraulic State | $\text{m/min}^2$ | $(h_t - 2h_{t-1} + h_{t-2}) / \Delta t^2$ | Second temporal derivative of flood wave front. |
| `neighbour_water_depth_mean_3x3` | Hydraulic State | $\text{m}$ | $\bar{h}_{3\times 3}$ | Local ambient flood stage in immediate surrounding cells. |
| `neighbour_water_depth_max_3x3` | Hydraulic State | $\text{m}$ | $\max_{3\times 3}(h)$ | Detects approaching flood wavefront in adjacent cells before arrival. |
| `hydraulic_head` | Hydraulic State | $\text{m}$ | $H(t, i, j) = z(i, j) + h(t, i, j)$ | Total piezometric water surface elevation. |
| `hydraulic_head_grad_mag` | Hydraulic State | $\text{m/m}$ | $\|\nabla(z + h)\|$ | Driving hydrodynamic momentum potential gradient. |
| `norm_coord_x` | Spatio-Temporal | dimensionless | $x / (W - 1) \in [0, 1]$ | Absolute normalized easting coordinate. |
| `norm_coord_y` | Spatio-Temporal | dimensionless | $y / (H - 1) \in [0, 1]$ | Absolute normalized northing coordinate. |
| `dist_to_center` | Spatio-Temporal | dimensionless | Euclidean distance to $(15.5, 15.5)$ in $[0, 1]$ | Spatial domain centrality descriptor. |
| `time_since_onset_min` | Spatio-Temporal | $\text{minutes}$ | $t \times 5.0$ | Elapsed time since storm rainfall began. |
| `time_fraction` | Spatio-Temporal | dimensionless | $t / (T - 1) \in [0, 1]$ | Temporal progress through the flood event. |

---

## 4. Prediction Targets & Forecast Horizons (Prompt 2 & 3 Contract)

Target generation is strictly separated from features to prevent circular leakage.

| Target Column | Type | Horizon | Physical Units | Mathematical Definition |
| :--- | :--- | :--- | :--- | :--- |
| **`target_depth_lead_30m`** | **Regression (PRIMARY)** | **+30 min (6 steps)** | **meters [m]** | $h(t + 6, i, j)$ — Standard municipal emergency action horizon. |
| `target_depth_lead_5m` | Regression | +5 min (1 step) | meters [m] | $h(t + 1, i, j)$ — Immediate frame continuity target. |
| `target_depth_lead_15m` | Regression | +15 min (3 steps) | meters [m] | $h(t + 3, i, j)$ — Rapid urban flash flood onset horizon. |
| `target_depth_lead_60m` | Regression | +60 min (12 steps) | meters [m] | $h(t + 12, i, j)$ — Strategic catchment-scale evacuation horizon. |
| `target_peak_remaining_depth` | Regression | End-of-event | meters [m] | $\max_{\tau \ge t} h(\tau, i, j)$ — Peak water level yet to be experienced. |
| `target_event_peak_depth` | Regression | Overall event | meters [m] | $\max_{0 \le \tau < T} h(\tau, i, j)$ — Maximum inundation reached during event. |
| `target_hazard_class_lead_30m` | Classification | +30 min (6 steps) | hazard class [0-3] | Discretized UKEA / FEMA depth categories (see below). |
| `target_is_flooded_lead_30m` | Classification | +30 min (6 steps) | binary [0 or 1] | $1.0$ if $h(t+6) \ge 0.03\text{ m}$ (3 cm), else $0.0$. |

### Hazard Classification Thresholds
- **Class 0 (`dry_negligible`)**: $h < 0.03\text{ m}$ ($< 3\text{ cm}$) — Negligible surface ponding; normal vehicular and pedestrian passage.
- **Class 1 (`shallow_caution`)**: $0.03\text{ m} \le h < 0.15\text{ m}$ ($3 - 15\text{ cm}$) — Curb-height ponding; caution for small passenger vehicles.
- **Class 2 (`moderate_hazard`)**: $0.15\text{ m} \le h < 0.30\text{ m}$ ($15 - 30\text{ cm}$) — Roadway overtopping; hazardous for standard passenger cars.
- **Class 3 (`critical_inundation`)**: $h \ge 0.30\text{ m}$ ($\ge 30\text{ cm}$) — Severe flooding; vehicles buoyant, basement flooding, immediate evacuation required.

---

## 5. Event-Aware Partitioning & Data Leakage Prevention

### The Golden Rule: Event Disjointness
Under no circumstances may rows from the same flood event be randomly partitioned between training and test sets.
Spatial autocorrelation across cells and temporal autocorrelation across consecutive frames within the same storm would lead to severe data leakage and artificially inflated accuracy.

$$\text{Events}_{\text{train}} \cap \text{Events}_{\text{val}} = \emptyset, \quad \text{Events}_{\text{train}} \cap \text{Events}_{\text{test}} = \emptyset, \quad \text{Events}_{\text{val}} \cap \text{Events}_{\text{test}} = \emptyset$$

### Train-Only Preprocessing
- **Median Imputation**: Imputation of missing DEM cells or nodata values is computed strictly using the training events' median, then applied to val/test.
- **Feature Scaling**: Tree-based models (XGBoost) do not require feature scaling, but where `TabularPreprocessor` is used with `RobustScaler` or `StandardScaler`, the scaler is fitted strictly on `X_train`.

---

## 6. Large Data Storage & Google Drive Synchronization

Per project rules:
- **GitHub Repository**: Stores source code, tests, documentation (`DATA_FORMAT.md`, `README.md`), configuration files (`baseline.yaml`), and lightweight synthetic test datasets (`data/sample/`).
- **Google Drive / Shared Storage**: Stores large processed tabular `.parquet` or `.h5` files, full hydrodynamic benchmark arrays, and trained `.json` / `.ubj` XGBoost model artifacts.

