"""
Build PyTorch Geometric graph with MONTHLY migration samples for CV.

Key improvements v3: Integration of along-channel features
- Sample size: 59 monthly samples
- Static features: 14-dim (geometry + vegetation + soil + land use)
- Hydrological features: 7-dim (Q, Q_prev, dQ, Q_cumsum3, Qs, season_sin, season_cos)
- Along-channel features (new):
  - Along-channel width W(s): width at each node per month [N_samples, 1000]
  - Along-channel depth D(s): depth at each node per month [N_samples, 1000] (optional)

Data structure:
- Static features: [1000, 14] (geometry + vegetation + soil + land use)
- Hydrological features: [N_samples, 7] (enhanced hydrological features)
- Along-channel width: [N_samples, 1000] (per node per month)
- Along-channel depth: [N_samples, 1000] (per node per month, optional)
- Migration: [N_samples, 1000] (per node per month)

CV strategy:
- Fold 1: Train 2016-2018, Validate 2019
- Fold 2: Train 2016-2019, Validate 2020

Usage:
    python build_graph_cv.py --fold 1
    python build_graph_cv.py --fold 2
    python build_graph_cv.py --fold 1 --use_depth  # Include along-channel depth
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
import argparse
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parents[4]
CODES_DIR = REPO_ROOT
PROCESSED_DIR = REPO_ROOT / 'data' / 'processed'
GRAPH_DIR = REPO_ROOT / 'data' / 'graph'
_CENTERLINE_DIR_CANDIDATES = [
    REPO_ROOT / 'data' / 'raw' / 'monthly_centerlines',
]


def _pick_centerline_dir():
    for d in _CENTERLINE_DIR_CANDIDATES:
        if d.exists() and any(d.glob('*_centerline_geo_1000pts.npy')):
            return d
    return _CENTERLINE_DIR_CANDIDATES[0]


CENTERLINE_DIR = _pick_centerline_dir()

# Along-channel feature data paths
_WIDTH_ALONG_CSV_CANDIDATES = [
    REPO_ROOT / 'data' / 'raw' / 'monthly_width_along_channel.csv',
]
_DEPTH_ALONG_CSV_CANDIDATES = [
    REPO_ROOT / 'data' / 'raw' / 'monthly_depth_along_channel.csv',
]


def _pick_depth_along_csv():
    for fp in _DEPTH_ALONG_CSV_CANDIDATES:
        if fp.exists():
            return fp
    return _DEPTH_ALONG_CSV_CANDIDATES[0]


DEPTH_ALONG_CSV = _pick_depth_along_csv()

# CV Fold definitions
# exclude_train: months excluded from training set (data quality issues)
# exclude_val: months excluded from validation set (abnormal initial position, extreme events, or low migration)
CV_FOLDS = {
    1: {
        'train_years': [2016, 2017, 2018], 
        'val_years': [2019],
        'exclude_train': ['201912'],  # Abnormal migration
        'exclude_val': ['201905', '201906', '201907', '201909']  # Low migration months, low SNR
    },
    2: {
        'train_years': [2016, 2017, 2018, 2019], 
        'val_years': [2020],
        'exclude_train': ['201912'],  # Abnormal migration
        'exclude_val': ['202001', '202005', '202006', '202008']  # 202001: abnormal start; 202005-202008: extreme flood period
    },
    3: {
        'train_years': [2016, 2017, 2018, 2019, 2020],
        'val_years': [2021],
        'exclude_train': ['201912'],
        'exclude_val': []
    },
}


def compute_hydro_stats(hydro_dict, years=None):
    if not hydro_dict:
        return {
            'Q_mean': 0, 'Q_std': 1,
            'Q_cumsum3_mean': 0, 'Q_cumsum3_std': 1,
            'Qs_mean': 0, 'Qs_std': 1
        }

    year_set = {str(y) for y in years} if years else None
    keys = [
        k for k in hydro_dict.keys()
        if (year_set is None or k[:4] in year_set)
    ]

    if not keys:
        keys = list(hydro_dict.keys())

    Q_all = [hydro_dict[k].get('Q', np.nan) for k in keys]
    Q_cumsum3_all = [hydro_dict[k].get('Q_cumsum3', np.nan) for k in keys]
    Qs_all = [hydro_dict[k].get('Qs', np.nan) for k in keys]

    Q_all = np.array(Q_all, dtype=float)
    Q_cumsum3_all = np.array(Q_cumsum3_all, dtype=float)
    Qs_all = np.array(Qs_all, dtype=float)

    Q_mean = float(np.nanmean(Q_all)) if np.isfinite(np.nanmean(Q_all)) else 0.0
    Q_std = float(np.nanstd(Q_all)) if np.isfinite(np.nanstd(Q_all)) else 1.0
    Q_std = Q_std if Q_std > 1e-6 else 1.0

    Q_cumsum3_mean = float(np.nanmean(Q_cumsum3_all)) if np.isfinite(np.nanmean(Q_cumsum3_all)) else 0.0
    Q_cumsum3_std = float(np.nanstd(Q_cumsum3_all)) if np.isfinite(np.nanstd(Q_cumsum3_all)) else 1.0
    Q_cumsum3_std = Q_cumsum3_std if Q_cumsum3_std > 1e-6 else 1.0

    Qs_mean = float(np.nanmean(Qs_all)) if np.isfinite(np.nanmean(Qs_all)) else 0.0
    Qs_std = float(np.nanstd(Qs_all)) if np.isfinite(np.nanstd(Qs_all)) else 1.0
    Qs_std = Qs_std if Qs_std > 1e-6 else 1.0

    return {
        'Q_mean': Q_mean,
        'Q_std': Q_std,
        'Q_cumsum3_mean': Q_cumsum3_mean,
        'Q_cumsum3_std': Q_cumsum3_std,
        'Qs_mean': Qs_mean,
        'Qs_std': Qs_std
    }


def build_edge_index(num_nodes, skip_connections=[5, 10, 20]):
    """Build edge index with adjacent and skip connections."""
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    for k in skip_connections:
        for i in range(num_nodes - k):
            edges.append([i, i + k])
            edges.append([i + k, i])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def load_centerline(year, month):
    """Load centerline coordinates for given year/month."""
    filename = f"{year}{month:02d}_centerline_geo_1000pts.npy"
    filepath = CENTERLINE_DIR / filename
    if not filepath.exists():
        return None
    return np.load(filepath)


def get_next_month(year, month):
    """Get next month's year and month."""
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _align_centerline_orientation(ref_cl, cand_cl):
    if ref_cl is None or cand_cl is None:
        return cand_cl
    if ref_cl.shape != cand_cl.shape:
        return cand_cl
    d1 = np.nanmean(np.linalg.norm(ref_cl - cand_cl, axis=1))
    d2 = np.nanmean(np.linalg.norm(ref_cl - cand_cl[::-1], axis=1))
    return cand_cl if d1 <= d2 else cand_cl[::-1]


def compute_monthly_migration(year, month, mode='corresp'):
    """
    Compute migration from current month to next month.
    Uses cdist to calculate minimum distance from each point to next month's centerline.
    """
    cl_current = load_centerline(year, month)
    next_year, next_month = get_next_month(year, month)
    cl_next = load_centerline(next_year, next_month)
    
    if cl_current is None:
        raise FileNotFoundError(f"Centerline not found: {year}-{month:02d}")
    if cl_next is None:
        raise FileNotFoundError(f"Centerline not found: {next_year}-{next_month:02d}")

    cl_next_aligned = _align_centerline_orientation(cl_current, cl_next)
    migration_corresp = np.linalg.norm(cl_next_aligned - cl_current, axis=1)

    distances = cdist(cl_current, cl_next)
    migration_min = np.min(distances, axis=1)
    if mode == 'min':
        return migration_min

    m_corresp = float(np.nanmean(migration_corresp))
    m_min = float(np.nanmean(migration_min))
    if np.isfinite(m_corresp) and np.isfinite(m_min) and m_min > 1e-6 and (m_corresp / m_min) > 5.0:
        return migration_min
    return migration_corresp


def load_static_features():
    """Load and merge all static feature files."""
    geo_file = PROCESSED_DIR / 'geometric_features.csv'
    if not geo_file.exists():
        raise FileNotFoundError(f"Run extract_geometric_features.py first: {geo_file}")
    
    df = pd.read_csv(geo_file)
    
    # Vegetation features
    veg_file = PROCESSED_DIR / 'vegetation_features.csv'
    if veg_file.exists():
        veg_df = pd.read_csv(veg_file)
        df = df.merge(veg_df, on='node_id', how='left')
    else:
        df['lai_mean'] = 2.0
        df['lai_std'] = 0.5
        df['lai_max'] = 4.0
    
    # Soil features
    soil_file = PROCESSED_DIR / 'soil_features.csv'
    if soil_file.exists():
        soil_df = pd.read_csv(soil_file)
        df = df.merge(soil_df, on='node_id', how='left')
    else:
        df['clay_pct'] = 30.0
        df['sand_pct'] = 40.0
        df['organic_matter'] = 2.0
        df['bulk_density'] = 1.4
    
    # Land use features
    landuse_file = PROCESSED_DIR / 'landuse_features.csv'
    if landuse_file.exists():
        lu_df = pd.read_csv(landuse_file)
        df = df.merge(lu_df, on='node_id', how='left')
    else:
        df['landuse_agriculture'] = 0.5
        df['landuse_forest'] = 0.3
        df['landuse_water'] = 0.2
    
    # Engineering features
    eng_file = PROCESSED_DIR / 'engineering_features.csv'
    if eng_file.exists():
        eng_df = pd.read_csv(eng_file)
        df = df.merge(eng_df, on='node_id', how='left')
    else:
        df['is_protected'] = 0
        
    return df


def load_monthly_hydro():
    """Load Shashi station monthly hydrological data and compute enhanced features."""
    hydro_file = REPO_ROOT / "data" / "raw" / "shashi_monthly_q_qs.csv"

    if not hydro_file.exists():
        print(f"  Warning: Hydrological data file not found: {hydro_file}")
        return None

    df = None
    last_exc = None
    for enc in ("utf-8-sig", "gb18030", "gbk"):
        try:
            df = pd.read_csv(hydro_file, encoding=enc)
            break
        except Exception as e:
            last_exc = e
            df = None
    if df is None:
        raise last_exc

    def _pick_col(cands, fallback_idx):
        for c in cands:
            if c in df.columns:
                return c
        return df.columns[fallback_idx]

    if len(df.columns) < 3:
        return None

    year_col = _pick_col(["Year", "year"], 0)
    month_col = _pick_col(["Month", "month"], 1)
    Q_col = _pick_col(["Q", "discharge"], 2)

    Qs_col = None
    for c in ("Qs",):
        if c in df.columns:
            Qs_col = c
            break
    if Qs_col is None and len(df.columns) >= 4:
        Qs_col = df.columns[3]

    df = df.sort_values([year_col, month_col]).reset_index(drop=True)
    Q_arr = pd.to_numeric(df[Q_col], errors="coerce").astype(float).to_numpy()

    hydro_dict = {}
    for idx, row in df.iterrows():
        year = int(row[year_col])
        month = int(row[month_col])
        key = f"{year}{month:02d}"

        Q = float(Q_arr[idx]) if np.isfinite(Q_arr[idx]) else np.nan

        if Qs_col is not None:
            Qs_raw = row.get(Qs_col, 0.0)
            Qs = float(Qs_raw) if pd.notna(Qs_raw) else 0.0
        else:
            Qs = 0.0

        if idx > 0 and np.isfinite(Q_arr[idx - 1]):
            Q_prev = float(Q_arr[idx - 1])
        else:
            Q_prev = Q

        dQ = (Q - Q_prev) / (Q_prev + 1e-6) if np.isfinite(Q) and np.isfinite(Q_prev) else 0.0
        start_idx = max(0, idx - 2)
        Q_cumsum3 = float(np.nansum(Q_arr[start_idx : idx + 1]))

        season_sin = float(np.sin(2 * np.pi * month / 12))
        season_cos = float(np.cos(2 * np.pi * month / 12))

        hydro_dict[key] = {
            "Q": Q,
            "Q_prev": Q_prev,
            "dQ": dQ,
            "Q_cumsum3": Q_cumsum3,
            "Qs": Qs,
            "season_sin": season_sin,
            "season_cos": season_cos,
        }

    return hydro_dict


def load_width_along_data():
    """Load along-channel width data."""
    width_csv = None
    for cand in _WIDTH_ALONG_CSV_CANDIDATES:
        if cand.exists():
            width_csv = cand
            break

    if width_csv is not None:
        df = pd.read_csv(width_csv, index_col=0)
        print(f"  Along-channel width data: {df.shape[0]} nodes x {df.shape[1]} months")
        return df

    width_files = sorted(CENTERLINE_DIR.glob('*_width_along_1000pts.npy'))
    if not width_files:
        print(f"  Warning: No along-channel width CSV or *_width_along_1000pts.npy found: {CENTERLINE_DIR}")
        return None

    width_dict = {}
    for fp in width_files:
        month_str = fp.name.split('_')[0]
        try:
            arr = np.load(fp)
            if arr.shape[0] != 1000:
                continue
            width_dict[month_str] = arr
        except Exception:
            continue

    if not width_dict:
        print(f"  Warning: Along-channel width npy read failed or dimension != 1000: {CENTERLINE_DIR}")
        return None

    sorted_months = sorted(width_dict.keys())
    df = pd.DataFrame({m: width_dict[m] for m in sorted_months})
    df.index.name = 'node_id'
    print(f"  Along-channel width data (aggregated from npy): {df.shape[0]} nodes x {df.shape[1]} months")
    print(f"  Month range: {sorted_months[0]} - {sorted_months[-1]}")
    return df


def load_depth_along_data():
    """Load along-channel depth data."""
    if not DEPTH_ALONG_CSV.exists():
        print(f"  Warning: Along-channel depth data not found: {DEPTH_ALONG_CSV}")
        return None
    
    df = pd.read_csv(DEPTH_ALONG_CSV, index_col=0)
    print(f"  Along-channel depth data: {df.shape[0]} nodes x {df.shape[1]} months")
    return df


def collect_monthly_samples(years, hydro_dict, width_along_df=None, depth_along_df=None,
                            anomaly_threshold=2000, mean_threshold=500,
                            exclude_months=None, migration_mode='corresp'):
    """
    Collect all monthly migration samples for specified years and filter anomalies.
    
    Args:
        years: List of years
        hydro_dict: Hydrological data dictionary (containing enhanced features)
        width_along_df: Along-channel width DataFrame (optional)
        depth_along_df: Along-channel depth DataFrame (optional)
        anomaly_threshold: Max migration anomaly threshold (m), values above are considered anomalous
        mean_threshold: Mean migration anomaly threshold (m)
        exclude_months: List of manually excluded months (data quality issues)
        migration_mode: Migration calculation mode (corresp/min)
    
    Returns:
        samples: list of dict, each containing {month_str, migration, hydro_features, width_along, depth_along}
        anomalies: list of dict, filtered anomalous samples
    """
    if exclude_months is None:
        exclude_months = []
    
    samples = []
    anomalies = []
    
    for year in years:
        for month in range(1, 13):
            month_str = f"{year}{month:02d}"
            
            # Manually exclude known problematic months
            if month_str in exclude_months:
                print(f"  ⚠️  Excluded {month_str}: data quality issue")
                continue
            
            try:
                migration = compute_monthly_migration(year, month, mode=migration_mode)
                
                # Get hydrological data (enhanced features)
                hydro = hydro_dict.get(month_str, {}) if hydro_dict else {}
                Q = hydro.get('Q', np.nan)
                
                sample = {
                    'month_str': month_str,
                    'migration': migration,
                    'Q': Q,
                    'Q_prev': hydro.get('Q_prev', Q),
                    'dQ': hydro.get('dQ', 0),
                    'Q_cumsum3': hydro.get('Q_cumsum3', Q * 3),
                    'Qs': hydro.get('Qs', 0),
                    'season_sin': hydro.get('season_sin', np.sin(2 * np.pi * month / 12)),
                    'season_cos': hydro.get('season_cos', np.cos(2 * np.pi * month / 12)),
                    'mean': np.mean(migration),
                    'max': np.max(migration)
                }
                
                # Get along-channel width
                if width_along_df is not None and month_str in width_along_df.columns:
                    sample['width_along'] = width_along_df[month_str].values
                else:
                    sample['width_along'] = None
                
                # Get along-channel depth
                if depth_along_df is not None and month_str in depth_along_df.columns:
                    sample['depth_along'] = depth_along_df[month_str].values
                else:
                    sample['depth_along'] = None
                
                # Anomaly detection
                if np.max(migration) > anomaly_threshold or np.mean(migration) > mean_threshold:
                    print(f"  ⚠️  Anomalous sample {month_str}: mean={np.mean(migration):.1f}m, max={np.max(migration):.1f}m")
                    anomalies.append(sample)
                else:
                    samples.append(sample)
                    width_info = f", W_mean={np.nanmean(sample['width_along']):.0f}m" if sample['width_along'] is not None else ""
                    print(f"  ✓ {month_str}: mean={np.mean(migration):.2f}m, Q={Q:.0f}, dQ={sample['dQ']:.2f}{width_info}")
                    
            except FileNotFoundError as e:
                print(f"  Skipped {month_str}: missing next-month centerline data")
    
    return samples, anomalies


def create_monthly_dataset(static_df, edge_index, samples, hydro_stats, 
                           use_width=True, use_depth=False, ablate_qs=False):
    """
    Create monthly migration dataset (enhanced hydrological + along-channel features version).
    
    Args:
        static_df: Static features DataFrame
        edge_index: Graph edge index
        samples: Monthly sample list
        hydro_stats: Hydrological feature normalization statistics
        use_width: Whether to include along-channel width features
        use_depth: Whether to include along-channel depth features
    
    Returns:
        Data object with monthly samples
    """
    feature_cols = [
        'curvature', 'abs_curvature', 'position', 'sinuosity_local',
        'lai_mean', 'lai_std', 'lai_max',
        'clay_pct', 'sand_pct', 'organic_matter', 'bulk_density',
        'landuse_agriculture', 'landuse_forest', 'landuse_water',
        'is_protected'
    ]
    
    # Static feature normalization
    x_static = torch.tensor(static_df[feature_cols].values, dtype=torch.float32)
    x_mean = x_static.mean(dim=0, keepdim=True)
    x_std = x_static.std(dim=0, keepdim=True)
    x_std[x_std < 1e-6] = 1.0
    x_static_norm = (x_static - x_mean) / x_std
    
    # Collect data from all samples
    migrations = []
    month_strs = []
    
    # Hydrological feature arrays
    Q_list, Q_prev_list, dQ_list = [], [], []
    Q_cumsum3_list, Qs_list = [], []
    season_sin_list, season_cos_list = [], []
    
    # Along-channel feature arrays
    width_along_list = []
    depth_along_list = []
    
    for sample in samples:
        migrations.append(sample['migration'])
        month_strs.append(sample['month_str'])
        
        Q_list.append(sample['Q'])
        Q_prev_list.append(sample['Q_prev'])
        dQ_list.append(sample['dQ'])
        Q_cumsum3_list.append(sample['Q_cumsum3'])
        Qs_list.append(sample['Qs'])
        season_sin_list.append(sample['season_sin'])
        season_cos_list.append(sample['season_cos'])
        
        # Along-channel width
        if use_width and sample.get('width_along') is not None:
            width_along_list.append(sample['width_along'])
        elif use_width:
            # Fill with NaN when missing
            width_along_list.append(np.full(1000, np.nan))
        
        # Along-channel depth
        if use_depth and sample.get('depth_along') is not None:
            depth_along_list.append(sample['depth_along'])
        elif use_depth:
            depth_along_list.append(np.full(1000, np.nan))
    
    # Convert to tensor
    y = torch.tensor(np.stack(migrations), dtype=torch.float32)  # [N_samples, 1000]
    
    # Hydrological feature normalization
    Q_norm = (np.array(Q_list) - hydro_stats['Q_mean']) / hydro_stats['Q_std']
    Q_prev_norm = (np.array(Q_prev_list) - hydro_stats['Q_mean']) / hydro_stats['Q_std']
    dQ_norm = np.clip(np.array(dQ_list), -2, 2)  # Clip range
    Q_cumsum3_norm = (np.array(Q_cumsum3_list) - hydro_stats['Q_cumsum3_mean']) / hydro_stats['Q_cumsum3_std']
    Qs_norm = (np.array(Qs_list) - hydro_stats['Qs_mean']) / hydro_stats['Qs_std']

    if ablate_qs:
        Qs_norm = np.zeros_like(Qs_norm)
    
    # Combine hydrological features [N_samples, 7]
    x_hydro = torch.tensor(np.stack([
        Q_norm,
        Q_prev_norm,
        dQ_norm,
        Q_cumsum3_norm,
        Qs_norm,
        np.array(season_sin_list),
        np.array(season_cos_list)
    ], axis=1), dtype=torch.float32)
    
    hydro_feature_names = ['Q', 'Q_prev', 'dQ', 'Q_cumsum3', 'Qs', 'season_sin', 'season_cos']
    
    # Create Data object
    data = Data(
        x_static=x_static_norm,      # [1000, 14] static features
        x_hydro=x_hydro,             # [N_samples, 7] enhanced hydrological features
        edge_index=edge_index,
        y=y,                         # [N_samples, 1000] migration
        num_nodes=len(static_df),
        num_samples=len(samples),
        months=month_strs,
        Q_values=Q_list,
        feature_names=feature_cols,
        hydro_feature_names=hydro_feature_names,
        x_mean=x_mean,
        x_std=x_std,
        hydro_stats=hydro_stats
    )
    
    # Add along-channel width features
    if use_width and width_along_list:
        width_array = np.stack(width_along_list)  # [N_samples, 1000]
        # Normalize using global mean and std
        width_mean = np.nanmean(width_array)
        width_std = np.nanstd(width_array)
        width_std = max(width_std, 1e-6)
        width_norm = (width_array - width_mean) / width_std
        # Replace NaN with 0 (normalized mean)
        width_norm = np.nan_to_num(width_norm, nan=0.0)
        
        data.x_width = torch.tensor(width_norm, dtype=torch.float32)
        data.width_mean = width_mean
        data.width_std = width_std
        print(f"    Along-channel width: mean={width_mean:.1f}m, std={width_std:.1f}m")
    
    # Add along-channel depth features
    if use_depth and depth_along_list:
        depth_array = np.stack(depth_along_list)  # [N_samples, 1000]
        depth_mask = np.isfinite(depth_array).astype(np.float32)
        depth_mean = np.nanmean(depth_array)
        depth_std = np.nanstd(depth_array)
        depth_std = max(depth_std, 1e-6)
        depth_norm = (depth_array - depth_mean) / depth_std
        depth_norm = np.nan_to_num(depth_norm, nan=0.0)
        
        data.x_depth = torch.tensor(depth_norm, dtype=torch.float32)
        data.x_depth_mask = torch.tensor(depth_mask, dtype=torch.float32)
        data.depth_mean = depth_mean
        data.depth_std = depth_std
        print(f"    Along-channel depth: mean={depth_mean:.2f}m, std={depth_std:.2f}m")
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Build monthly migration graph for CV')
    parser.add_argument('--fold', type=int, required=True, choices=[1, 2, 3],
                        help='CV fold number (1, 2, or 3)')
    parser.add_argument('--migration_mode', type=str, default='corresp', choices=['corresp', 'min'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional tag appended to output filenames to avoid overwrite')
    parser.add_argument('--train_years', nargs='+', type=int, default=None,
                        help='Override training years (e.g., 2016 2017 2018 2019)')
    parser.add_argument('--val_years', nargs='+', type=int, default=None,
                        help='Override validation years (e.g., 2020 2021)')
    parser.add_argument('--exclude_train', nargs='*', type=str, default=None,
                        help='Override excluded train months (e.g., 201912 202001)')
    parser.add_argument('--exclude_val', nargs='*', type=str, default=None,
                        help='Override excluded val months (e.g., 202005 202006)')
    parser.add_argument('--use_width', action='store_true', default=True,
                        help='Include along-channel width features (default: True)')
    parser.add_argument('--no_width', action='store_true',
                        help='Exclude along-channel width features')
    parser.add_argument('--use_depth', action='store_true',
                        help='Include along-channel depth features (default: False)')
    parser.add_argument('--ablate_qs', action='store_true',
                        help='Set Qs feature to zero for ablation study')
    args = parser.parse_args()
    
    # Process width argument
    use_width = not args.no_width
    use_depth = args.use_depth
    
    fold_config = CV_FOLDS[args.fold]

    train_years = args.train_years if args.train_years is not None else fold_config['train_years']
    val_years = args.val_years if args.val_years is not None else fold_config['val_years']
    exclude_train = args.exclude_train if args.exclude_train is not None else fold_config.get('exclude_train', [])
    exclude_val = args.exclude_val if args.exclude_val is not None else fold_config.get('exclude_val', [])

    output_dir = Path(args.output_dir) if args.output_dir else GRAPH_DIR / 'cv_monthly'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"Building monthly migration CV dataset - Fold {args.fold}")
    print(f"Training years: {train_years}")
    print(f"Validation years: {val_years}")
    print(f"Along-channel width: {'enabled' if use_width else 'disabled'}")
    print(f"Along-channel depth: {'enabled' if use_depth else 'disabled'}")
    print("="*60)
    
    # Load static features
    print("\nLoading static features...")
    static_df = load_static_features()
    num_nodes = len(static_df)
    print(f"  Number of nodes: {num_nodes}")
    
    # Build graph structure
    edge_index = build_edge_index(num_nodes)
    print(f"  Number of edges: {edge_index.shape[1]}")
    
    # Load hydrological data
    print("\nLoading hydrological data...")
    hydro_dict = load_monthly_hydro()
    hydro_stats = compute_hydro_stats(hydro_dict, years=train_years)
    print(f"  Q stats: mean={hydro_stats['Q_mean']:.0f}, std={hydro_stats['Q_std']:.0f}")
    print(f"  Q_cumsum3 stats: mean={hydro_stats['Q_cumsum3_mean']:.0f}")
    print(f"  Qs stats: mean={hydro_stats['Qs_mean']:.4f}")
    print(f"  Qs ablation: {'enabled' if args.ablate_qs else 'disabled'}")
    
    # Load along-channel feature data
    width_along_df = None
    depth_along_df = None
    
    if use_width:
        print("\nLoading along-channel width data...")
        width_along_df = load_width_along_data()
        if width_along_df is None:
            print("  Warning: Along-channel width unavailable, disabling this feature")
            use_width = False
    
    if use_depth:
        print("\nLoading along-channel depth data...")
        depth_along_df = load_depth_along_data()
        if depth_along_df is None:
            print("  Warning: Along-channel depth unavailable, disabling this feature")
            use_depth = False
    
    # Collect training samples
    print(f"\nCollecting training samples ({train_years})...")
    train_samples, train_anomalies = collect_monthly_samples(
        train_years, hydro_dict, 
        width_along_df=width_along_df, depth_along_df=depth_along_df,
        exclude_months=exclude_train,
        migration_mode=args.migration_mode
    )
    print(f"  Training samples: {len(train_samples)} (filtered anomalies: {len(train_anomalies)}, manually excluded: {len(exclude_train)})")
    
    if len(train_samples) == 0:
        raise ValueError("No valid training samples! Please check centerline data.")
    
    # Print training sample statistics
    train_migrations = [s['mean'] for s in train_samples]
    train_Qs = [s['Q'] for s in train_samples]
    print(f"  Migration: mean={np.mean(train_migrations):.2f}m, range=[{np.min(train_migrations):.2f}, {np.max(train_migrations):.2f}]")
    print(f"  Discharge Q: mean={np.mean(train_Qs):.0f}, range=[{np.min(train_Qs):.0f}, {np.max(train_Qs):.0f}]")
    
    # Collect validation samples
    print(f"\nCollecting validation samples ({val_years})...")
    val_samples, val_anomalies = collect_monthly_samples(
        val_years, hydro_dict,
        width_along_df=width_along_df, depth_along_df=depth_along_df,
        exclude_months=exclude_val,
        migration_mode=args.migration_mode
    )
    print(f"  Validation samples: {len(val_samples)} (filtered anomalies: {len(val_anomalies)}, manually excluded: {len(exclude_val)})")
    
    if len(val_samples) == 0:
        raise ValueError("No valid validation samples! Please check centerline data.")
    
    val_migrations = [s['mean'] for s in val_samples]
    val_Qs = [s['Q'] for s in val_samples]
    print(f"  Migration: mean={np.mean(val_migrations):.2f}m, range=[{np.min(val_migrations):.2f}, {np.max(val_migrations):.2f}]")
    print(f"  Discharge Q: mean={np.mean(val_Qs):.0f}, range=[{np.min(val_Qs):.0f}, {np.max(val_Qs):.0f}]")
    
    # Create datasets
    print("\nCreating datasets...")
    train_data = create_monthly_dataset(static_df, edge_index, train_samples, hydro_stats,
                                         use_width=use_width, use_depth=use_depth, ablate_qs=args.ablate_qs)
    val_data = create_monthly_dataset(static_df, edge_index, val_samples, hydro_stats,
                                       use_width=use_width, use_depth=use_depth, ablate_qs=args.ablate_qs)
    
    # Save
    suffix = ""
    if args.ablate_qs:
        suffix += "_noqs"
    if use_width:
        suffix += "_width"
    if use_depth:
        suffix += "_depth"

    tag_part = f"_{args.tag}" if args.tag else ""
     
    train_file = output_dir / f"fold{args.fold}_train{tag_part}{suffix}.pt"
    val_file = output_dir / f"fold{args.fold}_val{tag_part}{suffix}.pt"
    
    torch.save(train_data, train_file)
    torch.save(val_data, val_file)
    
    print(f"\n{'='*60}")
    print(f"Saved:")
    print(f"  Training set: {train_file}")
    print(f"    - {train_data.num_samples} monthly samples")
    print(f"    - Static features: {len(train_data.feature_names)} dims")
    print(f"    - Hydrological features: {len(train_data.hydro_feature_names)} dims")
    if hasattr(train_data, 'x_width'):
        print(f"    - Along-channel width: [N_samples, 1000]")
    if hasattr(train_data, 'x_depth'):
        print(f"    - Along-channel depth: [N_samples, 1000]")
    print(f"  Validation set: {val_file}")
    print(f"    - {val_data.num_samples} monthly samples")


if __name__ == '__main__':
    main()
