"""
Prepare Monthly Input Data for MCMM (Meander Centerline Migration Model)

Re-framed as Planform Process Prior for EMS submission.

Functionality:
1. Integrate multi-source data (Q, W, D, D50, migration rate).
2. Compute E_rate (back-calculated from observed geometric displacement;
   serves as an unconstrained geometric-evolution prior parameter).
3. Generate MCMM time-series input file (used as the Planform Prior
   before GNN correction in the forward pass).
   Note: Under sub-bankfull (monthly low-flow) conditions, MCMM outputs
   an unconstrained geometric potential that includes apparent water-level
   displacement, not a physically meaningful bank erosion rate. True
   migration is corrected downstream in the GNN with engineering constraints
   and water-level decoupling.

Data sources:
- Discharge Q / sediment load Qs: shashi_monthly_q_qs.csv
- Channel width W: monthly_width_stats.csv
- Water depth D: monthly_high_freq_water_stats.csv
- Grain size D50: d50_measured.csv / d50_prophet_forecast.csv
- Migration rate: centerline_migration_stats.csv

Output:
- jingjiang_monthly_*.csv: MCMM time-series input file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import io
import argparse
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration paths
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]

# Input data paths
Q_DATA_PATH_SSC = REPO_ROOT / "data" / "raw" / "shashi_monthly_q_qs.csv"
WIDTH_DATA_PATH = REPO_ROOT / "data" / "raw" / "monthly_width_stats.csv"
DEPTH_DATA_PATH = REPO_ROOT / "data" / "raw" / "monthly_high_freq_water_stats.csv"
D50_MEASURED_PATH = REPO_ROOT / "data" / "raw" / "d50_measured.csv"
D50_FORECAST_PATH = REPO_ROOT / "data" / "raw" / "d50_prophet_forecast.csv"
MIGRATION_DATA_PATH = REPO_ROOT / "data" / "raw" / "centerline_migration_stats.csv"

# Output path
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "jingjiang_monthly_2016_2024.csv"

# ============================================================================
# Physical constants and model parameters
# ============================================================================
SECONDS_PER_MONTH = 30.44 * 24 * 3600  # Average seconds per month
GRAVITY = 9.81  # Gravitational acceleration m/s²
DELTA_SG = 1.65  # Relative submerged density (ρs/ρ - 1)
SLOPE = 0.0000459  # Channel slope
KINEMATIC_VISCOSITY = 1.0e-6  # Kinematic viscosity m²/s

# ============================================================================
# Data loading functions
# ============================================================================

def _read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    for enc in (None, 'utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'cp936', 'gb18030', 'latin1'):
        try:
            if enc is None:
                return pd.read_csv(path, **kwargs)
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception:
            continue

    raw = Path(path).read_bytes()
    text = raw.decode('gb18030', errors='ignore')
    return pd.read_csv(io.StringIO(text), **kwargs)

def load_discharge_data():
    """Load discharge/sediment load data."""
    if Q_DATA_PATH_SSC.exists():
        df = _read_csv_with_fallback(Q_DATA_PATH_SSC, header=None)
        if df.shape[1] < 3:
            raise ValueError(f"Invalid Q csv format: {Q_DATA_PATH_SSC}")
        df = df.iloc[:, :4] if df.shape[1] >= 4 else df.iloc[:, :3]
        rename_map = {0: 'Year', 1: 'Month', 2: 'Q'}
        if df.shape[1] >= 4:
            rename_map[3] = 'Qs'
        df = df.rename(columns=rename_map)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        df['Q'] = pd.to_numeric(df['Q'], errors='coerce')
        if 'Qs' in df.columns:
            df['Qs'] = pd.to_numeric(df['Qs'], errors='coerce')
        df = df.dropna(subset=['Year', 'Month', 'Q'])
        df['Year'] = df['Year'].astype(int)
        df['Month'] = df['Month'].astype(int)
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].rename(columns={'Year': 'year', 'Month': 'month'}).assign(day=1))
        out_cols = ['Date', 'Q']
        if 'Qs' in df.columns:
            out_cols.append('Qs')
        return df[out_cols]

    raise FileNotFoundError(f"Missing discharge/Qs file: {Q_DATA_PATH_SSC}")

def load_width_data():
    """Load channel width data."""
    df = pd.read_csv(WIDTH_DATA_PATH)
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    df = df.rename(columns={'Mean_Width_Meters': 'W', 'StdDev_Width_Meters': 'W_std'})
    out_cols = ['Date', 'W']
    if 'W_std' in df.columns:
        out_cols.append('W_std')
    return df[out_cols]

def load_depth_data():
    """Load water depth data."""
    df = pd.read_csv(DEPTH_DATA_PATH)
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    if 'Mean_Depth_Along' in df.columns:
        df = df.rename(columns={'Mean_Depth_Along': 'D'})
    else:
        df = df.rename(columns={'Mean_Depth': 'D'})

    if 'StdDev_Depth_Along' in df.columns:
        df = df.rename(columns={'StdDev_Depth_Along': 'D_std'})
    elif 'StdDev_Depth' in df.columns:
        df = df.rename(columns={'StdDev_Depth': 'D_std'})

    out_cols = ['Date', 'D']
    if 'D_std' in df.columns:
        out_cols.append('D_std')
    return df[out_cols]

def load_d50_data():
    """Load D50 data (measured + forecast)."""
    # Measured data (2009-2014)
    df_measured = pd.read_csv(D50_MEASURED_PATH)
    
    def parse_d50_date(date_str):
        parts = date_str.split('-')
        year = 2000 + int(parts[0])
        month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
                     'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        month = month_map[parts[1]]
        return pd.Timestamp(year=year, month=month, day=1)
    
    df_measured['Date'] = df_measured['Date'].apply(parse_d50_date)
    df_measured['D50'] = df_measured['D50'] / 1000  # mm -> m
    
    # Forecast data (2015-2030)
    df_forecast = pd.read_csv(D50_FORECAST_PATH)
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    df_forecast['D50'] = df_forecast['Forecast Mean'] / 1000  # mm -> m
    
    # Merge
    df = pd.concat([
        df_measured[['Date', 'D50']],
        df_forecast[['Date', 'D50']]
    ]).drop_duplicates(subset='Date', keep='first')
    
    return df.sort_values('Date')

def load_migration_data():
    """Load migration rate data."""
    df = pd.read_csv(MIGRATION_DATA_PATH)
    
    # Parse time interval (e.g., "201601-201602")
    def parse_interval(interval_str):
        start = interval_str.split('-')[0]
        year = int(start[:4])
        month = int(start[4:6])
        return pd.Timestamp(year=year, month=month, day=1)
    
    df['Date'] = df['Time_Interval'].apply(parse_interval)
    df = df.rename(columns={'Mean_Migration_Meters': 'Migration'})

    df['Migration'] = pd.to_numeric(df['Migration'], errors='coerce')
    df['Migration_filtered'] = df['Migration'] >= 1000
    df.loc[df['Migration_filtered'], 'Migration'] = np.nan

    return df[['Date', 'Migration', 'Migration_filtered']]

# ============================================================================
# E_rate computation functions
# ============================================================================

def estimate_e_rate(migration_m, dt_seconds, du_estimate=1.0):
    """
    Estimate E_rate from observed migration.
    
    MCMM migration equation: dx = E * dU * dt
    Therefore: E_rate = migration / (dU * dt)
    
    Parameters:
    - migration_m: Monthly migration distance (m)
    - dt_seconds: Time step (s)
    - du_estimate: Excess flow velocity estimate (m/s), default 1.0
    
    Returns:
    - E_rate: Erosion rate (m/s)
    """
    # Simplified estimate: assume dU ≈ 1.0 m/s (typical meander excess velocity)
    # In practice, dU should be computed from the ZS model
    e_rate = migration_m / (du_estimate * dt_seconds)
    return e_rate


def _parse_yyyymm(yyyymm: str) -> pd.Timestamp:
    yyyymm = str(yyyymm).strip()
    if len(yyyymm) != 6 or not yyyymm.isdigit():
        raise ValueError(f"Invalid yyyymm: {yyyymm}")
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    return pd.Timestamp(year=year, month=month, day=1)

# ============================================================================
# Main program
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare monthly MCMM timeseries input")
    parser.add_argument("--start", default="201601", help="Start month (yyyymm), inclusive")
    parser.add_argument("--end", default="202412", help="End month (yyyymm), inclusive")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path")
    args = parser.parse_args()

    output_path = Path(args.output)
    qc_output_path = output_path.with_name("mcmm_input_qc_report.csv")
    start_limit = _parse_yyyymm(args.start)
    end_limit = _parse_yyyymm(args.end)
    if end_limit < start_limit:
        raise ValueError(f"Invalid range: {args.start} ~ {args.end}")

    print("=" * 60)
    print("MCMM Monthly Input Data Preparation")
    print("=" * 60)
    
    # 1. Load all data
    print("\n[1] Loading data...")
    
    try:
        df_q = load_discharge_data()
        print(f"  - Discharge data: {len(df_q)} records ({df_q['Date'].min()} ~ {df_q['Date'].max()})")
    except Exception as e:
        print(f"  - Discharge data loading failed: {e}")
        df_q = None
    
    try:
        df_w = load_width_data()
        print(f"  - Width data: {len(df_w)} records ({df_w['Date'].min()} ~ {df_w['Date'].max()})")
    except Exception as e:
        print(f"  - Width data loading failed: {e}")
        df_w = None
    
    try:
        df_d = load_depth_data()
        print(f"  - Depth data: {len(df_d)} records ({df_d['Date'].min()} ~ {df_d['Date'].max()})")
    except Exception as e:
        print(f"  - Depth data loading failed: {e}")
        df_d = None
    
    try:
        df_d50 = load_d50_data()
        print(f"  - D50 data: {len(df_d50)} records ({df_d50['Date'].min()} ~ {df_d50['Date'].max()})")
    except Exception as e:
        print(f"  - D50 data loading failed: {e}")
        df_d50 = None
    
    try:
        df_mig = load_migration_data()
        print(f"  - Migration data: {len(df_mig)} records ({df_mig['Date'].min()} ~ {df_mig['Date'].max()})")
    except Exception as e:
        print(f"  - Migration data loading failed: {e}")
        df_mig = None
    
    # 2. Determine overlapping time range
    print("\n[2] Determining data overlap range...")
    
    # Core data: W, D (2016-2020)
    if df_w is not None and df_d is not None:
        start_date = max(df_w['Date'].min(), df_d['Date'].min())
        end_date = min(df_w['Date'].max(), df_d['Date'].max())
        print(f"  - Geometry data overlap range: {start_date} ~ {end_date}")
    else:
        print("  - Error: Missing width or depth data")
        return
    
    # 3. Merge data
    print("\n[3] Merging data...")
    
    start_date = max(start_date, start_limit)
    end_date = min(end_date, end_limit)
    print(f"  - Applying user time range: {start_date} ~ {end_date}")

    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    df = pd.DataFrame({'Date': date_range})
    
    # Merge data sources
    df = df.merge(df_w, on='Date', how='left')
    df = df.merge(df_d, on='Date', how='left')
    
    if df_q is not None:
        df = df.merge(df_q, on='Date', how='left')
    else:
        df['Q'] = np.nan
    
    if df_d50 is not None:
        df = df.merge(df_d50, on='Date', how='left')
    else:
        df['D50'] = np.nan
    
    if df_mig is not None:
        df = df.merge(df_mig, on='Date', how='left')
    else:
        df['Migration'] = np.nan
    
    print(f"  - Records after merge: {len(df)}")
    print(f"  - Missing value statistics:")
    for col in ['Q', 'W', 'W_std', 'D', 'D_std', 'D50', 'Migration']:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"    {col}: {missing}/{len(df)} missing")
    
    # 4. Compute E_rate
    print("\n[4] Computing E_rate...")
    
    if 'Migration' in df.columns:
        # Back-calculate E_rate from migration
        df['E_rate'] = df['Migration'].apply(
            lambda x: estimate_e_rate(x, SECONDS_PER_MONTH) if pd.notna(x) else np.nan
        )
        
        # Statistics
        valid_e = df['E_rate'].dropna()
        if len(valid_e) > 0:
            print(f"  - E_rate range: {valid_e.min():.2e} ~ {valid_e.max():.2e} m/s")
            print(f"  - E_rate mean: {valid_e.mean():.2e} m/s")
            print(f"  - E_rate median: {valid_e.median():.2e} m/s")
    else:
        df['E_rate'] = np.nan
        print("  - No migration data, E_rate set to NaN")
    
    # 5. Fill missing values
    print("\n[5] Filling missing values...")

    qc = pd.DataFrame({'month': df['Date'].dt.strftime('%Y%m')})
    for col in ['Q', 'W', 'D', 'E_rate', 'D50', 'Migration', 'W_std', 'D_std']:
        if col in df.columns:
            qc[f'{col}_missing_before'] = df[col].isna()
        else:
            qc[f'{col}_missing_before'] = False
    if 'Migration_filtered' in df.columns:
        qc['Migration_filtered'] = df['Migration_filtered'].fillna(False).astype(bool)
    else:
        qc['Migration_filtered'] = False

    if df['W'].isna().any():
        df = df.sort_values('Date')
        df_tmp = df.set_index('Date')
        df_tmp['W'] = df_tmp['W'].interpolate(method='time')
        df = df_tmp.reset_index()
        if df_w is not None:
            df_w_tmp = df_w.copy()
            df_w_tmp['Month'] = df_w_tmp['Date'].dt.month
            monthly_avg_w = df_w_tmp.groupby('Month')['W'].mean()
            df['Month'] = df['Date'].dt.month
            df.loc[df['W'].isna(), 'W'] = df.loc[df['W'].isna(), 'Month'].map(monthly_avg_w)
            df = df.drop(columns=['Month'])
        df['W'] = df['W'].fillna(df['W'].mean())
        print("  - W: interpolation/multi-year monthly mean/mean fallback fill")

    if df['D'].isna().any():
        df = df.sort_values('Date')
        df_tmp = df.set_index('Date')
        df_tmp['D'] = df_tmp['D'].interpolate(method='time')
        df = df_tmp.reset_index()
        if df_d is not None:
            df_d_tmp = df_d.copy()
            df_d_tmp['Month'] = df_d_tmp['Date'].dt.month
            monthly_avg_d = df_d_tmp.groupby('Month')['D'].mean()
            df['Month'] = df['Date'].dt.month
            df.loc[df['D'].isna(), 'D'] = df.loc[df['D'].isna(), 'Month'].map(monthly_avg_d)
            df = df.drop(columns=['Month'])
        df['D'] = df['D'].fillna(df['D'].mean())
        print("  - D: interpolation/multi-year monthly mean/mean fallback fill")
    
    # Q: Fill with multi-year monthly mean
    if df['Q'].isna().any() and df_q is not None:
        # Compute multi-year monthly mean
        df_q['Month'] = df_q['Date'].dt.month
        monthly_avg_q = df_q.groupby('Month')['Q'].mean()
        
        df['Month'] = df['Date'].dt.month
        df['Q'] = df.apply(
            lambda row: monthly_avg_q.get(row['Month'], np.nan) if pd.isna(row['Q']) else row['Q'],
            axis=1
        )
        df = df.drop(columns=['Month'])
        print("  - Q: filled with multi-year monthly mean")
    
    # D50: Fill with forecast values
    if df['D50'].isna().any() and df_d50 is not None:
        df = df.drop(columns=['D50'])
        df = df.merge(df_d50, on='Date', how='left')
        print("  - D50: filled with forecast values")
    
    # E_rate: Fill with mean
    if df['E_rate'].isna().any():
        mean_e_rate = df['E_rate'].mean()
        df['E_rate'] = df['E_rate'].fillna(mean_e_rate)
        print(f"  - E_rate: filled with mean {mean_e_rate:.2e}")

    # W_std, D_std: Fill with mean (auxiliary columns for uncertainty/sensitivity analysis)
    if 'W_std' in df.columns and df['W_std'].isna().any():
        df['W_std'] = df['W_std'].fillna(df['W_std'].mean())
    if 'D_std' in df.columns and df['D_std'].isna().any():
        df['D_std'] = df['D_std'].fillna(df['D_std'].mean())

    for col in ['Q', 'W', 'D', 'E_rate', 'D50', 'Migration', 'W_std', 'D_std']:
        if col in df.columns:
            qc[f'{col}_filled'] = qc[f'{col}_missing_before'] & (~df[col].isna())
        else:
            qc[f'{col}_filled'] = False

    core_cols = [c for c in ['Q', 'W', 'D', 'E_rate'] if c in df.columns]
    if core_cols:
        qc['core_missing_after'] = df[core_cols].isna().any(axis=1)
    else:
        qc['core_missing_after'] = True

    qc.to_csv(qc_output_path, index=False)
    print(f"  - QC report: {qc_output_path}")
    missing_months = qc.loc[qc['core_missing_after'], 'month'].tolist()
    if len(missing_months) > 0:
        print(f"  - Warning: Number of months with missing core columns (Q/W/D/E_rate): {len(missing_months)}")
        print("    " + ",".join(missing_months))

    # 6. Format output
    print("\n[6] Generating MCMM input file...")
    
    # Create MCMM format
    df_output = pd.DataFrame({
        'month': df['Date'].dt.strftime('%Y%m'),
        'Q': df['Q'].round(2),
        'W': df['W'].round(2),
        'D': df['D'].round(4),
        'E_rate': df['E_rate'].apply(lambda x: f'{x:.6e}'),
        'physical_ds': df['D50'].apply(lambda x: f'{x:.6f}' if pd.notna(x) else '0.000250'),
        'W_std': df['W_std'].round(2) if 'W_std' in df.columns else np.nan,
        'D_std': df['D_std'].round(4) if 'D_std' in df.columns else np.nan,
    })
    
    # Save
    df_output.to_csv(output_path, index=False)
    print(f"  - Output file: {output_path}")
    print(f"  - Record count: {len(df_output)}")
    
    # 7. Display data summary
    print("\n[7] Data summary:")
    print(df_output.head(12).to_string(index=False))
    print("...")
    print(df_output.tail(6).to_string(index=False))
    
    # 8. Generate parameter suggestions
    print("\n[8] MCMM parameter suggestions:")
    print(f"  - slope = {SLOPE}")
    print(f"  - delta_sg = {DELTA_SG}")
    print(f"  - kinematic_viscosity = {KINEMATIC_VISCOSITY}")
    
    # E_scale estimation
    mean_e_rate = df['E_rate'].mean()
    suggested_e_scale = mean_e_rate / 1.0  # Assume E_rate input is on the order of 1.0
    print(f"  - E_scale (suggested) = {suggested_e_scale:.2e}")
    print(f"    (Based on mean E_rate = {mean_e_rate:.2e} m/s)")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
