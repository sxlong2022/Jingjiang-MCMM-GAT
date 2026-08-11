import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _codes_root() -> str:
    return os.path.dirname(os.path.dirname(SCRIPT_DIR))


def _abs_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = str(p)
    if p.strip() == '':
        return None
    return p if os.path.isabs(p) else os.path.join(SCRIPT_DIR, p)


def _as_month_int(s) -> Optional[int]:
    if s is None:
        return None
    try:
        if isinstance(s, (int, np.integer)):
            return int(s)
        ss = str(s).strip()
        if ss == '':
            return None
        if ss.isdigit():
            return int(ss)
        if '-' in ss:
            parts = ss.split('-')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]) * 100 + int(parts[1])
        return int(float(ss))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Write/replace D column in MCMM timeseries CSV with external monthly depth series (month,D_eff), outputting QC comparison table.'
    )

    codes_root = _codes_root()
    data_raw_dir = os.path.join(codes_root, 'data', 'raw')
    parser.add_argument(
        '--timeseries_csv',
        default=os.path.join(data_raw_dir, 'jingjiang_monthly_2016_2024_final.csv'),
        help='MCMM timeseries CSV (must include month,Q,W,D,E_rate,physical_ds as first 6 columns)',
    )
    parser.add_argument(
        '--depth_scheme2_csv',
        default=os.path.join(
            data_raw_dir,
            'mcmm_depth_timeseries_D_eff_scheme2.csv',
        ),
        help='Scheme 2 depth series CSV (must include month,D_eff)',
    )
    parser.add_argument(
        '--depth_scheme1_csv',
        default=os.path.join(
            data_raw_dir,
            'mcmm_depth_timeseries_D_eff.csv',
        ),
        help='Scheme 1 depth series CSV (optional, for QC; must include month,D_eff)',
    )
    parser.add_argument(
        '--out_timeseries_csv',
        default=os.path.join(SCRIPT_DIR, 'jingjiang_monthly_2016_2024_scheme2D.csv'),
        help='Output: new timeseries after replacing D (does not overwrite original by default)',
    )
    parser.add_argument(
        '--out_qc_csv',
        default=os.path.join(SCRIPT_DIR, 'qc_depth_scheme1_vs_scheme2_vs_original.csv'),
        help='Output: monthly QC comparison table',
    )
    parser.add_argument(
        '--month_col',
        default='month',
        help='month column name in depth series CSV',
    )
    parser.add_argument(
        '--depth_col',
        default='D_eff',
        help='depth column name in depth series CSV',
    )
    args = parser.parse_args()

    timeseries_csv = _abs_path(args.timeseries_csv)
    scheme2_csv = _abs_path(args.depth_scheme2_csv)
    scheme1_csv = _abs_path(args.depth_scheme1_csv)
    out_timeseries_csv = _abs_path(args.out_timeseries_csv)
    out_qc_csv = _abs_path(args.out_qc_csv)

    if not timeseries_csv or not os.path.exists(timeseries_csv):
        raise FileNotFoundError(f'timeseries_csv not found: {timeseries_csv}')
    if not scheme2_csv or not os.path.exists(scheme2_csv):
        raise FileNotFoundError(f'depth_scheme2_csv not found: {scheme2_csv}')

    df_ts = pd.read_csv(timeseries_csv, encoding='utf-8-sig')
    if 'month' not in df_ts.columns or 'D' not in df_ts.columns:
        raise ValueError('timeseries_csv must include columns: month, D')

    df_s2 = pd.read_csv(scheme2_csv, encoding='utf-8-sig')
    if args.month_col not in df_s2.columns or args.depth_col not in df_s2.columns:
        raise ValueError(f'scheme2 depth CSV must include columns: {args.month_col}, {args.depth_col}')
    df_s2 = df_s2[[args.month_col, args.depth_col]].copy()
    df_s2['month_int'] = df_s2[args.month_col].map(_as_month_int)
    df_s2['D_scheme2'] = pd.to_numeric(df_s2[args.depth_col], errors='coerce')
    df_s2 = df_s2.dropna(subset=['month_int']).drop_duplicates(subset=['month_int'])

    df_s1 = None
    if scheme1_csv and os.path.exists(scheme1_csv):
        try:
            tmp = pd.read_csv(scheme1_csv, encoding='utf-8-sig')
            if args.month_col in tmp.columns and args.depth_col in tmp.columns:
                tmp = tmp[[args.month_col, args.depth_col]].copy()
                tmp['month_int'] = tmp[args.month_col].map(_as_month_int)
                tmp['D_scheme1'] = pd.to_numeric(tmp[args.depth_col], errors='coerce')
                tmp = tmp.dropna(subset=['month_int']).drop_duplicates(subset=['month_int'])
                df_s1 = tmp[['month_int', 'D_scheme1']]
        except Exception:
            df_s1 = None

    df_ts['month_int'] = df_ts['month'].map(_as_month_int)
    if df_ts['month_int'].isna().any():
        raise ValueError('timeseries_csv contains unparsable month values')

    df_merge = df_ts.merge(df_s2[['month_int', 'D_scheme2']], on='month_int', how='left')
    if df_s1 is not None:
        df_merge = df_merge.merge(df_s1, on='month_int', how='left')

    df_merge['D_original'] = pd.to_numeric(df_merge['D'], errors='coerce')

    df_qc = pd.DataFrame({
        'month': df_merge['month'],
        'D_original': df_merge['D_original'],
        'D_scheme1': (df_merge['D_scheme1'] if 'D_scheme1' in df_merge.columns else np.nan),
        'D_scheme2': df_merge['D_scheme2'],
    })
    df_qc['s2_minus_orig'] = df_qc['D_scheme2'] - df_qc['D_original']
    df_qc['s2_div_orig'] = df_qc['D_scheme2'] / df_qc['D_original']
    if 'D_scheme1' in df_qc.columns:
        df_qc['s2_minus_s1'] = df_qc['D_scheme2'] - df_qc['D_scheme1']
        df_qc['s2_div_s1'] = df_qc['D_scheme2'] / df_qc['D_scheme1']

    out_df = df_merge.copy()
    out_df['D'] = out_df['D_scheme2']

    cols = list(df_ts.columns)
    out_cols = [c for c in cols if c in out_df.columns]
    out_df = out_df[out_cols]

    if out_df['D'].isna().any():
        missing = int(out_df['D'].isna().sum())
        raise RuntimeError(f'Scheme2 depth missing for {missing} months; abort to avoid writing invalid timeseries')

    os.makedirs(os.path.dirname(out_timeseries_csv) or '.', exist_ok=True)
    out_df.to_csv(out_timeseries_csv, index=False, encoding='utf-8-sig')

    os.makedirs(os.path.dirname(out_qc_csv) or '.', exist_ok=True)
    df_qc.to_csv(out_qc_csv, index=False, encoding='utf-8-sig')

    print(f'Wrote: {out_timeseries_csv}')
    print(f'Wrote: {out_qc_csv}')

    def _summary(col: str) -> str:
        x = pd.to_numeric(df_qc[col], errors='coerce').dropna().values
        if x.size == 0:
            return 'empty'
        return f"min={float(np.min(x)):.3f}  p50={float(np.median(x)):.3f}  max={float(np.max(x)):.3f}"

    print('D_original:', _summary('D_original'))
    if 'D_scheme1' in df_qc.columns:
        print('D_scheme1:', _summary('D_scheme1'))
    print('D_scheme2:', _summary('D_scheme2'))


if __name__ == '__main__':
    main()
