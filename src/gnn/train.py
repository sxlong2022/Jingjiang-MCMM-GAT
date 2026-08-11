"""
Monthly Migration CV Training with Enhanced Hydro Features.

Core improvements v3:
- Enhanced hydro features: 7-dim (Q, Q_prev, dQ, Q_cumsum3, Qs, season_sin, season_cos)
- Along-channel features: width W(s), depth D(s) as node-level time-varying features
- Monthly sample training (~47 samples)
- Model learns response relationship: hydro conditions + local channel morphology -> migration

Fold design:
  Fold 1: Train 2016-2018 (~35 samples), Val 2019 (11 samples)
  Fold 2: Train 2016-2019 (~47 samples), Val 2020 (11 samples)

Usage:
    python train_cv.py --fold 1
    python train_cv.py --fold 2
    python train_cv.py --fold 1 --use_width  # Enable along-channel width
    python train_cv.py --fold 1 --use_width --use_depth  # Enable width+depth
"""
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from datetime import datetime
import random
import os

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / 'src' / 'gnn'))

from models.gat_model import MigrationGATSimple, MigrationGATEnhanced
from models.loss_functions import combined_loss

GRAPH_DIR = REPO_ROOT / 'data' / 'graph' / 'cv_monthly'
OUTPUT_DIR = REPO_ROOT / 'outputs' / 'cv_monthly_results'


def resolve_device(device: str) -> torch.device:
    device = str(device).lower().strip()
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    return torch.device(device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        return self.early_stop


def load_fold_data(fold_num, use_width=False, use_depth=False, data_dir=None, tag=None):
    """Load training and validation data for a fold.
    
    Args:
        fold_num: Fold number (1 or 2)
        use_width: Whether to load along-channel width data
        use_depth: Whether to load along-channel depth data
        data_dir: Data directory (default: GRAPH_DIR)
        tag: Optional tag for matching fold{n}_train_{tag}*.pt
    """
    graph_dir = Path(data_dir) if data_dir else GRAPH_DIR
    tag_part = f"_{tag}" if tag else ""

    # Build filename suffix
    suffix = ""
    if use_width:
        suffix += "_width"
    if use_depth:
        suffix += "_depth"
    
    # Try suffixed files first, fall back to base files
    train_file = graph_dir / f"fold{fold_num}_train{tag_part}{suffix}.pt"
    val_file = graph_dir / f"fold{fold_num}_val{tag_part}{suffix}.pt"
    
    # If suffixed files don't exist, try base files
    if not train_file.exists():
        train_file = graph_dir / f"fold{fold_num}_train{tag_part}.pt"
        val_file = graph_dir / f"fold{fold_num}_val{tag_part}.pt"
 
    if not train_file.exists() and tag_part:
        train_file = graph_dir / f"fold{fold_num}_train{suffix}.pt"
        val_file = graph_dir / f"fold{fold_num}_val{suffix}.pt"

    if not train_file.exists() and tag_part:
        train_file = graph_dir / f"fold{fold_num}_train.pt"
        val_file = graph_dir / f"fold{fold_num}_val.pt"

    if (not train_file.exists() or not val_file.exists()) and tag_part:
        train_candidates = sorted(graph_dir.glob(f"fold{fold_num}_train{tag_part}*.pt"))

        # When user explicitly enables along-channel features, prefer candidates with matching suffix
        preferred = []
        if suffix:
            preferred = [tc for tc in train_candidates if tc.stem.endswith(suffix)]
        candidate_order = preferred + [tc for tc in train_candidates if tc not in preferred]

        for tc in candidate_order:
            vc = Path(str(tc).replace(f"fold{fold_num}_train", f"fold{fold_num}_val"))
            if vc.exists():
                train_file = tc
                val_file = vc
                break

    # Decide whether to warn about fallback to base data
    if (use_width or use_depth) and suffix:
        if not (train_file.stem.endswith(suffix) and val_file.stem.endswith(suffix)):
            print("  Note: No data files with along-channel features found, using base data")

    if tag_part or use_width or use_depth:
        print(f"  Loading data file: {train_file.name}")
    
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            f"Fold {fold_num} data not found. Run:\n"
            f"  python scripts/graph_construction/build_graph_cv.py --fold {fold_num}"
        )
    
    train_data = torch.load(train_file, weights_only=False)
    val_data = torch.load(val_file, weights_only=False)
    
    return train_data, val_data


def get_sample_features(data, sample_idx, use_width=False, use_depth=False, use_mcmm=False):
    """
    Get complete features for specified sample (static + hydrological + along-channel).
    
    Args:
        data: Data object with x_static, x_hydro, and optionally x_width, x_depth
        sample_idx: Sample index
        use_width: Whether to return along-channel width
        use_depth: Whether to return along-channel depth
    
    Returns:
        x: [num_nodes, num_features] complete features (14 static + 7 hydro = 21 dims)
        x_width: [num_nodes] along-channel width (if use_width=True and data exists)
        x_depth: [num_nodes] along-channel depth (if use_depth=True and data exists)
    """
    # Static features [num_nodes, 14]
    x_static = data.x_static
    
    # Hydrological features [7] -> broadcast to all nodes [num_nodes, 7]
    hydro_feat = data.x_hydro[sample_idx]  # [7]
    x_hydro = hydro_feat.unsqueeze(0).expand(data.num_nodes, -1)  # [num_nodes, 7]
    
    # Concatenate static + hydrological
    x = torch.cat([x_static, x_hydro], dim=1)  # [num_nodes, 21]

    if use_mcmm and hasattr(data, 'x_mcmm') and data.x_mcmm is not None:
        mcmm_feat = data.x_mcmm[sample_idx].unsqueeze(1)  # [num_nodes, 1]
        x = torch.cat([x, mcmm_feat], dim=1)

    if use_depth and hasattr(data, 'x_depth_mask') and data.x_depth_mask is not None:
        depth_mask_feat = data.x_depth_mask[sample_idx].unsqueeze(1)  # [num_nodes, 1]
        x = torch.cat([x, depth_mask_feat], dim=1)
    
    # Along-channel features
    x_width_out = None
    x_depth_out = None
    
    if use_width and hasattr(data, 'x_width') and data.x_width is not None:
        x_width_out = data.x_width[sample_idx]  # [num_nodes]
    
    if use_depth and hasattr(data, 'x_depth') and data.x_depth is not None:
        x_depth_out = data.x_depth[sample_idx]  # [num_nodes]
    
    return x, x_width_out, x_depth_out


def _get_target_name(data):
    if hasattr(data, 'target_name') and data.target_name is not None:
        return str(data.target_name)
    return 'obs'


def _get_obs_and_mcmm(data, sample_idx):
    y_target = data.y[sample_idx]
    y_obs = data.y_obs[sample_idx] if hasattr(data, 'y_obs') and data.y_obs is not None else y_target
    y_mcmm = data.y_mcmm[sample_idx] if hasattr(data, 'y_mcmm') and data.y_mcmm is not None else None
    return y_target, y_obs, y_mcmm


def _restore_pred_to_obs(pred, y_mcmm, target_name):
    if target_name == 'residual' and y_mcmm is not None:
        return pred + y_mcmm
    if target_name == 'ratio' and y_mcmm is not None:
        return pred * (y_mcmm + 1e-6)
    if target_name == 'dn':
        return pred
    return pred


def _get_Q_physical(data, sample_idx):
    if hasattr(data, 'Q_values') and data.Q_values is not None:
        try:
            return float(data.Q_values[sample_idx])
        except Exception:
            pass
    if hasattr(data, 'hydro_stats') and data.hydro_stats is not None and hasattr(data, 'x_hydro'):
        hs = data.hydro_stats
        if isinstance(hs, dict) and 'Q_mean' in hs and 'Q_std' in hs:
            try:
                return float(data.x_hydro[sample_idx, 0].item() * float(hs['Q_std']) + float(hs['Q_mean']))
            except Exception:
                return float('nan')
    return float('nan')


def _evaluate_mcmm_baseline(data):
    target_name = _get_target_name(data)
    if target_name == 'dn':
        return None
    if not (hasattr(data, 'y_obs') and data.y_obs is not None and hasattr(data, 'y_mcmm') and data.y_mcmm is not None):
        return None

    correlations = []
    rmses = []
    for idx in range(data.num_samples):
        y_obs = data.y_obs[idx]
        y_mcmm = data.y_mcmm[idx]

        pred_c = y_mcmm - y_mcmm.mean()
        obs_c = y_obs - y_obs.mean()
        corr = (torch.sum(pred_c * obs_c) / (torch.sqrt(torch.sum(pred_c**2) * torch.sum(obs_c**2)) + 1e-8)).item()
        correlations.append(corr)

        rmse = torch.sqrt(torch.mean((y_mcmm - y_obs) ** 2)).item()
        rmses.append(rmse)

    return {
        'correlation': float(np.mean(correlations)),
        'correlation_std': float(np.std(correlations)),
        'rmse': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
    }


def train_epoch(model, data, optimizer, config):
    """
    Train for one epoch over all monthly samples.
    Directly predict migration using migration-weighted loss + physics constraints.
    """
    model.train()
    total_loss = 0.0
    num_samples = data.num_samples
    
    use_width = config.get('use_width', False)
    use_depth = config.get('use_depth', False)
    use_mcmm = config.get('use_mcmm', False)

    target_name = _get_target_name(data)
    
    # Compute sample weights: weighted by migration magnitude
    if hasattr(data, 'y_obs') and data.y_obs is not None:
        if target_name == 'dn':
            mean_migrations = data.y_obs.abs().mean(dim=1)
        else:
            mean_migrations = data.y_obs.mean(dim=1)
    else:
        mean_migrations = data.y.mean(dim=1)  # [N_samples]
    weight_threshold = config.get('weight_threshold', 20.0)
    sample_weights = torch.clamp(mean_migrations / weight_threshold, min=0.3, max=2.0)
    
    # Get curvature data (for physics constraints)
    # Curvature is the 2nd column of static features (abs_curvature)
    curvature = data.x_static[:, 1]  # [num_nodes]
    
    # Randomly shuffle sample order
    indices = torch.randperm(num_samples)
    
    for idx in indices:
        optimizer.zero_grad()
        
        # Get features and targets for this month
        x, x_width, x_depth = get_sample_features(data, idx, use_width, use_depth, use_mcmm)
        _, y_obs, y_mcmm = _get_obs_and_mcmm(data, idx)
        weight = sample_weights[idx].item()
        
        # Get normalized discharge for this month
        Q_norm = data.x_hydro[idx, 0].item()  # First hydro feature is normalized Q
        
        # Forward - directly predict migration (pass along-channel features)
        pred = model(x, data.edge_index, x_width=x_width, x_depth=x_depth).squeeze()  # [num_nodes]

        pred_obs = _restore_pred_to_obs(pred, y_mcmm, target_name)
        
        # Loss - weighted loss + physics constraints
        loss, loss_dict = combined_loss(
            pred_obs, y_obs,
            lambda_smooth=config['lambda_smooth'],
            lambda_corr=config['lambda_corr'],
            sample_weight=weight,
            curvature=curvature,
            Q_norm=Q_norm,
            lambda_physics=config.get('lambda_physics', 0.1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss_dict['total']
    
    return {'total': total_loss / num_samples}


def evaluate(model, data, config=None):
    """
    Evaluate model on all samples, return per-sample and average metrics.
    Direct migration prediction version.
    """
    model.eval()
    correlations = []
    rmses = []
    
    use_width = config.get('use_width', False) if config else False
    use_depth = config.get('use_depth', False) if config else False
    use_mcmm = config.get('use_mcmm', False) if config else False

    target_name = _get_target_name(data)
    
    with torch.no_grad():
        for idx in range(data.num_samples):
            x, x_width, x_depth = get_sample_features(data, idx, use_width, use_depth, use_mcmm)
            _, y_obs, y_mcmm = _get_obs_and_mcmm(data, idx)
            
            # Direct prediction (pass along-channel features)
            pred = model(x, data.edge_index, x_width=x_width, x_depth=x_depth).squeeze()

            pred_obs = _restore_pred_to_obs(pred, y_mcmm, target_name)
            
            # Spatial correlation
            pred_c = pred_obs - pred_obs.mean()
            obs_c = y_obs - y_obs.mean()
            corr = (torch.sum(pred_c * obs_c) / 
                    (torch.sqrt(torch.sum(pred_c**2) * torch.sum(obs_c**2)) + 1e-8)).item()
            correlations.append(corr)
            
            # RMSE
            rmse = torch.sqrt(torch.mean((pred_obs - y_obs)**2)).item()
            rmses.append(rmse)
    
    return {
        'correlation': np.mean(correlations),
        'correlation_std': np.std(correlations),
        'correlations': correlations,
        'rmse': np.mean(rmses),
        'rmse_std': np.std(rmses)
    }


def train_fold(fold_num, config, verbose=True):
    """Train and evaluate one CV fold with monthly samples."""
    device = resolve_device(config.get('device', 'cpu'))

    use_width = config.get('use_width', False)
    use_depth = config.get('use_depth', False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}: Monthly Migration Training")
        feature_info = []
        if use_width:
            feature_info.append("along-channel width")
        if use_depth:
            feature_info.append("along-channel depth")
        if feature_info:
            print(f"Along-channel features: {', '.join(feature_info)}")
        print(f"{'='*60}")
    
    # Load data
    train_data, val_data = load_fold_data(
        fold_num,
        use_width,
        use_depth,
        data_dir=config.get('data_dir'),
        tag=config.get('tag')
    )

    if device.type == 'cuda':
        try:
            train_data = train_data.to(device)
            val_data = val_data.to(device)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to move data to GPU, falling back to CPU. Error: {e}")
            device = torch.device('cpu')
            config['device'] = str(device)
            train_data = train_data.to(device)
            val_data = val_data.to(device)
    else:
        train_data = train_data.to(device)
        val_data = val_data.to(device)
    
    # Check if along-channel features exist
    has_width = hasattr(train_data, 'x_width') and train_data.x_width is not None
    has_depth = hasattr(train_data, 'x_depth') and train_data.x_depth is not None
    has_mcmm = hasattr(train_data, 'x_mcmm') and train_data.x_mcmm is not None
    
    if use_width and not has_width:
        print("  Warning: Data does not contain along-channel width, disabling this feature")
        config['use_width'] = False
        use_width = False
    
    if use_depth and not has_depth:
        print("  Warning: Data does not contain along-channel depth, disabling this feature")
        config['use_depth'] = False
        use_depth = False

    use_mcmm = config.get('use_mcmm', False)
    if use_mcmm and not has_mcmm:
        print("  Warning: Data does not contain MCMM node feature x_mcmm, disabling this feature")
        config['use_mcmm'] = False
        use_mcmm = False
    
    if verbose:
        print(f"Training set: {train_data.num_samples} monthly samples")
        print(f"  Month range: {train_data.months[0]} - {train_data.months[-1]}")
        y_train_obs_mean = (train_data.y_obs.mean() if hasattr(train_data, 'y_obs') and train_data.y_obs is not None else train_data.y.mean())
        target_name = _get_target_name(train_data)
        if target_name == 'dn':
            print(f"  dn: mean={y_train_obs_mean:.2f}m")
        else:
            print(f"  Migration: mean={y_train_obs_mean:.2f}m")
        if has_width:
            w_mean = getattr(train_data, 'width_mean', float('nan'))
            w_std = getattr(train_data, 'width_std', float('nan'))
            print(f"  Along-channel width: mean={w_mean:.1f}m, std={w_std:.1f}m")
        if has_depth:
            d_mean = getattr(train_data, 'depth_mean', float('nan'))
            d_std = getattr(train_data, 'depth_std', float('nan'))
            print(f"  Along-channel depth: mean={d_mean:.2f}m, std={d_std:.2f}m")
        print(f"Validation set: {val_data.num_samples} monthly samples")
        print(f"  Month range: {val_data.months[0]} - {val_data.months[-1]}")
        print(f"Device: {device}")
        y_val_obs_mean = (val_data.y_obs.mean() if hasattr(val_data, 'y_obs') and val_data.y_obs is not None else val_data.y.mean())
        if target_name == 'dn':
            print(f"  dn: mean={y_val_obs_mean:.2f}m")
        else:
            print(f"  Migration: mean={y_val_obs_mean:.2f}m")

        baseline_train = _evaluate_mcmm_baseline(train_data)
        baseline_val = _evaluate_mcmm_baseline(val_data)
        if baseline_train is not None and baseline_val is not None:
            print(f"  MCMM baseline (train): corr={baseline_train['correlation']:.3f}±{baseline_train['correlation_std']:.3f}, rmse={baseline_train['rmse']:.2f}m")
            print(f"  MCMM baseline (val): corr={baseline_val['correlation']:.3f}±{baseline_val['correlation_std']:.3f}, rmse={baseline_val['rmse']:.2f}m")
    
    # Initialize model
    # Feature dimensions: 14 static + 7 hydro = 21 (+ optional node-level extras: MCMM, depth_mask)
    num_static = train_data.x_static.shape[1]
    num_hydro = train_data.x_hydro.shape[1]
    use_depth_mask = bool(use_depth and hasattr(train_data, 'x_depth_mask') and train_data.x_depth_mask is not None)
    extra_node_dim = (1 if use_mcmm else 0) + (1 if use_depth_mask else 0)
    in_channels = num_static + num_hydro + extra_node_dim
    config['use_depth_mask'] = use_depth_mask
    
    model_type = config.get('model_type', 'simple')

    target_name = _get_target_name(train_data)
    allow_negative = (target_name == 'residual' or target_name == 'dn')
    
    if model_type == 'enhanced':
        model = MigrationGATEnhanced(
            in_channels,
            hidden_channels=config['hidden_channels'],
            heads=config['heads'],
            dropout=config['dropout'],
            num_static=num_static,
            num_hydro=num_hydro,
            use_width=use_width,
            use_depth=use_depth,
            allow_negative=allow_negative,
            extra_node_dim=extra_node_dim
        )
    else:
        model = MigrationGATSimple(
            in_channels,
            hidden_channels=config['hidden_channels'],
            heads=config['heads'],
            dropout=config['dropout'],
            use_width=use_width,
            use_depth=use_depth,
            allow_negative=allow_negative
        )

    model = model.to(device)
    
    if verbose:
        num_params = sum(p.numel() for p in model.parameters())
        extra_info = []
        if use_width:
            extra_info.append("+W")
        if use_depth:
            extra_info.append("+D")
        if use_mcmm:
            extra_info.append("+M")
        extra_str = "".join(extra_info) if extra_info else ""
        print(f"\nModel: {model_type.upper()}{extra_str}, input_dims={in_channels} ({num_static} static + {num_hydro} hydro), params={num_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=0.001)
    
    # Training loop
    history = []
    best_epoch = 0
    
    for epoch in range(config['epochs']):
        loss_dict = train_epoch(model, train_data, optimizer, config)
        train_metrics = evaluate(model, train_data, config)
        val_metrics = evaluate(model, val_data, config)
        
        scheduler.step(loss_dict['total'])
        
        history.append({
            'epoch': epoch,
            'train_corr': train_metrics['correlation'],
            'train_corr_std': train_metrics['correlation_std'],
            'val_corr': val_metrics['correlation'],
            'val_corr_std': val_metrics['correlation_std'],
            'train_rmse': train_metrics['rmse'],
            'val_rmse': val_metrics['rmse'],
            **loss_dict
        })
        
        # Early stopping based on validation correlation
        if early_stopping(val_metrics['correlation'], model):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
        
        if val_metrics['correlation'] == early_stopping.best_score:
            best_epoch = epoch
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: train_corr={train_metrics['correlation']:.3f}±{train_metrics['correlation_std']:.3f}, "
                  f"val_corr={val_metrics['correlation']:.3f}±{val_metrics['correlation_std']:.3f}")
    
    # Load best model
    model.load_state_dict(early_stopping.best_model_state)
    final_train = evaluate(model, train_data, config)
    final_val = evaluate(model, val_data, config)
    
    results = {
        'fold': fold_num,
        'best_epoch': best_epoch,
        'train_samples': train_data.num_samples,
        'val_samples': val_data.num_samples,
        'train_corr': final_train['correlation'],
        'train_corr_std': final_train['correlation_std'],
        'val_corr': final_val['correlation'],
        'val_corr_std': final_val['correlation_std'],
        'train_rmse': final_train['rmse'],
        'val_rmse': final_val['rmse'],
        'val_correlations': final_val['correlations'],
        'val_months': val_data.months,
        'use_width': use_width,
        'use_depth': use_depth,
        'use_depth_mask': use_depth_mask,
        'history': history
    }
    
    if verbose:
        print(f"\nFold {fold_num} results (best @ epoch {best_epoch}):")
        print(f"  Train: corr={final_train['correlation']:.3f}±{final_train['correlation_std']:.3f}")
        print(f"  Val: corr={final_val['correlation']:.3f}±{final_val['correlation_std']:.3f}")
        print(f"  Val RMSE: {final_val['rmse']:.2f}m")
        
        # Print per-month validation correlations
        print(f"\n  Per-month validation correlations:")
        for month, corr in zip(val_data.months, final_val['correlations']):
            midx = val_data.months.index(month)
            Q = _get_Q_physical(val_data, midx)
            if np.isfinite(Q):
                print(f"    {month}: corr={corr:.3f}, Q={Q:.0f}")
            else:
                print(f"    {month}: corr={corr:.3f}")
    
    return results, model


def run_cv(folds, config):
    """Run cross-validation for specified folds."""
    out_dir = Path(config.get('out_dir')) if config.get('out_dir') else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for fold_num in folds:
        try:
            # Use config copy per fold to prevent one fold's missing data from modifying config for subsequent folds
            fold_config = dict(config)
            results, model = train_fold(fold_num, fold_config)
            all_results.append(results)
            
            # Save fold model
            model_file = out_dir / f"fold{fold_num}_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': fold_config,
                'results': {k: v for k, v in results.items() if k not in ['history', 'val_correlations']}
            }, model_file)
            
        except FileNotFoundError as e:
            print(f"\n[WARNING] Fold {fold_num} data not found: {e}")
            continue
    
    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("CV Summary (Monthly Migration)")
        print(f"{'='*70}")
        print(f"{'Fold':<6} {'Train':<8} {'Val':<8} {'Train Corr':<18} {'Val Corr':<18} {'Val RMSE':<10}")
        print("-" * 70)
        
        val_corrs = []
        for r in all_results:
            print(f"{r['fold']:<6} {r['train_samples']:<8} {r['val_samples']:<8} "
                  f"{r['train_corr']:.3f}±{r['train_corr_std']:.3f}       "
                  f"{r['val_corr']:.3f}±{r['val_corr_std']:.3f}       "
                  f"{r['val_rmse']:.2f}m")
            val_corrs.append(r['val_corr'])
        
        print("-" * 70)
        print(f"Mean Val Corr: {np.mean(val_corrs):.3f} ± {np.std(val_corrs):.3f}")
        
        # Target check
        target_corr = 0.7
        mean_corr = np.mean(val_corrs)
        
        print(f"\nTarget check:")
        print(f"  Spatial correlation >0.7: {'[PASS]' if mean_corr > target_corr else '[FAIL]'} ({mean_corr:.3f})")
        
        # Save summary
        summary_config = dict(config)
        summary_config['use_depth_mask'] = bool(any(r.get('use_depth_mask', False) for r in all_results))
        summary_config['use_depth_mask_all_folds'] = bool(all(r.get('use_depth_mask', False) for r in all_results))
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': summary_config,
            'folds': [{k: v for k, v in r.items() if k not in ['history', 'val_correlations']} 
                      for r in all_results],
            'mean_val_corr': float(np.mean(val_corrs)),
            'std_val_corr': float(np.std(val_corrs))
        }
        
        summary_file = out_dir / 'cv_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved: {summary_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Monthly Migration CV Training')
    parser.add_argument('--fold', nargs='+', type=int, default=[1, 2],
                        help='Folds to run (1 and/or 2)')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'enhanced'],
                        help='Model type: simple (2-layer) or enhanced (3-layer with FiLM)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                        help='Weight for physics constraint loss (curvature-migration correlation)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Optional directory containing fold*.pt (default: GNN/data/graph/cv_monthly)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Optional output directory for results (default: GNN/outputs/cv_monthly_results)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional tag used in filenames, e.g., fold1_train_<tag>_width.pt')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed for reproducibility (e.g., 0)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (auto=use cuda if available)')
    # Along-channel feature arguments
    parser.add_argument('--use_width', action='store_true',
                        help='Use along-channel width as node-level time-varying feature')
    parser.add_argument('--use_depth', action='store_true',
                        help='Use along-channel depth as node-level time-varying feature')
    parser.add_argument('--use_mcmm', action='store_true',
                        help='Use MCMM baseline migration as node-level time-varying feature (requires x_mcmm in dataset)')
    args = parser.parse_args()
    
    # Enhanced model uses larger default parameters
    if args.model == 'enhanced':
        hidden = args.hidden if args.hidden != 32 else 64
        heads = args.heads if args.heads != 2 else 4
    else:
        hidden = args.hidden
        heads = args.heads
    
    config = {
        'model_type': args.model,
        'epochs': args.epochs,
        'patience': args.patience,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'hidden_channels': hidden,
        'heads': heads,
        'dropout': args.dropout,
        'lambda_smooth': 0.05,
        'lambda_corr': 1.0,
        'lambda_physics': args.lambda_physics,
        'data_dir': args.data_dir,
        'out_dir': args.out_dir,
        'tag': args.tag,
        'seed': args.seed,
        'device': str(resolve_device(args.device)),
        'use_width': args.use_width,
        'use_depth': args.use_depth,
        'use_mcmm': args.use_mcmm
    }

    if args.seed is not None:
        set_seed(args.seed)
    
    print("="*70)
    print("Monthly Migration CV Training (Physics-Informed)")
    print("="*70)
    print(f"Folds: {args.fold}")
    print(f"Model: {args.model} (hidden={hidden}, heads={heads})")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Device: {config['device']}")
    
    feature_info = []
    if args.use_width:
        feature_info.append("along-channel width")
    if args.use_depth:
        feature_info.append("along-channel depth")
    if args.use_mcmm:
        feature_info.append("MCMM migration field")
    if feature_info:
        print(f"Along-channel features: {', '.join(feature_info)}")
    else:
        print("Along-channel features: none (baseline configuration)")
    
    print(f"Config: epochs={config['epochs']}, patience={config['patience']}, "
          f"lr={config['lr']}, lambda_physics={config['lambda_physics']}")
    
    run_cv(args.fold, config)


if __name__ == '__main__':
    main()
