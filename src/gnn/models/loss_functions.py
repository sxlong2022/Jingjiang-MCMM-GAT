"""
Physics-informed loss functions for GAT training.

Loss = MSE(predicted_migration, observed_migration) 
     + lambda_smooth * spatial_smoothness
     + lambda_reg * L2_regularization
"""

import torch
import torch.nn.functional as F


def physics_informed_loss(
    pred_migration,
    obs_migration,
    correction_factors,
    lambda_smooth=0.1,
    lambda_magnitude=0.01
):
    """
    Physics-informed loss function.
    
    Args:
        pred_migration: Predicted migration [N]
        obs_migration: Observed migration [N]
        correction_factors: E_rate correction factors [N]
        lambda_smooth: Weight for spatial smoothness regularization
        lambda_magnitude: Weight for correction magnitude regularization
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    # Primary loss: migration prediction error
    mse_loss = F.mse_loss(pred_migration, obs_migration)
    
    # Spatial smoothness: penalize large differences between adjacent nodes
    smoothness_loss = torch.mean(
        (correction_factors[1:] - correction_factors[:-1])**2
    )
    
    # Magnitude regularization: keep corrections close to 1.0
    magnitude_loss = torch.mean((correction_factors - 1.0)**2)
    
    # Total loss
    total_loss = mse_loss + lambda_smooth * smoothness_loss + lambda_magnitude * magnitude_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'mse': mse_loss.item(),
        'smoothness': smoothness_loss.item(),
        'magnitude': magnitude_loss.item()
    }
    
    return total_loss, loss_dict


def relative_error_loss(pred_migration, obs_migration, epsilon=1e-6):
    """
    Relative error loss - better for values with different magnitudes.
    
    Loss = mean(|pred - obs| / (|obs| + epsilon))
    """
    rel_error = torch.abs(pred_migration - obs_migration) / (torch.abs(obs_migration) + epsilon)
    return torch.mean(rel_error)


def spatial_correlation_loss(pred_migration, obs_migration):
    """
    Loss based on spatial correlation.
    We want to maximize correlation, so loss = 1 - correlation.
    """
    pred_centered = pred_migration - pred_migration.mean()
    obs_centered = obs_migration - obs_migration.mean()
    
    numerator = torch.sum(pred_centered * obs_centered)
    denominator = torch.sqrt(
        torch.sum(pred_centered**2) * torch.sum(obs_centered**2)
    ) + 1e-8
    
    correlation = numerator / denominator
    return 1.0 - correlation


def physics_constraint_loss(pred_migration, curvature, Q_norm):
    """
    Physics constraint loss: migration should be positively correlated with curvature.

    Based on the core MCMM assumption: E_rate ∝ curvature * f(Q).
    Higher curvature locations should exhibit larger migration.

    Args:
        pred_migration: Predicted migration [N]
        curvature: Absolute curvature [N]
        Q_norm: Normalized discharge (scalar)

    Returns:
        physics_loss: Physics constraint loss (lower is better)
    """
    # Expected migration pattern: proportional to curvature
    expected_pattern = curvature * (1 + Q_norm)  # Discharge modulation
    
    # Compute correlation between prediction and physical expectation
    pred_c = pred_migration - pred_migration.mean()
    exp_c = expected_pattern - expected_pattern.mean()
    
    corr = torch.sum(pred_c * exp_c) / (
        torch.sqrt(torch.sum(pred_c**2) * torch.sum(exp_c**2)) + 1e-8
    )
    
    # Loss = 1 - correlation (encourage positive correlation)
    return 1.0 - corr


def combined_loss(
    pred_migration,
    obs_migration,
    correction_factors=None,
    lambda_smooth=0.1,
    lambda_corr=0.5,
    lambda_magnitude=0.0,
    sample_weight=1.0,
    curvature=None,
    Q_norm=0.0,
    W_norm=None,
    lambda_physics=0.0
):
    """
    Combined loss for direct migration prediction.

    Main loss components:
    - MSE: Prediction error (normalized)
    - Correlation: Spatial correlation (most important)
    - Smoothness: Spatial smoothness
    - Physics: Physics constraint (curvature-migration relationship)

    Args:
        sample_weight: Sample weight; months with larger migration get higher weight
        curvature: Node curvature for physics constraint
        Q_norm: Normalized discharge
        lambda_physics: Physics constraint weight
    """
    # MSE loss (normalized)
    obs_std = obs_migration.std() + 1e-6
    mse_loss = F.mse_loss(pred_migration / obs_std, obs_migration / obs_std)
    
    # Correlation loss
    corr_loss = spatial_correlation_loss(pred_migration, obs_migration)
    
    # Spatial smoothness (on predictions)
    smoothness_loss = torch.mean((pred_migration[1:] - pred_migration[:-1])**2) / (obs_std**2 + 1e-6)
    
    # Physics constraint loss
    physics_loss = 0.0
    if curvature is not None and lambda_physics > 0:
        physics_loss = physics_constraint_loss(pred_migration, curvature, Q_norm)
    
    # Weighted total loss
    total = sample_weight * (mse_loss + lambda_corr * corr_loss) + \
            lambda_smooth * smoothness_loss + \
            lambda_physics * physics_loss
    
    return total, {
        'total': total.item(),
        'mse': mse_loss.item(),
        'correlation': (1 - corr_loss).item(),
        'smoothness': smoothness_loss.item(),
        'physics': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss,
        'weight': sample_weight
    }
