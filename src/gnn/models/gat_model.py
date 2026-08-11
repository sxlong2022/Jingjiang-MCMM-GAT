"""
Graph Attention Network (GAT) for spatially-varying E_rate prediction.

Architecture:
- Multi-head GAT layers for learning node representations
- Output: E_rate correction factor per node (positive, centered around 1.0)

v3 update: support along-channel features (width, depth) as node-level time-varying features

Reference: EGUsphere 2025 - GAT for hydrological prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class MigrationGAT(nn.Module):
    """
    GAT model for predicting E_rate correction factors.
    
    Args:
        in_channels: Number of input features per node
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (1 for E_rate correction)
        heads: Number of attention heads
        dropout: Dropout probability
        num_layers: Number of GAT layers
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        heads: int = 4,
        dropout: float = 0.3,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Input layer
        self.conv_in = GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # Hidden layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(
                hidden_channels * heads,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                concat=True
            ))
        
        # Output layer (single head, no concat)
        self.conv_out = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        # Final linear layer
        self.lin = nn.Linear(hidden_channels, out_channels)
        
        # Batch normalization
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels * heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            return_attention: If True, return attention weights
            
        Returns:
            correction: E_rate correction factors [N, 1], positive values
            attention_weights: (optional) Attention weights from last layer
        """
        attention_weights = None
        
        # Input layer
        x = self.conv_in(x, edge_index)
        x = self.bns[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i + 1](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        if return_attention:
            x, (edge_index_out, attention_weights) = self.conv_out(
                x, edge_index, return_attention_weights=True
            )
        else:
            x = self.conv_out(x, edge_index)
        
        x = F.elu(x)
        
        # Final linear + activation
        x = self.lin(x)
        
        # Softplus ensures positive output, add small constant to center around 1.0
        correction = F.softplus(x) + 0.1
        
        if return_attention:
            return correction, attention_weights
        return correction
    
    def get_attention_weights(self, x, edge_index):
        """Get attention weights for visualization."""
        _, attention = self.forward(x, edge_index, return_attention=True)
        return attention


class MigrationGATSimple(nn.Module):
    """
    Simplified GAT model for small datasets.
    Directly predict migration instead of correction factor.
    
    v3 update: support along-channel features (width, depth) as node-level time-varying features
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        heads: int = 2,
        dropout: float = 0.5,
        use_width: bool = False,
        use_depth: bool = False,
        allow_negative: bool = False
    ):
        super().__init__()
        
        self.dropout = dropout
        self.use_width = use_width
        self.use_depth = use_depth
        self.allow_negative = allow_negative
        
        # Compute extra along-channel feature dimensions
        extra_dim = 0
        if use_width:
            extra_dim += 1
        if use_depth:
            extra_dim += 1
        
        self.extra_dim = extra_dim
        total_in = in_channels + extra_dim
        
        self.conv1 = GATConv(total_in, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.lin = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index, x_width=None, x_depth=None):
        """
        Args:
            x: [N, in_channels] static + hydrological features
            edge_index: [2, E]
            x_width: [N] along-channel width for this month (optional)
            x_depth: [N] along-channel depth for this month (optional)
        """
        # Concatenate along-channel features
        features = [x]
        if self.use_width and x_width is not None:
            features.append(x_width.unsqueeze(-1))  # [N, 1]
        if self.use_depth and x_depth is not None:
            features.append(x_depth.unsqueeze(-1))  # [N, 1]
        
        if len(features) > 1:
            x = torch.cat(features, dim=-1)
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = self.lin(x)
        if self.allow_negative:
            return x
        return F.softplus(x)


class MigrationGATEnhanced(nn.Module):
    """
    Enhanced GAT model with:
    - 3 GAT layers with residual connections
    - Larger capacity (hidden=64, heads=4)
    - LayerNorm for stability
    - Hydro-feature modulation (FiLM-style)
    
    v3 update: support along-channel features (width, depth)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.3,
        num_static: int = 14,
        num_hydro: int = 7,
        use_width: bool = False,
        use_depth: bool = False,
        allow_negative: bool = False,
        extra_node_dim: int = 0
    ):
        super().__init__()
        
        self.dropout = dropout
        self.num_static = num_static
        self.num_hydro = num_hydro
        self.use_width = use_width
        self.use_depth = use_depth
        self.allow_negative = allow_negative
        self.extra_node_dim = int(extra_node_dim) if extra_node_dim else 0
        
        # Along-channel feature dimensions
        extra_dim = 0
        if use_width:
            extra_dim += 1
        if use_depth:
            extra_dim += 1
        extra_dim += self.extra_node_dim
        self.extra_dim = extra_dim
        
        # FiLM: modulate static features with hydrological features
        self.film_gamma = nn.Linear(num_hydro, num_static)
        self.film_beta = nn.Linear(num_hydro, num_static)
        
        # Input projection (modulated static features + along-channel features)
        self.input_proj = nn.Linear(num_static + extra_dim, hidden_channels)
        
        # GAT layers with residual
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        
        # Residual projections
        self.res1 = nn.Linear(hidden_channels, hidden_channels * heads)
        self.res2 = nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.res3 = nn.Linear(hidden_channels * heads, hidden_channels)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_channels * heads)
        self.ln2 = nn.LayerNorm(hidden_channels * heads)
        self.ln3 = nn.LayerNorm(hidden_channels)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x, edge_index, x_width=None, x_depth=None):
        """
        Args:
            x: [N, num_static + num_hydro] concatenated features
            edge_index: [2, E]
            x_width: [N] along-channel width for this month (optional)
            x_depth: [N] along-channel depth for this month (optional)
        """
        # Separate static and hydrological features
        x_static = x[:, :self.num_static]  # [N, 14]
        x_hydro = x[:, self.num_static:self.num_static + self.num_hydro]   # [N, 7] (identical across all nodes)
        x_extra = x[:, self.num_static + self.num_hydro:]
        
        # FiLM modulation: let hydrological features influence static feature representation
        gamma = self.film_gamma(x_hydro)  # [N, 14]
        beta = self.film_beta(x_hydro)    # [N, 14]
        x_mod = gamma * x_static + beta   # [N, 14]
        
        # Concatenate along-channel features
        features = [x_mod]
        if self.extra_node_dim > 0 and x_extra is not None and x_extra.numel() > 0:
            features.append(x_extra)
        if self.use_width and x_width is not None:
            features.append(x_width.unsqueeze(-1))
        if self.use_depth and x_depth is not None:
            features.append(x_depth.unsqueeze(-1))
        
        if len(features) > 1:
            x_mod = torch.cat(features, dim=-1)
        
        # Input projection
        h = self.input_proj(x_mod)  # [N, hidden]
        
        # GAT Layer 1 + Residual
        h1 = self.conv1(h, edge_index)
        h1 = self.ln1(h1 + self.res1(h))
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        # GAT Layer 2 + Residual
        h2 = self.conv2(h1, edge_index)
        h2 = self.ln2(h2 + self.res2(h1))
        h2 = F.elu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        # GAT Layer 3 + Residual
        h3 = self.conv3(h2, edge_index)
        h3 = self.ln3(h3 + self.res3(h2))
        h3 = F.elu(h3)
        
        # Output
        out = self.output(h3)
        if self.allow_negative:
            return out
        return F.softplus(out)


def create_model(in_channels, model_type='simple', **kwargs):
    """Factory function to create GAT model.
    
    Args:
        in_channels: Input feature dimensions (static + hydrological)
        model_type: 'simple' or 'enhanced'
        **kwargs: Extra parameters including use_width, use_depth, etc.
    """
    if model_type == 'simple':
        return MigrationGATSimple(in_channels, **kwargs)
    elif model_type == 'enhanced':
        return MigrationGATEnhanced(in_channels, **kwargs)
    elif model_type == 'full':
        return MigrationGAT(in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Test
if __name__ == '__main__':
    # Test model
    in_channels = 15
    num_nodes = 100
    num_edges = 300
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Simple model
    model = MigrationGATSimple(in_channels)
    out = model(x, edge_index)
    print(f"Simple model output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Full model
    model_full = MigrationGAT(in_channels)
    out_full, attn = model_full(x, edge_index, return_attention=True)
    print(f"Full model output shape: {out_full.shape}")
    print(f"Attention weights shape: {attn.shape}")
