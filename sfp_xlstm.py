"""
SFP-xLSTM: A fusion fault diagnosis framework of sparse focus modulation 
and xLSTM for vibration signals

Paper Implementation by: Based on Yubo Guan, Peng Li, Aiying Zhao, Shilin Wang
School of Automation and Electrical Engineering, Lanzhou Jiaotong University

This implementation follows the paper parameters:
- GAF Image Size: 256 × 256
- Encoding Method: GADF
- SFPM: 3 Focus Layers, Base Kernel Size 3, Dilation Rate Growth Factor 2, Sparsity Rate 0.3
- xLSTM: Hidden Dimension 256, 4 Memory Subspaces, Dropout 0.2
- Training: Batch Size 32, Learning Rate 0.003, 150 Epochs, Adam Optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# GADF Encoding Module
# ============================================================================
class GADFEncoder:
    """
    Gramian Angular Difference Field (GADF) Encoder
    
    Converts 1D vibration signals into 2D image representations.
    GADF is sensitive to high-frequency components and suitable for 
    capturing transient anomalies in bearing fault signals.
    
    Paper Reference: Section 2.1 - Data Pre-processing and GADF
    """
    
    def __init__(self, image_size: int = 256):
        """
        Args:
            image_size: Output image size (default: 256x256 as per paper)
        """
        self.image_size = image_size
    
    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal to [-1, 1] interval
        
        Formula (1) from paper:
        x̃_i = 2 * (x_i - max(S) + x_i - min(S)) / (max(S) - min(S)) - 1
        
        Simplified: x̃_i = 2 * (x_i - min(S)) / (max(S) - min(S)) - 1
        """
        s_min = np.min(signal)
        s_max = np.max(signal)
        if s_max - s_min == 0:
            return np.zeros_like(signal)
        x_normalized = 2 * (signal - s_min) / (s_max - s_min) - 1
        return np.clip(x_normalized, -1, 1)
    
    def polar_encoding(self, x_normalized: np.ndarray) -> np.ndarray:
        """
        Polar coordinate transformation
        
        Formula (2) from paper:
        θ_i = arccos(x̃_i), -1 ≤ x̃_i ≤ 1
        r_i = t_i / N
        """
        theta = np.arccos(x_normalized)
        return theta
    
    def encode(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert 1D signal to 2D GADF image
        
        Formula (3) from paper:
        G_GADF[i,j] = sin(θ_i - θ_j)
                    = sqrt(I - (x̃^T)²) · x̃ - x̃^T · sqrt(I - x̃²)
        
        Args:
            signal: 1D vibration signal of length n
            
        Returns:
            GADF matrix of size (image_size, image_size)
        """
        # Resample signal to image_size if needed
        if len(signal) != self.image_size:
            indices = np.linspace(0, len(signal) - 1, self.image_size).astype(int)
            signal = signal[indices]
        
        # Normalize to [-1, 1]
        x_normalized = self.normalize(signal)
        
        # Compute theta angles
        theta = self.polar_encoding(x_normalized)
        
        # Compute GADF matrix: sin(θ_i - θ_j)
        # Using identity: sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)  # This equals x_normalized
        
        # GADF[i,j] = sin(θ_i)cos(θ_j) - cos(θ_i)sin(θ_j)
        gadf = np.outer(sin_theta, cos_theta) - np.outer(cos_theta, sin_theta)
        
        return gadf
    
    def encode_to_grayscale(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert 1D signal to grayscale GADF image (as per paper)
        
        The paper converts RGB heatmaps to grayscale for:
        - Reduced storage requirements
        - Increased computational speed
        - Enhanced precision
        """
        gadf = self.encode(signal)
        
        # Normalize to [0, 255] for grayscale
        gadf_normalized = (gadf + 1) / 2 * 255
        grayscale = gadf_normalized.astype(np.uint8)
        
        return grayscale
    
    def batch_encode(self, signals: np.ndarray) -> torch.Tensor:
        """
        Batch encode multiple signals to GADF images
        
        Args:
            signals: Batch of 1D signals, shape (batch_size, signal_length)
            
        Returns:
            Batch of GADF images, shape (batch_size, 1, image_size, image_size)
        """
        batch_size = signals.shape[0]
        gadf_batch = np.zeros((batch_size, 1, self.image_size, self.image_size))
        
        for i in range(batch_size):
            gadf = self.encode(signals[i])
            gadf_batch[i, 0] = gadf
        
        return torch.FloatTensor(gadf_batch)


# ============================================================================
# Sparse Focal Point Modulation (SFPM) Module
# ============================================================================
class SFPM(nn.Module):
    """
    Sparse Focal Point Modulation Module
    
    Adaptively extracts multi-scale fault features from GADF images.
    Key features:
    - Multi-scale focal modulation for capturing spatial context
    - Adaptive Top-K sparsification strategy for noise robustness
    - Depthwise separable convolutions for efficiency
    
    Paper Reference: Section 2.2 - SFPM
    
    Parameters from paper (Table 4):
    - Number of Focus Layers (L): 3
    - Base Kernel Size (k): 3
    - Dilation Rate Growth Factor: 2
    - Sparsity Rate (ρ): 0.3
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_focal_levels: int = 3,
        base_kernel_size: int = 3,
        kernel_growth_factor: int = 2,
        dilation_growth_factor: int = 2,
        sparsity_rate: float = 0.3,
        min_features: int = 64,
        dropout_rate: float = 0.2
    ):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale GADF)
            hidden_channels: Number of hidden channels
            num_focal_levels: Number of focal levels L (default: 3)
            base_kernel_size: Base kernel size k₀ (default: 3)
            kernel_growth_factor: α in k_l = α*(l-1) + k₀ (default: 2)
            dilation_growth_factor: β in d_l = β^(l-1) (default: 2)
            sparsity_rate: ρ for Top-K selection (default: 0.3)
            min_features: Minimum features to retain K_min
            dropout_rate: Dropout rate (default: 0.2)
        """
        super(SFPM, self).__init__()
        
        self.num_focal_levels = num_focal_levels
        self.sparsity_rate = sparsity_rate
        self.min_features = min_features
        
        # Initial feature projection - Formula (6)
        # F = Conv1×1(X) ∈ R^(B×(2C+L+1)×H×W)
        self.input_proj = nn.Conv2d(
            in_channels, 
            2 * hidden_channels + num_focal_levels + 1, 
            kernel_size=1
        )
        
        self.hidden_channels = hidden_channels
        
        # Multi-scale depthwise separable convolutions - Formula (8), (9), (10)
        self.focal_convs = nn.ModuleList()
        for level in range(num_focal_levels):
            # k_l = α * (l-1) + k₀
            kernel_size = kernel_growth_factor * level + base_kernel_size
            # d_l = β^(l-1)
            dilation = dilation_growth_factor ** level
            
            # Depthwise separable convolution: DSConv(X) = PWConv(DWConv(X))
            self.focal_convs.append(
                nn.Sequential(
                    # Depthwise convolution
                    nn.Conv2d(
                        hidden_channels, 
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size // 2) * dilation,
                        dilation=dilation,
                        groups=hidden_channels,
                        bias=False
                    ),
                    # Pointwise convolution
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Importance assessment network for sparsification - Formula (11)
        # M = σ(Conv3×3(ReLU(Conv1×1(K))))
        self.importance_net = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Global context pooling - Formula (15)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Output projection - Formula (17)
        self.value_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.output_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final output convolution
        self.final_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
    
    def adaptive_topk(self, importance_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Top-K sparsification strategy
        
        Formula (12): S = K ⊙ TK(M)
        Formula (13): K = max(⌊ρ · H · W⌋, K_min)
        
        Args:
            importance_mask: Spatial attention mask from importance network
            features: Feature maps to sparsify
            
        Returns:
            Sparsified features
        """
        B, C, H, W = features.shape
        
        # Calculate K - Formula (13)
        k = max(int(self.sparsity_rate * H * W), self.min_features)
        
        # Flatten spatial dimensions
        mask_flat = importance_mask.view(B, -1)  # (B, H*W)
        
        # Get top-k indices
        _, topk_indices = torch.topk(mask_flat, k, dim=1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(mask_flat)
        sparse_mask.scatter_(1, topk_indices, 1.0)
        sparse_mask = sparse_mask.view(B, 1, H, W)
        
        # Apply sparse mask - Element-wise multiplication
        sparsified = features * sparse_mask
        
        return sparsified
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SFPM
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, hidden_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # Initial projection - Formula (6)
        F = self.input_proj(x)
        
        # Split into Q, K, G - Formula (7)
        Q = F[:, :self.hidden_channels]  # Query features
        K = F[:, self.hidden_channels:2*self.hidden_channels]  # Context features
        G = F[:, 2*self.hidden_channels:]  # Gating weights (L+1 channels)
        
        # Compute importance mask for sparsification - Formula (11)
        importance_mask = self.importance_net(K)
        
        # Apply sparsification - Formula (12)
        K_sparse = self.adaptive_topk(importance_mask, K)
        
        # Multi-scale focal feature extraction - Formula (9)
        focal_features = []
        K_current = K_sparse
        for level, conv in enumerate(self.focal_convs):
            K_current = conv(K_current)
            focal_features.append(K_current)
        
        # Gating mechanism for multi-scale aggregation - Formula (14)
        # Ĝ = Softmax(G, dim=1)
        G_normalized = F.softmax(G, dim=1)
        
        # Weighted summation: K_agg = Σ Ĝ[l] ⊙ K^(l)
        K_agg = torch.zeros_like(focal_features[0])
        for level in range(self.num_focal_levels):
            gate_weight = G_normalized[:, level:level+1]
            K_agg = K_agg + gate_weight * focal_features[level]
        
        # Global context incorporation - Formula (15), (16)
        K_global = self.global_pool(focal_features[-1])  # (B, C, 1, 1)
        K_global = K_global.expand(-1, -1, H, W)  # Broadcast to spatial dims
        
        # Final aggregation with global context
        global_gate = G_normalized[:, -1:]
        K_final = K_agg + global_gate * K_global
        
        # Query-value interaction and output - Formula (17)
        V = self.value_proj(K_final)
        QV = Q * V  # Element-wise multiplication
        
        # Project and add residual
        QV_flat = QV.permute(0, 2, 3, 1)  # (B, H, W, C)
        O = self.dropout(self.output_proj(QV_flat))
        O = O.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Residual connection needs input projection
        if x.shape[1] != self.hidden_channels:
            x_proj = self.final_conv(F[:, :self.hidden_channels])
        else:
            x_proj = x
        
        output = x_proj + O
        
        return output


# ============================================================================
# xLSTM Components: sLSTM and mLSTM
# ============================================================================
class sLSTMCell(nn.Module):
    """
    Scalar LSTM (sLSTM) with exponential gating
    
    Key features:
    - Exponential gating for improved adaptability
    - Wider dynamic range for input and forget gates
    - Alleviates vanishing gradient problem
    
    Paper Reference: Section 2.3 - Framework of the proposed xLSTM
    Formulas (18), (19), (20)
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(sLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrices for efficiency
        self.W_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        
        # Additional bias for forget gate (Formula 23)
        self.register_buffer('forget_bias', torch.zeros(hidden_size))
        
    def forward(
        self, 
        x: torch.Tensor, 
        states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of sLSTM cell
        
        Args:
            x: Input tensor (batch_size, input_size)
            states: Tuple of (h, c, n) - hidden state, cell state, normalizer
            
        Returns:
            h_new: New hidden state
            (h_new, c_new, n_new): New states
        """
        h, c, n = states
        
        # Combined linear transformations
        gates = self.W_ih(x) + self.W_hh(h)
        
        # Split into individual gates
        i_gate, f_gate, o_gate, c_tilde = gates.chunk(4, dim=1)
        
        # Exponential gating for input gate - Formula (19)
        i = torch.exp(i_gate)
        
        # Sigmoid or exponential for forget gate
        f = torch.sigmoid(f_gate + self.forget_bias)
        
        # Standard sigmoid for output gate
        o = torch.sigmoid(o_gate)
        
        # Candidate cell state
        c_tilde = torch.tanh(c_tilde)
        
        # Update cell state - Formula (19)
        c_new = f * c + i * c_tilde
        
        # Update normalizer state - Formula (20)
        n_new = f * n + i
        
        # Normalize cell state to prevent explosion
        c_normalized = c_new / (n_new + 1e-8)
        
        # Output
        h_new = o * torch.tanh(c_normalized)
        
        return h_new, (h_new, c_new, n_new)


class mLSTMCell(nn.Module):
    """
    Matrix LSTM (mLSTM) with matrix memory mechanism
    
    Key features:
    - Matrix memory structure C_t ∈ R^(n×n) instead of scalar
    - Covariance update rule for full parallelism
    - Query-key-value mechanism similar to Transformer
    
    Paper Reference: Section 2.3 - Framework of the proposed xLSTM
    Formula (21)
    """
    
    def __init__(self, input_size: int, hidden_size: int, head_dim: int = 64):
        super(mLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, head_dim)
        self.W_k = nn.Linear(input_size, head_dim)
        self.W_v = nn.Linear(input_size, head_dim)
        
        # Gate projections
        self.W_i = nn.Linear(input_size, 1)
        self.W_f = nn.Linear(input_size, 1)
        self.W_o = nn.Linear(input_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(head_dim, hidden_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of mLSTM cell
        
        Args:
            x: Input tensor (batch_size, input_size)
            states: Tuple of (h, C, n) - hidden state, matrix memory, normalizer
            
        Returns:
            h_new: New hidden state
            (h_new, C_new, n_new): New states
        """
        h, C, n = states
        batch_size = x.size(0)
        
        # Compute q, k, v
        q = self.W_q(x)  # (batch, head_dim)
        k = self.W_k(x)  # (batch, head_dim)
        v = self.W_v(x)  # (batch, head_dim)
        
        # Exponential gating
        i = torch.exp(self.W_i(x))  # (batch, 1)
        f = torch.sigmoid(self.W_f(x))  # (batch, 1)
        
        # Update matrix memory - Formula (21)
        # C_new = f * C + i * v * k^T
        vk_outer = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))  # (batch, head_dim, head_dim)
        C_new = f.unsqueeze(2) * C + i.unsqueeze(2) * vk_outer
        
        # Update normalizer
        n_new = f * n + i
        
        # Retrieve from memory
        # h_tilde = C * q / n
        Cq = torch.bmm(C_new, q.unsqueeze(2)).squeeze(2)  # (batch, head_dim)
        h_tilde = Cq / (n_new + 1e-8)
        
        # Output gate and final hidden state
        o = torch.sigmoid(self.W_o(x))
        h_new = o * torch.tanh(self.output_proj(h_tilde))
        
        return h_new, (h_new, C_new, n_new)


class xLSTM(nn.Module):
    """
    Extended LSTM (xLSTM) combining sLSTM and mLSTM
    
    Constructed by stacking sLSTM and mLSTM layers with residual connections,
    fusing multi-scale memory through attention mechanism.
    
    Paper Reference: Section 2.3 and Formula (24)
    
    Parameters from paper (Table 4):
    - Hidden Dimension: 256
    - Number of Memory Subspaces (N_s): 4
    - Dropout Rate: 0.2
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_subspaces: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension (default: 256)
            num_layers: Number of xLSTM blocks
            num_subspaces: Number of memory subspaces N_s (default: 4)
            dropout: Dropout rate (default: 0.2)
        """
        super(xLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_subspaces = num_subspaces
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Create alternating sLSTM and mLSTM layers
        self.lstm_layers = nn.ModuleList()
        self.layer_types = []  # Track layer types
        
        for i in range(num_layers):
            if i % 2 == 0:
                # sLSTM layer
                self.lstm_layers.append(sLSTMCell(hidden_size, hidden_size))
                self.layer_types.append('s')
            else:
                # mLSTM layer
                self.lstm_layers.append(mLSTMCell(hidden_size, hidden_size))
                self.layer_types.append('m')
        
        # Multi-scale memory subspaces - Formula (22)
        # Each subspace has different forget rates
        self.subspace_cells = nn.ModuleList([
            sLSTMCell(hidden_size, hidden_size // num_subspaces)
            for _ in range(num_subspaces)
        ])
        
        # Attention for memory fusion - Formula (24)
        self.memory_attention = nn.Sequential(
            nn.Linear(hidden_size // num_subspaces, hidden_size // num_subspaces),
            nn.Tanh(),
            nn.Linear(hidden_size // num_subspaces, 1)
        )
        
        # Set forget gate biases - Formula (23): b_f^(s) = -log(2^(s-1))
        for s, cell in enumerate(self.subspace_cells):
            bias_value = -np.log(2 ** s)
            cell.forget_bias.fill_(bias_value)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def init_states(self, batch_size: int, device: torch.device):
        """Initialize hidden states for all layers and subspaces"""
        states = {}
        
        # Main layer states
        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 's':
                h = torch.zeros(batch_size, self.hidden_size, device=device)
                c = torch.zeros(batch_size, self.hidden_size, device=device)
                n = torch.ones(batch_size, self.hidden_size, device=device)
                states[f'layer_{i}'] = (h, c, n)
            else:
                h = torch.zeros(batch_size, self.hidden_size, device=device)
                C = torch.zeros(batch_size, 64, 64, device=device)  # Matrix memory
                n = torch.ones(batch_size, 1, device=device)
                states[f'layer_{i}'] = (h, C, n)
        
        # Subspace states
        subspace_dim = self.hidden_size // self.num_subspaces
        for s in range(self.num_subspaces):
            h = torch.zeros(batch_size, subspace_dim, device=device)
            c = torch.zeros(batch_size, subspace_dim, device=device)
            n = torch.ones(batch_size, subspace_dim, device=device)
            states[f'subspace_{s}'] = (h, c, n)
        
        return states
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of xLSTM
        
        Args:
            x: Input sequence tensor (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize states
        states = self.init_states(batch_size, device)
        
        # Project input
        x = self.input_proj(x)
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Process through main layers
            for i, (layer, layer_type) in enumerate(zip(self.lstm_layers, self.layer_types)):
                state_key = f'layer_{i}'
                h_t, states[state_key] = layer(x_t, states[state_key])
                x_t = self.dropout(h_t) + x_t  # Residual connection
            
            # Process through subspaces - Formula (22)
            subspace_outputs = []
            subspace_dim = self.hidden_size // self.num_subspaces
            x_chunks = x_t.chunk(self.num_subspaces, dim=1)
            
            for s in range(self.num_subspaces):
                state_key = f'subspace_{s}'
                h_s, states[state_key] = self.subspace_cells[s](
                    x_chunks[s], states[state_key]
                )
                subspace_outputs.append(h_s)
            
            # Attention-based memory fusion - Formula (24)
            # α_t^(s) = softmax(w_a^T * tanh(W_a * c_t^(s)))
            attention_scores = []
            for s, h_s in enumerate(subspace_outputs):
                score = self.memory_attention(h_s)
                attention_scores.append(score)
            
            attention_weights = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
            
            # c_fused = Σ α_t^(s) * c_t^(s)
            h_fused = torch.zeros(batch_size, subspace_dim, device=device)
            for s, h_s in enumerate(subspace_outputs):
                h_fused = h_fused + attention_weights[:, s:s+1] * h_s
            
            # Expand fused output back to full hidden size
            h_final = h_fused.repeat(1, self.num_subspaces)
            
            outputs.append(h_final)
        
        # Return last output
        output = outputs[-1]
        output = self.layer_norm(output)
        
        return output


# ============================================================================
# Complete SFP-xLSTM Model
# ============================================================================
class SFPxLSTM(nn.Module):
    """
    SFP-xLSTM: Complete fault diagnosis model
    
    Combines:
    1. GADF encoding for signal-to-image conversion
    2. SFPM for multi-scale feature extraction
    3. xLSTM for temporal modeling and classification
    
    Paper Reference: Full model as shown in Figure 1
    Formula (35): ŷ = Softmax(W_cls · xLSTM(SFPM(GADF(x))) + b_cls)
    
    Default parameters from Table 4:
    - GAF Image Size: 256 × 256
    - SFPM: L=3, k=3, dilation_growth=2, ρ=0.3
    - xLSTM: hidden_dim=256, N_s=4, dropout=0.2
    - Training: batch_size=32, lr=0.003, epochs=150
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 256,
        # SFPM parameters
        sfpm_channels: int = 64,
        num_focal_levels: int = 3,
        base_kernel_size: int = 3,
        dilation_growth_factor: int = 2,
        sparsity_rate: float = 0.3,
        # xLSTM parameters
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        num_subspaces: int = 4,
        # Common parameters
        dropout: float = 0.2,
        pool_size: int = 8
    ):
        """
        Args:
            num_classes: Number of fault classes (default: 10 for CWRU)
            image_size: GADF image size (default: 256)
            sfpm_channels: SFPM hidden channels (default: 64)
            num_focal_levels: Number of focal levels L (default: 3)
            base_kernel_size: Base kernel size k (default: 3)
            dilation_growth_factor: Dilation growth β (default: 2)
            sparsity_rate: Sparsity rate ρ (default: 0.3)
            hidden_size: xLSTM hidden dimension (default: 256)
            num_lstm_layers: Number of xLSTM layers (default: 2)
            num_subspaces: Number of memory subspaces N_s (default: 4)
            dropout: Dropout rate (default: 0.2)
            pool_size: Adaptive pooling size P (default: 8)
        """
        super(SFPxLSTM, self).__init__()
        
        self.image_size = image_size
        self.pool_size = pool_size
        
        # SFPM module
        self.sfpm = SFPM(
            in_channels=1,
            hidden_channels=sfpm_channels,
            num_focal_levels=num_focal_levels,
            base_kernel_size=base_kernel_size,
            dilation_growth_factor=dilation_growth_factor,
            sparsity_rate=sparsity_rate,
            dropout_rate=dropout
        )
        
        # Adaptive pooling - Formula (34)
        # F_pool = AdaptiveAvgPool2d(X_out, (P, P))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        # xLSTM module
        # Input size = sfpm_channels (after reshaping to sequence)
        self.xlstm = xLSTM(
            input_size=sfpm_channels,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            num_subspaces=num_subspaces,
            dropout=dropout
        )
        
        # Classification head - Formula (35)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SFP-xLSTM
        
        Args:
            x: Input GADF images, shape (batch_size, 1, H, W)
            
        Returns:
            Class logits, shape (batch_size, num_classes)
        """
        # SFPM feature extraction
        features = self.sfpm(x)  # (B, C, H, W)
        
        # Adaptive pooling - Formula (34)
        pooled = self.adaptive_pool(features)  # (B, C, P, P)
        
        # Reshape to sequence
        # F_seq = Reshape(F_pool, [B, P², C])
        B, C, P, _ = pooled.shape
        sequence = pooled.view(B, C, P * P).permute(0, 2, 1)  # (B, P², C)
        
        # xLSTM temporal modeling
        lstm_out = self.xlstm(sequence)  # (B, hidden_size)
        
        # Classification
        logits = self.classifier(lstm_out)  # (B, num_classes)
        
        return logits


# ============================================================================
# Loss Functions
# ============================================================================
class CompositeLoss(nn.Module):
    """
    Composite Loss Function
    
    Formula (36): L = L_CE + λ₁L_sparse + λ₂L_temporal
    
    Components:
    - L_CE: Cross-Entropy Loss for classification
    - L_sparse: Sparsity regularization
    - L_temporal: Temporal smoothness constraint
    """
    
    def __init__(
        self,
        lambda_sparse: float = 0.01,
        lambda_temporal: float = 0.01,
        num_classes: int = 10
    ):
        super(CompositeLoss, self).__init__()
        
        self.lambda_sparse = lambda_sparse
        self.lambda_temporal = lambda_temporal
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        importance_maps: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute composite loss
        
        Args:
            logits: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)
            importance_maps: Sparsity attention maps (optional)
            temporal_features: Temporal features for smoothness (optional)
            
        Returns:
            Total loss value
        """
        # Classification loss
        loss_ce = self.ce_loss(logits, targets)
        
        total_loss = loss_ce
        
        # Sparsity regularization (L1 on importance maps)
        if importance_maps is not None and self.lambda_sparse > 0:
            loss_sparse = torch.mean(torch.abs(importance_maps))
            total_loss = total_loss + self.lambda_sparse * loss_sparse
        
        # Temporal smoothness (difference between consecutive features)
        if temporal_features is not None and self.lambda_temporal > 0:
            diff = temporal_features[:, 1:] - temporal_features[:, :-1]
            loss_temporal = torch.mean(diff ** 2)
            total_loss = total_loss + self.lambda_temporal * loss_temporal
        
        return total_loss


# ============================================================================
# Utility Functions
# ============================================================================
def create_model(
    num_classes: int = 10,
    pretrained: bool = False,
    **kwargs
) -> SFPxLSTM:
    """
    Factory function to create SFP-xLSTM model with paper default parameters
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        **kwargs: Override default parameters
        
    Returns:
        SFPxLSTM model instance
    """
    # Default parameters from paper Table 4
    default_config = {
        'num_classes': num_classes,
        'image_size': 256,
        'sfpm_channels': 64,
        'num_focal_levels': 3,
        'base_kernel_size': 3,
        'dilation_growth_factor': 2,
        'sparsity_rate': 0.3,
        'hidden_size': 256,
        'num_lstm_layers': 2,
        'num_subspaces': 4,
        'dropout': 0.2,
        'pool_size': 8
    }
    
    # Override with provided kwargs
    default_config.update(kwargs)
    
    model = SFPxLSTM(**default_config)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation and forward pass
    print("Testing SFP-xLSTM Model...")
    
    # Create model with paper parameters
    model = create_model(num_classes=10)
    print(f"Model Parameters: {count_parameters(model) / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
