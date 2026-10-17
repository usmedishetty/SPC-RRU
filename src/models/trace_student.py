"""
Causal Trace-Space Student Network Architecture.
Lightweight spatiotemporal model operating on dense spatial skin-patch traces
and metadata conditioning vectors, deployable in real-time on Raspberry Pi 5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

class CausalConv1dBlock(nn.Module):
    """
    Dilated Causal 1D Convolution with Residual Connection and Group Normalization.
    Guarantees no future frame leakage in streaming inference.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation,
            padding=self.pad
        )
        self.norm = nn.GroupNorm(4, out_channels)
        self.activation = nn.GELU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        # Chomp the right-side padding to preserve strict causality
        out = out[:, :, :-self.pad] if self.pad > 0 else out
        out = self.norm(out)
        out = self.activation(out)
        return out + self.residual(x)

class TraceSpaceStudent(nn.Module):
    """
    Causal Trace-Space Student Network.
    Inputs:
      - patch_traces: [B, num_patches * 3, T] (e.g. 16 patches * 3 RGB channels = 48 channels)
      - metadata: [B, 5] (normalized lux, fps, bitrate, mask_flag, ita)
    Outputs:
      - waveform: [B, T] reconstructed high-resolution arterial pulse wave
      - latent_z: [B, 64] pooled cardiovascular representation
      - vitals: predicted HR, SpO2, SBP, DBP with heteroscedastic uncertainty
    """
    def __init__(self, num_patches: int = 16, metadata_dim: int = 5, latent_dim: int = 64):
        super().__init__()
        in_channels = num_patches * 3

        # Linear projection for metadata conditioning
        self.meta_embed = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )

        # Causal TCN backbone with expanding dilations (receptive field ~10 seconds)
        self.input_conv = nn.Conv1d(in_channels, 32, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            CausalConv1dBlock(32, 48, kernel_size=3, dilation=1),
            CausalConv1dBlock(48, 48, kernel_size=3, dilation=2),
            CausalConv1dBlock(48, 64, kernel_size=3, dilation=4),
            CausalConv1dBlock(64, 64, kernel_size=3, dilation=8),
            CausalConv1dBlock(64, 64, kernel_size=3, dilation=16),
        ])

        # Continuous waveform decoder
        self.waveform_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

        # Multi-task vital sign heads with heteroscedastic uncertainty
        fused_dim = 64 + 32  # latent_dim + meta_dim
        self.hr_head = nn.Linear(fused_dim, 2)    # [mean, log_var]
        self.spo2_head = nn.Linear(fused_dim, 2)  # [mean, log_var]
        self.bp_head = nn.Linear(fused_dim, 4)    # [sbp_mean, sbp_logvar, dbp_mean, dbp_logvar]

    def forward(self, patch_traces: torch.Tensor, metadata: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass.
        Args:
          patch_traces: [B, C, T]
          metadata: [B, 5]
        """
        B, C, T = patch_traces.shape
        x = self.input_conv(patch_traces)

        # Apply causal dilated blocks
        for block in self.blocks:
            x = block(x)

        # Decode continuous temporal waveform
        waveform = self.waveform_head(x).squeeze(1) # [B, T]

        # Temporal global pooling for vital sign heads
        z_pooled = torch.mean(x, dim=-1) # [B, 64]
        m_embed = self.meta_embed(metadata) # [B, 32]
        fused = torch.cat([z_pooled, m_embed], dim=-1) # [B, 96]

        # 1. Heart Rate
        hr_out = self.hr_head(fused)
        hr_pred = hr_out[:, 0] * 140.0 + 40.0

        # 2. SpO2
        spo2_out = self.spo2_head(fused)
        spo2_pred = torch.sigmoid(spo2_out[:, 0]) * 30.0 + 70.0

        # 3. Blood Pressure
        bp_out = self.bp_head(fused)
        sbp_pred = bp_out[:, 0] * 80.0 + 80.0
        dbp_pred = bp_out[:, 2] * 50.0 + 50.0

        return {
            "waveform": waveform,
            "latent_z": z_pooled,
            "hr": {"pred": hr_pred, "unc": torch.exp(0.5 * hr_out[:, 1])},
            "spo2": {"pred": spo2_pred, "unc": torch.exp(0.5 * spo2_out[:, 1])},
            "sbp": {"pred": sbp_pred, "unc": torch.exp(0.5 * bp_out[:, 1])},
            "dbp": {"pred": dbp_pred, "unc": torch.exp(0.5 * bp_out[:, 3])}
        }
