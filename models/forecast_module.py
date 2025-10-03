# models/forecast_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, p_drop=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p_drop)
        self.res_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        out = self.drop(self.act(self.norm(self.conv(x))))
        return out + residual


class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3, num_layers=6, dilation_base=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(ResidualConvBlock(in_channels, hidden_dim, kernel_size, dilation))
        self.conv_stack = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, z_seq, frame_mask: torch.Tensor | None = None):
        """
        z_seq:      [B, T, N, D]
        frame_mask: [B, T] с 0/1 (1 — кадр «видим»), опционально
        """
        B, T, N, D = z_seq.shape

        if frame_mask is not None:
            m = frame_mask.to(z_seq.dtype).clamp(0, 1)
            z_seq = z_seq * m[:, :, None, None]                          

        z_seq = z_seq.permute(0, 2, 3, 1)                                 
        z_seq = z_seq.reshape(B * N, D, T)                               

        h = self.conv_stack(z_seq)                                         # [B*N, H, T]

        if frame_mask is not None:
            w = frame_mask.to(h.dtype).clamp(0, 1)                         
            w = w[:, None, None, :]                                        
            w = w.expand(B, N, 1, T).reshape(B * N, 1, T)                  
            h_num = (h * w).sum(dim=-1)                                    
            h_den = w.sum(dim=-1).clamp_min(1.0)                           
            h_pool = h_num / h_den                                        
        else:
            h_pool = h.mean(dim=-1)                                       

        h_out = self.out_proj(h_pool)                                   
        return h_out.view(B, N, D)                                         # [B, N, D]


class ForecastModule(nn.Module):
    def __init__(self, triplet_embed_dim):
        super().__init__()
        self.triplet_dynamics = TemporalConvEncoder(input_dim=triplet_embed_dim)

    def forward(self, z_seq, oc_last, ob_last, mask=None, frame_mask: torch.Tensor | None = None):
        z_pred = self.triplet_dynamics(z_seq, frame_mask=frame_mask)       # [B, N, D]
        return {"z_pred": z_pred}
