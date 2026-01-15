"""ResTranOCR: ResNet + Transformer architecture for multi-frame OCR."""
import torch
import torch.nn as nn

from src.models.components import (
    AttentionFusion,
    ResNetFeatureExtractor,
    STNBlock,
    TransformerSequenceModeler
)


class ResTranOCR(nn.Module):
    """ResNet-Transformer OCR model with multi-frame attention fusion and input STN.
    
    Architecture: 
    Input -> Shared STN -> ResNet Backbone -> Attention Fusion -> Transformer -> FC -> CTC
    """
    
    def __init__(
        self,
        num_classes: int,
        resnet_layers: int = 18,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
    ):
        """
        Args:
            num_classes: Number of output classes (including blank for CTC).
            resnet_layers: ResNet variant (18 or 34).
            transformer_heads: Number of attention heads.
            transformer_layers: Number of transformer encoder layers.
            transformer_ff_dim: Feedforward dimension in transformer.
            dropout: Dropout rate.
            use_stn: Whether to enable the input STN alignment block.
        """
        super().__init__()
        self.cnn_channels = 512  # ResNet output channels
        self.use_stn = use_stn
        
        # STN: Spatial Transformer Network for INPUT ALIGNMENT
        # Only initialize if enabled to save memory
        if use_stn:
            self.stn = STNBlock(in_channels=3)
        
        # Backbone: ResNet feature extractor
        self.backbone = ResNetFeatureExtractor(layers=resnet_layers)
        
        # Fusion: Multi-frame attention
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # ResNetFeatureExtractor now uses AdaptiveAvgPool to force H=1
        self.feature_height = 1 
        self.d_model = self.cnn_channels * self.feature_height
        
        # Neck: Transformer sequence modeler
        self.neck = TransformerSequenceModeler(
            d_model=self.d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout
        )
        
        # Head: Classification layer
        self.fc = nn.Linear(self.d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, 3, H, W] where T=5 frames.
        
        Returns:
            Log-softmax output [B, W', num_classes] for CTC loss.
        """
        b, t, c, h, w = x.size()
        
        # Reshape frames to batch-first for backbone input
        x_flat = x.view(b * t, c, h, w)

        if self.use_stn:
            # Shared STN: compute mean frame, predict transformation, apply to all frames
            x_mean = torch.mean(x, dim=1)  # [B, 3, H, W]
            xs = self.stn.localization(x_mean)
            theta = self.stn.fc_loc(xs).view(-1, 2, 3)  # [B, 2, 3]
            
            # Apply same transformation to all frames
            theta_repeated = theta.unsqueeze(1).repeat(1, t, 1, 1).view(b * t, 2, 3)
            grid = torch.nn.functional.affine_grid(theta_repeated, x_flat.size(), align_corners=False)
            aligned_images = torch.nn.functional.grid_sample(x_flat, grid, align_corners=False)
        else:
            aligned_images = x_flat
        
        # --- Backbone ---
        feat = self.backbone(aligned_images)  # [B*T, 512, 1, W']
        
        # --- Fusion ---
        fused = self.fusion(feat)  # [B, 512, 1, W']
        
        # --- Transformer ---
        # Reshape: [B, C, 1, W'] -> [B, W', C]
        b_out, c_out, h_f, w_f = fused.size()
        seq_input = fused.squeeze(2).permute(0, 2, 1) # [B, W', C]
        
        seq_out = self.neck(seq_input)  # [B, W', d_model]
        
        # --- Head ---
        out = self.fc(seq_out)  # [B, W', num_classes]
        
        return out.log_softmax(2)