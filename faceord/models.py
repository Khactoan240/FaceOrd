"""
FaceOrd architecture (Sections 3.2-3.4).

  GlobalCNNEncoder   global encoder (ConvNeXt-S by default)
  RegionalEncoder    CNN shared by the M facial regions
  LightTransformer   lightweight Transformer over the region tokens
  GatedFusion        sample-adaptive gate between global and local features
  OrdinalAgeHead     K-1 cumulative binary classifiers for the age group
  BinaryGenderHead   logistic gender classifier
  FaceOrd            complete model, including the ablation switches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU (no conv bias because of BatchNorm)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class GlobalCNNEncoder(nn.Module):
    """
    Global feature f^(g) in R^d of the aligned 224x224 face.

    backbone_name selects the architecture ("convnext_s", "resnet50",
    "efficientnet_b4", "convnext_b", or "scratch" for a small VGG-style CNN
    built from `channels`); `pretrained` loads ImageNet weights.

    Input  : (B, 3, 224, 224)
    Output : (B, d)
    """

    _BACKBONE_DIMS = {
        "resnet50":        2048,
        "efficientnet_b4": 1792,
        "convnext_s":       768,
        "convnext_b":      1024,
    }

    def __init__(self, d: int = 256, pretrained: bool = True,
                 channels: list = None, backbone_name: str = "convnext_s"):
        super().__init__()
        self.pretrained = pretrained

        if backbone_name != "scratch":
            name = backbone_name.lower()
            if name == "resnet50":
                backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
                self.features = nn.Sequential(*list(backbone.children())[:-2])
                feat_dim = 2048
            elif name == "efficientnet_b4":
                backbone = tv_models.efficientnet_b4(weights=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
                self.features = backbone.features
                feat_dim = 1792
            elif name == "convnext_s":
                backbone = tv_models.convnext_small(weights=tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
                self.features = backbone.features
                feat_dim = 768
            elif name == "convnext_b":
                backbone = tv_models.convnext_base(weights=tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
                self.features = backbone.features
                feat_dim = 1024
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}. "
                                 f"Choose from {list(self._BACKBONE_DIMS)} or 'scratch'")

            self.proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feat_dim, d),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
        else:
            if channels is None:
                channels = [64, 128, 256, 512]
            layers = []
            in_ch = 3
            for out_ch in channels:
                # two stacked 3x3 convolutions per stage
                layers += [
                    conv_bn_relu(in_ch, out_ch, 3),
                    conv_bn_relu(out_ch, out_ch, 3),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
                in_ch = out_ch
            self.features = nn.Sequential(*layers)
            self.proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_ch, d),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> f_g: (B, d)"""
        feat = self.features(x)    # (B, C, h, w)
        f_g  = self.proj(feat)     # (B, d)
        return f_g


class RegionalEncoder(nn.Module):
    """
    Shared CNN applied to the M region patches (Section 3.3).

    The regions are folded into the batch, (B, M, 3, R, R) -> (B*M, 3, R, R),
    encoded in one pass and reshaped back.

    Input  : (B, M, 3, R, R)
    Output : u_m in R^d for every region -> (B, M, d)
    """

    def __init__(self, d: int = 256, region_ch: list = None,
                 region_size: int = 64):
        super().__init__()
        if region_ch is None:
            region_ch = [32, 64, 128]

        layers = []
        in_ch = 3
        for out_ch in region_ch:
            layers += [
                conv_bn_relu(in_ch, out_ch, 3),
                nn.MaxPool2d(2, 2),
            ]
            in_ch = out_ch

        self.shared_cnn = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, d),
            nn.ReLU(inplace=True),
        )

    def forward(self, regions: torch.Tensor) -> torch.Tensor:
        """regions: (B, M, 3, R, R) -> (B, M, d)"""
        B, M, C, R, R2 = regions.shape
        x    = regions.view(B * M, C, R, R2)
        feat = self.shared_cnn(x)
        u    = self.proj(feat)                 # (B*M, d)
        return u.view(B, M, -1)


class _TransformerLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer: multi-head self-attention + FFN, both residual."""

    def __init__(self, d: int, num_heads: int, dim_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, num_heads, dropout=dropout,
                                           batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, d) -> (B, M, d)"""
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class LightTransformer(nn.Module):
    """
    L Transformer layers modelling dependencies between the M facial regions.

    Each token is z_m = u_m + e_m, where e_m is a learnable region embedding.
    The local feature is the mean of the output tokens: f^(l) in R^d.
    """

    def __init__(self, M: int, d: int, L: int = 2,
                 num_heads: int = 4, dim_ff: int = 512):
        super().__init__()
        self.region_emb = nn.Embedding(M, d)

        self.layers = nn.ModuleList([
            _TransformerLayer(d, num_heads, dim_ff)
            for _ in range(L)
        ])

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, M, d) -> f^(l): (B, d)"""
        B, M, d = u.shape
        idx = torch.arange(M, device=u.device)
        z   = u + self.region_emb(idx).unsqueeze(0)    # (B, M, d)

        for layer in self.layers:
            z = layer(z)

        return z.mean(dim=1)


class GatedFusion(nn.Module):
    """
    Sample-adaptive fusion of global and local features (Section 3.3):

        g = sigmoid(W_f [f^(g) || f^(l)] + b_f)  in (0, 1)^d
        f = g * f^(g) + (1 - g) * f^(l)
    """

    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid(),
        )

    def forward(self, f_g: torch.Tensor, f_l: torch.Tensor) -> torch.Tensor:
        """f_g, f_l: (B, d) -> f: (B, d)"""
        g = self.gate(torch.cat([f_g, f_l], dim=-1))
        return g * f_g + (1.0 - g) * f_l


class OrdinalAgeHead(nn.Module):
    """
    Ordinal age-group head with K-1 cumulative binary classifiers (Section 3.4).

    For threshold k in {0, ..., K-2} (zero-based groups):
        a_k = w_k^T f + b_k,   c_k = sigmoid(a_k) ~ P(y > k)

    Group masses are reconstructed from the cumulative probabilities,
        r_0 = 1 - c_0,   r_k = c_{k-1} - c_k (1 <= k <= K-2),   r_{K-1} = c_{K-2},
    then clamped to a small epsilon and renormalised so that they form a valid
    distribution (monotonicity is only encouraged by L_mono). The predicted
    group is argmax p.
    """

    def __init__(self, d: int, K: int):
        super().__init__()
        self.K = K
        self.classifiers = nn.Linear(d, K - 1)

    @staticmethod
    def masses_from_cumulative(c_hat: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Cumulative probabilities c_hat (B, K-1) -> group distribution p_hat (B, K)."""
        p1    = 1.0 - c_hat[:, 0:1]
        p_mid = c_hat[:, :-1] - c_hat[:, 1:]
        pK    = c_hat[:, -1:]
        p_hat = torch.cat([p1, p_mid, pK], dim=1).clamp_min(eps)
        return p_hat / p_hat.sum(dim=1, keepdim=True)

    def forward(self, f: torch.Tensor):
        """
        f: (B, d) -> (logits (B, K-1), c_hat (B, K-1), p_hat (B, K))
        """
        logits = self.classifiers(f)
        c_hat  = torch.sigmoid(logits)
        return logits, c_hat, self.masses_from_cumulative(c_hat)

    def predict(self, f: torch.Tensor) -> torch.Tensor:
        """Predicted age group (B,)."""
        _, _, p_hat = self.forward(f)
        return p_hat.argmax(dim=1)


class BinaryGenderHead(nn.Module):
    """
    Logistic gender classifier: y_g = sigmoid(w_g^T f + b_g) = P(female).
    """

    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f: (B, d) -> (B,) in (0, 1)"""
        return torch.sigmoid(self.fc(f)).squeeze(-1)

    def predict(self, f: torch.Tensor) -> torch.Tensor:
        """Predicted gender: 0 = male, 1 = female."""
        return (self.forward(f) >= 0.5).long()


class FaceOrd(nn.Module):
    """
    Landmark-guided ordinal multi-task model for age group and gender.

    Ablation switches:
      use_landmarks    : False -> global branch only (f = f^(g))
      flat_age         : True  -> nn.Linear(d, K) + softmax instead of the ordinal head
      use_attention    : False -> f^(l) is the mean of the region features (no Transformer)
      use_gated_fusion : False -> f = (f^(g) + f^(l)) / 2

    forward(global_img (B, 3, 224, 224), regions (B, M, 3, R, R)) returns:
      'age_logits'  : (B, K-1) ordinal logits | (B, K) when flat_age
      'age_c_hat'   : (B, K-1) cumulative probabilities | None when flat_age
      'age_p_hat'   : (B, K)   age-group distribution
      'gender_prob' : (B,)     P(female)
      'fused_feat'  : (B, d)   fused representation (used for t-SNE)
    """

    def __init__(self, d: int = 256, M: int = 6, K: int = 8,
                 L: int = 2, num_heads: int = 4, dim_ff: int = 512,
                 global_channels: list = None,
                 region_channels: list = None,
                 region_size: int = 64,
                 pretrained_backbone: bool = True,
                 backbone_name: str = "convnext_s",
                 use_landmarks: bool = True,
                 flat_age: bool = False,
                 use_attention: bool = True,
                 use_gated_fusion: bool = True):
        super().__init__()
        self.M             = M
        self.K             = K
        self.use_landmarks = use_landmarks
        self.flat_age      = flat_age
        self.use_attention = use_attention
        self.use_gated_fusion = use_gated_fusion

        self.global_encoder = GlobalCNNEncoder(d, pretrained=pretrained_backbone,
                                               channels=global_channels,
                                               backbone_name=backbone_name)

        if use_landmarks:
            self.regional_encoder = RegionalEncoder(d, region_channels, region_size)
            if use_attention:
                self.transformer = LightTransformer(M, d, L, num_heads, dim_ff)
            if use_gated_fusion:
                self.fusion = GatedFusion(d)

        if flat_age:
            self.age_head = nn.Linear(d, K)
        else:
            self.age_head = OrdinalAgeHead(d, K)

        self.gender_head = BinaryGenderHead(d)

    def forward(self, global_img: torch.Tensor,
                regions: torch.Tensor = None) -> dict:
        f_g = self.global_encoder(global_img)

        if self.use_landmarks and regions is not None:
            u   = self.regional_encoder(regions)
            f_l = self.transformer(u) if self.use_attention else u.mean(dim=1)
            f = self.fusion(f_g, f_l) if self.use_gated_fusion else 0.5 * (f_g + f_l)
        elif self.use_landmarks:
            raise ValueError("FaceOrd local branch requires region tensors")
        else:
            f = f_g

        if self.flat_age:
            age_logits = self.age_head(f)
            age_p_hat  = F.softmax(age_logits, dim=-1)
            age_c_hat  = None
        else:
            age_logits, age_c_hat, age_p_hat = self.age_head(f)

        gender_prob = self.gender_head(f)

        return {
            "age_logits"  : age_logits,
            "age_c_hat"   : age_c_hat,
            "age_p_hat"   : age_p_hat,
            "gender_prob" : gender_prob,
            "fused_feat"  : f,
        }

    @torch.no_grad()
    def predict(self, global_img: torch.Tensor,
                regions: torch.Tensor = None) -> dict:
        """Gradient-free inference: final age group, gender and probabilities."""
        out = self.forward(global_img, regions)
        return {
            "age_group"  : out["age_p_hat"].argmax(dim=1),
            "gender"     : (out["gender_prob"] >= 0.5).long(),
            "age_dist"   : out["age_p_hat"],
            "gender_prob": out["gender_prob"],
        }


def build_faceord(cfg, **variant) -> FaceOrd:
    """Build FaceOrd (or an ablation variant) from a Config.

    With USE_PRETRAINED_BACKBONE=False no weights are downloaded.
    """
    return FaceOrd(
        d=cfg.D, M=cfg.M, K=cfg.K, L=cfg.TRANSFORMER_LAYERS,
        num_heads=cfg.TRANSFORMER_HEADS, dim_ff=cfg.TRANSFORMER_DIM_FF,
        global_channels=cfg.GLOBAL_CHANNELS, region_channels=cfg.REGION_CHANNELS,
        region_size=cfg.REGION_SIZE, pretrained_backbone=cfg.USE_PRETRAINED_BACKBONE,
        backbone_name=cfg.BACKBONE, **variant,
    )
