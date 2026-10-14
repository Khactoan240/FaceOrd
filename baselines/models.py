"""Comparison architectures with a flat age-group head and a binary gender head.

  vgg19, resnet50, densenet169   ImageNet-pretrained torchvision CNNs
  optimized_vgg19                OptimizedVGG19 (Hiremath & Patil, 2025), trained from scratch
  swin_t, convnext_small         ImageNet-pretrained Swin-T and ConvNeXt-S
  convnext_tiny                  ImageNet-pretrained ConvNeXt-T
  vit_face_approx                torchvision ViT-B/16 face-only classifier

MiVOLO and FaceXFormer use their official code (see official_adapters.py).
Every model returns {"age_logits": (B, K), "age_prob": (B, K), "gender_prob": (B,)}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ParallelConvBlock(nn.Module):
    """
    Dual-path 3x3 block of OptimizedVGG19 (Hiremath & Patil, 2025, Fig. 4b).

    Two parallel 1x1 -> 3x3 -> 3x3 branches are concatenated and projected to
    `out_ch` with a 1x1 convolution; this keeps the receptive field of a 5x5
    convolution with fewer parameters.
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.branch_a = nn.Sequential(
            _conv_bn_relu(in_ch, mid_ch, k=1, p=0),
            _conv_bn_relu(mid_ch, mid_ch, k=3, p=1),
            _conv_bn_relu(mid_ch, mid_ch, k=3, p=1),
        )
        self.branch_b = nn.Sequential(
            _conv_bn_relu(in_ch, mid_ch, k=1, p=0),
            _conv_bn_relu(mid_ch, mid_ch, k=3, p=1),
            _conv_bn_relu(mid_ch, mid_ch, k=3, p=1),
        )
        self.merge = _conv_bn_relu(mid_ch * 2, out_ch, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat([self.branch_a(x),
                                     self.branch_b(x)], dim=1))


class _MultiTaskHead(nn.Module):
    """Shared head: Dropout -> age logits (K) and gender probability."""

    def __init__(self, in_features: int, K: int, dropout: float = 0.5):
        super().__init__()
        self.drop      = nn.Dropout(dropout)
        self.age_fc    = nn.Linear(in_features, K)
        self.gender_fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.drop(x)
        age_logits  = self.age_fc(x)
        gender_prob = torch.sigmoid(self.gender_fc(x)).squeeze(-1)
        return {
            "age_logits"  : age_logits,
            "age_prob"    : F.softmax(age_logits, dim=1),
            "gender_prob" : gender_prob,
        }


class OptimizedVGG19(nn.Module):
    """
    OptimizedVGG19 multi-task classifier (Hiremath & Patil, 2025, Table 3).

      stages 1-3 : standard 3x3 Conv-BN-ReLU blocks
      stages 4-5 : ParallelConvBlock
      FC 4096 -> 4096 -> age / gender heads

    Input (B, 3, 224, 224); flat K-class age head, logistic gender head.
    """

    def __init__(self, K: int = 8, dropout: float = 0.5):
        super().__init__()
        self.K = K

        self.stage1 = nn.Sequential(          # 224 -> 112, 3 -> 64
            _conv_bn_relu(3,  64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(2, 2),
        )
        self.stage2 = nn.Sequential(          # 112 -> 56, 64 -> 128
            _conv_bn_relu(64,  128),
            _conv_bn_relu(128, 128),
            nn.MaxPool2d(2, 2),
        )
        self.stage3 = nn.Sequential(          # 56 -> 28, 128 -> 256
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(2, 2),
        )
        self.stage4 = nn.Sequential(          # 28 -> 14, 256 -> 512
            ParallelConvBlock(256, 128, 512),
            ParallelConvBlock(512, 128, 512),
            nn.MaxPool2d(2, 2),
        )
        self.stage5 = nn.Sequential(          # 14 -> 7, 512 -> 512
            ParallelConvBlock(512, 128, 512),
            ParallelConvBlock(512, 128, 512),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.head = _MultiTaskHead(4096, K, dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.stage1(x)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.stage5(feat)
        feat = self.fc(feat)
        return self.head(feat)


class VGG19Baseline(nn.Module):
    """ImageNet VGG-19 with the multi-task head."""

    def __init__(self, K: int = 8, pretrained: bool = True):
        super().__init__()
        weights          = tv_models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        backbone         = tv_models.vgg19(weights=weights)
        self.features    = backbone.features
        self.avgpool     = backbone.avgpool
        self.fc          = nn.Sequential(*list(backbone.classifier.children())[:-1])
        self.head        = _MultiTaskHead(4096, K, dropout=0.5)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.fc(feat)
        return self.head(feat)


class ResNet50Baseline(nn.Module):
    """ImageNet ResNet-50 with the multi-task head."""

    def __init__(self, K: int = 8, pretrained: bool = True):
        super().__init__()
        weights      = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone     = tv_models.resnet50(weights=weights)
        in_features  = backbone.fc.in_features
        backbone.fc  = nn.Identity()
        self.backbone = backbone
        self.head     = _MultiTaskHead(in_features, K, dropout=0.5)

    def forward(self, x: torch.Tensor) -> dict:
        return self.head(self.backbone(x))


class DenseNet169Baseline(nn.Module):
    """ImageNet DenseNet-169 with the multi-task head."""

    def __init__(self, K: int = 8, pretrained: bool = True):
        super().__init__()
        weights      = tv_models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        backbone     = tv_models.densenet169(weights=weights)
        in_features  = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head     = _MultiTaskHead(in_features, K, dropout=0.5)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone.features(x)
        feat = F.relu(feat, inplace=True)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.head(feat)


class ViTFaceApprox(nn.Module):
    """ImageNet ViT-B/16 on the face crop, 768 -> 512 projection and the multi-task head."""

    def __init__(self, K: int = 8, pretrained: bool = True):
        super().__init__()
        weights  = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.vit_b_16(weights=weights)

        in_feat        = backbone.heads.head.in_features   # 768
        backbone.heads = nn.Identity()
        self.backbone  = backbone
        self.proj      = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head      = _MultiTaskHead(512, K, dropout=0.1)

    def forward(self, x: torch.Tensor) -> dict:
        return self.head(self.proj(self.backbone(x)))


class SwinTBaseline(nn.Module):
    """ImageNet Swin-T (Liu et al., ICCV 2021) with the multi-task head."""

    def __init__(self, K: int = 8, pretrained: bool = True):
        super().__init__()
        weights  = tv_models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.swin_t(weights=weights)

        in_feat           = backbone.head.in_features   # 768
        backbone.head     = nn.Identity()
        self.backbone     = backbone
        self.head         = _MultiTaskHead(in_feat, K, dropout=0.3)

    def forward(self, x: torch.Tensor) -> dict:
        return self.head(self.backbone(x))


class ConvNeXtTinyBaseline(nn.Module):
    """ImageNet ConvNeXt-T (Liu et al., CVPR 2022) with the multi-task head."""

    def __init__(self, K: int = 8, pretrained: bool = True):
        super().__init__()
        weights  = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.convnext_tiny(weights=weights)

        in_feat               = backbone.classifier[-1].in_features  # 768
        backbone.classifier[-1] = nn.Identity()   # keep LayerNorm2d + Flatten
        self.backbone         = backbone
        self.head             = _MultiTaskHead(in_feat, K, dropout=0.3)

    def forward(self, x: torch.Tensor) -> dict:
        return self.head(self.backbone(x))


class ConvNeXtSmallBaseline(nn.Module):
    """ImageNet ConvNeXt-S with the multi-task head."""

    def __init__(self, K=8, pretrained=True):
        super().__init__()
        weights = tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.convnext_small(weights=weights)
        dim = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        self.head = _MultiTaskHead(dim, K, dropout=0.3)

    def forward(self, x):
        return self.head(self.backbone(x))


MODELS = {
    "optimized_vgg19"  : OptimizedVGG19,
    "vgg19"            : VGG19Baseline,
    "resnet50"         : ResNet50Baseline,
    "densenet169"      : DenseNet169Baseline,
    "vit_face_approx"  : ViTFaceApprox,
    "swin_t"           : SwinTBaseline,
    "convnext_tiny"    : ConvNeXtTinyBaseline,
    "convnext_small"   : ConvNeXtSmallBaseline,
}

# Trained from scratch, as in the original OptimizedVGG19 work
_SCRATCH_MODELS = {"optimized_vgg19"}


def build_model(name: str, K: int = 8, pretrained: bool = True) -> nn.Module:
    """
    Build a comparison model by name (see MODELS); `pretrained` loads ImageNet
    weights for the torchvision backbones.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    if name in _SCRATCH_MODELS:
        return MODELS[name](K=K)
    return MODELS[name](K=K, pretrained=pretrained)
