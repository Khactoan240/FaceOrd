"""Adapters for the official MiVOLO and FaceXFormer implementations.

The upstream repositories are not included; clone them and point the configuration
(MIVOLO_SOURCE, FACEXFORMER_SOURCE) or the environment variables
FACEORD_MIVOLO_SOURCE / FACEORD_FACEXFORMER_SOURCE to them:

    git clone https://github.com/WildChlamydia/MiVOLO external/MiVOLO
    git clone https://github.com/Kartik-3004/facexformer external/facexformer

Both adapters return the common output contract {"age_logits": (B, 8),
"gender_prob": (B,)}. A digest of the upstream Python sources is stored in each
checkpoint and checked when it is loaded.
"""
import hashlib
import importlib
import os
from pathlib import Path
import sys
import torch
from torch import nn


def source_tree(configured_path, env_key, package):
    """Put an upstream source tree on sys.path and return a SHA-256 digest of its Python files."""
    root = Path(os.environ.get(env_key, configured_path)).resolve()
    if not (root / package / '__init__.py').is_file():
        raise FileNotFoundError(f'Provide the official source tree at {root} or set {env_key}')
    loaded = sys.modules.get(package)
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f'{package} is already imported from a different source tree')
    sys.path.insert(0, str(root))
    files = sorted((root / package).rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {'package': package, 'python_source_sha256': digest.hexdigest()}


class MiVOLOFaceAdapter(nn.Module):
    """Face-only MiVOLO (3-channel input) with 8 age logits and 2 gender logits.

    MIVOLO_VARIANT selects the 224px architecture (default mivolo_d3_224); ImageNet
    initialisation follows the upstream variant configuration.
    """

    def __init__(self, cfg):
        super().__init__()
        self.source_provenance = source_tree(cfg.MIVOLO_SOURCE, 'FACEORD_MIVOLO_SOURCE', 'mivolo')
        module = importlib.import_module('mivolo.model.mivolo_model')
        if not cfg.MIVOLO_VARIANT.endswith('_224'):
            raise ValueError('This protocol requires a 224px MiVOLO variant')
        factory = getattr(module, cfg.MIVOLO_VARIANT)
        self.net = factory(pretrained=cfg.USE_PRETRAINED_BACKBONE,
                           pretrained_cfg=module.default_cfgs[cfg.MIVOLO_VARIANT],
                           in_chans=3, num_classes=cfg.K + 2)
        # K age logits followed by two gender logits; face crops only, no person branch.
        self.k = cfg.K

    def forward(self, image):
        logits = self.net(image)
        if logits.ndim != 2 or logits.shape[1] != self.k + 2:
            raise RuntimeError('Unexpected MiVOLO output shape')
        return {'age_logits': logits[:, :self.k],
                'gender_prob': logits[:, self.k:].softmax(1)[:, 1]}


class FaceXFormerAdapter(nn.Module):
    """FaceXFormer (Swin-B encoder + FaceX decoder) with re-initialised age (8) and gender (2) outputs.

    The age/gender/race task token (task id 4) produces the predictions.
    """

    def __init__(self, cfg):
        super().__init__()
        self.source_provenance = source_tree(cfg.FACEXFORMER_SOURCE, 'FACEORD_FACEXFORMER_SOURCE', 'network')
        module = importlib.import_module('network.models.facexformer')
        # The upstream constructor always loads ImageNet Swin-B weights; patch the factory so
        # that USE_PRETRAINED_BACKBONE=False (checkpoint loading) builds it without a download.
        original = module.swin_b
        def swin_for_this_run(*args, **kwargs):
            kwargs['weights'] = 'IMAGENET1K_V1' if cfg.USE_PRETRAINED_BACKBONE else None
            return original(*args, **kwargs)
        module.swin_b = swin_for_this_run
        try:
            self.net = module.FaceXFormer()
        finally:
            module.swin_b = original
        for name, outputs in [('age_prediction_head', cfg.K), ('gender_prediction_head', 2)]:
            head = getattr(self.net.face_decoder, name)
            old = head.layers[-1]
            head.layers[-1] = nn.Linear(old.in_features, outputs)

    def forward(self, image):
        tasks = torch.full((len(image),), 4, device=image.device, dtype=torch.long)
        outputs = self.net(image, {}, tasks)
        age, gender = outputs[4], outputs[5]
        if age.shape != (len(image), 8) or gender.shape != (len(image), 2):
            raise RuntimeError('Unexpected FaceXFormer output shape')
        return {'age_logits': age, 'gender_prob': gender.softmax(1)[:, 1]}


def build_official(name, cfg):
    """'mivolo_face' or 'facexformer' -> adapter instance."""
    return {'mivolo_face': MiVOLOFaceAdapter, 'facexformer': FaceXFormerAdapter}[name](cfg)
