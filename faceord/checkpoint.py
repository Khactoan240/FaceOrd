"""State-dict checkpoints with the full configuration; loading never downloads weights."""
from pathlib import Path
import torch
from .config import Config
from .experiment import make_model


def save_checkpoint(path, model, criterion, cfg, model_name, variant, metadata):
    """Save model/criterion state, configuration, model name, variant and run metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 2, 'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(), 'config': cfg.to_dict(),
                'model_name': model_name, 'variant': variant, 'metadata': metadata}, path)


def load_checkpoint(path, device='cpu'):
    """Rebuild the model from a checkpoint; returns (model in eval mode, Config, payload)."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 2:
        raise ValueError(f'Unsupported checkpoint format in {path}')
    cfg = Config(**payload['config'])
    cfg.validate()
    original_pretrained = cfg.USE_PRETRAINED_BACKBONE
    cfg.USE_PRETRAINED_BACKBONE = False
    model = make_model(cfg, payload['model_name'], payload['variant'])
    cfg.USE_PRETRAINED_BACKBONE = original_pretrained
    model.load_state_dict(payload['model_state_dict'], strict=True)
    expected = payload.get('metadata', {}).get('upstream_source')
    if expected and expected != getattr(model, 'source_provenance', None):
        raise ValueError('The external model source tree differs from the one used for training')
    return model.to(device).eval(), cfg, payload
