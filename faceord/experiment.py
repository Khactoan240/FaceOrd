"""Model/loss factories, forward pass and the cumulative ablation variants (a)-(f).

  a_global_flat        global branch, flat age head, fixed weights
  b_ordinal            + ordinal age head with monotonicity penalty
  c_label_distribution + soft label-distribution loss
  d_local_regions      + landmark regions (mean of region features, mean fusion)
  e_attention_gate     + regional Transformer and gated fusion
  f_full               + uncertainty-aware task weighting (FaceOrd)
"""
from .models import build_faceord
from .losses import FlatAgeMultiTaskLoss, OrdinalFixedMultiTaskLoss, UncertaintyMultiTaskLoss
from baselines.models import build_model

VARIANTS = {
    'a_global_flat': dict(use_landmarks=False, flat_age=True),
    'b_ordinal': dict(use_landmarks=False),
    'c_label_distribution': dict(use_landmarks=False),
    'd_local_regions': dict(use_landmarks=True, use_attention=False, use_gated_fusion=False),
    'e_attention_gate': dict(use_landmarks=True),
    'f_full': dict(use_landmarks=True),
}
BASELINES = ['vgg19', 'resnet50', 'densenet169', 'optimized_vgg19',
             'mivolo_face', 'facexformer', 'swin_t', 'convnext_small',
             'vit_face_approx', 'convnext_tiny']


def make_model(cfg, model_name='faceord', variant='f_full'):
    """Build FaceOrd (or an ablation variant) or a comparison model."""
    if model_name == 'faceord':
        return build_faceord(cfg, **VARIANTS[variant])
    if model_name in {'mivolo_face', 'facexformer'}:
        from baselines.official_adapters import build_official
        return build_official(model_name, cfg)
    return build_model(model_name, cfg.K, pretrained=cfg.USE_PRETRAINED_BACKBONE)


def make_criterion(cfg, model_name, variant, class_weights=None):
    """Loss for a model/variant: flat CE, fixed-weight ordinal or uncertainty-weighted ordinal."""
    if model_name != 'faceord' or variant == 'a_global_flat':
        return FlatAgeMultiTaskLoss(w_age=cfg.FIXED_AGE_WEIGHT,
                                    w_gender=cfg.FIXED_GENDER_WEIGHT, class_weights=class_weights)
    kwargs = dict(K=cfg.K, lambda_mono=cfg.LAMBDA_MONO,
                  lambda_dist=0.0 if variant == 'b_ordinal' else cfg.LAMBDA_DIST,
                  tau=cfg.LABEL_DIST_TAU, class_weights=class_weights)
    if variant == 'f_full' and cfg.WEIGHTING == 'uncertainty':
        return UncertaintyMultiTaskLoss(**kwargs)
    return OrdinalFixedMultiTaskLoss(**kwargs, w_age=cfg.FIXED_AGE_WEIGHT,
                                    w_gender=cfg.FIXED_GENDER_WEIGHT)


def forward_model(model, batch, device, model_name):
    """Run a batch through the model; comparison models only receive the global image."""
    img = batch['global_img'].to(device)
    out = model(img, batch['regions'].to(device)) if model_name == 'faceord' else model(img)
    if model_name != 'faceord':
        out['age_p_hat'] = out['age_logits'].softmax(dim=1)
        out['age_c_hat'] = None
    return out


def compute_loss(criterion, out, batch, device):
    """Evaluate the criterion in float32 (BCE on probabilities is not autocast-safe)."""
    age, gender = batch['age_group'].to(device), batch['gender'].to(device)
    if isinstance(criterion, FlatAgeMultiTaskLoss):
        return criterion(out['age_logits'].float(), None, out['age_p_hat'].float(),
                         out['gender_prob'].float(), age, gender)
    return criterion(out['age_c_hat'].float(), out['age_p_hat'].float(),
                     out['gender_prob'].float(), age, gender)
