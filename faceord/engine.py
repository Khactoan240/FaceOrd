"""Shared training and evaluation protocol.

Each outer fold trains on its inner training subset, selects the checkpoint on the
stratified validation subset and evaluates the outer test fold once. Used by
kfold, ablation and the sensitivity studies.
"""
from contextlib import nullcontext
import json
import math
import os
import platform
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from .calibration import calibrated_ece, fit_temperature
from .checkpoint import load_checkpoint, save_checkpoint
from .data_cleaner import DatasetCleaner
from .dataset import UTKFaceDataset
from .experiment import make_model, make_criterion, forward_model, compute_loss
from .metrics import compute_metrics
from .splits import load_or_create_folds


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch and enable deterministic CUDA/cuDNN kernels where supported."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Derive NumPy/Python seeds of a DataLoader worker from its PyTorch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(name):
    """'auto' selects CUDA, then MPS, then CPU."""
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def write_json(path, value):
    """Write strict JSON (no NaN), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def make_loader(dataset, indices, cfg, shuffle=False, seed=42):
    """DataLoader over `indices` with a seeded generator for shuffling and workers."""
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, list(indices)), batch_size=cfg.BATCH_SIZE,
                      shuffle=shuffle, num_workers=cfg.NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(), generator=gen,
                      worker_init_fn=seed_worker)


def class_weights(labels, k, mode):
    """Age-group weights from training labels (inverse or square-root inverse frequency, mean 1)."""
    if mode == 'none':
        return None
    counts = np.bincount(labels, minlength=k).astype(float)
    if np.any(counts == 0):
        raise ValueError('Training data must contain every age group for class weighting')
    weights = 1 / counts
    if mode == 'sqrt':
        weights = np.sqrt(weights)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def collect_predictions(model, loader, device, model_name='faceord', include_cumulatives=False,
                        include_logits=False):
    """Predictions in loader order: (age_true, gender_true, age_prob, gender_prob).

    include_cumulatives appends the raw cumulative outputs c_hat (None for flat age
    heads); include_logits appends c_hat and then the raw age logits.
    """
    model.eval()
    ages, genders, age_probs, gender_probs = [], [], [], []
    cumulative, logits = [], []
    for batch in loader:
        out = forward_model(model, batch, device, model_name)
        ages.append(batch['age_group'].numpy())
        genders.append(batch['gender'].numpy())
        age_probs.append(out['age_p_hat'].float().cpu().numpy())
        gender_probs.append(out['gender_prob'].float().cpu().numpy())
        if (include_cumulatives or include_logits) and out.get('age_c_hat') is not None:
            cumulative.append(out['age_c_hat'].float().cpu().numpy())
        if include_logits:
            logits.append(out['age_logits'].float().cpu().numpy())
    if not ages:
        raise ValueError('Evaluation subset is empty')
    pred = (np.concatenate(ages), np.concatenate(genders),
            np.concatenate(age_probs), np.concatenate(gender_probs))
    if include_cumulatives or include_logits:
        pred += (np.concatenate(cumulative) if cumulative else None,)
    if include_logits:
        pred += (np.concatenate(logits),)
    return pred


def prediction_metrics(pred):
    """Age and gender metrics, plus ordinal violation diagnostics when c_hat is available."""
    age, gender, ap, gp = pred[:4]
    result = {'age': compute_metrics(age, ap), 'gender': compute_metrics(gender, gp, binary=True)}
    if len(pred) > 4 and pred[4] is not None:
        from .metrics import cumulative_diagnostics
        result['cumulative_diagnostics'] = cumulative_diagnostics(pred[4])
    return result


def validation_score(pred):
    """Checkpoint selection uses unweighted age NLL + binary gender NLL."""
    age, gender, ap, gp = pred
    age_nll = -np.log(np.clip(ap[np.arange(len(age)), age], 1e-7, 1)).mean()
    true_gp = np.where(gender == 1, gp, 1 - gp)
    return float(age_nll - np.log(np.clip(true_gp, 1e-7, 1)).mean())


def lr_for_epoch(epoch, cfg, base_lr):
    """Linear warm-up for WARMUP_EPOCHS, then cosine decay from base_lr to LR_MIN."""
    warmup = min(cfg.WARMUP_EPOCHS, max(0, cfg.EPOCHS - 1))
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, cfg.EPOCHS - warmup - 1)
    return cfg.LR_MIN + 0.5 * (base_lr - cfg.LR_MIN) * (1 + math.cos(math.pi * progress))


def freeze_warmup(model, cfg, freeze):
    """Freeze or unfreeze the first two ConvNeXt-S stages of the FaceOrd global encoder.

    torchvision ConvNeXt features are [stem, stage1, downsample, stage2, ...].
    """
    if cfg.BACKBONE != 'convnext_s' or not hasattr(model, 'global_encoder'):
        return
    for module in list(model.global_encoder.features.children())[:4]:
        for parameter in module.parameters():
            parameter.requires_grad_(not freeze)


def train_one(model, criterion, train_ds, eval_ds, train_indices, val_indices,
              cfg, model_name, variant, destination, metadata):
    """Train one model and keep the epoch with the lowest validation NLL.

    Optimiser: SGD (momentum 0.9, lr 1e-4) for VGG19/ResNet50/DenseNet169, AdamW
    (cfg.LR) otherwise; AdamW models use the configured warm-up. Gradients are
    clipped to GRAD_CLIP and AMP is used on CUDA. After epoch CLEAN_AFTER_EPOCH
    the DataCleaner resamples the training indices for every following epoch.
    Writes best.pt, history.json and cleaner/epoch_<n>.json to `destination`;
    returns the reloaded best model and its validation metrics.
    """
    destination = Path(destination)
    if (destination / 'best.pt').exists() or (destination / 'history.json').exists():
        raise FileExistsError(f'Existing experiment in {destination}; choose a new output folder')
    destination.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.DEVICE)
    model, criterion = model.to(device), criterion.to(device)
    cw_params = list(criterion.parameters())
    groups = [{'params': list(model.parameters()), 'weight_decay': cfg.WEIGHT_DECAY}]
    if cw_params:
        groups.append({'params': cw_params, 'weight_decay': 0.0})
    conventional = model_name in {'vgg19', 'resnet50', 'densenet169'}
    base_lr = 1e-4 if conventional else cfg.LR
    optimizer = (torch.optim.SGD(groups, lr=base_lr, momentum=0.9) if conventional
                 else torch.optim.AdamW(groups, lr=base_lr))
    amp_enabled = cfg.AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    val_loader = make_loader(eval_ds, val_indices, cfg)
    current_indices = list(train_indices)
    train_loader = make_loader(train_ds, current_indices, cfg, True, cfg.TRAIN_SEED)
    metadata = {**metadata, 'training_seed': cfg.TRAIN_SEED}
    if hasattr(model, 'source_provenance'):
        metadata['upstream_source'] = model.source_provenance
    best, best_epoch, history = float('inf'), 0, []
    for epoch in range(cfg.EPOCHS):
        freeze_warmup(model, cfg, epoch < cfg.WARMUP_EPOCHS)
        learning_rate = lr_for_epoch(epoch, cfg, base_lr)
        if conventional:
            progress = epoch / max(1, cfg.EPOCHS - 1)
            learning_rate = cfg.LR_MIN + 0.5 * (base_lr - cfg.LR_MIN) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        model.train()
        running, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast('cuda', dtype=torch.float16) if amp_enabled else nullcontext()
            with ctx:
                out = forward_model(model, batch, device, model_name)
            losses = compute_loss(criterion, out, batch, device)
            if not torch.isfinite(losses['total']):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch + 1}')
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + cw_params, cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            batch_n = len(batch['age_group'])
            running += float(losses['total'].detach()) * batch_n
            count += batch_n
        val_pred = collect_predictions(model, val_loader, device, model_name)
        score = validation_score(val_pred)
        record = {'epoch': epoch + 1, 'lr': learning_rate, 'train_loss': running / count,
                  'validation_nll': score, 'validation': prediction_metrics(val_pred)}
        if hasattr(criterion, 'log_sigma_a'):
            record['sigma_age'] = float(criterion.log_sigma_a.detach().clamp(-2, 2).exp().cpu())
            record['sigma_gender'] = float(criterion.log_sigma_g.detach().clamp(-2, 2).exp().cpu())
        history.append(record)
        if score < best:
            best, best_epoch = score, epoch + 1
            save_checkpoint(destination / 'best.pt', model, criterion, cfg, model_name, variant,
                            {**metadata, 'best_epoch': best_epoch, 'validation_nll': best})
        print(f"  epoch {epoch + 1}/{cfg.EPOCHS}: train={running/count:.4f} val_nll={score:.4f}", flush=True)
        if cfg.CLEAN and epoch + 1 >= cfg.CLEAN_AFTER_EPOCH and epoch + 1 < cfg.EPOCHS:
            cleaner = DatasetCleaner(cfg.CLEAN_FRACTION, cfg.TRAIN_SEED + epoch + 1,
                                     cfg.CLEAN_EASY_CONFIDENCE)
            current_indices = cleaner.clean(model, eval_ds, list(train_indices), device, model_name,
                                            cfg.BATCH_SIZE, cfg.NUM_WORKERS)
            if set(current_indices) - set(train_indices) or len(current_indices) != len(train_indices):
                raise RuntimeError('DataCleaner must preserve size and training membership')
            record['cleaner_for_epoch'] = epoch + 2
            record['cleaner'] = cleaner.last_info
            write_json(destination / 'cleaner' / f'epoch_{epoch + 2}.json',
                       {'indices': current_indices, **cleaner.last_info})
            train_loader = make_loader(train_ds, current_indices, cfg, True, cfg.TRAIN_SEED + epoch + 1)
    freeze_warmup(model, cfg, False)
    write_json(destination / 'history.json', history)
    selected, _, _ = load_checkpoint(destination / 'best.pt', device)
    return selected, history[best_epoch - 1]['validation']


def export_prediction_rows(pred, indices, samples, root, fold):
    """One row per held-out sample: labels, predictions, probabilities, raw c_hat and race label."""
    age, gender, ap, gp = pred[:4]
    rows = []
    for j, index in enumerate(indices):
        row = {'fold': fold, 'sample_idx': int(index),
               'filename': Path(samples[index][0]).relative_to(root).as_posix(),
               'age_true': int(age[j]), 'age_pred': int(ap[j].argmax()),
               'gender_true': int(gender[j]), 'gender_pred': int(gp[j] >= 0.5),
               'gender_prob': float(gp[j])}
        row.update({f'age_prob_{k}': float(value) for k, value in enumerate(ap[j])})
        parts = Path(samples[index][0]).name.split('_')
        row['race_label'] = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        if len(pred) > 4 and pred[4] is not None:
            row.update({f'age_c_{k}': float(value) for k, value in enumerate(pred[4][j])})
        rows.append(row)
    return rows


def aggregate_fold_metrics(folds):
    """Mean and sample standard deviation (ddof=1) of every metric across folds."""
    result = {'age': {}, 'gender': {}}
    for task in result:
        for key in ('accuracy', 'f1', 'auc', 'map', 'brier', 'ece', 'ece_calibrated', 'mae_class', 'adj_acc'):
            values = [f[task].get(key) for f in folds]
            values = [v for v in values if v is not None]
            if values:
                result[task][key] = {'mean': float(np.mean(values)),
                                     'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    return result


def runtime_metadata():
    """Library versions, CUDA and GPU of the current run."""
    import torchvision
    from importlib.metadata import PackageNotFoundError, version
    try:
        face_alignment = version('face-alignment')
    except PackageNotFoundError:
        face_alignment = None
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
            'face_alignment': face_alignment, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'platform': platform.platform()}


def run_cv(cfg, model_name='faceord', variant='f_full', split_path=None, pool_indices=None,
           evaluate_outer_test=True, tau_selection=None, hyperparameter_selection=None):
    """K-fold cross-validation with persisted, shared splits.

    Fold i trains with seed FOLD_SEEDS[i-1]. `pool_indices` restricts the dataset
    (inner folds of the sensitivity studies); `tau_selection` and
    `hyperparameter_selection` supply per-fold settings chosen on training data.
    After training, a temperature for the age head is fitted on the validation
    subset and gives the calibrated ECE of the test fold (age.ece_calibrated).
    Writes per-fold metrics, metrics.json and the out-of-fold predictions
    (oof_predictions.csv/.xlsx) to cfg.SAVE_DIR.
    """
    cfg.validate()
    seed_everything(cfg.SEED)
    eval_ds = UTKFaceDataset(cfg.DATA_ROOT, train=False, cfg=cfg)
    train_ds = UTKFaceDataset(cfg.DATA_ROOT, train=True, cfg=cfg)
    # Nested sensitivity runs restrict the dataset before creating INNER folds.
    if pool_indices is not None:
        eval_ds.samples = [eval_ds.samples[i] for i in pool_indices]
        train_ds.samples = [train_ds.samples[i] for i in pool_indices]
    out_dir = Path(cfg.SAVE_DIR)
    split_path = split_path or out_dir / 'splits.json'
    folds, split_meta = load_or_create_folds(split_path, eval_ds.samples, cfg.DATA_ROOT, cfg)
    for fold in folds:
        if (out_dir / f"fold_{fold['fold']}" / 'best.pt').exists():
            raise FileExistsError(f'Existing run at {out_dir}; choose a new output path')
    write_json(out_dir / 'run_metadata.json', {'config': cfg.to_dict(), 'model': model_name,
               'variant': variant, 'environment': runtime_metadata(), 'split': split_meta,
               'split_manifest': str(split_path), 'outer_test_evaluated': evaluate_outer_test})
    base_tau = cfg.LABEL_DIST_TAU
    selectable = ['LABEL_DIST_TAU', 'LAMBDA_MONO', 'LAMBDA_DIST', 'WEIGHTING',
                  'FIXED_AGE_WEIGHT', 'FIXED_GENDER_WEIGHT']
    base_settings = {key: getattr(cfg, key) for key in selectable}
    selected_settings = {}
    if hyperparameter_selection:
        selection = json.loads(Path(hyperparameter_selection).read_text())
        if selection['outer_split'] != split_meta:
            raise ValueError('Hyperparameters were selected on different outer splits')
        selected_settings = selection['selected_settings']
        if set(selected_settings) != {str(i) for i in range(1, cfg.KFOLD_K + 1)}:
            raise ValueError('Hyperparameter selection must cover all outer folds')
        for settings in selected_settings.values():
            if set(settings) - set(selectable):
                raise ValueError('Unsupported selected hyperparameters')
    tau_map = {}
    if tau_selection is not None:
        selection = json.loads(Path(tau_selection).read_text())
        if selection['outer_split'] != split_meta:
            raise ValueError('Tau selection uses different outer splits')
        tau_map = selection['selected_tau']
        if set(tau_map) != {str(i) for i in range(1, cfg.KFOLD_K + 1)}:
            raise ValueError('Tau selection must cover every outer fold')
    fold_results, rows = [], []
    for split in folds:
        fold = split['fold']
        for key, value in {**base_settings, **selected_settings.get(str(fold), {})}.items():
            setattr(cfg, key, value)
        cfg.LABEL_DIST_TAU = float(tau_map.get(str(fold), base_tau))
        if 'LABEL_DIST_TAU' in selected_settings.get(str(fold), {}):
            cfg.LABEL_DIST_TAU = selected_settings[str(fold)]['LABEL_DIST_TAU']
        cfg.validate()
        if cfg.LABEL_DIST_TAU <= 0:
            raise ValueError('Selected tau must be positive')
        cfg.TRAIN_SEED = cfg.FOLD_SEEDS[fold - 1]
        seed_everything(cfg.TRAIN_SEED)
        print(f'Fold {fold}/{cfg.KFOLD_K}: train={len(split["train"])} '
              f'val={len(split["val"])} test={len(split["test"])}', flush=True)
        weights = class_weights([eval_ds.samples[i][1] for i in split['train']], cfg.K,
                                cfg.CLASS_WEIGHTS if model_name == 'faceord' else 'none')
        model = make_model(cfg, model_name, variant)
        criterion = make_criterion(cfg, model_name, variant, weights)
        metadata = {'fold': fold, 'split': split_meta, 'purpose': 'outer_cv',
                    'train_indices': split['train'], 'val_indices': split['val'],
                    'test_indices': split['test']}
        selected, validation = train_one(model, criterion, train_ds, eval_ds,
                                          split['train'], split['val'], cfg, model_name, variant,
                                          out_dir / f'fold_{fold}', metadata)
        if evaluate_outer_test:
            device = resolve_device(cfg.DEVICE)
            val_pred = collect_predictions(selected, make_loader(eval_ds, split['val'], cfg),
                                           device, model_name, include_logits=True)
            ordinal = val_pred[4] is not None
            temperature = fit_temperature(val_pred[0], val_pred[5], ordinal)
            pred = collect_predictions(selected, make_loader(eval_ds, split['test'], cfg),
                                       device, model_name, include_cumulatives=True, include_logits=True)
            metrics = prediction_metrics(pred)
            metrics['age']['ece_calibrated'] = calibrated_ece(pred[0], pred[5], temperature, ordinal)
            metrics['temperature'] = temperature
            rows.extend(export_prediction_rows(pred, split['test'], eval_ds.samples,
                                                Path(cfg.DATA_ROOT), fold))
        else:
            metrics = validation
        fold_results.append({'fold': fold, 'training_seed': cfg.TRAIN_SEED, 'tau': cfg.LABEL_DIST_TAU,
                             'selected_settings': {key: getattr(cfg, key) for key in selectable},
                             **metrics, 'validation': validation})
        write_json(out_dir / f'fold_{fold}' / 'metrics.json', fold_results[-1])
        del model, criterion, selected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for key, value in base_settings.items():
        setattr(cfg, key, value)
    result = {'model': model_name, 'variant': variant, 'config': cfg.to_dict(),
              'split': split_meta, 'evaluation': 'outer_test' if evaluate_outer_test else 'inner_validation',
              'folds': fold_results, 'summary': aggregate_fold_metrics(fold_results)}
    write_json(out_dir / 'metrics.json', result)
    if evaluate_outer_test:
        df = pd.DataFrame(rows).sort_values('sample_idx')
        if df['sample_idx'].tolist() != list(range(len(eval_ds))):
            raise RuntimeError('OOF predictions must cover every sample exactly once')
        df.to_csv(out_dir / 'oof_predictions.csv', index=False)
        df.to_excel(out_dir / 'oof_predictions.xlsx', sheet_name='predictions', index=False)
    return result


def run_final(cfg, model_name='faceord', variant='f_full'):
    """Train one source-domain checkpoint for zero-shot transfer, with source validation."""
    from sklearn.model_selection import train_test_split
    from .splits import dataset_fingerprint
    cfg.validate()
    seed_everything(cfg.TRAIN_SEED)
    eval_ds = UTKFaceDataset(cfg.DATA_ROOT, train=False, cfg=cfg)
    train_ds = UTKFaceDataset(cfg.DATA_ROOT, train=True, cfg=cfg)
    indices = np.arange(len(eval_ds))
    train, val = train_test_split(indices, stratify=[s[1] for s in eval_ds.samples],
                                 test_size=cfg.VAL_FRACTION, random_state=cfg.SEED)
    weights = class_weights([eval_ds.samples[i][1] for i in train], cfg.K,
                            cfg.CLASS_WEIGHTS if model_name == 'faceord' else 'none')
    metadata = {'purpose': 'final_source_domain',
                'dataset_sha256': dataset_fingerprint(eval_ds.samples, Path(cfg.DATA_ROOT)),
                'train_indices': train.tolist(), 'val_indices': val.tolist()}
    out_dir = Path(cfg.SAVE_DIR)
    model, validation = train_one(make_model(cfg, model_name, variant),
                                  make_criterion(cfg, model_name, variant, weights),
                                  train_ds, eval_ds, train.tolist(), val.tolist(),
                                  cfg, model_name, variant, out_dir, metadata)
    write_json(out_dir / 'run_metadata.json', {'config': cfg.to_dict(), 'model': model_name,
               'variant': variant, 'environment': runtime_metadata(), 'validation': validation,
               'purpose': 'final_source_domain'})
    return model
