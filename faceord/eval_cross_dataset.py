"""Direct transfer of UTKFace checkpoints to MORPH-II, Adience or FairFace (no fine-tuning or calibration).

    python -m faceord.eval_cross_dataset --dataset adience \
        --checkpoints results/faceord/fold_*/best.pt \
        --annotations /data/adience/annotations.csv --images /data/adience \
        --landmarks /data/adience/landmarks.npz

Annotation files are produced by scripts/prepare_external_annotations.py. Every
checkpoint is evaluated separately; the summary is the mean and sample standard
deviation of the per-checkpoint metrics.
"""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .checkpoint import load_checkpoint
from .engine import collect_predictions, prediction_metrics, resolve_device, write_json, aggregate_fold_metrics
from .landmarks import load_cache
from .preprocessing import FaceOrdPreprocessor


class ExternalDataset(Dataset):
    """Target-domain images from an annotation CSV.

    Columns: file (path relative to the image root), age_group (0..7), gender (0/1).
    Landmarks are read from `landmarks` (cache keyed by `file`) when available.
    """
    def __init__(self, annotations, image_root, cfg, landmarks=None):
        root = Path(image_root).resolve()
        df = pd.read_csv(annotations)
        if not {'file', 'age_group', 'gender'} <= set(df):
            raise ValueError('Annotation CSV requires file, age_group and gender')
        if df.empty or df['file'].duplicated().any():
            raise ValueError('Annotation file must contain nonempty, unique image paths')
        for col, lo, hi in [('age_group', 0, 7), ('gender', 0, 1)]:
            values = df[col].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f'{col} must contain integer labels')
            if np.any(values < lo) or np.any(values > hi):
                raise ValueError(f'Invalid {col} labels')
        self.rows = df
        self.samples = []
        for row in df.itertuples():
            path = (root / str(row.file)).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Missing image or path outside image root: {row.file}')
            self.samples.append((str(path), int(row.age_group), int(row.gender)))
        self.preprocessor = FaceOrdPreprocessor.from_config(cfg)
        self.landmarks = load_cache(landmarks, gamma=cfg.GAMMA, backend=cfg.LANDMARK_BACKEND)
        self.keys = [Path(str(name)).as_posix() for name in df['file']]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, age, gender = self.samples[index]
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f'Cannot decode {path}')
        global_img, regions = self.preprocessor(image, self.landmarks.get(self.keys[index]))
        return {'global_img': global_img, 'regions': regions,
                'age_group': torch.tensor(age), 'gender': torch.tensor(gender)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoints', '--checkpoint', required=True, nargs='+', type=Path)
    p.add_argument('--annotations', required=True, type=Path)
    p.add_argument('--images', required=True)
    p.add_argument('--landmarks', help='Landmark cache of the target images (faceord.landmarks --recursive)')
    p.add_argument('--dataset', choices=['morph', 'adience', 'fairface'], required=True)
    p.add_argument('--age-mapping-description', default='',
                   help='Optional note on the label mapping, stored with the results')
    p.add_argument('--output', default='results/cross_dataset')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--workers', default=0, type=int)
    p.add_argument('--batch-size', default=32, type=int)
    args = p.parse_args()
    device = resolve_device(args.device)
    output = Path(args.output)
    if (output / f'{args.dataset}_metrics.json').exists():
        raise FileExistsError('Choose a new output directory for this evaluation')
    hashes, seeds, measurements, identities = set(), set(), [], []
    for number, checkpoint in enumerate(args.checkpoints, 1):
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model, cfg, payload = load_checkpoint(checkpoint, device)
        metadata = payload['metadata']
        if metadata.get('purpose') not in {'final_source_domain', 'outer_cv'}:
            raise ValueError('Expected a UTKFace checkpoint from faceord.kfold (outer fold or --final)')
        seed = metadata.get('training_seed')
        if seed is None or seed in seeds or digest in hashes:
            raise ValueError('Checkpoints must have distinct hashes and recorded training seeds')
        seeds.add(seed)
        hashes.add(digest)
        fields = ['IMG_SIZE', 'REGION_SIZE', 'GAMMA', 'D', 'K', 'M', 'BACKBONE', 'MIVOLO_VARIANT',
                  'GLOBAL_CHANNELS', 'REGION_CHANNELS', 'TRANSFORMER_LAYERS',
                  'TRANSFORMER_HEADS', 'TRANSFORMER_DIM_FF', 'LANDMARK_BACKEND',
                  'LANDMARK_FACE_DETECTOR', 'LANDMARK_JITTER', 'LANDMARK_DROPOUT', 'LANDMARK_SCALE']
        identity = {'model': payload['model_name'], 'variant': payload['variant'],
                    'source_purpose': metadata['purpose'], 'upstream_source': metadata.get('upstream_source'),
                    'configuration': {key: getattr(cfg, key) for key in fields},
                    'source_dataset_sha256': metadata.get('dataset_sha256', metadata.get('split', {}).get('dataset_sha256'))}
        if not identity['source_dataset_sha256']:
            raise ValueError('Source checkpoint must record its UTKFace dataset fingerprint')
        if identities and identity != identities[0]:
            raise ValueError('Transfer checkpoints must share architecture, preprocessing and source dataset')
        identities.append(identity)
        if cfg.LANDMARK_JITTER or cfg.LANDMARK_DROPOUT:
            raise ValueError('Cross-dataset evaluation uses unperturbed landmarks')
        cfg.NUM_WORKERS = args.workers
        cfg.validate()
        dataset = ExternalDataset(args.annotations, args.images, cfg, args.landmarks)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
        pred = collect_predictions(model, loader, device, payload['model_name'])
        measurements.append({'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                             'training_seed': seed, 'source_purpose': metadata['purpose'],
                             **prediction_metrics(pred)})
        age, gender, ap, gp = pred
        rows = pd.DataFrame({'file': dataset.rows['file'], 'age_true': age, 'age_pred': ap.argmax(1),
                             'gender_true': gender, 'gender_pred': (gp >= 0.5).astype(int), 'gender_prob': gp})
        for k in range(cfg.K):
            rows[f'age_prob_{k}'] = ap[:, k]
        output.mkdir(parents=True, exist_ok=True)
        rows.to_csv(output / f'{args.dataset}_checkpoint_{number}_predictions.csv', index=False)
        del model
    summary = aggregate_fold_metrics(measurements)
    if len(measurements) == 1:
        for task in summary.values():
            for metric in task.values():
                metric['std'] = None
    result = {'dataset': args.dataset, 'n_samples': len(dataset), 'n_checkpoints': len(measurements),
              'zero_shot': True, 'checkpoint_measurements': measurements, 'summary': summary,
              'aggregation': 'mean and sample std (ddof=1) of checkpoint metrics; no probability ensemble',
              'annotation_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              'age_mapping': args.age_mapping_description, 'calibration': 'none', 'source': identities[0]}
    write_json(output / f'{args.dataset}_metrics.json', result)
    print(summary)


if __name__ == '__main__':
    main()
