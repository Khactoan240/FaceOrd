"""End-to-end latency per image (batch size 1).

    python -m faceord.benchmark_pipeline --checkpoint results/faceord/fold_1/best.pt \
        --images data/UTKFace/25_0_2_20170116174525125.jpg.chip.jpg --device cuda

Each iteration decodes the image and runs resize + CLAHE/gamma, landmark
estimation (2D-FAN or dlib), affine alignment, region extraction, host-to-device transfer, model
inference (including ordinal post-processing) and the final decisions. Every stage
is timed separately after warm-up; one-time model loading is excluded.
"""
import argparse
from contextlib import nullcontext
import hashlib
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .checkpoint import load_checkpoint
from .engine import resolve_device, write_json, runtime_metadata
from .preprocessing import FaceOrdPreprocessor, clahe_gamma, detect_landmarks, affine_align, extract_regions


def synchronize(device):
    """Wait for queued GPU work so wall-clock timings are accurate."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


@torch.inference_mode()
def timed_sample(path, model, cfg, model_name, device, amp):
    """Process one image through the full pipeline; returns per-stage times in ms."""
    preprocess = FaceOrdPreprocessor.from_config(cfg)
    synchronize(device)
    start = previous = time.perf_counter()
    timings = {}
    def mark(name):
        nonlocal previous
        synchronize(device)
        current = time.perf_counter()
        timings[name] = (current - previous) * 1000
        previous = current
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    image = clahe_gamma(cv2.resize(image, (224, 224)), cfg.GAMMA)
    mark('decode_resize_photometric_ms')
    points = detect_landmarks(image, cfg.LANDMARK_BACKEND, cfg.LANDMARK_DEVICE,
                              cfg.LANDMARK_FACE_DETECTOR, cfg.DLIB_PREDICTOR)
    mark('landmark_ms')
    image, points = affine_align(image, points)
    mark('affine_ms')
    patches = extract_regions(image, points, cfg.REGION_SIZE)
    global_img = preprocess.global_tf(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    regions = torch.stack([preprocess.region_tf(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))) for patch in patches]).unsqueeze(0)
    mark('regions_tensor_ms')
    global_img, regions = global_img.to(device), regions.to(device)
    mark('host_to_device_ms')
    with torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext():
        out = model(global_img, regions) if model_name == 'faceord' else model(global_img)
    mark('model_including_ordinal_rectification_ms')
    probabilities = out['age_p_hat'] if model_name == 'faceord' else out['age_logits'].softmax(1)
    probabilities.float().cpu().numpy().argmax(1)
    (out['gender_prob'].float().cpu().numpy() >= .5)
    mark('output_transfer_decisions_ms')
    timings['total_ms'] = (previous - start) * 1000
    return timings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--images', required=True, nargs='+', type=Path)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    p.add_argument('--landmark-device', default='cuda', choices=['cuda', 'cpu'], help='2D-FAN device')
    p.add_argument('--dlib-predictor', help='Override DLIB_PREDICTOR of the checkpoint')
    p.add_argument('--precision', choices=['float32', 'amp'], default='amp')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--output', type=Path, default=Path('results/pipeline_timing.json'))
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        p.error('Use at least one warm-up and two measured iterations')
    device = resolve_device(args.device)
    model, cfg, payload = load_checkpoint(args.checkpoint, device)
    cfg.LANDMARK_DEVICE, cfg.NUM_WORKERS = args.landmark_device, 0
    if args.dlib_predictor:
        cfg.DLIB_PREDICTOR = args.dlib_predictor
    cfg.validate()
    if cfg.LANDMARK_JITTER or cfg.LANDMARK_DROPOUT:
        raise ValueError('Timing uses unperturbed landmarks')
    amp = args.precision == 'amp'
    if amp and device.type != 'cuda':
        raise ValueError('AMP timing requires CUDA')
    for number in range(args.warmup):
        timed_sample(args.images[number % len(args.images)], model, cfg, payload['model_name'], device, amp)
    samples = [timed_sample(args.images[number % len(args.images)], model, cfg, payload['model_name'], device, amp)
               for number in range(args.repeats)]
    summary = {key: {'mean': float(np.mean([row[key] for row in samples])),
                     'std': float(np.std([row[key] for row in samples], ddof=1))} for key in samples[0]}
    write_json(args.output, {'batch_size': 1, 'warmup': args.warmup, 'repeats': args.repeats,
               'precision': args.precision, 'landmark_precision': 'float32', 'device': str(device),
               'landmark_device': cfg.LANDMARK_DEVICE, 'environment': runtime_metadata(),
               'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               'images': [str(path) for path in args.images], 'summary_ms': summary, 'samples_ms': samples,
               'landmark_backend': cfg.LANDMARK_BACKEND,
               'scope': 'image decoding, CLAHE/gamma, landmarks, alignment, regions, model, ordinal post-processing and output decisions'})


if __name__ == '__main__':
    main()
