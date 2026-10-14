"""Model-side cost: parameters, FLOPs counted by the PyTorch profiler and inference latency.

    python -m faceord.flops --config configs/faceord.json --device cuda
    python -m faceord.flops --model resnet50 --config configs/resnet50.json --device cuda

Batch size 1, float32; landmark detection, alignment and preprocessing are excluded
(see faceord.benchmark_pipeline for end-to-end timing).
"""
import argparse
import time
from pathlib import Path
import torch
from .config import Config
from .engine import resolve_device, runtime_metadata, write_json
from .experiment import make_model, BASELINES


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/faceord.json')
    p.add_argument('--model', choices=['faceord', *BASELINES], default='faceord')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--output', default='results/computational_cost.json')
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        p.error('repeats must be positive and warmup nonnegative')
    cfg = Config.from_json(args.config)
    cfg.USE_PRETRAINED_BACKBONE = False
    device = resolve_device(args.device)
    model = make_model(cfg, args.model).to(device).eval()
    inputs = (torch.randn(1, 3, 224, 224, device=device),)
    if args.model == 'faceord':
        inputs += (torch.randn(1, 6, 3, cfg.REGION_SIZE, cfg.REGION_SIZE, device=device),)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            model(*inputs)
        counted_flops = sum(event.flops for event in profiler.key_averages())
        for _ in range(args.warmup):
            model(*inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            model(*inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - start) * 1000 / args.repeats
    result = {'model': args.model, 'parameters': sum(p.numel() for p in model.parameters()),
              'profiler_counted_gflops': counted_flops / 1e9, 'latency_ms': milliseconds,
              'batch_size': 1, 'precision': 'float32', 'warmup': args.warmup, 'repeats': args.repeats,
              'device': str(device), 'environment': runtime_metadata(),
              'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
              'excludes': ['landmark detection', 'alignment', 'regional extraction', 'preprocessing'],
              'flop_coverage': 'operations with PyTorch profiler FLOP formulas (convolutions, matrix products)'}
    write_json(Path(args.output), result)
    print(result)


if __name__ == '__main__':
    main()
