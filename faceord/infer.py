"""Predict the age group and gender of one face image.

    python -m faceord.infer --checkpoint results/faceord/fold_1/best.pt --image face.jpg
"""
import argparse
from pathlib import Path
import cv2
import torch
from .checkpoint import load_checkpoint
from .dataset import GROUP_LABELS
from .engine import resolve_device, write_json
from .preprocessing import FaceOrdPreprocessor


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', default='results/inference.json')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = p.parse_args()
    device = resolve_device(args.device)
    model, cfg, payload = load_checkpoint(args.checkpoint, device)
    image = cv2.imread(str(Path(args.image)))
    if image is None:
        raise ValueError(f'Cannot decode {args.image}')
    preprocess = FaceOrdPreprocessor.from_config(cfg)
    global_img, regions = preprocess(image)
    with torch.inference_mode():
        if payload['model_name'] == 'faceord':
            out = model(global_img.unsqueeze(0).to(device), regions.unsqueeze(0).to(device))
            probs = out['age_p_hat'][0].cpu().tolist()
        else:
            out = model(global_img.unsqueeze(0).to(device))
            probs = out['age_logits'][0].softmax(0).cpu().tolist()
        gender_prob = float(out['gender_prob'][0].cpu())
    group = max(range(8), key=probs.__getitem__)
    result = {'age_group': group, 'age_range': GROUP_LABELS[group], 'age_probabilities': probs,
              'gender': int(gender_prob >= 0.5), 'gender_prob': gender_prob,
              'gender_encoding': {'0': 'male', '1': 'female'}}
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
