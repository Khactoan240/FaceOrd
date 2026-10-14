"""Print (and optionally run) the commands of an experiment group.

    python run_experiment.py --experiment baselines              # print the commands
    python run_experiment.py --experiment main --execute         # run them

Groups: main (FaceOrd five-fold CV), baselines (all comparison models), ablation,
tau (sensitivity study) and final (five source-domain checkpoints for transfer).
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

BASELINES = ['vgg19', 'resnet50', 'densenet169', 'optimized_vgg19',
             'mivolo_face', 'facexformer', 'swin_t', 'convnext_small']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', choices=['main', 'baselines', 'ablation', 'tau', 'final'],
                   default='main')
    p.add_argument('--data', default='data/UTKFace')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps', 'auto'])
    p.add_argument('--execute', action='store_true', help='Run the printed commands')
    args = p.parse_args()
    shared = ['--data', args.data, '--device', args.device]
    splits = ['--splits', 'results/shared_splits.json']
    if args.experiment == 'main':
        plans = [['-m', 'faceord.kfold', '--config', 'configs/faceord.json', *shared, *splits]]
    elif args.experiment == 'baselines':
        plans = [['-m', 'faceord.kfold', '--model', name, '--config', f'configs/{name}.json',
                  *shared, *splits] for name in BASELINES]
        plans.append(['-m', 'faceord.kfold', '--model', 'optimized_vgg19', '--config',
                      'configs/optimized_vgg19_no_clean.json', *shared, *splits])
    elif args.experiment == 'ablation':
        plans = [['-m', 'faceord.ablation', *shared, *splits]]
    elif args.experiment == 'tau':
        plans = [['-m', 'faceord.tau_sweep', *shared, '--outer-splits', 'results/shared_splits.json']]
    else:
        plans = [['-m', 'faceord.kfold', '--final', '--train-seed', str(seed), '--output',
                  f'results/final_source_seed_{seed}', *shared] for seed in range(3407, 3412)]
    for plan in plans:
        command = [sys.executable, *plan]
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == '__main__':
    main()
