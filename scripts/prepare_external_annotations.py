"""Build the annotation CSV (file, age_group, gender) used by faceord.eval_cross_dataset.

Generic CSV with columns file, age, gender (file relative to the image root). The
age is an exact age for MORPH-II, a native category such as "(25, 32)" for
Adience, or a range such as "20-29" for FairFace:

    python scripts/prepare_external_annotations.py --dataset morph \
        --input /data/morph/annotations_raw.csv --output /data/morph/annotations.csv
    python scripts/prepare_external_annotations.py --dataset fairface \
        --input /data/fairface/fairface_label_val.csv --output /data/fairface/annotations.csv

Original Adience fold files (tab-separated fold_*_data.txt):

    python scripts/prepare_external_annotations.py --dataset adience \
        --adience-folds /data/adience/fold_*_data.txt --images /data/adience \
        --output /data/adience/annotations.csv

Adience rows whose age is not one of the eight native categories, or whose gender
is not m/f, are skipped and counted. FairFace ranges are mapped approximately to
the eight groups (faceord.age_mapping.FAIRFACE_AGE_GROUPS). A
<output>.mapping.json file records the mapping and the number of samples.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faceord.age_mapping import (FAIRFACE_AGE_GROUPS, adience_age_group, canonical_gender,
                                 exact_age_group, fairface_age_group)

ADIENCE_LAYOUT = {'aligned': ('aligned', 'landmark_aligned_face'),
                  'faces': ('faces', 'coarse_tilt_aligned_face')}
AGE_MAPPERS = {'adience': adience_age_group, 'morph': exact_age_group, 'fairface': fairface_age_group}
MAPPING_NOTES = {
    'adience': 'eight native Adience categories mapped one-to-one to groups 0-7',
    'morph': 'exact ages, bins [0, 3, 7, 14, 22, 34, 45, 54]',
    'fairface': f'approximate mapping of FairFace ranges: {FAIRFACE_AGE_GROUPS}',
}


def read_generic(path):
    """Rows of a file/age/gender CSV."""
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'file', 'age', 'gender'} <= set(rows[0]):
        raise ValueError('Input CSV requires the columns file, age and gender')
    return rows, {}


def read_adience_folds(paths, variant, image_root=None):
    """Rows of the original Adience fold files with image paths relative to the dataset root."""
    folder, prefix = ADIENCE_LAYOUT[variant]
    rows, skipped = [], {'age_label': 0, 'gender_label': 0, 'missing_file': 0}
    for fold_file in sorted(paths):
        with Path(fold_file).open(newline='') as handle:
            for record in csv.DictReader(handle, delimiter='\t'):
                record = {key.strip(): (value or '').strip() for key, value in record.items() if key}
                try:
                    adience_age_group(record['age'])
                except (ValueError, SyntaxError):
                    skipped['age_label'] += 1
                    continue
                if record.get('gender', '').lower() not in {'m', 'f'}:
                    skipped['gender_label'] += 1
                    continue
                name = f"{folder}/{record['user_id']}/{prefix}.{record['face_id']}.{record['original_image']}"
                if image_root is not None and not (Path(image_root) / name).is_file():
                    skipped['missing_file'] += 1
                    continue
                rows.append({'file': name, 'age': record['age'], 'gender': record['gender']})
    return rows, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='CSV with file, age, gender')
    source.add_argument('--adience-folds', nargs='+', type=Path, help='Original Adience fold_*_data.txt files')
    p.add_argument('--adience-variant', choices=list(ADIENCE_LAYOUT), default='aligned')
    p.add_argument('--images', type=Path, help='Image root; rows without an image file are skipped')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--dataset', required=True, choices=list(AGE_MAPPERS))
    args = p.parse_args()
    if args.adience_folds and args.dataset != 'adience':
        p.error('--adience-folds requires --dataset adience')

    if args.input:
        rows, skipped = read_generic(args.input)
        sources = [args.input]
    else:
        rows, skipped = read_adience_folds(args.adience_folds, args.adience_variant, args.images)
        sources = args.adience_folds
    if not rows:
        raise ValueError('No usable annotations')
    if len({r['file'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image paths')

    converted = []
    for row in rows:
        path = Path(row['file'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('file must be a relative path within the image root')
        group = AGE_MAPPERS[args.dataset](row['age'])
        converted.append({'file': row['file'], 'age_group': group,
                          'gender': canonical_gender(row['gender']), 'original_age': row['age']})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    digest = hashlib.sha256()
    for path in sources:
        digest.update(Path(path).read_bytes())
    metadata = {'dataset': args.dataset, 'input_sha256': digest.hexdigest(),
                'mapping': MAPPING_NOTES[args.dataset],
                'gender': '0=male, 1=female', 'n_samples': len(converted), 'skipped': skipped}
    args.output.with_suffix('.mapping.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{len(converted)} annotations -> {args.output}; skipped: {skipped}')


if __name__ == '__main__':
    main()
