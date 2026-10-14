"""Outer cross-validation folds stored in a JSON manifest shared by all methods.

Folds are stratified by age group (seed cfg.SEED); each outer training portion is
split again into inner training and a stratified validation subset. The manifest
records a fingerprint of the dataset so it cannot be reused with different data.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def dataset_fingerprint(samples, root):
    """SHA-256 of the ordered (relative path, age group, gender) records."""
    records = [[Path(path).relative_to(root).as_posix(), int(age), int(gender)]
               for path, age, gender in samples]
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_split(train, val, test, n):
    """Train, validation and test must be disjoint, non-empty and cover range(n)."""
    if any(any(not isinstance(i, (int, np.integer)) for i in seq) for seq in (train, val, test)):
        raise ValueError("Split indices must be integers")
    groups = [set(map(int, g)) for g in (train, val, test)]
    if any(not g for g in groups):
        raise ValueError("Every fold must have nonempty training, validation and test subsets")
    if any(len(g) != len(seq) for g, seq in zip(groups, (train, val, test))):
        raise ValueError("Base split contains duplicate indices")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Train, validation and outer test subsets must be disjoint")
    if set.union(*groups) != set(range(n)):
        raise ValueError("Fold does not cover the dataset exactly")


def create_folds(labels, k=5, seed=42, val_fraction=0.1):
    """Stratified k outer folds, each with a stratified validation split of its training part."""
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    if np.bincount(labels).min() < k:
        raise ValueError("Every age group needs at least k samples for stratified CV")
    result = []
    for fold, (outer_train, test) in enumerate(
        StratifiedKFold(k, shuffle=True, random_state=seed).split(indices, labels), 1
    ):
        train, val = train_test_split(outer_train, test_size=val_fraction,
                                     stratify=labels[outer_train], random_state=seed + fold)
        validate_split(train, val, test, len(labels))
        result.append({"fold": fold, "train": sorted(train.tolist()),
                       "val": sorted(val.tolist()), "test": sorted(test.tolist())})
    return result


def load_or_create_folds(path, samples, root, cfg):
    """Load the manifest at `path` (verifying dataset and settings) or create it."""
    path = Path(path)
    fingerprint = dataset_fingerprint(samples, Path(root))
    metadata = {"dataset_sha256": fingerprint, "n_samples": len(samples),
                "k": cfg.KFOLD_K, "seed": cfg.SEED, "val_fraction": cfg.VAL_FRACTION}
    if path.exists():
        stored = json.loads(path.read_text())
        if any(stored.get(key) != value for key, value in metadata.items()):
            raise ValueError("Split manifest differs from dataset/configuration; use a new path")
        folds = stored["folds"]
    else:
        folds = create_folds([s[1] for s in samples], cfg.KFOLD_K, cfg.SEED, cfg.VAL_FRACTION)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**metadata, "folds": folds}, indent=2))
    if len(folds) != cfg.KFOLD_K or [f["fold"] for f in folds] != list(range(1, cfg.KFOLD_K + 1)):
        raise ValueError("Invalid fold count or order in split manifest")
    for f in folds:
        validate_split(f["train"], f["val"], f["test"], len(samples))
    test_indices = [i for f in folds for i in f["test"]]
    if sorted(test_indices) != list(range(len(samples))):
        raise ValueError("Each sample must occur in exactly one outer test fold")
    metadata["split_sha256"] = hashlib.sha256(
        json.dumps(folds, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return folds, metadata
