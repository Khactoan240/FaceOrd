"""Experiment configuration.

Default values are the final FaceOrd configuration. Every experiment loads a
JSON file from ``configs/`` whose keys must match the fields below.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    # Input images
    IMG_SIZE: int = 224                 # global face resolution
    REGION_SIZE: int = 64               # side length of each local region patch
    GAMMA: float = 1.2                  # gamma correction applied after CLAHE

    # Architecture
    D: int = 256                        # shared feature dimension d
    M: int = 6                          # number of landmark-defined regions
    K: int = 8                          # number of ordered age groups
    AGE_BINS: list = field(default_factory=lambda: [0, 3, 7, 14, 22, 34, 45, 54])
    BACKBONE: str = "convnext_s"        # global encoder (see models.GlobalCNNEncoder)
    USE_PRETRAINED_BACKBONE: bool = True
    GLOBAL_CHANNELS: list = field(default_factory=lambda: [64, 128, 256, 512])  # "scratch" backbone only
    REGION_CHANNELS: list = field(default_factory=lambda: [32, 64, 128])
    TRANSFORMER_LAYERS: int = 2
    TRANSFORMER_HEADS: int = 4
    TRANSFORMER_DIM_FF: int = 512

    # Age loss
    LABEL_DIST_TAU: float = 0.5         # width of the soft label distribution
    LAMBDA_MONO: float = 0.1            # weight of the monotonicity penalty
    LAMBDA_DIST: float = 0.5            # weight of the label-distribution loss
    CLASS_WEIGHTS: str = "sqrt"         # age-group weighting: none | sqrt | inverse

    # Optimisation
    EPOCHS: int = 30
    BATCH_SIZE: int = 32
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    LR_MIN: float = 1e-6                # cosine decay floor
    WARMUP_EPOCHS: int = 3              # linear warm-up; ConvNeXt stages 1-2 frozen meanwhile
    GRAD_CLIP: float = 5.0
    AMP: bool = True                    # automatic mixed precision (CUDA only)

    # DataCleaner
    CLEAN: bool = True
    CLEAN_AFTER_EPOCH: int = 3          # first resampled training set is used in epoch 4
    CLEAN_FRACTION: float = 0.2         # share of easy samples replaced per epoch
    CLEAN_EASY_CONFIDENCE: float = 0.90

    # Evaluation protocol
    KFOLD_K: int = 5
    VAL_FRACTION: float = 0.1           # stratified validation split of each outer training portion

    # Landmarks
    LANDMARK_BACKEND: str = "2d_fan"    # "2d_fan": pretrained 2D-FAN (face-alignment 1.4.1)
                                        # "dlib": HOG face detector + 68-point shape predictor
    LANDMARK_DEVICE: str = "cpu"        # 2D-FAN only
    LANDMARK_FACE_DETECTOR: str = "sfd" # 2D-FAN only
    DLIB_PREDICTOR: str = "data/shape_predictor_68_face_landmarks.dat"  # dlib only
    LANDMARK_CACHE: str = "data/landmarks_utkface.npz"  # built by `python -m faceord.landmarks`
    LANDMARK_JITTER: float = 0.0        # evaluation-time perturbation (robustness study)
    LANDMARK_DROPOUT: float = 0.0
    LANDMARK_SCALE: str = "face_diagonal"

    # Task weighting
    WEIGHTING: str = "uncertainty"      # uncertainty | fixed
    FIXED_AGE_WEIGHT: float = 1.0
    FIXED_GENDER_WEIGHT: float = 1.0

    # Official comparison models (local source trees)
    FACEXFORMER_SOURCE: str = "external/facexformer"
    MIVOLO_SOURCE: str = "external/MiVOLO"
    MIVOLO_VARIANT: str = "mivolo_d3_224"

    # Runtime
    NUM_WORKERS: int = 4
    SEED: int = 42                      # outer split seed, shared by all methods
    FOLD_SEEDS: list = field(default_factory=lambda: [3407, 3408, 3409, 3410, 3411])
    TRAIN_SEED: int = 3407              # set per fold from FOLD_SEEDS
    DATA_ROOT: str = "data/UTKFace"
    SAVE_DIR: str = "results/faceord"
    DEVICE: str = "auto"

    @classmethod
    def from_json(cls, path):
        """Load a configuration file; unknown keys are rejected."""
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def to_dict(self):
        return asdict(self)

    def validate(self):
        """Check value ranges and the fixed structure (8 groups, 6 regions, 224 px)."""
        import math
        for key in ['LABEL_DIST_TAU', 'LAMBDA_MONO', 'LAMBDA_DIST', 'CLEAN_FRACTION',
                    'CLEAN_EASY_CONFIDENCE', 'FIXED_AGE_WEIGHT', 'FIXED_GENDER_WEIGHT',
                    'LANDMARK_JITTER', 'LANDMARK_DROPOUT']:
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f'{key} must be finite')
        if self.LAMBDA_MONO < 0 or self.LAMBDA_DIST < 0:
            raise ValueError('Loss coefficients must be nonnegative')
        if not all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in [self.SEED, self.TRAIN_SEED, *self.FOLD_SEEDS]):
            raise ValueError('Seeds must be integers in [0, 2**32)')
        if self.K != 8 or self.M != 6 or self.IMG_SIZE != 224:
            raise ValueError("FaceOrd uses 8 age groups, 6 regions and 224px global inputs")
        if self.AGE_BINS != [0, 3, 7, 14, 22, 34, 45, 54]:
            raise ValueError("AGE_BINS must match the age-group definition")
        if self.LABEL_DIST_TAU <= 0 or not 0 < self.VAL_FRACTION < 1:
            raise ValueError("tau must be positive and VAL_FRACTION must lie in (0, 1)")
        if self.EPOCHS < 1 or self.BATCH_SIZE < 1 or self.KFOLD_K < 2:
            raise ValueError("epochs/batch size must be positive; CV needs at least two folds")
        if self.NUM_WORKERS < 0 or self.WARMUP_EPOCHS < 0:
            raise ValueError("worker and warm-up counts must be nonnegative")
        if not 0 <= self.CLEAN_FRACTION <= 1 or self.CLEAN_AFTER_EPOCH < 1:
            raise ValueError("Invalid DataCleaner configuration")
        if self.CLASS_WEIGHTS not in {"none", "sqrt", "inverse"}:
            raise ValueError("CLASS_WEIGHTS must be none, sqrt or inverse")
        if len(self.FOLD_SEEDS) < self.KFOLD_K or len(set(self.FOLD_SEEDS)) != len(self.FOLD_SEEDS):
            raise ValueError("FOLD_SEEDS must contain distinct seeds for every fold")
        if not 0 <= self.CLEAN_EASY_CONFIDENCE <= 1:
            raise ValueError("Invalid easy confidence threshold")
        if self.LANDMARK_BACKEND not in {"2d_fan", "dlib"}:
            raise ValueError("LANDMARK_BACKEND must be 2d_fan or dlib")
        if self.LANDMARK_BACKEND == "2d_fan" and self.LANDMARK_DEVICE.startswith("cuda") and self.NUM_WORKERS:
            raise ValueError("CUDA landmark inference requires NUM_WORKERS=0")
        if self.LANDMARK_JITTER < 0 or not 0 <= self.LANDMARK_DROPOUT < 1:
            raise ValueError("Invalid landmark perturbation")
        if self.LANDMARK_SCALE not in {"face_diagonal", "interocular"}:
            raise ValueError("LANDMARK_SCALE must be face_diagonal or interocular")
        if self.WEIGHTING not in {"fixed", "uncertainty"} or min(self.FIXED_AGE_WEIGHT, self.FIXED_GENDER_WEIGHT) <= 0:
            raise ValueError("Invalid task weighting")
        if self.D % self.TRANSFORMER_HEADS:
            raise ValueError("Feature dimension must be divisible by the attention head count")
