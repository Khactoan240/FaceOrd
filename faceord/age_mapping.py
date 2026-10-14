"""Age-group and gender labels shared by all datasets.

Exact ages (UTKFace, MORPH-II) use the bins 0-2, 3-6, 7-13, 14-21, 22-33, 34-44,
45-53, 54+. The eight native Adience categories map one-to-one onto groups 0-7.
FairFace age ranges do not align with these groups and are mapped approximately
(FAIRFACE_AGE_GROUPS).
"""
import ast
import bisect

AGE_BINS = [0, 3, 7, 14, 22, 34, 45, 54]
ADIENCE_CATEGORIES = [(0, 2), (4, 6), (8, 12), (15, 20),
                      (25, 32), (38, 43), (48, 53), (60, 100)]
FAIRFACE_AGE_GROUPS = {'0-2': 0, '3-9': 1, '10-19': 3, '20-29': 4, '30-39': 5,
                       '40-49': 5, '50-59': 6, '60-69': 7, 'more than 70': 7}


def exact_age_group(age):
    """Integer age in [0, 116] -> group index."""
    value = float(age)
    if not 0 <= value <= 116 or value != int(value):
        raise ValueError('Expected an integer age in [0, 116]')
    return bisect.bisect_right(AGE_BINS, int(value)) - 1


def adience_age_group(category):
    """Native Adience category such as "(25, 32)" -> group index."""
    pair = ast.literal_eval(category) if isinstance(category, str) else category
    if not isinstance(pair, (tuple, list)) or len(pair) != 2 or tuple(pair) not in ADIENCE_CATEGORIES:
        raise ValueError(f'Unknown native Adience age category: {category!r}')
    return ADIENCE_CATEGORIES.index(tuple(pair))


def fairface_age_group(label):
    """FairFace age range such as "20-29" -> group index (approximate)."""
    text = str(label).strip().lower()
    if text not in FAIRFACE_AGE_GROUPS:
        raise ValueError(f'Unknown FairFace age range: {label!r}')
    return FAIRFACE_AGE_GROUPS[text]


def canonical_gender(value):
    """m/male/0 -> 0, f/female/1 -> 1."""
    text = str(value).strip().lower()
    if text in {'m', 'male', '0'}:
        return 0
    if text in {'f', 'female', '1'}:
        return 1
    raise ValueError(f'Unknown gender annotation: {value!r}')
