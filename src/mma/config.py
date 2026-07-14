"""Central constants shared by training and inference."""

PHASE_LABELS = [
    "Striking",
    "Grappling/Ground Work",
    "Clinch",
    "Transition/Takedown",
    "Neutral/Measuring Distance",
]
PRESSURE_LABELS = ["Fighter 1", "Fighter 2", "Mutual"]

PHASE2IDX = {label: index for index, label in enumerate(PHASE_LABELS)}
IDX2PHASE = {index: label for label, index in PHASE2IDX.items()}
PRESSURE2IDX = {label: index for index, label in enumerate(PRESSURE_LABELS)}
IDX2PRESSURE = {index: label for label, index in PRESSURE2IDX.items()}

NUM_PHASE_CLASSES = len(PHASE_LABELS)
NUM_PRESSURE_CLASSES = len(PRESSURE_LABELS)

# Clip geometry
CLIP_SECONDS = 5
NUM_FRAMES = 16  # frames sampled per clip (cached once by scripts/preprocess.py)
CACHE_SHORT_SIDE = 128  # cached frame height for 16:9 sources (width follows aspect)
CROP_SIZE = 112  # model input resolution
GATE_FRAMES = 4  # frames per clip used by the fight/no-fight gate

# Normalization stats per backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

RANDOM_SEED = 42

# Validation protocol used by the submission experiments.  This fight is never
# used for model selection, early stopping, threshold selection, or tuning.  The
# other ten fights form five development folds with exactly two validation
# fights per fold.
DEFAULT_HOLDOUT_FIGHT = "Paddy Pimblett vs Michael Chandler"
DEV_FOLDS = 5
