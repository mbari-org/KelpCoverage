# Model
DEFAULT_MODEL_NAME = "mobile_sam"

# pixel mean / std taken directly from SAM github
# https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/sam.py#L27
SAM_PIXEL_MEAN = [123.675, 116.28, 103.53]   # shape (3,1,1) when used as tensor
SAM_PIXEL_STD  = [58.395,  57.12,  57.375]   # shape (3,1,1) when used as tensor

# Slicing
SLICE_SIZE    = 1024   # fine pass slice size (px)
SLICE_OVERLAP = 0.2    # fractional overlap between adjacent slices
PADDING       = 0      # constant-color border added to each slice (px)

# Candidate selection
GRID_SIZE                = 64    # grid cell size (px) for uniformity checking
UNIFORMITY_STD_THRESHOLD = 4.0   # max L* std for a grid cell to be "uniform"
UNIFORM_GRID_THRESH      = 0.98  # fraction of uniform grids required for shortcut
WATER_GRID_THRESH        = 0.98  # fraction of water-colored grids required for shortcut
THRESHOLD                = 20    # LAB distance threshold for water color match
NUM_POINTS               = 3     # number of SAM prompt points per slice

# Fallback
FALLBACK_BRIGHTNESS_THRESHOLD = 100.0  # avg LAB L* above this → classify as water
FALLBACK_DISTANCE_THRESHOLD   = 55.0   # avg LAB dist below this → classify as water

# SAM
GPU_BATCH_SIZE = 32   # number of slices per GPU encode batch

# Image Preprocessing
DOWNSAMPLE_FACTOR    = 1.0     # divisor; 1.0 = no downsample, 2.0 = half resolution
CLAHE_ENABLED        = False
CLAHE_CLIP_LIMIT     = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Hierarchical Mode
HIERARCHICAL              = True
HIERARCHICAL_SLICE_SIZE   = 4096   # coarse pass slice size (px)
USE_EROSION_MERGE         = True
EROSION_KERNEL_SIZE       = 51     # square kernel via max_pool2d; must be odd
USE_COLOR_VALIDATION      = True
MERGE_COLOR_THRESHOLD     = 15     # chromatic (a,b only) LAB distance
MERGE_LIGHTNESS_THRESHOLD = 75.0   # L* below this → candidate for kelp in disagreement zone

# Poisson disk sampling 
POISSON_DISK_K = 30   # max rejection attempts per new sample point

# Checkpointing 
CHECKPOINT_INTERVAL = 25   # save results.json every N images

# CIE L*a*b* conversion constants 
# sRGB gamma
SRGB_GAMMA_THRESHOLD = 0.04045
SRGB_GAMMA_EXPONENT  = 2.4
SRGB_GAMMA_OFFSET    = 0.055
SRGB_GAMMA_DIVISOR   = 1.055
SRGB_LINEAR_SCALE    = 12.92

# RGB -> XYZ matrix (D65 illuminant)
RGB_TO_XYZ = [
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
]

# D65 white point normalization
D65_X = 0.950456
D65_Y = 1.000000
D65_Z = 1.088754

# XYZ -> LAB
CIE_T       = 0.008856   # threshold T = (6/29)^3
CIE_KAPPA   = 903.3      # when Y/Yn <= T: L = kappa * Y
CIE_F_SLOPE = 7.787      # linear slope below T: f(t) = 7.787*t + 16/116
CIE_F_SHIFT = 16.0 / 116.0

# OpenCV LAB encoding 
# Convert back to true CIE LAB:
#   true_L = opencv_L * 100.0 / 255.0
#   true_a = opencv_a - 128
#   true_b = opencv_b - 128
OPENCV_L_SCALE = 100.0 / 255.0
OPENCV_AB_SHIFT = 128
