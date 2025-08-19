# advanced_latent_noise.py

import torch # type: ignore
import numpy as np # type: ignore
import math
from enum import Enum

# --- Enums for Categorical Inputs ---
class DeviceChoice(Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"

class InterpolationType(Enum):
    SMOOTHSTEP = "Smoothstep"
    LINEAR = "Linear"

class BaseNoiseAlgorithm(Enum):
    GAUSSIAN = "Gaussian"
    UNIFORM = "Uniform"
    VALUE = "Value"
    PERLIN = "Perlin"
    SIMPLEX = "Simplex"
    VORONOI_F1 = "Voronoi F1"
    VORONOI_F2 = "Voronoi F2"
    VORONOI_F2_F1 = "Voronoi F2-F1"
    WHITE = "White" # Equivalent to Gaussian for randn
    PINK = "Pink"
    BROWN = "Brown"

class FBMSummingMode(Enum):
    STANDARD = "Standard"
    TURBULENCE = "Turbulence"
    RIDGE = "Ridge"

class WarpNoiseAlgorithm(Enum):
    VALUE = "Value"
    PERLIN = "Perlin"
    SIMPLEX = "Simplex"

class CompositionGuideType(Enum):
    NONE = "None"
    RULE_OF_THIRDS = "Rule of Thirds"
    GOLDEN_RATIO_LINES = "Golden Ratio Lines"
    CENTRAL_FOCUS = "Central Focus"
    DIAGONAL_FLOW_TL_BR = "Diagonal Flow TL-BR"
    COMBINED = "Combined"

class GuideEffect(Enum):
    INTENSITY_BOOST = "Intensity Boost"
    INTENSITY_REDUCE = "Intensity Reduce"

class MaskEffect(Enum):
    MODULATE_INTENSITY_BY_MASK = "Modulate Intensity By Mask"
    APPLY_TO_MASKED_AREA = "Apply To Masked Area"
    APPLY_TO_UNMASKED_AREA = "Apply To Unmasked Area"

# --- Constants ---
NODE_NAME = "Advanced_Latent_Noise"
NODE_DISPLAY_NAME = "Advanced Latent Noise"
CATEGORY = "latent/noise"
MAX_SEED_VALUE = np.iinfo(np.int64).max # Max seed value for UI

# Simplex Noise constants
_F2_SIMPLEX = (math.sqrt(3.0) - 1.0) / 2.0
_G2_SIMPLEX = (3.0 - math.sqrt(3.0)) / 6.0
_SIMPLEX_R_SQ_THRESHOLD = 0.6 # Radius squared threshold for Simplex kernel
_SIMPLEX_NORMALIZATION_FACTOR = 28.0 # Typical normalization factor for 2D Simplex

# Perlin Noise constants
_PERLIN_NORMALIZATION_FACTOR = 1.414 # sqrt(2), adjusts range for gradient noise

# Permutation and Gradient tables (initialized later)
_STANDARD_PERLIN_PERM_256_LIST = [ # This list must contain 256 unique numbers from 0-255
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32, 57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
    74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122, 60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
    65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
    52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,254,163, 70,224,156,150,162,138,155, 49,191,179, 51,180,199, 12,221, 45,181,172,144,204,236,129,153, 43,214,
    28,232,108,241,113,222,218,210,243, 22, 79,178,107,184,167,157,195,101, 45, 12, 81,176,121, 19,253,206,238, 50,114,205,  9,127,
    78, 72, 39,145,141,115,112, 61,154,193,249,106, 34,185,104, 14, 67,  4,200,246,110,192,153, 29, 93, 97, 98, 51, 24,215,235,239
]
assert len(_STANDARD_PERLIN_PERM_256_LIST) == 256, "Base permutation list is not 256 elements long!"
_PERMUTATION_TABLE_DATA_NP = np.array(_STANDARD_PERLIN_PERM_256_LIST * 2, dtype=np.int64)

_GRADIENT_VECTORS_2D_PERLIN_NP = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]], dtype=np.float32)

_GRAD_SIMPLEX_NP_UNNORMALIZED = np.array([[0,1],[0,-1],[1,0],[-1,0],[1,1],[-1,1],[1,-1],[-1,-1],[1,2],[-1,2],[1,-2],[-1,-2],[2,1],[-2,1],[2,-1],[-2,-1]],dtype=np.float32)
_GRAD_SIMPLEX_NP = _GRAD_SIMPLEX_NP_UNNORMALIZED / np.sqrt(np.sum(_GRAD_SIMPLEX_NP_UNNORMALIZED**2,axis=1,keepdims=True))

# Device-specific cache for PyTorch tables
_DEVICE_TABLE_CACHE = {}

# --- Helper: Device Handling ---
def get_torch_device(device_choice: DeviceChoice) -> torch.device:
    assert isinstance(device_choice, DeviceChoice), "device_choice must be a DeviceChoice Enum member."
    if device_choice == DeviceChoice.CUDA:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print(f"Warning ({NODE_NAME}): CUDA selected but not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_choice == DeviceChoice.CPU:
        return torch.device("cpu")
    # Auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper: Tensor Normalization ---
def normalize_tensor(tensor: torch.Tensor, new_min: float = 0.0, new_max: float = 1.0) -> torch.Tensor:
    assert isinstance(tensor, torch.Tensor), "Input must be a PyTorch tensor."
    assert new_min <= new_max, "new_min must be less than or equal to new_max."

    if tensor.numel() == 0: return tensor
    tensor_min, tensor_max = tensor.min(), tensor.max()
    
    range_val = tensor_max - tensor_min
    # Using a small epsilon for floating point comparison
    if torch.isclose(range_val, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype), atol=1e-7):
        return torch.full_like(tensor, (new_min + new_max) / 2.0)
    
    normalized_tensor = (tensor - tensor_min) / range_val
    return normalized_tensor * (new_max - new_min) + new_min

# --- Helper: Interpolation ---
def _interpolate_smoothstep(t: torch.Tensor) -> torch.Tensor:
    # Ken Perlin's Smootherstep: 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)

def _interpolate_linear(t: torch.Tensor) -> torch.Tensor:
    return t

_INTERPOLATION_FUNCTIONS = {
    InterpolationType.SMOOTHSTEP: _interpolate_smoothstep,
    InterpolationType.LINEAR: _interpolate_linear,
}

# --- Table Initialization for Perlin/Simplex ---
def _get_device_specific_table(table_name: str, numpy_data: np.ndarray, device: torch.device) -> torch.Tensor:
    # Simplified caching for debugging - ensure device object is hashable.
    device_key = (device.type, device.index if device.index is not None else -1)

    if device_key not in _DEVICE_TABLE_CACHE:
        _DEVICE_TABLE_CACHE[device_key] = {}

    if table_name not in _DEVICE_TABLE_CACHE[device_key]:
        print(f"({NODE_NAME}) Initializing table '{table_name}' on device '{device}'. Numpy data shape: {numpy_data.shape}")
        tensor_table = torch.from_numpy(numpy_data).to(device)
        _DEVICE_TABLE_CACHE[device_key][table_name] = tensor_table
        
        # Verification for permutation table
        if table_name == "perm_table":
            assert tensor_table.shape[0] == 512, f"Newly created perm_table has wrong size: {tensor_table.shape[0]} on {device}"
            assert tensor_table.dtype == torch.long, f"Newly created perm_table has wrong dtype: {tensor_table.dtype} on {device}"
    
    retrieved_table = _DEVICE_TABLE_CACHE[device_key][table_name]
    # Additional check every time for perm_table (can be removed after debugging)
    if table_name == "perm_table":
         assert retrieved_table.shape[0] == 512, f"Retrieved perm_table has wrong size: {retrieved_table.shape[0]} on {device}"
         assert retrieved_table.device == device, f"Retrieved perm_table on wrong device: {retrieved_table.device} vs {device}"

    return retrieved_table

def _get_perm_table(device: torch.device) -> torch.Tensor:
    return _get_device_specific_table("perm_table", _PERMUTATION_TABLE_DATA_NP, device)

def _get_grad_vectors_perlin(device: torch.device) -> torch.Tensor:
    return _get_device_specific_table("grad_vectors_perlin", _GRADIENT_VECTORS_2D_PERLIN_NP, device)

def _get_grad_vectors_simplex(device: torch.device) -> torch.Tensor:
    return _get_device_specific_table("grad_vectors_simplex", _GRAD_SIMPLEX_NP, device)

# --- PyTorch-based Noise Algorithm Implementations ---

# Section: Value Noise
def _generate_value_noise_coordinates(shape: tuple, scale: float, device: torch.device):
    batch_size, channels, height, width = shape
    x_coords = torch.linspace(0.0, scale, width, device=device)
    y_coords = torch.linspace(0.0, scale, height, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)
    
    x0 = torch.floor(grid_x).long()
    y0 = torch.floor(grid_y).long()
    
    return grid_x, grid_y, x0, y0

def _pytorch_value_noise_2d(shape: tuple, seed: int, device: torch.device,
                            scale: float, effective_seed: int,
                            interpolation_fn) -> torch.Tensor:
    assert len(shape) == 4, "Shape must be (batch, channels, height, width)."
    assert scale > 0, "Scale must be positive."

    batch_size, channels, height, width = shape
    grid_x, grid_y, x0, y0 = _generate_value_noise_coordinates(shape, scale, device)
    
    x1, y1 = x0 + 1, y0 + 1
    xf, yf = grid_x - x0.float(), grid_y - y0.float()

    table_w_size = max(2, int(math.ceil(scale)) + 1)
    table_h_size = max(2, int(math.ceil(scale)) + 1)
    
    torch.manual_seed(effective_seed) # effective_seed = main_seed + octave_offset
    rand_table = torch.rand((batch_size, channels, table_h_size, table_w_size), device=device) * 2.0 - 1.0
    
    # Modulo for safety, ensuring indices wrap around the table
    idx_x0, idx_y0 = x0 % table_w_size, y0 % table_h_size
    idx_x1, idx_y1 = x1 % table_w_size, y1 % table_h_size
    
    # Fancy indexing for batch and channels
    b_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
    c_idx = torch.arange(channels, device=device).view(1, -1, 1, 1)
    
    v00 = rand_table[b_idx, c_idx, idx_y0, idx_x0]
    v10 = rand_table[b_idx, c_idx, idx_y0, idx_x1]
    v01 = rand_table[b_idx, c_idx, idx_y1, idx_x0]
    v11 = rand_table[b_idx, c_idx, idx_y1, idx_x1]
    
    u, v = interpolation_fn(xf), interpolation_fn(yf)
    
    x_interp1 = v00 * (1 - u) + v10 * u
    x_interp2 = v01 * (1 - u) + v11 * u
    
    return x_interp1 * (1 - v) + x_interp2 * v

# Section: Perlin (Gradient) Noise
def _hash_coords_to_gradient_idx_perlin(ix: torch.Tensor, iy: torch.Tensor,
                                       effective_seed: int, device: torch.device) -> torch.Tensor:
    perm_table = _get_perm_table(device)
    assert perm_table.shape[0] == 512, f"Permutation table size mismatch: expected 512, got {perm_table.shape[0]}"
    grad_vectors = _get_grad_vectors_perlin(device)
    # Add effective_seed to coordinates before hashing to vary noise pattern
    # Modulo 255 ensures indices stay within the first half of doubled perm_table
    ix_s = (ix + effective_seed) & 255
    iy_s = (iy + effective_seed) & 255
    return perm_table[perm_table[ix_s] + iy_s] % grad_vectors.shape[0]

def _calculate_perlin_dot_products(grid_coords: torch.Tensor, corner_coords: torch.Tensor,
                                   grad_vectors_indices: torch.Tensor, device: torch.device):
    grad_vectors_all = _get_grad_vectors_perlin(device)
    selected_grad_vectors = grad_vectors_all[grad_vectors_indices]
    
    # grid_coords are (B, C, H, W), corner_coords are (B, C, H, W)
    # We need to calculate (grid_x - corner_x, grid_y - corner_y)
    # For n00: (xf, yf)
    # For n10: (xf-1, yf)
    # For n01: (xf, yf-1)
    # For n11: (xf-1, yf-1)
    # These relative vectors are passed as `grid_coords - corner_coords_float` which is what `xf, yf` etc. represent.
    # The function will be called with appropriate diff vectors.
    # Here, corner_coords is used as a placeholder for the (xf_e, yf_e), (xf_e-1, yf_e) etc.
    # Let's rename parameters for clarity
    # diff_vector should be shape (B, C, H, W, 2)
    # selected_grad_vectors shape (B, C, H, W, 2)
    # dot product: sum(selected_grad_vectors * diff_vector, dim=-1)
    # This part is a bit tricky due to how it's called in the original structure.
    # Original: n00 = torch.sum(g00*torch.cat((xf_e,yf_e),dim=-1),dim=-1)
    # It seems `corner_coords` here is not needed, calculation is done by caller.
    # This function seems to be just a lookup:
    # return grad_vectors_all[grad_vectors_indices]
    # No, this is not correct. It was for calculating dot products. Let's re-evaluate.

    # The logic in the main Perlin function is:
    # g00 = _grad_vectors_2d_perlin_torch[idx00]
    # n00 = torch.sum(g00*torch.cat((xf_e,yf_e),dim=-1),dim=-1)
    # This is straightforward. The dot product is simple.
    # The existing structure for this part is fine.
    pass # No specific sub-function needed here, original structure is concise.

def _pytorch_perlin_gradient_noise_2d(shape: tuple, seed: int, device: torch.device, scale: float,
                                     effective_seed: int, interpolation_fn) -> torch.Tensor:
    assert len(shape) == 4, "Shape must be (batch, channels, height, width)."
    assert scale > 0, "Scale must be positive."

    batch_size, channels, height, width = shape
    
    x_coords = torch.linspace(0.0, scale, width, device=device)
    y_coords = torch.linspace(0.0, scale, height, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)
    
    x0, y0 = torch.floor(grid_x).long(), torch.floor(grid_y).long()
    x1, y1 = x0 + 1, y0 + 1
    
    xf, yf = grid_x - x0.float(), grid_y - y0.float()

    # Get gradient indices for each corner of the grid cell
    idx00 = _hash_coords_to_gradient_idx_perlin(x0, y0, effective_seed, device)
    idx10 = _hash_coords_to_gradient_idx_perlin(x1, y0, effective_seed, device)
    idx01 = _hash_coords_to_gradient_idx_perlin(x0, y1, effective_seed, device)
    idx11 = _hash_coords_to_gradient_idx_perlin(x1, y1, effective_seed, device)

    grad_vectors_all = _get_grad_vectors_perlin(device)
    g00, g10 = grad_vectors_all[idx00], grad_vectors_all[idx10]
    g01, g11 = grad_vectors_all[idx01], grad_vectors_all[idx11]

    # Expand dimensions of xf, yf for broadcasting with gradient vectors
    xf_e, yf_e = xf.unsqueeze(-1), yf.unsqueeze(-1)

    # Dot products
    # n00: influence of g00 at (xf, yf)
    # n10: influence of g10 at (xf-1, yf)
    # n01: influence of g01 at (xf, yf-1)
    # n11: influence of g11 at (xf-1, yf-1)
    n00 = torch.sum(g00 * torch.cat((xf_e,     yf_e),     dim=-1), dim=-1)
    n10 = torch.sum(g10 * torch.cat((xf_e - 1, yf_e),     dim=-1), dim=-1)
    n01 = torch.sum(g01 * torch.cat((xf_e,     yf_e - 1), dim=-1), dim=-1)
    n11 = torch.sum(g11 * torch.cat((xf_e - 1, yf_e - 1), dim=-1), dim=-1)
    
    u, v = interpolation_fn(xf), interpolation_fn(yf)
    
    x_interp1 = n00 * (1 - u) + n10 * u
    x_interp2 = n01 * (1 - u) + n11 * u
    
    return (x_interp1 * (1 - v) + x_interp2 * v) * _PERLIN_NORMALIZATION_FACTOR

# Section: Voronoi Noise
def _pytorch_voronoi_noise_2d_single_channel(height: int, width: int, num_points: int, 
                                             effective_seed: int, device: torch.device, 
                                             variant: BaseNoiseAlgorithm) -> torch.Tensor:
    assert height > 0 and width > 0, "Height and width must be positive."
    assert num_points >= 1, "Number of points must be at least 1."

    torch.manual_seed(effective_seed)
    # Points are in [0,1] range for this specific channel
    points = torch.rand(num_points, 2, device=device) # (num_points, [x, y])

    px_x = torch.linspace(0.0, 1.0, width, device=device)
    px_y = torch.linspace(0.0, 1.0, height, device=device)
    gy, gx = torch.meshgrid(px_y, px_x, indexing='ij') # Pixel grid

    # Reshape for broadcasting: gx, gy become (H, W, 1), points_x, points_y become (1, 1, num_points)
    g_x = gx.unsqueeze(-1)
    g_y = gy.unsqueeze(-1)
    p_x = points[:, 0].view(1, 1, -1)
    p_y = points[:, 1].view(1, 1, -1)

    dist_sq = (g_x - p_x)**2 + (g_y - p_y)**2 # Shape (H, W, num_points)
    
    k_val = 2 if variant != BaseNoiseAlgorithm.VORONOI_F1 else 1
    # topk returns (values, indices), we only need values
    top_k_dist_sq, _ = torch.topk(dist_sq, k_val, dim=-1, largest=False) 
    
    f1_dist = torch.sqrt(top_k_dist_sq[..., 0])
    
    if variant == BaseNoiseAlgorithm.VORONOI_F1:
        noise_cell = f1_dist
    elif variant == BaseNoiseAlgorithm.VORONOI_F2:
        noise_cell = torch.sqrt(top_k_dist_sq[..., 1])
    elif variant == BaseNoiseAlgorithm.VORONOI_F2_F1:
        f2_dist = torch.sqrt(top_k_dist_sq[..., 1])
        noise_cell = f2_dist - f1_dist
    else: # Should not happen if called correctly
        noise_cell = f1_dist

    # Normalize to [-1, 1] range for this channel
    return normalize_tensor(noise_cell, 0.0, 1.0) * 2.0 - 1.0


def _pytorch_voronoi_noise_2d(shape: tuple, seed: int, device: torch.device, 
                               scale_for_point_density: float, effective_seed_base: int, 
                               variant: BaseNoiseAlgorithm) -> torch.Tensor:
    assert len(shape) == 4, "Shape must be (batch, channels, height, width)."
    assert scale_for_point_density > 0, "Scale for point density must be positive."

    batch_size, channels, height, width = shape
    # `scale_for_point_density` (from initial_frequency_scale) determines num_points
    num_points = int(max(4, scale_for_point_density * scale_for_point_density))
    
    output_noise = torch.zeros(shape, device=device)

    for b in range(batch_size):
        for c in range(channels):
            # Unique seed for each batch item and channel
            channel_seed = effective_seed_base + b * (channels * 10 + 1) + c * 7 
            output_noise[b, c] = _pytorch_voronoi_noise_2d_single_channel(
                height, width, num_points, channel_seed, device, variant
            )
    return output_noise

# Section: Simplex Noise
def _hash_coords_simplex(ix: torch.Tensor, iy: torch.Tensor,
                         effective_seed: int, device: torch.device) -> torch.Tensor:
    perm_table = _get_perm_table(device)
    assert perm_table.shape[0] == 512, f"Permutation table size mismatch: expected 512, got {perm_table.shape[0]}"
    grad_vectors = _get_grad_vectors_simplex(device)
    # Add effective_seed to vary noise pattern, & 255 for table indexing
    ix_s = (ix + effective_seed) & 255
    iy_s = (iy + effective_seed) & 255
    return perm_table[perm_table[ix_s] + iy_s] % grad_vectors.shape[0]

def _calculate_simplex_contribution(x_rel: torch.Tensor, y_rel: torch.Tensor,
                                    ix_corner: torch.Tensor, iy_corner: torch.Tensor,
                                    effective_seed: int, device: torch.device) -> torch.Tensor:
    # x_rel, y_rel: relative coordinates from point to corner vertex
    # ix_corner, iy_corner: integer coordinates of the corner vertex
    contribution = torch.zeros_like(x_rel)

    # Kernel: (r_sq_thresh - r_sq)^4 if r_sq < r_sq_thresh, else 0
    r_sq = x_rel * x_rel + y_rel * y_rel
    mask = r_sq < _SIMPLEX_R_SQ_THRESHOLD

    if torch.any(mask):
        # Ensure inputs to .long() are finite
        ix_masked = ix_corner[mask]
        iy_masked = iy_corner[mask]

        if not (torch.all(torch.isfinite(ix_masked)) and torch.all(torch.isfinite(iy_masked))):
            print(f"Warning ({NODE_NAME}): Non-finite values detected in Simplex corner coordinates before hashing.")
            ix_masked = torch.where(torch.isfinite(ix_masked), ix_masked, torch.zeros_like(ix_masked))
            iy_masked = torch.where(torch.isfinite(iy_masked), iy_masked, torch.zeros_like(iy_masked))

        term_kernel = (_SIMPLEX_R_SQ_THRESHOLD - r_sq[mask])**4

        ix_corner_masked_long = ix_masked.long()
        iy_corner_masked_long = iy_masked.long()

        grad_indices = _hash_coords_simplex(ix_corner_masked_long, iy_corner_masked_long, effective_seed, device)
        grad_vectors_all = _get_grad_vectors_simplex(device)
        selected_grads = grad_vectors_all[grad_indices] # Shape: (N_masked, 2)

        # Stack relative coords for dot product: (N_masked, 2)
        rel_coords_masked = torch.stack((x_rel[mask], y_rel[mask]), dim=-1)

        dot_product = torch.sum(selected_grads * rel_coords_masked, dim=-1)
        contribution[mask] = term_kernel * dot_product

    return contribution

def _pytorch_simplex_noise_2d(shape: tuple, seed: int, device: torch.device, scale: float,
                              effective_seed: int) -> torch.Tensor:
    assert len(shape) == 4, "Shape must be (batch, channels, height, width)."
    assert scale > 0, "Scale must be positive."

    B, C, H, W = shape
    
    # 1. Input coordinates
    xin = torch.linspace(0.0, scale, W, device=device)
    yin = torch.linspace(0.0, scale, H, device=device)
    gy_in, gx_in = torch.meshgrid(yin, xin, indexing='ij')
    gx_in = gx_in.view(1,1,H,W).repeat(B,C,1,1)
    gy_in = gy_in.view(1,1,H,W).repeat(B,C,1,1)

    # 2. Skew input coordinates to grid
    s = (gx_in + gy_in) * _F2_SIMPLEX
    i_sk, j_sk = torch.floor(gx_in + s), torch.floor(gy_in + s) # Simplex cell origin (skewed)

    # 3. Unskew cell origin back to input space
    t_unsk = (i_sk + j_sk) * _G2_SIMPLEX
    X0_unsk, Y0_unsk = i_sk - t_unsk, j_sk - t_unsk # Simplex cell origin (unskewed)
    
    # 4. Relative coordinates from input point to unskewed cell origin
    x0_rel, y0_rel = gx_in - X0_unsk, gy_in - Y0_unsk

    # 5. Determine second and third simplex vertices (offsets from origin i_sk, j_sk)
    # i1_offset, j1_offset determine the middle vertex; (1,1) is always the far vertex
    i1_offset = torch.where(x0_rel > y0_rel, 1, 0).long()
    j1_offset = torch.where(x0_rel > y0_rel, 0, 1).long()
    
    # 6. Relative coordinates from input point to other two simplex vertices
    # Vertex 1 (closest): (x0_rel, y0_rel) -> already have
    # Vertex 2 (middle):
    x1_rel = x0_rel - i1_offset.float() + _G2_SIMPLEX
    y1_rel = y0_rel - j1_offset.float() + _G2_SIMPLEX
    # Vertex 3 (farthest):
    x2_rel = x0_rel - 1.0 + 2.0 * _G2_SIMPLEX
    y2_rel = y0_rel - 1.0 + 2.0 * _G2_SIMPLEX

    # 7. Sum contributions from the three simplex vertices
    noise_sum = torch.zeros_like(gx_in)
    
    # Contribution from (i_sk, j_sk)
    noise_sum += _calculate_simplex_contribution(x0_rel, y0_rel, i_sk, j_sk, effective_seed, device)
    
    # Contribution from (i_sk + i1_offset, j_sk + j1_offset)
    noise_sum += _calculate_simplex_contribution(x1_rel, y1_rel, i_sk + i1_offset, j_sk + j1_offset, effective_seed, device)
    
    # Contribution from (i_sk + 1, j_sk + 1)
    noise_sum += _calculate_simplex_contribution(x2_rel, y2_rel, i_sk + 1, j_sk + 1, effective_seed, device)
            
    return noise_sum * _SIMPLEX_NORMALIZATION_FACTOR

# Section: Basic and FFT-based Noise
def _generate_white_gaussian_uniform_noise(shape: tuple, device: torch.device,
                                           noise_type: BaseNoiseAlgorithm) -> torch.Tensor:
    if noise_type == BaseNoiseAlgorithm.GAUSSIAN or noise_type == BaseNoiseAlgorithm.WHITE:
        # White noise is often considered flat spectrum, randn gives Gaussian distribution
        # For true "white noise" in signal processing sense (uniform power spectrum), 
        # this is a common way to generate it in spatial domain.
        return torch.randn(shape, device=device)
    elif noise_type == BaseNoiseAlgorithm.UNIFORM:
        return torch.rand(shape, device=device) * 2.0 - 1.0
    else: # Should not be reached if called correctly
        print(f"Warning ({NODE_NAME}): Unexpected type {noise_type} in _generate_white_gaussian_uniform_noise. Defaulting to Gaussian.")
        return torch.randn(shape, device=device)

def _generate_colored_fft_noise(shape: tuple, device: torch.device,
                                noise_type: BaseNoiseAlgorithm,
                                initial_frequency_scale: float) -> torch.Tensor: # TODO: Use initial_frequency_scale
    assert len(shape) == 4, "Shape must be (batch, channels, height, width)."
    batch_size, channels, height, width = shape

    if height <= 1 or width <= 1: # FFT not meaningful for 1D lines/pixels
        print(f"Warning ({NODE_NAME}): FFT noise for degenerate dimensions ({height}x{width}). Falling back to Gaussian.")
        return torch.randn(shape, device=device)
        
    spatial_noise = torch.randn(shape, device=device) # Start with Gaussian white noise
    spectrum = torch.fft.rfft2(spatial_noise, norm="ortho")

    # Frequency grids
    freq_y = torch.fft.fftfreq(height, d=1.0/height, device=device)
    freq_x = torch.fft.rfftfreq(width,  d=1.0/width,  device=device)
    ky_grid, kx_grid = torch.meshgrid(freq_y, freq_x, indexing='ij')
    
    # Magnitude of frequencies, add epsilon to avoid division by zero at DC
    k_magnitude_sq = kx_grid**2 + ky_grid**2 + 1e-7 # k^2
    
    # Determine exponent alpha for 1/f^alpha power spectrum
    # Power spectrum is |F(k)|^2. Amplitude is |F(k)|.
    # So, amplitude filter is k_magnitude_sq ^ (-alpha / 4) for amplitude, or k_magnitude ^ (-alpha/2)
    # If noise power P(f) ~ 1/f^beta, then amplitude A(f) ~ 1/f^(beta/2)
    # Pink noise: beta=1. Brown noise: beta=2.
    beta = 1.0 if noise_type == BaseNoiseAlgorithm.PINK else 2.0
    
    # Amplitude filter: (k_sq)^(-beta/4) = k^(-beta/2)
    amplitude_filter = k_magnitude_sq ** (-beta / 4.0)
    
    # TODO: The 'initial_frequency_scale' parameter is not currently used here.
    # Its role in FFT noise needs to be defined:
    # e.g., modify beta, act as a frequency cutoff, or scale k_magnitude_sq.
    # Example: amplitude_filter = (k_magnitude_sq / (initial_frequency_scale**2 + 1e-7)) ** (-beta / 4.0)
    # This was an untested idea in the original. For now, it's unused to preserve behavior.

    if amplitude_filter.ndim > 1 and amplitude_filter.shape[0] > 0 and amplitude_filter.shape[1] > 0:
        amplitude_filter[0, 0] = 1.0 # Keep DC component (average value) unchanged, avoid NaN from 0^power

    # Apply filter and inverse FFT
    spectrum_filtered = spectrum * amplitude_filter.unsqueeze(0).unsqueeze(0) # Add batch/channel dims
    filtered_noise = torch.fft.irfft2(spectrum_filtered, s=(height, width), norm="ortho")
    
    return filtered_noise

def _generate_basic_or_fft_noise(shape: tuple, effective_seed: int, device: torch.device,
                                 noise_type: BaseNoiseAlgorithm, initial_frequency_scale: float) -> torch.Tensor:
    torch.manual_seed(effective_seed)
    if noise_type in [BaseNoiseAlgorithm.GAUSSIAN, BaseNoiseAlgorithm.UNIFORM, BaseNoiseAlgorithm.WHITE]:
        return _generate_white_gaussian_uniform_noise(shape, device, noise_type)
    elif noise_type in [BaseNoiseAlgorithm.PINK, BaseNoiseAlgorithm.BROWN]:
        return _generate_colored_fft_noise(shape, device, noise_type, initial_frequency_scale)
    else:
        # This case should ideally not be reached if noise_type is from the enum
        # and handled by _generate_single_noise_layer dispatch
        print(f"Warning ({NODE_NAME}): Noise type '{noise_type.value}' fell through in _generate_basic_or_fft_noise. Defaulting to Gaussian.")
        return torch.randn(shape, device=device)

# --- Main Class Definition ---
class Advanced_Latent_Noise:
    # Helper to get interpolation function by enum
    def _get_interpolation_function_from_type(self, interp_type: InterpolationType):
        assert isinstance(interp_type, InterpolationType), "interp_type must be an InterpolationType Enum member."
        return _INTERPOLATION_FUNCTIONS[interp_type]

    def _generate_single_noise_layer(self, shape: tuple, seed: int, device: torch.device,
                                     noise_type: BaseNoiseAlgorithm, scale: float,
                                     octave_seed_offset: int = 0, 
                                     interpolation_type: InterpolationType = InterpolationType.SMOOTHSTEP
                                     ) -> torch.Tensor:
        # effective_seed combines main seed and octave-specific offset
        effective_seed = seed + octave_seed_offset
        interpolation_fn = self._get_interpolation_function_from_type(interpolation_type)

        if noise_type == BaseNoiseAlgorithm.VALUE:
            return _pytorch_value_noise_2d(shape, seed, device, scale, effective_seed, interpolation_fn)
        elif noise_type == BaseNoiseAlgorithm.PERLIN:
             return _pytorch_perlin_gradient_noise_2d(shape, seed, device, scale, effective_seed, interpolation_fn)
        elif noise_type == BaseNoiseAlgorithm.SIMPLEX:
            return _pytorch_simplex_noise_2d(shape, seed, device, scale, effective_seed) # Simplex uses its own kernel, not general interpolation_fn
        elif noise_type in [BaseNoiseAlgorithm.VORONOI_F1, BaseNoiseAlgorithm.VORONOI_F2, BaseNoiseAlgorithm.VORONOI_F2_F1]:
            # For Voronoi, 'scale' is used for point density, not coordinate range.
            return _pytorch_voronoi_noise_2d(shape, seed, device, scale, effective_seed, noise_type)
        else: # Gaussian, Uniform, White, Pink, Brown
            return _generate_basic_or_fft_noise(shape, effective_seed, device, noise_type, scale)

    def _apply_fbm(self, shape: tuple, seed: int, device: torch.device,
                   base_noise_type: BaseNoiseAlgorithm, initial_scale: float,
                   octaves: int, persistence: float, lacunarity: float,
                   summing_mode: FBMSummingMode, interpolation_type: InterpolationType) -> torch.Tensor:
        assert octaves >= 1, "Number of octaves must be at least 1."
        assert 0.0 < persistence <= 1.0, "Persistence must be between 0 (exclusive) and 1 (inclusive)."
        assert lacunarity >= 1.0, "Lacunarity must be at least 1.0."

        total_noise = torch.zeros(shape, device=device)
        current_amplitude = 1.0
        current_frequency_scale = initial_scale
        
        octave_seed_base_offset = 73 # Arbitrary prime to differentiate octave seeds

        for i in range(octaves):
            octave_offset = i * octave_seed_base_offset
            noise_layer = self._generate_single_noise_layer(
                shape, seed, device, base_noise_type, current_frequency_scale,
                octave_seed_offset=octave_offset, interpolation_type=interpolation_type
            )
            
            if summing_mode == FBMSummingMode.TURBULENCE:
                total_noise += torch.abs(noise_layer) * current_amplitude
            elif summing_mode == FBMSummingMode.RIDGE:
                total_noise += (1.0 - torch.abs(noise_layer)) * current_amplitude
            else: # Standard
                total_noise += noise_layer * current_amplitude
            
            current_amplitude *= persistence
            current_frequency_scale *= lacunarity
            
        return total_noise

    def _generate_domain_warp_field(self, warp_shape: tuple, seed: int, device: torch.device,
                                   warp_noise_type: WarpNoiseAlgorithm, warp_octaves: int,
                                   warp_persistence: float, warp_lacunarity: float,
                                   warp_scale: float, interpolation_type: InterpolationType) -> torch.Tensor:
        # Map WarpNoiseAlgorithm to BaseNoiseAlgorithm for FBM generation
        base_warp_noise_type_map = {
            WarpNoiseAlgorithm.VALUE: BaseNoiseAlgorithm.VALUE,
            WarpNoiseAlgorithm.PERLIN: BaseNoiseAlgorithm.PERLIN,
            WarpNoiseAlgorithm.SIMPLEX: BaseNoiseAlgorithm.SIMPLEX,
        }
        base_warp_noise_type = base_warp_noise_type_map[warp_noise_type]

        # Warp field needs 2 channels (dx, dy)
        assert warp_shape[1] == 2, "Warp field shape must have 2 channels."

        # Use a different seed for the warp field to avoid correlation with base noise
        warp_field_seed = seed + 101 
        
        field = self._apply_fbm(
            warp_shape, warp_field_seed, device, base_warp_noise_type, warp_scale,
            warp_octaves, warp_persistence, warp_lacunarity,
            FBMSummingMode.STANDARD, interpolation_type
        )
        
        # Normalize each component of the field (dx, dy) to [-1, 1]
        # This ensures warp_strength has a consistent meaning
        for i in range(warp_shape[0]): # Batch dimension
            field[i, 0] = normalize_tensor(field[i, 0], -1.0, 1.0)
            field[i, 1] = normalize_tensor(field[i, 1], -1.0, 1.0)
        return field

    def _apply_domain_warp_using_field(self, noise_tensor: torch.Tensor, warp_field: torch.Tensor,
                                       warp_strength: float, device: torch.device) -> torch.Tensor:
        B, C, H, W = noise_tensor.shape
        
        dx = warp_field[:, 0:1] * warp_strength # (B, 1, H, W)
        dy = warp_field[:, 1:2] * warp_strength # (B, 1, H, W)

        # Create base grid for sampling, in range [-1, 1] for grid_sample
        gy_base, gx_base = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        # grid_base shape (H, W, 2), then expand for batch
        grid_base = torch.stack((gx_base, gy_base), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        # Normalize displacements: warp_strength is in pixels.
        # grid_sample expects offsets in [-1, 1] range (normalized coordinates)
        # If offset is W/2 pixels, it's a normalized offset of 1.0.
        norm_dx = dx / (W / 2.0)
        norm_dy = dy / (H / 2.0)
        
        # Combine normalized offsets and permute to (B, H, W, 2) for grid_sample
        # Offsets shape: (B, 2, H, W) -> (B, H, W, 2)
        offsets_permuted = torch.cat((norm_dx, norm_dy), dim=1).permute(0, 2, 3, 1)
        
        warped_grid = grid_base + offsets_permuted
        
        # align_corners=False is generally recommended for feature-based warping.
        # padding_mode='border' replicates edge pixels, 'reflection' might also be good.
        return torch.nn.functional.grid_sample(
            noise_tensor, warped_grid, mode='bilinear',
            padding_mode='border', align_corners=False
        )

    def _apply_domain_warp(self, noise_tensor: torch.Tensor, seed: int, device: torch.device,
                           warp_noise_type: WarpNoiseAlgorithm, warp_octaves: int, warp_persistence: float, 
                           warp_lacunarity: float, warp_strength: float, warp_scale: float, 
                           interpolation_type: InterpolationType) -> torch.Tensor:
        B, C, H, W = noise_tensor.shape
        warp_field_shape = (B, 2, H, W) # 2 channels for dx, dy

        warp_field = self._generate_domain_warp_field(
            warp_field_shape, seed, device, warp_noise_type, warp_octaves,
            warp_persistence, warp_lacunarity, warp_scale, interpolation_type
        )
        
        return self._apply_domain_warp_using_field(noise_tensor, warp_field, warp_strength, device)

    # --- Composition Guide Helpers ---
    def _apply_fade_to_line(self, current_map_segment: torch.Tensor, coord_distance: torch.Tensor,
                            thickness_pixels: float, fade_pixels: float) -> torch.Tensor:
        """ Helper to draw a line with fading edges onto a map segment. """
        half_thick = thickness_pixels / 2.0
        if fade_pixels <= 0: # Sharp line
            line_intensity = (coord_distance <= half_thick).float()
        else:
            # Smoothstep from edge of line to edge of fade region
            # Total extent from center = half_thick + fade_pixels
            normalized_dist = torch.clamp(coord_distance / (half_thick + fade_pixels + 1e-6), 0, 1)
            line_intensity = 1.0 - _interpolate_smoothstep(normalized_dist)
        return torch.max(current_map_segment, line_intensity)

    def _generate_central_focus_map(self, H: int, W: int, dev: torch.device, thick_px: int, fade_px: int) -> torch.Tensor:
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
        center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
        dist_sq_from_center = (y_coords - center_y)**2 + (x_coords - center_x)**2
        # Max distance squared to a corner, for normalization
        max_dist_sq = ((H / 2.0)**2 + (W / 2.0)**2)
        normalized_dist = torch.sqrt(dist_sq_from_center / (max_dist_sq + 1e-6))
        # Focus is strongest at center, fades outwards
        return 1.0 - _interpolate_smoothstep(torch.clamp(normalized_dist, 0, 1))

    def _generate_rule_of_thirds_map(self, H: int, W: int, dev: torch.device, thick_px: int, fade_px: int) -> torch.Tensor:
        weight_map = torch.zeros((H,W), device=dev)
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
        for i in [1, 2]:
            line_y_pos = H * i / 3.0
            weight_map = self._apply_fade_to_line(weight_map, torch.abs(y_coords - line_y_pos), thick_px, fade_px)
            line_x_pos = W * i / 3.0
            weight_map = self._apply_fade_to_line(weight_map, torch.abs(x_coords - line_x_pos), thick_px, fade_px)
        return weight_map

    def _generate_golden_ratio_map(self, H: int, W: int, dev: torch.device, thick_px: int, fade_px: int) -> torch.Tensor:
        weight_map = torch.zeros((H,W), device=dev)
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
        phi = (1 + math.sqrt(5)) / 2.0
        golden_lines_y = [H / phi, H - (H / phi)]
        golden_lines_x = [W / phi, W - (W / phi)]
        for line_pos in golden_lines_y:
            weight_map = self._apply_fade_to_line(weight_map, torch.abs(y_coords - line_pos), thick_px, fade_px)
        for line_pos in golden_lines_x:
            weight_map = self._apply_fade_to_line(weight_map, torch.abs(x_coords - line_pos), thick_px, fade_px)
        return weight_map

    def _generate_diagonal_flow_map(self, H: int, W: int, dev: torch.device, thick_px: int, fade_px: int) -> torch.Tensor:
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
        # Normalized coordinates [0, 1]
        norm_y = y_coords / (H - 1 + 1e-6)
        norm_x = x_coords / (W - 1 + 1e-6)
        
        # Distance to TL-BR diagonal (y=x in normalized space)
        # For line y=x, distance |y-x|/sqrt(2). We can use |y-x| as proxy if thickness is relative.
        dist_to_diag = torch.abs(norm_y - norm_x) 
        
        # Effective thickness/fade in normalized space
        # This needs careful interpretation of thick_px relative to image size for diagonal
        # Let's use a fraction of min_dim for normalized thickness
        norm_thick = thick_px / min(H, W) 
        norm_fade = fade_px / min(H, W)

        diag_intensity = torch.zeros((H,W), device=dev)
        diag_intensity = self._apply_fade_to_line(diag_intensity, dist_to_diag, norm_thick, norm_fade)
        
        # Add a vignette to emphasize corners along the diagonal (e.g. TL and BR)
        # This creates a "flow" effect along the diagonal
        # Distance from center (0.5, 0.5) in normalized space
        center_dist_norm = torch.sqrt(((norm_x - 0.5) * 2)**2 + ((norm_y - 0.5) * 2)**2) / math.sqrt(2) # Range [0,1]
        vignette = _interpolate_smoothstep(torch.clamp(center_dist_norm, 0, 1)) # Darker at center, brighter at corners
        
        # Combine diagonal line with vignette that enhances its ends
        return diag_intensity * vignette


    _GUIDE_GENERATOR_FUNCTIONS = {
        CompositionGuideType.CENTRAL_FOCUS: _generate_central_focus_map,
        CompositionGuideType.RULE_OF_THIRDS: _generate_rule_of_thirds_map,
        CompositionGuideType.GOLDEN_RATIO_LINES: _generate_golden_ratio_map,
        CompositionGuideType.DIAGONAL_FLOW_TL_BR: _generate_diagonal_flow_map,
    }

    def _create_composition_weight_map(self, guide_type: CompositionGuideType, H: int, W: int, 
                                       dev: torch.device, thickness_pixels: int, fade_pixels: int) -> torch.Tensor:
        assert H > 0 and W > 0, "Height and width must be positive for composition map."
        assert thickness_pixels >= 0 and fade_pixels >= 0, "Thickness and fade must be non-negative."

        if guide_type == CompositionGuideType.COMBINED:
            all_maps = []
            for g_type_single in self._GUIDE_GENERATOR_FUNCTIONS.keys():
                # Pass self to call instance methods for individual map generation
                map_func = self._GUIDE_GENERATOR_FUNCTIONS[g_type_single]
                all_maps.append(map_func(self, H, W, dev, thickness_pixels, fade_pixels))
            if not all_maps: return torch.zeros((1, 1, H, W), device=dev)
            # Stack along a new dimension (dim=0) and take max across this dimension
            combined_map_2d = torch.max(torch.stack(all_maps, dim=0), dim=0).values
        elif guide_type in self._GUIDE_GENERATOR_FUNCTIONS:
            map_func = self._GUIDE_GENERATOR_FUNCTIONS[guide_type]
            combined_map_2d = map_func(self, H, W, dev, thickness_pixels, fade_pixels)
        else: # NONE or unrecognized
            return torch.zeros((1, 1, H, W), device=dev)
        
        return combined_map_2d.unsqueeze(0).unsqueeze(0) # Add batch and channel dims

    def _apply_composition_guide(self, noise_tensor: torch.Tensor, dev: torch.device, 
                                 guide_type: CompositionGuideType, guide_strength: float, 
                                 guide_effect: GuideEffect, latent_W: int, latent_H: int, 
                                 guide_line_thickness_factor: float, guide_line_fading_percent: float
                                 ) -> torch.Tensor:
        if guide_type == CompositionGuideType.NONE or guide_strength == 0.0:
            return noise_tensor

        assert 0.0 <= guide_strength <= 1.0, "Guide strength must be between 0 and 1."
        assert 0.005 <= guide_line_thickness_factor <= 0.2, "Thickness factor out of range."
        assert 0.0 <= guide_line_fading_percent <= 100.0, "Fading percent out of range."

        # Calculate absolute thickness and fade in pixels for the latent dimensions
        abs_thickness_pixels = max(1, int(min(latent_H, latent_W) * guide_line_thickness_factor))
        abs_fade_pixels = int(abs_thickness_pixels * (guide_line_fading_percent / 100.0))

        weight_map = self._create_composition_weight_map(
            guide_type, latent_H, latent_W, dev, abs_thickness_pixels, abs_fade_pixels
        ) # Shape (1, 1, H, W)

        # Apply effect
        if guide_effect == GuideEffect.INTENSITY_BOOST:
            # Factor 0.5 to make strength more intuitive (0-1 range more gradual)
            noise_tensor += (weight_map * guide_strength * 0.5) 
        elif guide_effect == GuideEffect.INTENSITY_REDUCE:
            noise_tensor *= (1.0 - (weight_map * guide_strength))
        
        return noise_tensor

    def _apply_masking(self, noise_tensor: torch.Tensor, dev: torch.device, 
                       optional_mask: torch.Tensor, mask_effect: MaskEffect, 
                       latent_H: int, latent_W: int) -> torch.Tensor:
        if optional_mask is None:
            return noise_tensor

        mask = optional_mask.to(dev)
        # Ensure mask is 4D (B, C, H, W)
        if mask.dim() == 2: # H, W -> 1, 1, H, W
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3: # B, H, W -> B, 1, H, W (assuming batch first) or H, W, C (needs permute)
            # ComfyUI masks are usually (B, H, W) or (H, W)
            if mask.shape[0] == noise_tensor.shape[0] and mask.shape[1] == latent_H and mask.shape[2] == latent_W : # B, H, W
                 mask = mask.unsqueeze(1) # B, 1, H, W
            else: # Fallback or error - for now, assume it needs unsqueeze(0) if not batch
                 mask = mask.unsqueeze(1) # Assuming C, H, W -> C, 1, H, W or similar, might need review if mask shapes vary wildly.
                                          # More robust: check mask.shape against latent_H, latent_W

        mask = mask.float()
        
        # Normalize mask to [0, 1] if it's not already (e.g. 0-255)
        if mask.max() > 1.001 or mask.min() < -0.001: # Allow small float inaccuracies
            mask = normalize_tensor(mask, 0.0, 1.0)
        
        # Resize mask to latent dimensions if necessary
        if mask.shape[2] != latent_H or mask.shape[3] != latent_W:
            mask = torch.nn.functional.interpolate(
                mask, size=(latent_H, latent_W), mode='bilinear', align_corners=False
            )
        
        # Expand mask channels if noise has more channels (e.g. mask is grayscale, noise is RGB-like)
        if mask.shape[1] == 1 and noise_tensor.shape[1] > 1:
            mask = mask.expand_as(noise_tensor)
        
        # Apply mask effect
        if mask_effect == MaskEffect.MODULATE_INTENSITY_BY_MASK:
            noise_tensor *= mask
        elif mask_effect == MaskEffect.APPLY_TO_MASKED_AREA:
            # Apply noise where mask is > 0.5, zero out elsewhere
            noise_tensor = torch.where(mask > 0.5, noise_tensor, torch.zeros_like(noise_tensor))
        elif mask_effect == MaskEffect.APPLY_TO_UNMASKED_AREA:
            # Apply noise where mask is < 0.5, zero out elsewhere
            noise_tensor = torch.where(mask < 0.5, noise_tensor, torch.zeros_like(noise_tensor))
            
        return noise_tensor

    def _create_image_preview(self, noise_tensor_final: torch.Tensor, 
                              optional_vae: torch.nn.Module, 
                              target_height: int, target_width: int) -> torch.Tensor:
        # Detach from graph for preview
        noise_detached = noise_tensor_final.detach()
        
        # Attempt VAE decode if VAE is provided
        if optional_vae is not None:
            try:
                # Standard VAE scaling factor for latents
                latents_for_vae = noise_detached / 0.18215 
                vae_device = next(optional_vae.parameters()).device
                # Use VAE's dtype if available, else default (e.g. float32)
                vae_dtype = getattr(optional_vae, 'dtype', torch.float32)
                
                with torch.no_grad(): # Ensure no gradients for VAE decode
                    images_decoded = optional_vae.decode(latents_for_vae.to(vae_device, dtype=vae_dtype)).float()
                
                # Clamp and permute to (B, H, W, C) for image format, move to CPU
                preview_images = torch.clamp(images_decoded.cpu().permute(0, 2, 3, 1), 0.0, 1.0)
                # Resize if needed (though VAE output should match target if latents are correctly sized)
                if preview_images.shape[1] != target_height or preview_images.shape[2] != target_width:
                     preview_images = torch.nn.functional.interpolate(
                        preview_images.permute(0,3,1,2), # B,C,H,W for interpolate
                        size=(target_height, target_width), mode='bilinear', align_corners=False
                     ).permute(0,2,3,1) # B,H,W,C

                return preview_images
            except Exception as e:
                print(f"Error ({NODE_NAME}): VAE decode failed: {e}. Falling back to direct noise preview.")
                # Fall through to direct noise preview
        
        # Fallback: Direct preview of noise tensor (first 3 channels)
        # Ensure noise is on CPU for this path
        noise_for_preview_cpu = noise_detached.cpu()
        
        num_channels_preview = noise_for_preview_cpu.shape[1]
        if num_channels_preview == 0: # Should not happen with valid noise
            preview_tensor_ch3 = torch.zeros((noise_for_preview_cpu.shape[0], 3, noise_for_preview_cpu.shape[2], noise_for_preview_cpu.shape[3]), device="cpu")
        elif num_channels_preview == 1: # Grayscale -> Repeat to 3 channels
            preview_tensor_ch3 = noise_for_preview_cpu.repeat(1, 3, 1, 1)
        elif num_channels_preview == 2: # 2 Channels -> Add a zero channel
            zeros_channel = torch.zeros_like(noise_for_preview_cpu[:, :1])
            preview_tensor_ch3 = torch.cat((noise_for_preview_cpu, zeros_channel), dim=1)
        else: # >= 3 channels, take the first 3
            preview_tensor_ch3 = noise_for_preview_cpu[:, :3]

        # Normalize each image in batch independently to [0,1] for preview
        for i in range(preview_tensor_ch3.shape[0]):
            preview_tensor_ch3[i] = normalize_tensor(preview_tensor_ch3[i], 0.0, 1.0)
        
        # Resize to target preview dimensions and permute to (B, H, W, C)
        # Interpolate expects (B, C, H, W)
        resized_preview = torch.nn.functional.interpolate(
            preview_tensor_ch3, size=(target_height, target_width),
            mode='bilinear', align_corners=False
        )
        return resized_preview.permute(0, 2, 3, 1) # To (B, H, W, C)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_VALUE}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "device": ([e.value for e in DeviceChoice], {"default": DeviceChoice.AUTO.value}),
                "base_noise_type": ([e.value for e in BaseNoiseAlgorithm], {"default": BaseNoiseAlgorithm.SIMPLEX.value}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "normalize_output_range": ("BOOLEAN", {"default": False}),
                "monochrome_noise": ("BOOLEAN", {"default": False}),
                "initial_frequency_scale": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 256.0, "step": 0.1}),

                "enable_fbm": ("BOOLEAN", {"default": True}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 16}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.01}),
                "fbm_summing_mode": ([e.value for e in FBMSummingMode], {"default": FBMSummingMode.STANDARD.value}),
                "fbm_interpolation": ([e.value for e in InterpolationType], {"default": InterpolationType.SMOOTHSTEP.value}),

                "enable_domain_warp": ("BOOLEAN", {"default": False}),
                "warp_noise_type": ([e.value for e in WarpNoiseAlgorithm], {"default": WarpNoiseAlgorithm.SIMPLEX.value}),
                "warp_octaves": ("INT", {"default": 3, "min": 1, "max": 8}),
                "warp_strength": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 200.0, "step": 0.1}),
                "warp_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "warp_persistence": ("FLOAT", {"default": 0.5, "min":0.01, "max":1.0, "step":0.01}),
                "warp_lacunarity": ("FLOAT", {"default": 2.0, "min":1.0, "max":4.0, "step":0.01}),
                # Note: Warp interpolation uses fbm_interpolation setting

                "enable_composition_guide": ("BOOLEAN", {"default": False}),
                "composition_guide_type": ([e.value for e in CompositionGuideType], {"default": CompositionGuideType.NONE.value}),
                "guide_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guide_effect_on_noise": ([e.value for e in GuideEffect], {"default": GuideEffect.INTENSITY_BOOST.value}),
                "guide_line_thickness_factor": ("FLOAT", {"default": 0.05, "min":0.005, "max":0.2, "step":0.005}), # Relative to min(H,W)
                "guide_line_fading_percent": ("FLOAT", {"default": 50.0, "min":0.0, "max":100.0, "step":1.0}), # Relative to thickness

                "mask_effect": ([e.value for e in MaskEffect], {"default": MaskEffect.MODULATE_INTENSITY_BY_MASK.value}),
            }, 
            "optional": {"opt_vae": ("VAE",), "opt_mask": ("MASK",)}
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "generate_advanced_noise_entrypoint" # Renamed to avoid conflict if class and func have same name
    CATEGORY = CATEGORY

    def generate_advanced_noise_entrypoint(self, seed: int, width: int, height: int, batch_size: int,
                                device: str,  # Changed from device_str
                                base_noise_type: str,  # Changed from base_noise_type_str
                                strength: float, offset: float, normalize_output_range: bool,
                                monochrome_noise: bool, initial_frequency_scale: float,
                                enable_fbm: bool, octaves: int, persistence: float, lacunarity: float,
                                fbm_summing_mode: str,  # Changed from fbm_summing_mode_str
                                fbm_interpolation: str,  # Changed from fbm_interpolation_str
                                enable_domain_warp: bool,
                                warp_noise_type: str,  # Changed from warp_noise_type_str
                                warp_octaves: int,
                                warp_strength: float, warp_scale: float, warp_persistence: float, warp_lacunarity:float,
                                enable_composition_guide: bool,
                                composition_guide_type: str,  # Changed from composition_guide_type_str
                                guide_strength: float,
                                guide_effect_on_noise: str,  # Changed from guide_effect_on_noise_str
                                guide_line_thickness_factor: float, guide_line_fading_percent: float,
                                mask_effect: str,  # Changed from mask_effect_str
                                opt_vae=None, opt_mask=None):

        # --- Parameter Conversion and Validation ---
        assert width >= 64 and height >= 64, "Width and height must be at least 64."
        assert batch_size >= 1, "Batch size must be at least 1."
        assert octaves >=1, "FBM octaves must be at least 1."
        assert warp_octaves >=1, "Warp octaves must be at least 1."

        # Parameters are now directly the strings from INPUT_TYPES
        torch_device = get_torch_device(DeviceChoice(device))
        _base_noise_type_enum = BaseNoiseAlgorithm(base_noise_type) # Renamed internal var to avoid conflict
        _fbm_summing_mode_enum = FBMSummingMode(fbm_summing_mode) # Renamed internal var
        _fbm_interpolation_type_enum = InterpolationType(fbm_interpolation) # Renamed internal var

        _warp_noise_type_enum = None # Initialize
        if enable_domain_warp:
            _warp_noise_type_enum = WarpNoiseAlgorithm(warp_noise_type) # Renamed internal var

        _composition_guide_type_enum = CompositionGuideType.NONE # Initialize
        _guide_effect_on_noise_enum = None # Initialize
        if enable_composition_guide:
            _composition_guide_type_enum = CompositionGuideType(composition_guide_type) # Renamed internal var
            _guide_effect_on_noise_enum = GuideEffect(guide_effect_on_noise) # Renamed internal var

        _mask_effect_enum = MaskEffect(mask_effect) # Renamed internal var

        # --- Latent Space Setup ---
        # Standard SD latent channels, adjust if for other models
        SD_LATENT_CHANNELS = 4
        latent_height, latent_width = height // 8, width // 8

        channels_for_generation = 1 if monochrome_noise else SD_LATENT_CHANNELS
        noise_shape = (batch_size, channels_for_generation, latent_height, latent_width)

        # --- Base Noise Generation ---
        if enable_fbm:
            current_noise = self._apply_fbm(
                noise_shape, seed, torch_device, _base_noise_type_enum,
                initial_frequency_scale, octaves, persistence, lacunarity,
                _fbm_summing_mode_enum, _fbm_interpolation_type_enum
            )
        else:
            # Single layer of noise, octave_seed_offset=0
            current_noise = self._generate_single_noise_layer(
                noise_shape, seed, torch_device, _base_noise_type_enum,
                initial_frequency_scale, 0, _fbm_interpolation_type_enum
            )

        assert current_noise.shape == noise_shape, f"Generated noise shape {current_noise.shape} mismatch expected {noise_shape}."

        # --- Monochrome Handling: Expand to target channels if needed ---
        if monochrome_noise and current_noise.shape[1] == 1 and SD_LATENT_CHANNELS > 1:
            current_noise = current_noise.repeat(1, SD_LATENT_CHANNELS, 1, 1)
        elif not monochrome_noise and current_noise.shape[1] != SD_LATENT_CHANNELS and SD_LATENT_CHANNELS > 0:
            if current_noise.shape[1] > 0:
                current_noise = current_noise[:, 0:1].repeat(1, SD_LATENT_CHANNELS, 1, 1)
            else:
                current_noise = torch.zeros((batch_size, SD_LATENT_CHANNELS, latent_height, latent_width), device=torch_device)

        expected_final_channel_shape = (batch_size, SD_LATENT_CHANNELS, latent_height, latent_width)
        assert current_noise.shape == expected_final_channel_shape, "Noise shape after monochrome handling is incorrect."


        # --- Domain Warping ---
        if enable_domain_warp:
            current_noise = self._apply_domain_warp(
                current_noise, seed, torch_device, _warp_noise_type_enum, # Use the enum version
                warp_octaves, warp_persistence, warp_lacunarity,
                warp_strength, warp_scale, _fbm_interpolation_type_enum
            )
            assert current_noise.shape == expected_final_channel_shape, "Noise shape after domain warp is incorrect."


        # --- Composition Guide ---
        if enable_composition_guide and _composition_guide_type_enum != CompositionGuideType.NONE:
            current_noise = self._apply_composition_guide(
                current_noise, torch_device, _composition_guide_type_enum, # Use the enum version
                guide_strength, _guide_effect_on_noise_enum, latent_width, # Use the enum version
                latent_height, guide_line_thickness_factor, guide_line_fading_percent
            )
            assert current_noise.shape == expected_final_channel_shape, "Noise shape after composition guide is incorrect."

        # --- Masking ---
        if opt_mask is not None:
            current_noise = self._apply_masking(
                current_noise, torch_device, opt_mask, _mask_effect_enum, # Use the enum version
                latent_height, latent_width
            )
            assert current_noise.shape == expected_final_channel_shape, "Noise shape after masking is incorrect."

        # --- Final Adjustments (Strength, Offset, Normalization) ---
        current_noise = current_noise * strength + offset

        if normalize_output_range:
            for i in range(current_noise.shape[0]):
                current_noise[i] = normalize_tensor(current_noise[i], -1.0, 1.0)

        # --- Output ---
        latent_output_dict = {"samples": current_noise}

        image_preview_tensor = self._create_image_preview(current_noise, opt_vae, height, width)
        assert image_preview_tensor.shape == (batch_size, height, width, 3), "Image preview shape is incorrect."

        return (latent_output_dict, image_preview_tensor)

NODE_CLASS_MAPPINGS = {NODE_NAME: Advanced_Latent_Noise}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: NODE_DISPLAY_NAME}