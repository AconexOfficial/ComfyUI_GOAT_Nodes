import torch
import torch.nn.functional as F
import comfy.utils
import math
import comfy.model_management

# Dependency checks (cv2, scipy, numpy)
try: import cv2
except ImportError: cv2 = None
try:
    import scipy.ndimage
    import numpy as np
except ImportError: scipy, np = None, None

# --- Constants and Default Settings ---
class DefaultSettings:
    # Image Processing Constants
    LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]) # Standard Rec.709
    LAPLACIAN_KERNEL_T = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    SOBEL_X_KERNEL_T = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
    SOBEL_Y_KERNEL_T = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)

    # Guided Filter Defaults
    GUIDED_FILTER_DEFAULT_RADIUS = 3
    GUIDED_FILTER_DEFAULT_EPS_DENOISE = 0.01**2
    GUIDED_FILTER_DEFAULT_EPS_SHARPEN = 0.1**2 # Base Epsilon for sharpening
    GUIDED_FILTER_DEFAULT_EPS_MASK_SMOOTH = 0.001**2

    # LBP Defaults
    LBP_DEFAULT_RADIUS = 1
    LBP_DEFAULT_NEIGHBORS = 8

    # Stage Aware Control (SAC) Defaults
    SAC_CHANGE_THRESH = 0.05
    SAC_FLATNESS_GUIDE_BLUR_SIGMA = 1.0
    SAC_NOISE_MAP_BLUR_SIGMA = 0.5
    SAC_DETAIL_MAP_WINDOW_SIZE = 5
    SAC_DETAIL_MAP_BLUR_SIGMA = 1.0
    SAC_USM_HALO_MAP_BLUR_SIGMA = 1.0
    SAC_EDGE_MASK_BLUR_SIGMA = 1.0
    SAC_USE_GUIDED_FILTER_FOR_MASKS = True

    # Tiling Defaults
    TILING_MIN_OVERLAP_PX = 16
    TILING_OVERLAP_PERCENT_OF_TILE = 0.125

    # --- Input Parameter Scaling ---
    OVERALL_STRENGTH_SCALAR = 8.0
    PREPROCESS_DENOISE_SCALAR = 0.4
    GF_SHARPEN_SCALAR = 2.0
    USM_SHARPEN_SCALAR = 3.2
    FFT_SHARPEN_SCALAR = 0.8
    EDGE_SHARPEN_SCALAR = 2.0
    SAC_MODERATION_SCALAR = 2.0
    POSTPROCESS_DENOISE_SCALAR = 1.0

    # --- Sharpening Presets ---
    DEFAULT_PRESET_NAME = "crisp"
    PRESET_PARAMS = {
        "clean": { # Very low noise, clean edges, minimal edge brightening
            "gf_radius": 1,             "gf_eps_sharpen": 0.005**2, "gf_strength_mult": 1.8,  # Extremely low eps, very high strength
            "usm_spatial_sigma": 1.0,   "usm_threshold": 0.0,     "usm_strength_mult": 2.2,  # Finer sigma, zero thresh, very high strength
            "fft_cutoff_freq": 0.08,    "fft_strength_mult": 2.2,  # Very low cutoff, very high strength
            "edge_gf_radius": 1,        "edge_gf_eps": 0.005**2,  "edge_sobel_thresh": 0.05, "edge_strength_mult": 1.8,  # Match GF eps, very low sobel, very high strength
        },
        "crisp": { # low noise, sharp edges, moderate edge brightening
            "gf_radius": 1, "gf_eps_sharpen": GUIDED_FILTER_DEFAULT_EPS_SHARPEN, "gf_strength_mult": 1.0,
            "usm_spatial_sigma": 1.5, "usm_threshold": 0.01, "usm_strength_mult": 1.0,
            "fft_cutoff_freq": 0.15, "fft_strength_mult": 1.0,
            "edge_gf_radius": 1, "edge_gf_eps": GUIDED_FILTER_DEFAULT_EPS_SHARPEN, "edge_sobel_thresh": 0.1, "edge_strength_mult": 1.0,
        },
        "gritty": { # Stylized, textured look, strong contrast, high edge brightening
            "gf_radius": 3, "gf_eps_sharpen": GUIDED_FILTER_DEFAULT_EPS_SHARPEN, "gf_strength_mult": 1.4,
            "usm_spatial_sigma": 2.0, "usm_threshold": 0.0, "usm_strength_mult": 1.6,
            "fft_cutoff_freq": 0.15, "fft_strength_mult": 0.75,
            "edge_gf_radius": 3, "edge_gf_eps": GUIDED_FILTER_DEFAULT_EPS_SHARPEN, "edge_sobel_thresh": 0.1, "edge_strength_mult": 1.3,
        }
    }

class Advanced_Sharpen:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.luma_weights = DefaultSettings.LUMA_WEIGHTS.to(self.device).view(1, 1, 1, 3)
        self.laplacian_kernel = DefaultSettings.LAPLACIAN_KERNEL_T.to(self.device).unsqueeze(0).unsqueeze(0)
        self.sobel_x_kernel = DefaultSettings.SOBEL_X_KERNEL_T.to(self.device).unsqueeze(0).unsqueeze(0)
        self.sobel_y_kernel = DefaultSettings.SOBEL_Y_KERNEL_T.to(self.device).unsqueeze(0).unsqueeze(0)
        if scipy is None or np is None:
            print("Warning: 🐐 Advanced_Sharpen - SciPy/NumPy not found. Some features (SciPy Gaussian Blur, potentially fallback USM blur) might be unavailable or replaced by Torch alternatives.")
        if cv2 is None:
            print("Warning: 🐐 Advanced_Sharpen - OpenCV (cv2) not found. Bilateral filtering for USM and potential future CV2-based features will be unavailable or fall back to alternatives.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "overall_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Master control for the final blend between original and sharpened image."}),
                # --- Preprocessing ---
                "preprocess_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of selective denoising applied before sharpening (targets flat areas). May deteriorate small details at high values."}),
                # --- Sharpening Stages ---
                "preset": (list(DefaultSettings.PRESET_PARAMS.keys()), {"default": DefaultSettings.DEFAULT_PRESET_NAME, "tooltip":"Select a sharpening characteristic preset. Modifies internal parameters of sharpening stages."}),
                "luma_sharpen": ("BOOLEAN", {"default": True, "tooltip":"Perform sharpening only on the luminance (brightness) channel to avoid color artifacts."}),
                "gf_sharpen": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Strength of Guided Filter sharpening (good for general detail enhancement)."}),
                "usm_sharpen": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Strength of Unsharp Mask (effective for edge contrast and granularity)."}),
                "fft_sharpen": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Strength of Frequency Domain (FFT) sharpening (enhances high frequencies)."}),
                "edge_sharpen": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Strength of Edge Refinement sharpening (selectively sharpens detected edges)."}),
                # --- Stage Aware Control (SAC) ---
                "stage_aware_control": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Controls the intensity of Stage-Aware Control. 0.0 disables SAC. Higher values activate content-aware moderation (flatness, edges, noise, detail, halos) to reduce artifacts."}),
                # --- Postprocessing ---
                "postprocess_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Strength of subtle final smoothing pass to reduce minor artifacts."}),
                # --- Technical ---
                "tiled_sharpen": ("BOOLEAN", {"default": False, "tooltip":"Process the image in seamless tiles to save memory on large images."}),
                "tile_size": ("INT", {"default": 1024, "min": 128,"max": 4096,"step": 128, "tooltip":"Size of tiles used if Tiled Sharpen is enabled."}),
            }
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("IMAGE","INT","INT",), ("IMAGE","WIDTH","HEIGHT",), "execute_sharpening", '🐐 GOAT Nodes/Postprocessing'
    DESCRIPTION = "Advanced sharpening suite with selective denoising, multiple sharpening methods (GF, USM, FFT, Edge), Stage-Aware Control (SAC) for artifact and noise reduction, Luma-only mode as well as optional tiling."

    # --- Utility Methods ---
    def _assert_tensor_nhwc(self, tensor: torch.Tensor, name: str, expected_channels: int = None):
        assert isinstance(tensor, torch.Tensor), f"{name} must be a Torch tensor."
        assert tensor.ndim == 4, f"{name} must be 4D NHWC, got {tensor.ndim}D."
        if expected_channels is not None:
            assert tensor.shape[3] == expected_channels, f"{name} must have {expected_channels} channels, got {tensor.shape[3]}."

    def _to_grayscale_torch(self, tensor_nhwc: torch.Tensor) -> torch.Tensor:
        self._assert_tensor_nhwc(tensor_nhwc, "Grayscale input")
        if tensor_nhwc.shape[3] == 1: return tensor_nhwc
        gray_nhwc = torch.sum(tensor_nhwc * self.luma_weights.to(tensor_nhwc.device), dim=3, keepdim=True)
        self._assert_tensor_nhwc(gray_nhwc, "Grayscale output", 1)
        return gray_nhwc

    def _normalize_map_torch(self, tensor_map: torch.Tensor, per_batch_item: bool = True, eps: float = 1e-6) -> torch.Tensor:
        assert isinstance(tensor_map, torch.Tensor), "Input for normalization must be a tensor."
        if per_batch_item and tensor_map.ndim > 0 and tensor_map.shape[0] > 1 :
            normalized_maps = []
            for i in range(tensor_map.shape[0]):
                item = tensor_map[i]
                min_val, max_val = torch.min(item), torch.max(item)
                if (max_val - min_val).abs() > eps:
                    normalized_maps.append((item - min_val) / (max_val - min_val + eps))
                else:
                    normalized_maps.append(torch.zeros_like(item))
            output = torch.stack(normalized_maps, dim=0)
        else:
            min_val, max_val = torch.min(tensor_map), torch.max(tensor_map)
            if (max_val - min_val).abs() > eps:
                output = (tensor_map - min_val) / (max_val - min_val + eps)
            else:
                output = torch.zeros_like(tensor_map)
        assert output.shape == tensor_map.shape, "Normalized map shape mismatch."
        return output

    def _rgb_to_ycbcr(self, image_rgb_nhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._assert_tensor_nhwc(image_rgb_nhwc, "RGB to YCbCr input", 3)
        r, g, b = image_rgb_nhwc[..., 0], image_rgb_nhwc[..., 1], image_rgb_nhwc[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return y.unsqueeze(-1), cb.unsqueeze(-1), cr.unsqueeze(-1)

    def _ycbcr_to_rgb(self, y_nhwc: torch.Tensor, cb_nhwc: torch.Tensor, cr_nhwc: torch.Tensor) -> torch.Tensor:
        self._assert_tensor_nhwc(y_nhwc, "YCbCr_Y to RGB input", 1)
        self._assert_tensor_nhwc(cb_nhwc, "YCbCr_Cb to RGB input", 1)
        self._assert_tensor_nhwc(cr_nhwc, "YCbCr_Cr to RGB input", 1)
        cb_c = cb_nhwc - 0.5
        cr_c = cr_nhwc - 0.5
        r = y_nhwc + 1.402 * cr_c
        g = y_nhwc - 0.344136 * cb_c - 0.714136 * cr_c
        b = y_nhwc + 1.772 * cb_c
        rgb_image = torch.cat([r,g,b], dim=-1)
        return torch.clamp(rgb_image, 0.0, 1.0)

    def _box_filter_torch(self, tensor_nchw: torch.Tensor, radius: int) -> torch.Tensor:
        assert tensor_nchw.ndim == 4, "Box filter input must be NCHW"
        kernel_size = 2 * radius + 1
        return F.avg_pool2d(tensor_nchw, kernel_size=kernel_size, stride=1, padding=radius, count_include_pad=False)

    def _guided_filter_torch(self, x_nhwc: torch.Tensor, guide_nhwc: torch.Tensor, radius: int, eps: float, guide_is_gray: bool = False) -> torch.Tensor:
        self._assert_tensor_nhwc(x_nhwc, "GF input x")
        self._assert_tensor_nhwc(guide_nhwc, "GF input guide")
        assert x_nhwc.shape[:3] == guide_nhwc.shape[:3], "GF input x and guide must match N, H, W dimensions."

        x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
        if guide_is_gray:
            guide_nchw = guide_nhwc.permute(0, 3, 1, 2).contiguous()
        else:
            if guide_nhwc.shape[3] == 3:
                guide_nchw_orig = guide_nhwc.permute(0, 3, 1, 2).contiguous()
                # If input x is grayscale, guide should also be grayscale
                if x_nchw.shape[1] == 1:
                    guide_gray_nhwc = self._to_grayscale_torch(guide_nhwc)
                    guide_nchw = guide_gray_nhwc.permute(0, 3, 1, 2).contiguous()
                else: # Both x and guide are color
                     guide_nchw = guide_nchw_orig
            elif guide_nhwc.shape[3] == 1: # Guide is already gray
                guide_nchw = guide_nhwc.permute(0,3,1,2).contiguous()
            else:
                raise ValueError(f"Unsupported guide channels: {guide_nhwc.shape[3]}")

        mean_I = self._box_filter_torch(guide_nchw, radius)
        mean_p = self._box_filter_torch(x_nchw, radius)
        mean_Ip = self._box_filter_torch(guide_nchw * x_nchw, radius)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self._box_filter_torch(guide_nchw * guide_nchw, radius)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = self._box_filter_torch(a, radius)
        mean_b = self._box_filter_torch(b, radius)

        output_nchw = mean_a * guide_nchw + mean_b
        return output_nchw.permute(0, 2, 3, 1)

    def _torch_gaussian_blur(self, tensor_nhwc: torch.Tensor, sigma: float, kernel_size: int = 0) -> torch.Tensor:
        self._assert_tensor_nhwc(tensor_nhwc, "Gaussian blur input")
        if sigma <= 1e-6: return tensor_nhwc

        # Determine kernel size if not provided
        if kernel_size == 0:
             kernel_size = max(3, int(round(sigma * 3.5)) * 2 + 1)
        elif kernel_size % 2 == 0:
             kernel_size +=1 # Ensure odd kernel size

        # Create 1D Gaussian kernel
        center = kernel_size // 2
        x_coords = torch.arange(0, kernel_size, dtype=tensor_nhwc.dtype, device=tensor_nhwc.device)
        kernel_1d = torch.exp(-((x_coords - center) ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum() # Normalize

        img_nchw = tensor_nhwc.permute(0, 3, 1, 2).contiguous() # BHWC -> BCHW
        b, channels, h, w = img_nchw.shape
        padding = kernel_size // 2

        # === Horizontal Blur ===
        kernel_h = kernel_1d.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1) # Shape: [C, 1, 1, K]

        # Manual padding for horizontal blur (pad only width dimension)
        padded_img_h = F.pad(img_nchw, (padding, padding, 0, 0), mode='reflect')

        # Convolve with padding=0
        blurred_h = F.conv2d(padded_img_h, kernel_h, padding=0, groups=channels)

        # === Vertical Blur ===
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1) # Shape: [C, 1, K, 1]

        # Manual padding for vertical blur (pad only height dimension)
        padded_img_v = F.pad(blurred_h, (0, 0, padding, padding), mode='reflect')

        # Convolve with padding=0
        blurred_hv = F.conv2d(padded_img_v, kernel_v, padding=0, groups=channels)

        return blurred_hv.permute(0, 2, 3, 1) # BCHW -> BHWC

    def _smooth_map_torch(self, map_nhwc: torch.Tensor, guide_nhwc: torch.Tensor, params: dict, key_prefix:str = "mask") -> torch.Tensor:
        self._assert_tensor_nhwc(map_nhwc, f"{key_prefix} map to smooth")
        self._assert_tensor_nhwc(guide_nhwc, f"{key_prefix} guide for smoothing")

        if DefaultSettings.SAC_USE_GUIDED_FILTER_FOR_MASKS:
            radius = DefaultSettings.GUIDED_FILTER_DEFAULT_RADIUS
            eps = DefaultSettings.GUIDED_FILTER_DEFAULT_EPS_MASK_SMOOTH
            guide_gray_nhwc = self._to_grayscale_torch(guide_nhwc)
            smoothed_map = self._guided_filter_torch(map_nhwc, guide_gray_nhwc, radius, eps, guide_is_gray=True)
        else:
            # Fallback to Gaussian blur if GF is disabled for masks
            blur_sigma = params.get(f"sac_{key_prefix}_blur_sigma", DefaultSettings.SAC_EDGE_MASK_BLUR_SIGMA) # Reuse edge blur sigma as default
            if key_prefix == "edge": blur_sigma = params.get("edge_mask_blur_sigma_sac", DefaultSettings.SAC_EDGE_MASK_BLUR_SIGMA) # Specific sigma for edge mask if defined elsewhere

            if blur_sigma > 0:
                 smoothed_map = self._torch_gaussian_blur(map_nhwc, blur_sigma)
            else:
                 smoothed_map = map_nhwc # No smoothing if sigma is zero

        return self._normalize_map_torch(smoothed_map) # Normalize after smoothing

    def _torch_sobel(self, tensor_nhwc_gray: torch.Tensor, add_epsilon: float = 1e-6) -> torch.Tensor:
        self._assert_tensor_nhwc(tensor_nhwc_gray, "Sobel input", 1)
        img_nchw = tensor_nhwc_gray.permute(0, 3, 1, 2).contiguous() # BHWC to BCHW

        # Sobel kernels should be [out_channels, in_channels, kH, kW]
        k_x = self.sobel_x_kernel.to(img_nchw.device) # Shape [1, 1, 3, 3]
        k_y = self.sobel_y_kernel.to(img_nchw.device) # Shape [1, 1, 3, 3]

        # For BCHW (dim 0, 1, 2, 3), we want to pad dims 2 (H) and 3 (W) by 1 pixel on each side.
        padding_amount = 1
        padded_img_nchw = F.pad(img_nchw, (padding_amount, padding_amount, padding_amount, padding_amount), mode='reflect')

        # Apply convolution with padding=0 because the input is already padded
        grad_x = F.conv2d(padded_img_nchw, k_x, stride=1, padding=0)
        grad_y = F.conv2d(padded_img_nchw, k_y, stride=1, padding=0)

        # Calculate magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + add_epsilon)
        return magnitude.permute(0, 2, 3, 1) # Back to NHWC

    def _torch_lbp(self, img_nhwc_gray: torch.Tensor, radius: int = 1, neighbors: int = 8) -> torch.Tensor:
        self._assert_tensor_nhwc(img_nhwc_gray, "LBP input", 1)
        if neighbors != 8 or radius != 1:
            print("Warning: 🐐 Advanced_Sharpen - Current LBP implementation is optimized for radius=1, neighbors=8. Results for other values may not be standard LBP.")

        img_nchw = img_nhwc_gray.permute(0, 3, 1, 2)
        b, _, h, w = img_nchw.shape

        # Offsets for 8 neighbors at radius 1
        offsets = [(0, -radius), (-radius, -radius), (-radius, 0), (-radius, radius),
                   (0, radius), (radius, radius), (radius, 0), (radius, -radius)]

        lbp_map_nchw = torch.zeros_like(img_nchw, dtype=torch.float32)
        padded_img = F.pad(img_nchw, (radius, radius, radius, radius), mode='reflect')

        # Calculate LBP code
        for i, (dy, dx) in enumerate(offsets):
            # Extract neighbor values using slicing on the padded image
            neighbor_values = padded_img[:, :, radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            # Compare neighbor with center pixel
            comparison = (neighbor_values >= img_nchw).float()
            # Add weighted comparison to LBP map
            lbp_map_nchw += comparison * (2**i)

        lbp_map_nhwc = lbp_map_nchw.permute(0,2,3,1)
        # Normalize the LBP map to 0-1 range
        return self._normalize_map_torch(lbp_map_nhwc)

    # --- NumPy/SciPy/CV2 Helpers ---
    @staticmethod
    def _tensor_to_numpy_bhwc(t):
        return t.detach().cpu().numpy() # Ensure detached before converting

    @staticmethod
    def _numpy_to_tensor_nhwc(n, device):
        return torch.from_numpy(n).to(device)

    def _apply_scipy_gaussian_blur(self, t_nhwc, sigma):
        if sigma <= 0 or scipy is None: return t_nhwc
        original_device = t_nhwc.device
        numpy_batch = self._tensor_to_numpy_bhwc(t_nhwc)
        blurred_batch = np.zeros_like(numpy_batch)
        # Process each image in the batch
        for i in range(numpy_batch.shape[0]):
             # Apply Gaussian filter (sigma=(y, x, c))
             blurred_batch[i] = scipy.ndimage.gaussian_filter(numpy_batch[i], sigma=(sigma, sigma, 0), mode='reflect')
        return self._numpy_to_tensor_nhwc(blurred_batch, original_device)

    def _apply_cv2_bilateral_blur(self, t_nhwc, spatial_sigma, color_sigma):
        self._assert_tensor_nhwc(t_nhwc, "CV2 Bilateral input")
        if spatial_sigma <= 0 or color_sigma <= 0 or cv2 is None: return t_nhwc

        original_device = t_nhwc.device
        is_single_channel = t_nhwc.shape[3] == 1
        numpy_batch = self._tensor_to_numpy_bhwc(t_nhwc)

        # Determine diameter 'd' based on spatial sigma (common practice)
        d = max(5, int(spatial_sigma * 2.5) * 2 + 1)

        # Ensure input is float32 for cv2.bilateralFilter
        if not numpy_batch.dtype == np.float32:
            numpy_batch = numpy_batch.astype(np.float32)

        blurred_batch = np.zeros_like(numpy_batch, dtype=np.float32)

        # Process each image in the batch
        for i in range(numpy_batch.shape[0]):
            image_slice = numpy_batch[i]
            # OpenCV bilateralFilter expects HWC format
            if is_single_channel:
                # Needs 2D input if single channel (remove channel dim)
                input_for_cv2 = np.squeeze(image_slice, axis=-1) if image_slice.ndim == 3 else image_slice
                filtered_slice_2d = cv2.bilateralFilter(input_for_cv2, d, color_sigma, spatial_sigma, borderType=cv2.BORDER_REFLECT)
                # Add channel dimension back
                blurred_batch[i] = filtered_slice_2d[:, :, np.newaxis]
            else: # Color image (HWC)
                blurred_batch[i] = cv2.bilateralFilter(image_slice, d, color_sigma, spatial_sigma, borderType=cv2.BORDER_REFLECT)

        return self._numpy_to_tensor_nhwc(blurred_batch, original_device)

    # --- Preprocessing Stage ---
    def _selective_denoise(self, img_nhwc: torch.Tensor, strength: float, params: dict) -> torch.Tensor:
        if strength <= 1e-6: return img_nhwc
        self._assert_tensor_nhwc(img_nhwc, "Selective denoise input")

        radius = DefaultSettings.GUIDED_FILTER_DEFAULT_RADIUS
        eps = DefaultSettings.GUIDED_FILTER_DEFAULT_EPS_DENOISE

        guide_for_denoise = img_nhwc
        guide_is_gray = False
        if img_nhwc.shape[3] == 3:
             guide_for_denoise_gray = self._to_grayscale_torch(img_nhwc)
             guide_for_denoise = guide_for_denoise_gray
             guide_is_gray = True

        denoised_img = self._guided_filter_torch(img_nhwc, guide_for_denoise, radius, eps, guide_is_gray=guide_is_gray)

        # Use flatness map to control blending: apply more denoising to flatter areas
        img_gray_for_flat_map = self._to_grayscale_torch(img_nhwc)
        # Use the original image as guide for smoothing the flatness map
        flat_map = self._generate_flatness_map(img_gray_for_flat_map, DefaultSettings.SAC_FLATNESS_GUIDE_BLUR_SIGMA, params, guide_nhwc=img_nhwc)

        # Ensure flat_map matches image channels for blending
        if img_nhwc.shape[3] > 1 and flat_map.shape[3] == 1:
            flat_map = flat_map.repeat(1,1,1,img_nhwc.shape[3])

        # Blend original and denoised based on flatness and strength
        output_img = img_nhwc * (1.0 - flat_map * strength) + denoised_img * (flat_map * strength)

        return torch.clamp(output_img, 0.0, 1.0)

    # --- Main Sharpening Stage Methods ---
    # --- Stage 1: Guided Filter Sharpen ---
    def _guided_filter_sharpen(self, img_nhwc: torch.Tensor, params: dict) -> torch.Tensor:
        self._assert_tensor_nhwc(img_nhwc, "Guided Filter sharpen input")

        selected_preset_name = params.get('preset', DefaultSettings.DEFAULT_PRESET_NAME)
        preset_config = DefaultSettings.PRESET_PARAMS.get(selected_preset_name, DefaultSettings.PRESET_PARAMS[DefaultSettings.DEFAULT_PRESET_NAME])

        # Get 0-1 input, scale it by base scalar and preset multiplier
        amount = params.get('gf_sharpen', 0.0) * DefaultSettings.GF_SHARPEN_SCALAR * preset_config["gf_strength_mult"]
        if amount <= 1e-6: return img_nhwc

        radius = preset_config["gf_radius"]
        eps = preset_config["gf_eps_sharpen"]

        # Use grayscale guide for sharpening both gray and color images
        guide_img = self._to_grayscale_torch(img_nhwc) if img_nhwc.shape[3] == 3 else img_nhwc
        guide_is_gray = True

        # Apply guided filter to get smoothed version
        smoothed_img = self._guided_filter_torch(img_nhwc, guide_img, radius, eps, guide_is_gray=guide_is_gray)

        # Calculate details (high frequencies)
        details = img_nhwc - smoothed_img

        # Add scaled details back to original image
        sharpened_image = img_nhwc + details * amount
        return torch.clamp(sharpened_image, 0.0, 1.0)

    # --- Stage 2: Unsharp Mask ---
    def _unsharp_mask(self, img_nhwc, params: dict):
        self._assert_tensor_nhwc(img_nhwc, "USM input")

        selected_preset_name = params.get('preset', DefaultSettings.DEFAULT_PRESET_NAME)
        preset_config = DefaultSettings.PRESET_PARAMS.get(selected_preset_name, DefaultSettings.PRESET_PARAMS[DefaultSettings.DEFAULT_PRESET_NAME])

        # Get 0-1 input, scale it by base scalar and preset multiplier
        amount = params.get('usm_sharpen', 0.0) * DefaultSettings.USM_SHARPEN_SCALAR * preset_config["usm_strength_mult"]
        if amount <= 1e-6: return img_nhwc

        # USM parameters from preset
        spatial_sigma = preset_config["usm_spatial_sigma"]
        threshold = preset_config["usm_threshold"]
        color_sigma = 0.1    # Color sigma for Bilateral (if used) - kept standard for now

        # Prefer Bilateral if CV2 available, fallback to Gaussian (SciPy then Torch)
        use_bilateral = cv2 is not None
        blurred = None

        if use_bilateral:
            blurred = self._apply_cv2_bilateral_blur(img_nhwc, spatial_sigma, color_sigma)
        elif scipy is not None:
            blurred = self._apply_scipy_gaussian_blur(img_nhwc, spatial_sigma)
        else:
            # Fallback to Torch Gaussian if SciPy/CV2 are missing
            blurred = self._torch_gaussian_blur(img_nhwc, spatial_sigma)

        assert blurred is not None and blurred.shape == img_nhwc.shape, "Blurring failed or shape changed in USM."

        # Calculate details (difference)
        details = img_nhwc - blurred

        # Apply threshold to details
        if threshold > 0: # Check threshold > 0 as it can be 0.0 for gritty
            details = details * (torch.abs(details) >= threshold).float()
        elif threshold < 0:
             pass

        # Add scaled details back
        sharpened_image = img_nhwc + details * amount
        return torch.clamp(sharpened_image, 0.0, 1.0)

    # --- Stage 3: FFT Sharpen ---
    def _fft_sharpen(self, img_nhwc: torch.Tensor, params: dict) -> torch.Tensor:
        self._assert_tensor_nhwc(img_nhwc, "FFT input")

        selected_preset_name = params.get('preset', DefaultSettings.DEFAULT_PRESET_NAME)
        preset_config = DefaultSettings.PRESET_PARAMS.get(selected_preset_name, DefaultSettings.PRESET_PARAMS[DefaultSettings.DEFAULT_PRESET_NAME])

        # Get 0-1 input, scale it by base scalar and preset multiplier
        amount = params.get('fft_sharpen', 0.0) * DefaultSettings.FFT_SHARPEN_SCALAR * preset_config["fft_strength_mult"]
        if amount <= 1e-6: return img_nhwc

        cutoff_freq = preset_config["fft_cutoff_freq"]
        b,h,w,c = img_nhwc.shape
        img_nchw = img_nhwc.permute(0,3,1,2).contiguous()

        # Create frequency coordinates
        y_freqs = torch.fft.fftshift(torch.fft.fftfreq(h, dtype=img_nchw.dtype, device=img_nchw.device))
        x_freqs = torch.fft.fftshift(torch.fft.fftfreq(w, dtype=img_nchw.dtype, device=img_nchw.device))
        y_mesh, x_mesh = torch.meshgrid(y_freqs, x_freqs, indexing='ij')

        # Create high-pass filter mask (Gaussian high-pass)
        sigma_freq = cutoff_freq / 2.0 # Sigma in frequency domain
        if sigma_freq < 1e-6: sigma_freq = 1e-6 # Avoid division by zero
        radius_sq = y_mesh**2 + x_mesh**2
        low_pass_filter_shifted = torch.exp(-radius_sq / (2 * sigma_freq**2))
        high_pass_filter_shifted = 1.0 - low_pass_filter_shifted

        hpf_mask = high_pass_filter_shifted.unsqueeze(0).unsqueeze(0).expand(b,c,h,w)

        # Apply FFT, filtering, and inverse FFT
        fft_img_complex = torch.fft.fft2(img_nchw, dim=(-2,-1))
        fft_img_shifted = torch.fft.fftshift(fft_img_complex, dim=(-2,-1))

        # Enhance high frequencies based on the mask and amount
        fft_sharpened_shifted = fft_img_shifted * (1.0 + amount * hpf_mask)

        fft_sharpened_unshifted = torch.fft.ifftshift(fft_sharpened_shifted, dim=(-2,-1))
        sharpened_img_nchw = torch.fft.ifft2(fft_sharpened_unshifted, dim=(-2,-1)).real # Take real part

        return torch.clamp(sharpened_img_nchw.permute(0,2,3,1), 0.0, 1.0)

    # --- Stage 4: Edge Refine ---
    def _edge_refine(self, img_nhwc: torch.Tensor, orig_for_mask_nhwc: torch.Tensor, params: dict) -> torch.Tensor:
        self._assert_tensor_nhwc(img_nhwc, "Edge refine current image input")
        self._assert_tensor_nhwc(orig_for_mask_nhwc, "Edge refine original image input for mask", 1) # Expect Luma

        selected_preset_name = params.get('preset', DefaultSettings.DEFAULT_PRESET_NAME)
        preset_config = DefaultSettings.PRESET_PARAMS.get(selected_preset_name, DefaultSettings.PRESET_PARAMS[DefaultSettings.DEFAULT_PRESET_NAME])

        # Get 0-1 input, scale it by base scalar and preset multiplier
        amount = params.get('edge_sharpen', 0.0) * DefaultSettings.EDGE_SHARPEN_SCALAR * preset_config["edge_strength_mult"]
        if amount <= 1e-6: return img_nhwc

        # Parameters for internal GF from preset
        radius = preset_config["edge_gf_radius"]
        eps = preset_config["edge_gf_eps"]

        # Calculate details using Guided Filter (similar to GF sharpen stage)
        img_gray_for_guide = self._to_grayscale_torch(img_nhwc) if img_nhwc.shape[3] == 3 else img_nhwc
        guide_is_gray = True # Always use gray guide here
        smoothed_for_detail = self._guided_filter_torch(img_nhwc, img_gray_for_guide, radius, eps, guide_is_gray=guide_is_gray)
        details = img_nhwc - smoothed_for_detail

        # --- Generate Edge Mask ---
        # Use the provided original Luma image for edge detection
        sobel_input_img = self._torch_gaussian_blur(orig_for_mask_nhwc, sigma=0.5) # Standard pre-blur
        edge_mask = self._torch_sobel(sobel_input_img)
        edge_mask = self._normalize_map_torch(edge_mask) # Normalize Sobel magnitude

        # Threshold the edge map using preset value
        sobel_threshold = preset_config["edge_sobel_thresh"]
        edge_mask = (edge_mask > sobel_threshold).float()

        # Smooth the binary edge mask using the chosen method (GF or Gaussian)
        edge_mask = self._smooth_map_torch(edge_mask, params['__original_guide_for_masks__'], params, key_prefix="edge")

        # Ensure mask matches image channels for blending
        if img_nhwc.shape[3] > 1 and edge_mask.shape[3] == 1:
            edge_mask = edge_mask.repeat(1,1,1,img_nhwc.shape[3])

        # Apply sharpened details only in edge regions
        refined_image = img_nhwc + details * edge_mask * amount
        return torch.clamp(refined_image, 0.0, 1.0)

    # --- SAC Guidance Map Generation ---
    def _generate_flatness_map(self, img_nhwc_gray: torch.Tensor, blur_sigma: float, params: dict, guide_nhwc: torch.Tensor) -> torch.Tensor:
        self._assert_tensor_nhwc(img_nhwc_gray, "Flatness map input", 1)
        # Flatness is inversely related to edge strength (Sobel magnitude)
        sobel_mag = self._torch_sobel(img_nhwc_gray)
        normalized_sobel = self._normalize_map_torch(sobel_mag)
        flat_map = 1.0 - normalized_sobel
        # Smooth the flatness map using the provided guide
        if blur_sigma > 0:
            flat_map = self._smooth_map_torch(flat_map, guide_nhwc, params, key_prefix="flatness_guide")
        return self._normalize_map_torch(flat_map) # Normalize final map

    def _generate_noise_map(self, img_nhwc_gray: torch.Tensor, M_flat: torch.Tensor, blur_sigma: float, params:dict, guide_nhwc: torch.Tensor) -> torch.Tensor:
        self._assert_tensor_nhwc(img_nhwc_gray, "Noise map input gray", 1)
        self._assert_tensor_nhwc(M_flat, "Noise map input M_flat", 1)

        # Estimate noise using Laplacian response in flat areas
        img_nchw_gray = img_nhwc_gray.permute(0,3,1,2).contiguous() # BHWC -> BCHW
        # Laplacian kernel needs shape [out_channels, in_channels, kH, kW]
        lap_k = self.laplacian_kernel.to(img_nchw_gray.device) # Shape [1, 1, 3, 3]

        # --- Manual Padding for Laplacian ---
        padding_amount = 1 # For 3x3 kernel
        padded_img_nchw_gray = F.pad(img_nchw_gray, (padding_amount, padding_amount, padding_amount, padding_amount), mode='reflect')

        # Apply convolution with padding=0 on the manually padded image
        lap_response_nchw = torch.abs(F.conv2d(padded_img_nchw_gray, lap_k, padding=0))
        lap_response_nhwc = lap_response_nchw.permute(0,2,3,1) # BCHW -> BHWC

        # Ensure shapes match M_flat (should be okay now with manual padding)
        if M_flat.shape[1:3] != lap_response_nhwc.shape[1:3]:
            # This check might still be useful if resizing occurs elsewhere, but less likely needed for padding itself
            print(f"Warning: Resizing lap_response in _generate_noise_map (Post-Conv). M_flat: {M_flat.shape[1:3]}, Lap_Resp: {lap_response_nhwc.shape[1:3]}. This might indicate issues.")
            target_h, target_w = M_flat.shape[1], M_flat.shape[2]
            # Ensure we permute back to NCHW for interpolate if resizing is needed
            lap_response_resized_nchw = F.interpolate(lap_response_nhwc.permute(0, 3, 1, 2), size=(target_h, target_w), mode='bilinear', align_corners=False)
            lap_response_nhwc = lap_response_resized_nchw.permute(0, 2, 3, 1) # Back to NHWC

        # Noise is high where flatness is high AND Laplacian response is high
        noise_map_raw = M_flat * lap_response_nhwc

        # Smooth the noise map
        if blur_sigma > 0:
            # Ensure the guide tensor is passed correctly for smoothing
            noise_map_raw = self._smooth_map_torch(noise_map_raw, guide_nhwc, params, key_prefix="noise_map")
        return self._normalize_map_torch(noise_map_raw)

    def _generate_detail_map_stddev(self, img_nhwc_gray: torch.Tensor, window_size: int, blur_sigma: float, params: dict, guide_nhwc: torch.Tensor) -> torch.Tensor:
        self._assert_tensor_nhwc(img_nhwc_gray, "Detail map input gray", 1)
        assert window_size % 2 == 1, "Window size for detail map must be odd."

        img_nchw_gray = img_nhwc_gray.permute(0,3,1,2).contiguous()
        pad = window_size // 2

        try:
            # Calculate local standard deviation using avg_pool2d for mean(x) and mean(x^2)
            mean_x = F.avg_pool2d(img_nchw_gray, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)
            mean_x2 = F.avg_pool2d(img_nchw_gray**2, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)
            var_x = torch.relu(mean_x2 - mean_x**2) # Ensure non-negative variance due to precision
            std_dev_nchw = torch.sqrt(var_x + 1e-6) # Add epsilon for stability
            detail_map_nhwc = std_dev_nchw.permute(0,2,3,1)
        except Exception as e:
            print(f"Warning: Failed to compute std dev map (window {window_size}): {e}. Returning zero map.")
            return torch.zeros_like(img_nhwc_gray)

        # Smooth the detail map
        if blur_sigma > 0:
            detail_map_nhwc = self._smooth_map_torch(detail_map_nhwc, guide_nhwc, params, key_prefix="detail_map")
        return self._normalize_map_torch(detail_map_nhwc)

    def _generate_usm_halo_potential_map(self, I_original_nhwc_gray: torch.Tensor, I_blurred_usm_nhwc_gray: torch.Tensor,
                                         M_flat: torch.Tensor, M_edge_strength: torch.Tensor, blur_sigma: float, params: dict, guide_nhwc: torch.Tensor) -> torch.Tensor:
        self._assert_tensor_nhwc(I_original_nhwc_gray, "Halo map input orig_gray", 1)
        self._assert_tensor_nhwc(I_blurred_usm_nhwc_gray, "Halo map input blur_gray", 1)
        self._assert_tensor_nhwc(M_flat, "Halo map input M_flat", 1)
        self._assert_tensor_nhwc(M_edge_strength, "Halo map input M_edge", 1)
        
        # 1. Get USM detail signal (approx)
        diff_orig_blur = torch.abs(I_original_nhwc_gray - I_blurred_usm_nhwc_gray)

        # 2. Calculate Laplacian of the detail signal
        diff_nchw = diff_orig_blur.permute(0,3,1,2).contiguous() # BHWC -> BCHW
        # Ensure laplacian kernel has correct shape [out, in, kH, kW]
        lap_k = self.laplacian_kernel.to(diff_nchw.device) # Shape [1, 1, 3, 3]

        # --- Manual Padding for Laplacian ---
        padding_amount = 1 # For 3x3 kernel
        padded_diff_nchw = F.pad(diff_nchw, (padding_amount, padding_amount, padding_amount, padding_amount), mode='reflect')

        # Apply convolution with padding=0 on the manually padded image
        lap_of_detail_nchw = torch.abs(F.conv2d(padded_diff_nchw, lap_k, padding=0))
        lap_of_detail_nhwc = lap_of_detail_nchw.permute(0,2,3,1) # BCHW -> BHWC
        normalized_lap_of_detail = self._normalize_map_torch(lap_of_detail_nhwc)

        # 3. Identify flat areas near edges (potential halo zones)
        m_flat_nchw = M_flat.permute(0,3,1,2).contiguous()
        pool_kernel_size = 3 # Simple 3x3 max pooling for dilation
        h, w = m_flat_nchw.shape[-2], m_flat_nchw.shape[-1]
        if h < pool_kernel_size or w < pool_kernel_size: # Avoid error on small images/tiles
            dilated_m_flat_nchw = m_flat_nchw
        else:
            # Dilation using max pooling requires padding=1 for 3x3 kernel to maintain size conceptually
            dilated_m_flat_nchw = F.max_pool2d(m_flat_nchw, kernel_size=pool_kernel_size, stride=1, padding=1)
        dilated_m_flat_nhwc = dilated_m_flat_nchw.permute(0,2,3,1) # Back to NHWC
        # Combine dilated flatness with edge strength map
        m_flat_near_edge = dilated_m_flat_nhwc * M_edge_strength

        # Ensure shapes match before combining (less likely needed now, but good safeguard)
        if m_flat_near_edge.shape[1:3] != normalized_lap_of_detail.shape[1:3]:
               print(f"Warning: Resizing lap_of_detail in _generate_usm_halo_potential_map (Post-Conv). FlatNearEdge: {m_flat_near_edge.shape[1:3]}, LapDetail: {normalized_lap_of_detail.shape[1:3]}.")
               target_h, target_w = m_flat_near_edge.shape[1:3]
               # Permute back to NCHW for interpolate
               lap_of_detail_resized_nchw = F.interpolate(normalized_lap_of_detail.permute(0, 3, 1, 2), size=(target_h, target_w), mode='bilinear', align_corners=False)
               normalized_lap_of_detail = lap_of_detail_resized_nchw.permute(0, 2, 3, 1) # Back to NHWC

        # 4. Combine: High halo potential where Laplacian of detail is high AND it's a flat area near an edge
        halo_map_raw = m_flat_near_edge * normalized_lap_of_detail

        # 5. Smooth the final halo map
        if blur_sigma > 0:
            halo_map_raw = self._smooth_map_torch(halo_map_raw, guide_nhwc, params, key_prefix="usm_halo_map")
        return self._normalize_map_torch(halo_map_raw)

    def _sac_generate_all_guidance_maps(self, I_original_nhwc: torch.Tensor, I_blurred_usm_nhwc_gray_approx: torch.Tensor | None, params: dict) -> dict:
        maps = {}
        # Use original image (pre-denoise) as the base guide for smoothing SAC maps for structural fidelity.
        guide_for_sac_maps = I_original_nhwc
        params['__original_guide_for_masks__'] = guide_for_sac_maps

        # Generate maps based on the grayscale version of the original image
        I_original_gray = self._to_grayscale_torch(I_original_nhwc)

        # --- Always generate Flatness and Edge Strength for SAC ---
        maps['M_flat'] = self._generate_flatness_map(I_original_gray, DefaultSettings.SAC_FLATNESS_GUIDE_BLUR_SIGMA, params, guide_for_sac_maps)
        maps['M_edge_strength'] = 1.0 - maps['M_flat'] # Edge strength is inverse of flatness

        # --- Generate other maps ---
        advanced_sac_active = params.get('stage_aware_control', 0.0) > 1e-6 # Check if advanced SAC is ON

        maps['M_noise'] = torch.zeros_like(maps['M_flat'])
        maps['M_detail'] = torch.zeros_like(maps['M_flat'])
        maps['M_lbp'] = torch.zeros_like(maps['M_flat'])
        maps['M_halo_usm'] = torch.zeros_like(maps['M_flat'])

        if advanced_sac_active:
            # Noise Map
            maps['M_noise'] = self._generate_noise_map(I_original_gray, maps['M_flat'], DefaultSettings.SAC_NOISE_MAP_BLUR_SIGMA, params, guide_for_sac_maps)

            # Detail Map
            maps['M_detail'] = self._generate_detail_map_stddev(
                I_original_gray,
                DefaultSettings.SAC_DETAIL_MAP_WINDOW_SIZE,
                DefaultSettings.SAC_DETAIL_MAP_BLUR_SIGMA, params, guide_for_sac_maps
            )

            # LBP Map (Texture/Complex Pattern detection)
            maps['M_lbp'] = self._torch_lbp(I_original_gray, DefaultSettings.LBP_DEFAULT_RADIUS, DefaultSettings.LBP_DEFAULT_NEIGHBORS)
            # Smooth LBP map slightly
            maps['M_lbp'] = self._smooth_map_torch(maps['M_lbp'], guide_for_sac_maps, params, key_prefix="lbp_map")

            # USM Halo Potential Map (only if USM stage is active)
            if params.get('usm_sharpen', 0.0) > 0:
                if I_blurred_usm_nhwc_gray_approx is not None:
                     maps['M_halo_usm'] = self._generate_usm_halo_potential_map(
                        I_original_gray, I_blurred_usm_nhwc_gray_approx,
                        maps['M_flat'], maps['M_edge_strength'],
                        DefaultSettings.SAC_USM_HALO_MAP_BLUR_SIGMA, params, guide_for_sac_maps
                    )
                else:
                    # This shouldn't happen if USM is active, but good to have a fallback warning
                    print("Warning: 🐐 Advanced_Sharpen - Cannot generate USM halo map for SAC, blurred approximation unavailable.")

        # Clean up temporary guide key from params
        if '__original_guide_for_masks__' in params:
            del params['__original_guide_for_masks__']

        return maps

    def _sac_calculate_moderator(self, M_change_stage: torch.Tensor, sac_control_strength: float,
                                 guidance_maps: dict, params: dict, stage_name: str) -> torch.Tensor:
        # sac_control_strength is the raw 0-1 stage_aware_control input value

        # Normalize the change map to make thresholding more consistent
        change_map_norm = self._normalize_map_torch(M_change_stage)
        moderator = torch.zeros_like(change_map_norm) # Initialize moderator to zeros

        # --- Apply All Moderation only if sac_control_strength > 0 ---
        if sac_control_strength > 1e-6: # Use epsilon for float comparison
            sac_intensity = sac_control_strength * DefaultSettings.SAC_MODERATION_SCALAR

            # Initial moderator based on thresholded change *and* strength
            change_moderator_strength = (change_map_norm > DefaultSettings.SAC_CHANGE_THRESH).float() * sac_intensity
            moderator = torch.max(moderator, change_moderator_strength) # Start with change-based moderation

            # --- Flatness/Edge influence (for non-edge stages) ---
            if stage_name in ['gf_sharpen', 'usm_sharpen', 'fft_sharpen']:
                # Higher flatness -> potentially more moderation.
                flatness_influence_scaled = guidance_maps.get('M_flat', 0.0) * sac_intensity
                # Combine with a small thresholded change detector to avoid moderating *everything*
                change_detected_for_flatness = (change_map_norm > DefaultSettings.SAC_CHANGE_THRESH * 0.5).float()
                moderator = torch.max(moderator, flatness_influence_scaled * change_detected_for_flatness)

            # Noise: Increase moderation in noisy areas
            noise_influence = guidance_maps.get('M_noise', 0.0) * sac_intensity
            moderator = torch.max(moderator, noise_influence)

            # Detail: Decrease moderation (allow more sharpening) in detailed areas
            detail_factor = guidance_maps.get('M_detail', 0.0) * sac_intensity
            moderator = torch.relu(moderator * (1.0 - detail_factor)) # Reduce moderator proportionally to detail

            # LBP: Increase moderation in high LBP areas (complex textures/patterns)
            lbp_influence = guidance_maps.get('M_lbp', 0.0) * sac_intensity
            moderator = torch.max(moderator, lbp_influence)

            # Halo (USM only): Increase moderation strongly where halo potential is high
            if stage_name == 'usm_sharpen': # Match the stage name used in the map key
                halo_factor = 1.5 # Give halo reduction a bit more weight
                halo_influence = guidance_maps.get('M_halo_usm', 0.0) * sac_intensity * halo_factor
                moderator = torch.max(moderator, halo_influence)

        return torch.clamp(moderator, 0.0, 1.0)

    def _stage_adaptive_artifact_control(self, stage_outputs: dict, guidance_maps: dict, params: dict, luma_active: bool) -> torch.Tensor:
        # Start with the final output of the sharpening pipeline (before SAC)
        I_current_moderated = stage_outputs.get('final_pipeline_unsafe').clone()

        # Assert starting state based on luma_active flag
        if luma_active:
            self._assert_tensor_nhwc(I_current_moderated, "SAC starting image (Luma)", 1)
        else:
            self._assert_tensor_nhwc(I_current_moderated, "SAC starting image (RGB/Grayscale)", stage_outputs['original'].shape[3])

        stage_to_input_param_map = {
            'edge': 'edge_sharpen',
            'fft': 'fft_sharpen',
            'usm': 'usm_sharpen',
            'gf': 'gf_sharpen'
        }

        rollback_stages_ordered = []
        if params.get(stage_to_input_param_map['edge'], 0.0) > 0.0: rollback_stages_ordered.append('edge')
        if params.get(stage_to_input_param_map['fft'], 0.0) > 0.0: rollback_stages_ordered.append('fft')
        if params.get(stage_to_input_param_map['usm'], 0.0) > 0.0: rollback_stages_ordered.append('usm')
        if params.get(stage_to_input_param_map['gf'], 0.0) > 0.0: rollback_stages_ordered.append('gf')

        # Get the SAC intensity control from the 'stage_aware_control' parameter
        current_sac_strength = params.get('stage_aware_control', 0.0)
        sac_stage_retention = {} # Track how much effect was kept per stage

        for stage_name in rollback_stages_ordered:
            target_rollback_img_candidate = None
            target_key = "N/A"
            if stage_name == 'edge':
                target_rollback_img_candidate = stage_outputs.get('fft', stage_outputs.get('usm', stage_outputs.get('gf', stage_outputs.get('preprocessed'))))
                target_key = "fft/usm/gf/preprocessed"
            elif stage_name == 'fft':
                target_rollback_img_candidate = stage_outputs.get('usm', stage_outputs.get('gf', stage_outputs.get('preprocessed')))
                target_key = "usm/gf/preprocessed"
            elif stage_name == 'usm':
                target_rollback_img_candidate = stage_outputs.get('gf', stage_outputs.get('preprocessed'))
                target_key = "gf/preprocessed"
            elif stage_name == 'gf':
                target_rollback_img_candidate = stage_outputs.get('preprocessed')
                target_key = "preprocessed"

            if target_rollback_img_candidate is None:
                if target_key == "preprocessed":
                    target_rollback_img_candidate = stage_outputs.get('original')
                    if target_rollback_img_candidate: target_key = "original (fallback)"
                if target_rollback_img_candidate is None:
                    print(f"Warning: 🐐 Advanced_Sharpen SAC - Could not find target rollback image for stage {stage_name}. Skipping moderation for this stage.")
                    sac_stage_retention[stage_name] = 1.0
                    continue

            target_rollback_img = target_rollback_img_candidate.clone()

            if luma_active and target_rollback_img.shape[3] != 1:
                target_rollback_img = self._to_grayscale_torch(target_rollback_img)
            elif not luma_active and target_rollback_img.shape[3] != I_current_moderated.shape[3]:
                 if I_current_moderated.shape[3] != 1 and target_rollback_img.shape[3] == 1:
                     target_rollback_img = target_rollback_img.repeat(1,1,1, I_current_moderated.shape[3])
                 elif target_rollback_img.shape[3] != I_current_moderated.shape[3]:
                      print(f"Warning: 🐐 Advanced_Sharpen SAC - Channel mismatch between current ({I_current_moderated.shape[3]}) and target ({target_rollback_img.shape[3]}) for stage {stage_name}. Attempting grayscale conversion.")
                      target_rollback_img = self._to_grayscale_torch(target_rollback_img)
                      if I_current_moderated.shape[3] != 1:
                          target_rollback_img = target_rollback_img.repeat(1,1,1, I_current_moderated.shape[3])

            if target_rollback_img.shape != I_current_moderated.shape:
                print(f"Error: 🐐 Advanced_Sharpen SAC - Target ({target_rollback_img.shape}) and Current ({I_current_moderated.shape}) shape mismatch for stage {stage_name} after adjustment. Skipping moderation.")
                sac_stage_retention[stage_name] = 1.0
                continue

            M_change_stage_raw = torch.abs(I_current_moderated - target_rollback_img)
            input_param_name = stage_to_input_param_map.get(stage_name, stage_name)
            final_moderator = self._sac_calculate_moderator(
                M_change_stage_raw, current_sac_strength, # Pass the current SAC strength
                guidance_maps, params, input_param_name
            )

            avg_retention = torch.mean(1.0 - final_moderator).item()
            sac_stage_retention[stage_name] = avg_retention

            if final_moderator.shape[3] != I_current_moderated.shape[3]:
                if I_current_moderated.shape[3] == 1 and final_moderator.shape[3] > 1:
                    final_moderator = torch.mean(final_moderator, dim=3, keepdim=True)
                elif I_current_moderated.shape[3] > 1 and final_moderator.shape[3] == 1:
                    final_moderator = final_moderator.repeat(1,1,1, I_current_moderated.shape[3])
                else:
                     final_moderator = torch.mean(final_moderator, dim=3, keepdim=True).repeat(1,1,1, I_current_moderated.shape[3])

            assert final_moderator.shape == I_current_moderated.shape, f"Moderator shape {final_moderator.shape} != Image shape {I_current_moderated.shape} in stage {stage_name} after adjustment"
            I_current_moderated = I_current_moderated * (1.0 - final_moderator) + target_rollback_img * final_moderator

        # --- Print SAC contribution analysis ---
        print("\n--- SAC Stage Retention Analysis ---")
        pipeline_order = ['gf', 'usm', 'fft', 'edge']
        active_stages_count = 0

        if current_sac_strength > 1e-6:
             print(f"  (Stage Aware Control Strength: {current_sac_strength:.2f}, Scaled Intensity: ~{current_sac_strength * DefaultSettings.SAC_MODERATION_SCALAR:.2f})")
             print("  (Retention = Avg % of stage effect kept after SAC adjustment)")
        else:
             print(f"  (Stage Aware Control Strength: {current_sac_strength:.2f} - SAC moderation is OFF)")

        for stage in pipeline_order:
            input_param = stage_to_input_param_map.get(stage)
            is_active = input_param and params.get(input_param, 0.0) > 0.0
            if is_active:
                active_stages_count += 1
                if current_sac_strength > 1e-6: # Only print retention if SAC was active
                    retention = sac_stage_retention.get(stage)
                    if retention is not None:
                        print(f"  - {stage:<15}: {retention * 100.0:>6.2f}% retention")
                    else:
                         print(f"  - {stage:<15}: --- (Error calculating retention)")


        if active_stages_count == 0:
             print("  No sharpening stages were active.")
        elif current_sac_strength > 1e-6 and not sac_stage_retention: # Check if retention dict is empty when SAC was meant to be active
             print("  Error: SAC was active but failed to calculate retention for any stage.")

        print("------------------------------------\n")

        if luma_active:
            self._assert_tensor_nhwc(I_current_moderated, "SAC final output (Luma)", 1)
        else:
            self._assert_tensor_nhwc(I_current_moderated, "SAC final output (RGB/Grayscale)", stage_outputs['original'].shape[3])

        return torch.clamp(I_current_moderated, 0.0, 1.0)

    # --- Postprocessing: Subtle Artifact Reduction ---
    def _subtle_artifact_reduction(self, img_nhwc: torch.Tensor, strength: float, params: dict) -> torch.Tensor:
        # Note: strength is already scaled by ARTIFACT_REDUCTION_SCALAR
        if strength <= 1e-6: return img_nhwc
        self._assert_tensor_nhwc(img_nhwc, "Artifact reduction input")

        radius = 1 # Small radius for subtle effect
        eps = (DefaultSettings.GUIDED_FILTER_DEFAULT_EPS_DENOISE / 2.0)

        # Use the image itself as the guide
        guide_img = img_nhwc
        guide_is_gray = guide_img.shape[3] == 1

        smoothed_img = self._guided_filter_torch(img_nhwc, guide_img, radius, eps, guide_is_gray=guide_is_gray)

        # Blend based on strength
        output_img = img_nhwc * (1.0 - strength) + smoothed_img * strength
        return torch.clamp(output_img, 0.0, 1.0)

    # --- Main Pipeline and Execution Logic ---
    def _apply_sharpening_pipeline(self, current_image_nhwc: torch.Tensor, params: dict) -> tuple[dict, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        """Applies the full sharpening pipeline (Denoise -> Stages -> SAC -> Recombine -> Artifact Reduction)."""
        stage_outputs = {'original': current_image_nhwc.clone()}
        processed_image = current_image_nhwc.clone()
        I_blurred_usm_nhwc_gray_approx = None # For SAC halo map generation
        original_colors_cbcr = None # For Luma mode recombination

        # === 1. Preprocessing: Selective Denoise ===
        denoise_strength_param = params.get('preprocess_denoise', 0.0)
        if denoise_strength_param > 0.0:
            scaled_denoise_strength = denoise_strength_param * DefaultSettings.PREPROCESS_DENOISE_SCALAR
            processed_image = self._selective_denoise(processed_image, scaled_denoise_strength, params)
        stage_outputs['preprocessed'] = processed_image.clone()

        # === 2. Handle Luma Sharpening Split ===
        luma_sharpening_active = params.get('luma_sharpen', True) and processed_image.shape[3] == 3
        # Store the guide for mask smoothing (use preprocessed image for structure)
        params['__original_guide_for_masks__'] = stage_outputs['preprocessed'].clone()

        if luma_sharpening_active:
            y, cb, cr = self._rgb_to_ycbcr(processed_image)
            original_colors_cbcr = (cb.clone(), cr.clone())
            processed_image = y # Process only Y channel from now on
            self._assert_tensor_nhwc(processed_image, "Luma channel after split", 1)

        # === Sharpening Stages (Order: GF -> USM -> FFT -> EdgeRefine) ===

        # === 3. Stage: Guided Filter Sharpen ===
        if params.get('gf_sharpen', 0.0) > 0.0:
            processed_image = self._guided_filter_sharpen(processed_image, params)
        stage_outputs['gf'] = processed_image.clone()

        # === 4. Stage: Unsharp Mask ===
        if params.get('usm_sharpen', 0.0) > 0.0:
            # --- Pre-calculate USM blur approximation IF needed for SAC halo map ---
            source_for_usm_blur_approx = processed_image # Current state (post-GF)
            # Only calculate if advanced SAC is active and USM halo map isn't already generated
            if I_blurred_usm_nhwc_gray_approx is None and params.get('stage_aware_control', 0.0) > 1e-6: # Check new control
                 temp_blurred = None
                 usm_blur_type = 'gaussian' # Default
                 usm_sp_s = 1.5
                 usm_co_s = 0.1
                 use_bilateral = cv2 is not None

                 if use_bilateral: temp_blurred = self._apply_cv2_bilateral_blur(source_for_usm_blur_approx, usm_sp_s, usm_co_s)
                 elif scipy is not None: temp_blurred = self._apply_scipy_gaussian_blur(source_for_usm_blur_approx, usm_sp_s)
                 else: temp_blurred = self._torch_gaussian_blur(source_for_usm_blur_approx, usm_sp_s)

                 if temp_blurred is not None:
                      # Ensure the approximation is grayscale for the halo map function
                      if temp_blurred.shape[3] == 1: I_blurred_usm_nhwc_gray_approx = temp_blurred.clone()
                      else: I_blurred_usm_nhwc_gray_approx = self._to_grayscale_torch(temp_blurred).clone()
            # --- Apply USM Sharpening ---
            processed_image = self._unsharp_mask(processed_image, params)
        stage_outputs['usm'] = processed_image.clone()

        # === 5. Stage: FFT Sharpen ===
        if params.get('fft_sharpen', 0.0) > 0.0:
            processed_image = self._fft_sharpen(processed_image, params)
        stage_outputs['fft'] = processed_image.clone()

        # === 6. Stage: Edge Refine ===
        if params.get('edge_sharpen', 0.0) > 0.0:
            # Mask is always generated from Luma of the *preprocessed* image for consistency
            mask_source_img_luma = self._to_grayscale_torch(stage_outputs['preprocessed'])
            processed_image = self._edge_refine(processed_image, mask_source_img_luma, params)
        stage_outputs['edge'] = processed_image.clone()

        # === End Sharpening Stages ===

        # Store final pipeline output (pre-SAC, pre-recombine)
        stage_outputs['final_pipeline_unsafe'] = processed_image.clone()

        # Clean up temporary guide key
        if '__original_guide_for_masks__' in params:
            del params['__original_guide_for_masks__']

        # Return all intermediate results, the USM blur approx, and original CbCr if split
        return stage_outputs, I_blurred_usm_nhwc_gray_approx, original_colors_cbcr

    def _process_full_image(self, original_image_nhwc: torch.Tensor, params: dict) -> torch.Tensor:
        """Processes the entire image using the sharpening pipeline."""

        # === 1. Run the Sharpening Pipeline (Denoise -> Stages) ===
        stage_outputs, I_blurred_usm_approx, original_cbcr = self._apply_sharpening_pipeline(original_image_nhwc, params)

        processed_image_before_sac = stage_outputs['final_pipeline_unsafe']
        luma_sharpening_was_active = original_cbcr is not None

        # === 2. Apply Stage Aware Control (SAC) ===
        guidance_map_source_img = stage_outputs.get('original')
        params['__original_guide_for_masks__'] = guidance_map_source_img.clone()
        guidance_maps = self._sac_generate_all_guidance_maps(guidance_map_source_img, I_blurred_usm_approx, params)
        if '__original_guide_for_masks__' in params: # Clean up temp key
            del params['__original_guide_for_masks__']

        # Apply SAC moderation.
        final_processed_image_after_sac = self._stage_adaptive_artifact_control(stage_outputs, guidance_maps, params, luma_sharpening_was_active)

        # === 3. Recombine Colors (If Luma Sharpening was active) ===
        final_image_rgb = None
        if luma_sharpening_was_active:
            self._assert_tensor_nhwc(final_processed_image_after_sac, "Final Y before recombine", 1)
            cb_orig, cr_orig = original_cbcr
            if cb_orig.shape[1:3] != final_processed_image_after_sac.shape[1:3]:
                 target_size = final_processed_image_after_sac.shape[1:3]
                 print(f"Warning: Resizing Cb/Cr channels to match processed Luma size {target_size} before recombination.")
                 cb_orig = F.interpolate(cb_orig.permute(0,3,1,2), size=target_size, mode='bilinear', align_corners=False).permute(0,2,3,1)
                 cr_orig = F.interpolate(cr_orig.permute(0,3,1,2), size=target_size, mode='bilinear', align_corners=False).permute(0,2,3,1)
            final_image_rgb = self._ycbcr_to_rgb(final_processed_image_after_sac, cb_orig, cr_orig)
        else:
            final_image_rgb = final_processed_image_after_sac
            assert final_image_rgb.shape[3] == original_image_nhwc.shape[3], \
                   f"Channel mismatch: Expected {original_image_nhwc.shape[3]}, got {final_image_rgb.shape[3]} when Luma sharpening was off."

        # === 4. Postprocessing: Subtle Artifact Reduction ===
        artifact_reduction_param = params.get('postprocess_denoise', 0.5)
        if artifact_reduction_param > 0.0:
            scaled_ar_strength = artifact_reduction_param * DefaultSettings.POSTPROCESS_DENOISE_SCALAR
            final_image_rgb = self._subtle_artifact_reduction(final_image_rgb, scaled_ar_strength, params)

        final_image_rgb = torch.clamp(final_image_rgb, 0.0, 1.0)
        assert final_image_rgb.shape == original_image_nhwc.shape, \
            f"Final output shape {final_image_rgb.shape} doesn't match original input shape {original_image_nhwc.shape}"

        return final_image_rgb

    def _process_tiled(self, image_nhwc: torch.Tensor, params: dict) -> torch.Tensor:
        """Processes the image in tiles with sine window blending."""
        b, original_height, original_width, channels = image_nhwc.shape
        tile_s_param = params.get('tile_size', 1024)

        # Ensure tile size is not larger than image dimensions
        effective_tile_h = min(tile_s_param, original_height)
        effective_tile_w = min(tile_s_param, original_width)

        # Calculate overlap size (ensure it's reasonable and not too large)
        overlap_h_ideal = max(DefaultSettings.TILING_MIN_OVERLAP_PX, int(effective_tile_h * DefaultSettings.TILING_OVERLAP_PERCENT_OF_TILE))
        overlap_w_ideal = max(DefaultSettings.TILING_MIN_OVERLAP_PX, int(effective_tile_w * DefaultSettings.TILING_OVERLAP_PERCENT_OF_TILE))
        # Overlap cannot be larger than tile size minus 1 pixel for step calculation
        overlap_h = max(0, min(overlap_h_ideal, effective_tile_h - 1 if effective_tile_h > 1 else 0))
        overlap_w = max(0, min(overlap_w_ideal, effective_tile_w - 1 if effective_tile_w > 1 else 0))

        # Calculate step size (how much to move window each time)
        step_h = max(1, effective_tile_h - overlap_h)
        step_w = max(1, effective_tile_w - overlap_w)

        # Initialize accumulators for blending
        final_output_accumulator = torch.zeros_like(image_nhwc)
        weight_sum_accumulator = torch.zeros_like(image_nhwc)

        # Calculate number of tiles needed
        num_tiles_y = math.ceil(original_height / step_h) if step_h > 0 else 1
        num_tiles_x = math.ceil(original_width / step_w) if step_w > 0 else 1

        print(f"🐐 Advanced_Sharpen: Processing in {num_tiles_y}x{num_tiles_x} tiles ({effective_tile_w}x{effective_tile_h} px) with overlap ({overlap_w}x{overlap_h} px).")
        pbar = comfy.utils.ProgressBar(num_tiles_y * num_tiles_x)

        for y_idx in range(num_tiles_y):
            for x_idx in range(num_tiles_x):
                # Calculate coordinates for extracting the current tile
                y_start_orig = y_idx * step_h
                x_start_orig = x_idx * step_w
                # Ensure extraction does not go beyond image bounds
                y_extract_end = min(y_start_orig + effective_tile_h, original_height)
                x_extract_end = min(x_start_orig + effective_tile_w, original_width)
                # Adjust start if needed for last tile to ensure full tile size (important for some filters)
                y_extract_start = max(0, y_extract_end - effective_tile_h)
                x_extract_start = max(0, x_extract_end - effective_tile_w)

                # Extract the tile
                current_tile_for_processing = image_nhwc[:, y_extract_start:y_extract_end, x_extract_start:x_extract_end, :]
                actual_tile_h, actual_tile_w = current_tile_for_processing.shape[1:3]

                # --- Process the extracted tile ---
                # _process_full_image handles the entire pipeline for this tile
                processed_tile = self._process_full_image(current_tile_for_processing.clone(), params)

                # --- Create sine window blending mask for this tile ---
                # Sine window provides smooth transitions in overlapping regions
                blend_curve_h = torch.sin(torch.linspace(0, math.pi, actual_tile_h, device=image_nhwc.device))
                blend_curve_w = torch.sin(torch.linspace(0, math.pi, actual_tile_w, device=image_nhwc.device))
                current_blend_mask = torch.outer(blend_curve_h, blend_curve_w).unsqueeze(0).unsqueeze(-1) # Shape (1, H, W, 1)
                # Repeat mask for all channels
                if channels > 1:
                    current_blend_mask = current_blend_mask.repeat(1,1,1,channels)

                # Add processed tile weighted by the blend mask to the accumulator
                final_output_accumulator[:, y_extract_start:y_extract_end, x_extract_start:x_extract_end, :] += \
                    processed_tile * current_blend_mask
                # Add the blend mask itself to the weight accumulator
                weight_sum_accumulator[:, y_extract_start:y_extract_end, x_extract_start:x_extract_end, :] += \
                    current_blend_mask

                pbar.update(1) # Update progress bar

        # Normalize the accumulated output by the sum of weights
        epsilon = 1e-7 # Avoid division by zero
        final_output_normalized = final_output_accumulator / (weight_sum_accumulator + epsilon)

        # Handle potential edge cases where weights might be zero (e.g., image smaller than tile)
        final_output_normalized = torch.where(weight_sum_accumulator > epsilon/2, final_output_normalized, image_nhwc)

        return torch.clamp(final_output_normalized, 0.0, 1.0)

    def execute_sharpening(self, image: torch.Tensor, **kwargs):
        """Main execution function called by ComfyUI."""
        print(f"🐐 Advanced_Sharpen: Processing...")
        if image is None:
            print("Error: 🐐 Advanced_Sharpen - Input image is missing.")
            # Return a dummy tensor if no image is provided
            dummy_h, dummy_w = 64, 64
            return (torch.zeros((1, dummy_h, dummy_w, 3), dtype=torch.float32, device=self.device).cpu(), dummy_w, dummy_h)

        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            print(f"Error: 🐐 Advanced_Sharpen - Input image must be a 4D Tensor (BHWC), got {type(image)} with ndim={image.ndim if hasattr(image, 'ndim') else 'N/A'}.")
            # Attempt to return original image data if possible, else fallback
            h = image.shape[1] if hasattr(image,'ndim') and image.ndim==4 else 64
            w = image.shape[2] if hasattr(image,'ndim') and image.ndim==4 else 64
            try:
                return (image.cpu(), w, h) # Try returning original on CPU
            except:
                return (torch.zeros((1, h, w, 3), dtype=torch.float32, device=self.device).cpu(), w, h) # Fallback

        params = kwargs # Collect all keyword arguments as parameters
        original_image_nhwc = image.to(self.device) # Ensure image is on the correct device
        b, height, width, c = original_image_nhwc.shape

        final_image_processed = None
        try:
            # Determine if tiling is needed based on size and parameter
            tile_size = params.get('tile_size', 1024)
            use_tiled = params.get('tiled_sharpen', False) and \
                        (height > tile_size or width > tile_size)

            if use_tiled:
                final_image_processed = self._process_tiled(original_image_nhwc.clone(), params)
            else:
                # if params.get('tiled_sharpen', False): print("Info: Advanced_Sharpen - Tiling enabled but image smaller than tile size. Processing full image.")
                final_image_processed = self._process_full_image(original_image_nhwc.clone(), params)

            # --- Final Blend with Overall Strength ---
            # Apply the main strength control as a blend between original and fully processed
            overall_strength_param = params.get('overall_strength', 0.5)
            # Scale the 0-1 input strength
            scaled_overall_strength = overall_strength_param * DefaultSettings.OVERALL_STRENGTH_SCALAR

            # Use linear interpolation for blending
            final_image = torch.lerp(original_image_nhwc, final_image_processed, scaled_overall_strength)
            final_image = torch.clamp(final_image, 0.0, 1.0)

        except Exception as e:
            print(f"\n!!! Error: 🐐 Advanced_Sharpen - Exception during processing !!!")
            import traceback
            print(traceback.format_exc())
            print(f"!!! Falling back to original image. !!!\n")
            final_image = original_image_nhwc # Fallback to original on error

        # Final assertions and return
        self._assert_tensor_nhwc(final_image, "Final output image", c)
        assert final_image.shape[1:3] == (height, width), f"Final output shape error. Expected {(height, width)}, got {final_image.shape[1:3]}"

        # Move final result to CPU before returning to ComfyUI
        return (final_image.cpu(), width, height)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {"Advanced_Sharpen": Advanced_Sharpen}
NODE_DISPLAY_NAME_MAPPINGS = {"Advanced_Sharpen": "🐐 Advanced Sharpen"}
