import torch # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
from sklearn.cluster import KMeans # type: ignore

# --- Constants ---
# Step 0: Initial Analysis
BW_THRESHOLD_STD_DEV_A_B = 2.5
DARK_L_THRESHOLD = 30.0
BRIGHT_L_THRESHOLD = 75.0
# Step 1: White Balance
SHADES_OF_GRAY_POWER_P = 6
SHADES_OF_GRAY_EPSILON = 1e-6
GP_SATURATION_THRESHOLD = 0.10  # Max saturation (e.g., (max(RGB)-min(RGB))/max(RGB)) for a pixel to be gray candidate
GP_LUMINANCE_MIN_THRESHOLD = 0.05 # Min luminance (0-1 range)
GP_LUMINANCE_MAX_THRESHOLD = 0.95 # Max luminance (0-1 range) to avoid clipped highlights
GP_MIN_GRAY_PIXEL_FRACTION = 0.001 # Minimum fraction of total pixels that must be "gray" to trust the estimate
GP_PERCENTILE_SELECTION = 20 # Use top Nth percentile of gray candidates (e.g., 10th for lowest diffs)
M_BRADFORD = torch.tensor([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296]
], dtype=torch.float32)
M_BRADFORD_INV = torch.linalg.inv(M_BRADFORD)
M_SRGB_TO_XYZ = torch.tensor([ # sRGB D65 primaries to XYZ D65
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=torch.float32)
M_XYZ_TO_SRGB = torch.linalg.inv(M_SRGB_TO_XYZ)
XYZ_D65_TARGET = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32) # Target illuminant
SOG_FALLBACK_EFFECT_SCALE = 0.50 # Reduce SoG effect by this factor when it's a fallback (0.0 to 1.0)
# Step 2: Statistical Transfer - Adaptive Luminance
NIGHT_LOW_KEY_L_MEAN_INPUT_FACTOR = 0.6
NIGHT_LOW_KEY_L_MEAN_TARGET_FACTOR = 0.4
NIGHT_LOW_KEY_PREDEFINED_L_TARGET_SCALE = 0.4
NIGHT_LOW_KEY_L_STD_SCALE = 0.9
NORMAL_L_MEAN_TARGET_ABSOLUTE = 55.0 # A general target for "normal" L* mean
NORMAL_L_MEAN_INPUT_FACTOR = 0.7
NORMAL_L_MEAN_TARGET_FACTOR = 0.3
NORMAL_L_STD_SCALE = 0.95 # Slightly compress/expand towards base for normal
BRIGHT_HIGH_KEY_L_MEAN_INPUT_FACTOR = 0.6
BRIGHT_HIGH_KEY_L_MEAN_TARGET_FACTOR = 0.4
BRIGHT_HIGH_KEY_PREDEFINED_L_TARGET_SCALE = 1.45
BRIGHT_HIGH_KEY_L_STD_SCALE = 0.85
PRESERVE_L_STD_SCALE = 0.95 # Slight compression for "Preserve Original" L_std
# Step 2: Statistical Transfer - Color Styles
VIBRANT_L_STD_SCALE = 1.20 # This can also affect L if not overridden by brightness mode
VIBRANT_A_STD_SCALE = 1.50
VIBRANT_B_STD_SCALE = 1.50
SOFT_L_STD_SCALE = 0.85
SOFT_A_STD_SCALE = 0.65
SOFT_B_STD_SCALE = 0.65
ENHANCED_NATURALISM_L_STD_SCALE = 1.10
ENHANCED_NATURALISM_A_STD_SCALE = 1.25
ENHANCED_NATURALISM_B_STD_SCALE = 1.25
# Step 2: Statistical Transfer - B&W Processing
BW_TARGET_A_MEAN = 0.0
BW_TARGET_A_STD = 1.0
BW_TARGET_B_MEAN = 0.0
BW_TARGET_B_STD = 1.0
# Step 3: Selective Hue/Saturation
SHS_K_MEANS_CLUSTERS = 8
SHS_MIN_CLUSTER_SIZE_FRACTION = 0.01
SHS_CONFIDENT_MATCH_DISTANCE_THRESHOLD_LAB = 30.0
SHS_KMEANS_MAX_ITER = 100
SHS_KMEANS_N_INIT = 'auto'
SHS_CHROMA_SCALE_CLAMP_MIN = 0.2 # Min factor for chroma scaling
SHS_CHROMA_SCALE_CLAMP_MAX = 2.5 # Max factor for chroma scaling
SHS_FALLOFF_CHARACTERISTIC_DISTANCE_LAB = 15.0 # Smaller values = faster falloff. For L*a*b* distances.
SHS_CLUSTER_INFLUENCE_RADIUS_LAB = 20.0 # Smaller = sharper transitions, larger = softer, more blended.
# Step 4: CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
# General
EPSILON = 1e-7

def supports_amp():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return capability[0] >= 7
    return False

class LabColorStats:
    def __init__(self, l_mean, l_std, a_mean, a_std, b_mean, b_std):
        self.mean_L = l_mean
        self.std_L = l_std
        self.mean_a = a_mean
        self.std_a = a_std
        self.mean_b = b_mean
        self.std_b = b_std

    def __repr__(self):
        return (f"LabStats(L_mean={self.mean_L:.2f}, L_std={self.std_L:.2f}, "
                f"a_mean={self.mean_a:.2f}, a_std={self.std_a:.2f}, "
                f"b_mean={self.mean_b:.2f}, b_std={self.std_b:.2f})")

class Advanced_Color_Correction:
    VIRTUAL_COLOR_CARD_LAB = {
        "A1_White": (100.0, 0.0, 0.0), "A2_Christmas_Silver": (89.18, 0.0, 0.0),
        "A3_Brushed_Metal": (80.6, 0.0, 0.0), "A4_Ultimate_Grey": (69.24, 0.0, 0.0),
        "A5_Grey": (53.59, 0.0, 0.0), "A6_Shadow_Mountain": (37.41, 0.0, 0.0),
        "A7_Off_Black": (19.87, 0.0, 0.0), "A8_Black": (0.0, 0.0, 0.0),
        "B1_Peach_Glow": (89.35, 5.93, 27.77), "B2_Necrophilic_Brown": (73.79, 11.28, 41.53),
        "B3_Afternoon_Sky": (79.21, -14.84, -21.28), "B4_Fig_Leaf": (42.23, -18.83, 30.6),
        "B5_Brown_Countdown": (31.56, 10.88, 26.19), "B6_Azul_Petroleo": (28.39, -3.25, -7.96),
        "B7_Petrichor": (76.29, -29.58, -9.13), "B8_Chanterelle": (70.82, 8.52, 68.76),
        "C1_Broadleaf_Forest": (24.48, -28.75, 15.87), "C2_Phoenix_Red": (60.75, 41.6, 32.8),
        "C3_Teal": (48.25, -28.85, -8.48), "C4_Maroon": (25.54, 48.05, 38.06),
        "C5_Hot_Orange": (67.82, 36.91, 56.85), "C6_Fuchsia_Red": (44.44, 55.23, -12.34),
        "C7_Calming_Silver_Lavender": (68.9, 11.91, -14.28), "C8_Pottery_Red": (49.05, 36.04, 16.79),
        "D1_Holy_White": (95.95, -4.19, 12.05), "D2_Scallion": (54.65, -28.22, 49.69),
        "D3_Fantastic_Pink": (83.01, 11.78, 3.82), "D4_Bracing_Blue": (27.37, 9.11, -41.01),
        "D5_Tiki_Monster": (72.09, -23.82, 18.04), "D6_Texas_Ranger_Brown": (43.8, 29.32, 35.64),
        "D7_Light_Steel_Blue": (78.45, -1.28, -15.21), "D8_Amber": (81.03, 10.39, 83.03)
    }

    def __init__(self):
        self.base_target_stats = self._calculate_color_card_stats()
        self._prepare_colored_card_patches() # Prepare here once

    def _calculate_color_card_stats(self) -> LabColorStats:
        lab_values = np.array(list(self.VIRTUAL_COLOR_CARD_LAB.values()))
        mean_l, mean_a, mean_b = np.mean(lab_values, axis=0)
        std_l, std_a, std_b = np.std(lab_values, axis=0)
        return LabColorStats(mean_l, std_l, mean_a, std_a, mean_b, std_b)

    @classmethod
    def INPUT_TYPES(self):
        brightness_options = [
            "auto", "original", "dark",
            "normal", "bright"
        ]
        color_style_options = [
            "natural", "vibrant", "cinematic",
            "enhanced_naturalism", "black_white_tonal"
        ]
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness_mode": (brightness_options, {"default": "auto"}),
                "color_style": (color_style_options, {"default": "natural"}),
                "overall_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength_white_balance": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength_stat_transfer": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength_selective_hue": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength_clahe": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamut_clipping": ("BOOLEAN", {"default": True}),
                "debug_prints": (["Disabled", "Enabled"], {"default": "Disabled"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Postprocessing'

    def _initial_analysis(self, image_tensor_lab_chw: torch.Tensor, 
                          brightness_mode_input: str, color_style_input: str,
                          debug_enabled: bool) -> tuple[float, float, float, float, str, bool]:
        assert image_tensor_lab_chw.ndim == 3 and image_tensor_lab_chw.shape[0] == 3
        
        L_channel, a_channel, b_channel = image_tensor_lab_chw[0], image_tensor_lab_chw[1], image_tensor_lab_chw[2]
        
        initial_mean_l = torch.mean(L_channel).item()
        initial_std_l = torch.std(L_channel).item()
        initial_std_a = torch.std(a_channel).item()
        initial_std_b = torch.std(b_channel).item()
        
        is_bw_candidate_by_chroma = (initial_std_a < BW_THRESHOLD_STD_DEV_A_B) and \
                                    (initial_std_b < BW_THRESHOLD_STD_DEV_A_B)

        detected_brightness_category = "normal"
        if initial_mean_l < DARK_L_THRESHOLD:
            detected_brightness_category = "dark"
        elif initial_mean_l > BRIGHT_L_THRESHOLD:
            detected_brightness_category = "bright"

        is_bw_processing_pipeline = (color_style_input == "black_white_tonal")

        if debug_enabled:
            print(f"[ACC Debug] Initial L* Mean: {initial_mean_l:.2f}, Initial L* Std: {initial_std_l:.2f}")
            print(f"[ACC Debug] Initial a* Std: {initial_std_a:.2f}, Initial b* Std: {initial_std_b:.2f}")
            print(f"[ACC Debug] Is B&W Candidate (low chroma std): {is_bw_candidate_by_chroma}")
            if brightness_mode_input == "auto":
                print(f"[ACC Debug] Auto-Detected Brightness Category for L*: {detected_brightness_category}")
            print(f"[ACC Debug] Effective Brightness Mode for L*: {brightness_mode_input if brightness_mode_input != 'auto' else detected_brightness_category}")
            print(f"[ACC Debug] Selected Color Style: {color_style_input}")
            print(f"[ACC Debug] Pipeline B&W Processing: {is_bw_processing_pipeline}")

        return initial_mean_l, initial_std_l, initial_std_a, initial_std_b, \
               detected_brightness_category, is_bw_processing_pipeline


    def _srgb_to_linear(self, srgb_tensor: torch.Tensor) -> torch.Tensor:
        limit = 0.04045
        return torch.where(srgb_tensor > limit,
                           torch.pow((srgb_tensor + 0.055) / 1.055, 2.4),
                           srgb_tensor / 12.92)

    def _linear_to_srgb(self, linear_tensor: torch.Tensor) -> torch.Tensor:
        limit = 0.0031308
        return torch.where(linear_tensor > limit,
                           1.055 * (torch.pow(linear_tensor, (1.0 / 2.4))) - 0.055,
                           12.92 * linear_tensor)

    def _find_gray_pixels_illuminant(self, image_tensor_rgb_chw_linear: torch.Tensor,
                                     debug_enabled: bool = False) -> torch.Tensor | None:
        """
        Estimates scene illuminant by finding "gray" pixels in LINEAR RGB.
        Returns a (3,) tensor (linear RGB illuminant) or None.
        """
        C, H, W = image_tensor_rgb_chw_linear.shape
        num_pixels = H * W
        device = image_tensor_rgb_chw_linear.device

        if num_pixels == 0: return None

        pixels_rgb_flat = image_tensor_rgb_chw_linear.reshape(C, -1).T # (N, 3)

        # Calculate luminance (approximate for linear RGB)
        # Using sRGB Y weights for simplicity, or could use actual linear Y coeffs
        luminance = 0.2126 * pixels_rgb_flat[:, 0] + \
                    0.7152 * pixels_rgb_flat[:, 1] + \
                    0.0722 * pixels_rgb_flat[:, 2]

        # Calculate max difference between channels as a proxy for saturation
        # For linear data, simple difference is often used.
        rgb_max = torch.max(pixels_rgb_flat, dim=1).values
        rgb_min = torch.min(pixels_rgb_flat, dim=1).values
        channel_diff = rgb_max - rgb_min
        
        # Thresholds should be appropriate for linear RGB (0-1 range typically)
        # GP_SATURATION_THRESHOLD might need to be adjusted for linear data (e.g. 0.02-0.05 for diff)
        # The original GP_SATURATION_THRESHOLD = 0.05 was for (max-min)/max, which is relative.
        # For absolute difference, it's harder to set a universal threshold.
        # Let's use a relative measure for saturation to be more robust:
        saturation_metric = channel_diff / (rgb_max + EPSILON) # (max-min)/max

        # Create a mask for gray pixel candidates
        gray_candidate_mask = (saturation_metric < GP_SATURATION_THRESHOLD) & \
                              (luminance > GP_LUMINANCE_MIN_THRESHOLD) & \
                              (luminance < GP_LUMINANCE_MAX_THRESHOLD)
        
        gray_pixels = pixels_rgb_flat[gray_candidate_mask] # (M, 3)
        
        num_gray_candidates = gray_pixels.shape[0]

        if num_gray_candidates < (num_pixels * GP_MIN_GRAY_PIXEL_FRACTION):
            if debug_enabled: print(f"[ACC Debug WB] Not enough gray pixel candidates found: {num_gray_candidates}")
            return None

        # Select a robust subset of gray pixels (e.g., top N% with smallest channel_diff)
        # To do this, we need the channel_diff values for the candidates
        candidate_channel_diffs = channel_diff[gray_candidate_mask]
        
        # Determine the number of pixels to select based on percentile
        num_to_select = max(1, int(num_gray_candidates * (GP_PERCENTILE_SELECTION / 100.0)))
        
        if num_gray_candidates <= num_to_select : # if few candidates, use all
            selected_gray_pixels = gray_pixels
        else:
            # Find the threshold for the Nth percentile of channel differences
            # Sort differences and pick the threshold
            # Using `kthvalue` is efficient
            if num_to_select < candidate_channel_diffs.shape[0]:
                diff_threshold_for_selection = torch.kthvalue(candidate_channel_diffs, num_to_select).values
                # Select pixels whose channel difference is less than or equal to this threshold
                robust_selection_mask = candidate_channel_diffs <= diff_threshold_for_selection
                selected_gray_pixels = gray_pixels[robust_selection_mask]
                if selected_gray_pixels.shape[0] == 0: # Should not happen if num_to_select >= 1
                     selected_gray_pixels = gray_pixels # Fallback to all candidates
            else: # Should not be reached if num_to_select logic is correct
                selected_gray_pixels = gray_pixels


        if selected_gray_pixels.shape[0] == 0:
            if debug_enabled: print(f"[ACC Debug WB] No robust gray pixels selected after percentile.")
            return None
        
        # The average of these selected gray pixels is our illuminant estimate
        illuminant_estimate_linear_rgb = torch.mean(selected_gray_pixels, dim=0) # (3,)
        
        if debug_enabled:
            print(f"[ACC Debug WB] Gray Pixel Illuminant (Linear RGB): R={illuminant_estimate_linear_rgb[0]:.3f} G={illuminant_estimate_linear_rgb[1]:.3f} B={illuminant_estimate_linear_rgb[2]:.3f} from {selected_gray_pixels.shape[0]} pixels")
        return illuminant_estimate_linear_rgb


    def _step1_white_balance(self, image_tensor_rgb_chw_srgb: torch.Tensor, strength: float, 
                             is_bw_processing: bool, debug_enabled: bool) -> torch.Tensor:
        assert image_tensor_rgb_chw_srgb.ndim == 3 and image_tensor_rgb_chw_srgb.shape[0] == 3
        assert 0.0 <= strength <= 1.0

        if strength == 0:
            return image_tensor_rgb_chw_srgb.clone()

        device = image_tensor_rgb_chw_srgb.device
        
        m_bradford = M_BRADFORD.to(device)
        m_bradford_inv = M_BRADFORD_INV.to(device)
        m_srgb_to_xyz = M_SRGB_TO_XYZ.to(device)
        m_xyz_to_srgb = M_XYZ_TO_SRGB.to(device)

        img_linear_rgb = self._srgb_to_linear(image_tensor_rgb_chw_srgb)

        source_illuminant_rgb_linear_estimate = None
        using_gray_pixel_method = False 

        local_illuminant_linear_rgb = self._find_gray_pixels_illuminant(img_linear_rgb, debug_enabled)

        if local_illuminant_linear_rgb is not None:
            source_illuminant_rgb_linear_estimate = local_illuminant_linear_rgb
            using_gray_pixel_method = True
            if debug_enabled: print(f"[ACC Debug WB] Using Gray Pixel based illuminant (unnormalized): R={source_illuminant_rgb_linear_estimate[0]:.3f} G={source_illuminant_rgb_linear_estimate[1]:.3f} B={source_illuminant_rgb_linear_estimate[2]:.3f}")
        else:
            if debug_enabled: print(f"[ACC Debug WB] Gray Pixel method failed, falling back to Shades of Gray.")
            img_rgb_p = torch.pow(img_linear_rgb + SHADES_OF_GRAY_EPSILON, SHADES_OF_GRAY_POWER_P)
            mean_rgb_p = torch.mean(img_rgb_p, dim=(1, 2))
            sog_illuminant_linear = torch.pow(mean_rgb_p, 1.0 / SHADES_OF_GRAY_POWER_P)
            source_illuminant_rgb_linear_estimate = sog_illuminant_linear
            if debug_enabled: print(f"[ACC Debug WB] SoG Illuminant (unnormalized Linear RGB): R={sog_illuminant_linear[0]:.3f} G={sog_illuminant_linear[1]:.3f} B={sog_illuminant_linear[2]:.3f}")

        if source_illuminant_rgb_linear_estimate is None:
             return image_tensor_rgb_chw_srgb 

        norm_factor = source_illuminant_rgb_linear_estimate[1] 
        if norm_factor < EPSILON: norm_factor = torch.mean(source_illuminant_rgb_linear_estimate)
        if norm_factor < EPSILON: norm_factor = 1.0 
        
        source_illuminant_rgb_linear_normalized = source_illuminant_rgb_linear_estimate / (norm_factor + EPSILON)
        
        if debug_enabled:
            method_name = "Gray Pixel" if using_gray_pixel_method else "Shades of Gray"
            print(f"[ACC Debug WB] {method_name} Normalized Linear Illuminant: R={source_illuminant_rgb_linear_normalized[0]:.3f} G={source_illuminant_rgb_linear_normalized[1]:.3f} B={source_illuminant_rgb_linear_normalized[2]:.3f}")

        source_illuminant_xyz = torch.matmul(m_srgb_to_xyz, source_illuminant_rgb_linear_normalized.unsqueeze(1)).squeeze()
        source_illuminant_lms = torch.matmul(m_bradford, source_illuminant_xyz.unsqueeze(1)).squeeze()
        
        d65_rgb_linear_normalized_temp = torch.ones(3, device=device) 
        target_illuminant_xyz_for_ratio = torch.matmul(m_srgb_to_xyz, d65_rgb_linear_normalized_temp.unsqueeze(1)).squeeze()
        target_illuminant_lms = torch.matmul(m_bradford, target_illuminant_xyz_for_ratio.unsqueeze(1)).squeeze()

        lms_scaling_factors = target_illuminant_lms / (source_illuminant_lms + EPSILON)

        # ---- Apply internal strength reduction IF SoG was the fallback ----
        if not using_gray_pixel_method:
            # Lerp scaling factors towards 1.0 (no change)
            # SOG_FALLBACK_EFFECT_SCALE of 0.75 means 75% of the original SoG correction strength
            neutral_scaling_factors = torch.ones_like(lms_scaling_factors)
            lms_scaling_factors = torch.lerp(neutral_scaling_factors, lms_scaling_factors, SOG_FALLBACK_EFFECT_SCALE)
            if debug_enabled:
                print(f"[ACC Debug WB] SoG fallback: LMS Scaling Factors (dampened by {SOG_FALLBACK_EFFECT_SCALE:.2f}): L={lms_scaling_factors[0]:.3f} M={lms_scaling_factors[1]:.3f} S={lms_scaling_factors[2]:.3f}")
        elif debug_enabled: # Only print original factors if Gray Pixel was used (or SoG before dampening)
             print(f"[ACC Debug WB] LMS Scaling Factors: L={lms_scaling_factors[0]:.3f} M={lms_scaling_factors[1]:.3f} S={lms_scaling_factors[2]:.3f}")


        C, H, W = img_linear_rgb.shape
        img_linear_rgb_flat = img_linear_rgb.reshape(C, H * W)
        img_xyz_flat = torch.matmul(m_srgb_to_xyz, img_linear_rgb_flat)
        img_lms_flat = torch.matmul(m_bradford, img_xyz_flat)
        adapted_img_lms_flat = img_lms_flat * lms_scaling_factors.unsqueeze(1) 
        adapted_img_xyz_flat = torch.matmul(m_bradford_inv, adapted_img_lms_flat)
        adapted_img_linear_rgb_flat = torch.matmul(m_xyz_to_srgb, adapted_img_xyz_flat)
        adapted_img_linear_rgb = adapted_img_linear_rgb_flat.reshape(C, H, W)

        corrected_srgb_chw = self._linear_to_srgb(adapted_img_linear_rgb)
        corrected_srgb_chw = torch.clamp(corrected_srgb_chw, 0.0, 1.0) 
            
        # The final 'strength' parameter still applies to the overall effect of this step
        wb_image_tensor_rgb_chw = torch.lerp(image_tensor_rgb_chw_srgb, corrected_srgb_chw, strength)
        
        return wb_image_tensor_rgb_chw

    def _apply_stat_transfer_channel(self, channel_data: torch.Tensor, target_mean: float, target_std: float, 
                                     min_val: float, max_val: float) -> torch.Tensor:
        mean_in = torch.mean(channel_data)
        std_in = torch.std(channel_data)
        
        if std_in < EPSILON:
            return torch.clamp(channel_data - mean_in + target_mean, min_val, max_val)

        transferred_channel = (channel_data - mean_in) * (target_std / (std_in + EPSILON)) + target_mean
        return torch.clamp(transferred_channel, min_val, max_val)

    def _step2_statistical_transfer(self, image_tensor_lab_chw: torch.Tensor, strength: float,
                                    brightness_mode: str, color_style: str,
                                    initial_mean_l: float, initial_std_l: float,
                                    detected_brightness_category: str,
                                    is_bw_processing_pipeline: bool) -> torch.Tensor:
        assert image_tensor_lab_chw.ndim == 3 and image_tensor_lab_chw.shape[0] == 3
        assert 0.0 <= strength <= 1.0

        if strength == 0:
            return image_tensor_lab_chw.clone()

        l_in, a_in, b_in = image_tensor_lab_chw[0], image_tensor_lab_chw[1], image_tensor_lab_chw[2]
        
        # Initialize base targets
        target_L_mean = self.base_target_stats.mean_L
        target_L_std = self.base_target_stats.std_L
        target_a_mean = self.base_target_stats.mean_a
        target_a_std = self.base_target_stats.std_a
        target_b_mean = self.base_target_stats.mean_b
        target_b_std = self.base_target_stats.std_b

        # --- 1. Determine L* targets based on brightness_mode ---
        effective_brightness_condition = brightness_mode
        if brightness_mode == "auto":
            effective_brightness_condition = detected_brightness_category

        if effective_brightness_condition == "dark" or brightness_mode == "dark":
            predefined_low_L = self.base_target_stats.mean_L * NIGHT_LOW_KEY_PREDEFINED_L_TARGET_SCALE
            target_L_mean = initial_mean_l * NIGHT_LOW_KEY_L_MEAN_INPUT_FACTOR + \
                            predefined_low_L * NIGHT_LOW_KEY_L_MEAN_TARGET_FACTOR
            target_L_std = self.base_target_stats.std_L * NIGHT_LOW_KEY_L_STD_SCALE
        elif effective_brightness_condition == "bright" or brightness_mode == "bright":
            predefined_high_L = self.base_target_stats.mean_L * BRIGHT_HIGH_KEY_PREDEFINED_L_TARGET_SCALE # Now 1.45
            target_L_mean = initial_mean_l * BRIGHT_HIGH_KEY_L_MEAN_INPUT_FACTOR + \
                            predefined_high_L * BRIGHT_HIGH_KEY_L_MEAN_TARGET_FACTOR
            target_L_std = self.base_target_stats.std_L * BRIGHT_HIGH_KEY_L_STD_SCALE # Now 0.85
        elif brightness_mode == "original":
            target_L_mean = initial_mean_l
            target_L_std = initial_std_l * PRESERVE_L_STD_SCALE 
        elif effective_brightness_condition == "normal" or brightness_mode == "normal":
            # Gently guide towards a standard normal L* mean, also considering input
            target_L_mean = initial_mean_l * NORMAL_L_MEAN_INPUT_FACTOR + \
                            NORMAL_L_MEAN_TARGET_ABSOLUTE * NORMAL_L_MEAN_TARGET_FACTOR
            target_L_std = self.base_target_stats.std_L * NORMAL_L_STD_SCALE
        # else: (should not happen if brightness_mode is one of the defined options)
            # Default to base target stats if logic error
            # target_L_mean = self.base_target_stats.mean_L
            # target_L_std = self.base_target_stats.std_L
        # --- 2. Adjust a*, b* targets based on color_style ---
        # Note: Some color styles also suggest L_std scaling, which can be an additional layer
        # or could be primarily driven by the brightness_mode. For now, let L_std be primarily from brightness.
        if color_style == "vibrant":
            # target_L_std *= VIBRANT_L_STD_SCALE # Optional: consider if this should override brightness L_std
            target_a_std *= VIBRANT_A_STD_SCALE
            target_b_std *= VIBRANT_B_STD_SCALE
        elif color_style == "cinematic":
            target_a_std *= SOFT_A_STD_SCALE
            target_b_std *= SOFT_B_STD_SCALE
        elif color_style == "enhanced_naturalism":
            target_a_std *= ENHANCED_NATURALISM_A_STD_SCALE
            target_b_std *= ENHANCED_NATURALISM_B_STD_SCALE
        # For "Natural", use base a*, b* std (already initialized)

        # --- 3. Handle B&W Processing for Chrominance ---
        if is_bw_processing_pipeline: # This flag is True if color_style is "B&W Tonal"
            target_a_mean = BW_TARGET_A_MEAN
            target_a_std = BW_TARGET_A_STD
            target_b_mean = BW_TARGET_B_MEAN
            target_b_std = BW_TARGET_B_STD

        # Apply statistical transfer per channel
        l_out = self._apply_stat_transfer_channel(l_in, target_L_mean, target_L_std, 0.0, 100.0)
        a_out = self._apply_stat_transfer_channel(a_in, target_a_mean, target_a_std, -128.0, 127.0)
        b_out = self._apply_stat_transfer_channel(b_in, target_b_mean, target_b_std, -128.0, 127.0)

        transferred_lab_chw = torch.stack([l_out, a_out, b_out], dim=0)
        st_image_tensor_lab = torch.lerp(image_tensor_lab_chw, transferred_lab_chw, strength)
        return st_image_tensor_lab

    @staticmethod
    def _shs_lab_to_lch_numpy(lab_array: np.ndarray) -> np.ndarray:
        # ... (no change)
        L = lab_array[..., 0]
        a = lab_array[..., 1]
        b = lab_array[..., 2]
        C = np.sqrt(a**2 + b**2 + EPSILON)
        h_rad = np.arctan2(b, a)
        h_deg = np.rad2deg(h_rad)
        h_deg = (h_deg + 360.0) % 360.0
        return np.stack([L, C, h_deg], axis=-1).astype(np.float32)

    @staticmethod
    def _shs_lch_to_lab_numpy(lch_array: np.ndarray) -> np.ndarray:
        # ... (no change)
        L = lch_array[..., 0]
        C = lch_array[..., 1]
        h_deg = lch_array[..., 2]
        h_rad = np.deg2rad(h_deg)
        a = C * np.cos(h_rad)
        b = C * np.sin(h_rad)
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    # _shs_lerp_hue_degrees_single_np might not be directly needed with radian shifts

    def _prepare_colored_card_patches(self):
        # ... (no change)
        if hasattr(self, 'np_colored_card_patches_lab') and self.np_colored_card_patches_lab is not None:
            return
        temp_colored_patches_lab_list = []
        for key, lab_val in self.VIRTUAL_COLOR_CARD_LAB.items():
            if key.startswith("B") or key.startswith("C") or key.startswith("D"):
                temp_colored_patches_lab_list.append(list(lab_val))
        self.np_colored_card_patches_lab = np.array(temp_colored_patches_lab_list, dtype=np.float32).reshape(-1,3) if temp_colored_patches_lab_list else np.array([], dtype=np.float32).reshape(0,3)


    def _step3_selective_hue_saturation(self, image_tensor_lab_chw: torch.Tensor, strength: float, is_bw_processing_pipeline: bool) -> torch.Tensor:
        if is_bw_processing_pipeline or strength == 0.0 or self.np_colored_card_patches_lab.shape[0] == 0:
            return image_tensor_lab_chw.clone()

        original_device = image_tensor_lab_chw.device
        C_channels, H, W = image_tensor_lab_chw.shape
        num_pixels = H * W

        pixels_lab_hw_c_tensor = image_tensor_lab_chw.permute(1, 2, 0)
        pixels_lab_np_hw_c = pixels_lab_hw_c_tensor.cpu().numpy()
        pixels_lab_np_flat = pixels_lab_np_hw_c.reshape(num_pixels, C_channels) # Shape: (N, 3)
        
        # Original LCh values of all pixels
        pixels_lch_np_flat = Advanced_Color_Correction._shs_lab_to_lch_numpy(pixels_lab_np_flat) # Shape: (N, 3)

        # 1. K-Means to find representative cluster centers
        try:
            k_for_kmeans = min(SHS_K_MEANS_CLUSTERS, max(1, num_pixels // 20 if num_pixels > SHS_K_MEANS_CLUSTERS else 1))
            if num_pixels < k_for_kmeans : return image_tensor_lab_chw.clone()
            
            kmeans = KMeans(n_clusters=k_for_kmeans, random_state=0, n_init=SHS_KMEANS_N_INIT, max_iter=SHS_KMEANS_MAX_ITER)
            # We don't strictly need pixel_labels_np anymore for this blended approach,
            # but K-Means fit provides the centers.
            kmeans.fit(pixels_lab_np_flat) 
            cluster_centers_lab_np = kmeans.cluster_centers_ # Shape: (K, 3)
        except Exception:
            return image_tensor_lab_chw.clone()

        # 2. Calculate target Chroma scales and Hue shifts for each cluster center
        # These are the "ideal" adjustments if a pixel were 100% like that cluster center
        target_cluster_chroma_scales = np.ones(k_for_kmeans, dtype=np.float32)
        target_cluster_hue_shifts_rad = np.zeros(k_for_kmeans, dtype=np.float32)
        
        # Optimization: Check if any card patches were found
        # No need to proceed with complex matching if card is empty
        if self.np_colored_card_patches_lab.shape[0] == 0:
             return image_tensor_lab_chw.clone()

        for i in range(k_for_kmeans):
            current_cluster_center_lab = cluster_centers_lab_np[i]
            
            # Find closest card patch to this cluster center
            distances_to_card = np.linalg.norm(self.np_colored_card_patches_lab - current_cluster_center_lab, axis=1)
            closest_card_idx = np.argmin(distances_to_card)
            
            if distances_to_card[closest_card_idx] < SHS_CONFIDENT_MATCH_DISTANCE_THRESHOLD_LAB:
                matched_card_patch_lab = self.np_colored_card_patches_lab[closest_card_idx]
                
                current_cluster_center_lch = Advanced_Color_Correction._shs_lab_to_lch_numpy(current_cluster_center_lab)
                matched_card_patch_lch = Advanced_Color_Correction._shs_lab_to_lch_numpy(matched_card_patch_lab)
                
                scale = matched_card_patch_lch[1] / (current_cluster_center_lch[1] + EPSILON)
                target_cluster_chroma_scales[i] = np.clip(scale, SHS_CHROMA_SCALE_CLAMP_MIN, SHS_CHROMA_SCALE_CLAMP_MAX)
                
                h_diff_deg = matched_card_patch_lch[2] - current_cluster_center_lch[2]
                if h_diff_deg > 180.0: h_diff_deg -= 360.0
                elif h_diff_deg < -180.0: h_diff_deg += 360.0
                target_cluster_hue_shifts_rad[i] = np.deg2rad(h_diff_deg)

        # 3. Calculate influence weights of each cluster on each pixel
        # pixels_lab_np_flat: (N, 3), cluster_centers_lab_np: (K, 3)
        # We want distances_sq_pixels_to_clusters: (N, K)
        # (N, 1, 3) - (1, K, 3) => (N, K, 3) -> sum over last dim => (N, K)
        diff_pixels_clusters = pixels_lab_np_flat[:, np.newaxis, :] - cluster_centers_lab_np[np.newaxis, :, :]
        distances_sq_pixels_to_clusters = np.sum(diff_pixels_clusters**2, axis=2) # Shape: (N, K)

        # Gaussian influence weights: W_pk = exp(-d_pk^2 / R^2)
        influence_radius_sq = SHS_CLUSTER_INFLUENCE_RADIUS_LAB**2 + EPSILON
        pixel_cluster_weights = np.exp(-distances_sq_pixels_to_clusters / influence_radius_sq) # Shape: (N, K)
        
        # Normalize weights for each pixel so sum_k(W_pk) = 1
        sum_weights_per_pixel = np.sum(pixel_cluster_weights, axis=1, keepdims=True) + EPSILON # Shape: (N, 1)
        normalized_pixel_cluster_weights = pixel_cluster_weights / sum_weights_per_pixel # Shape: (N, K)

        # 4. Calculate blended Chroma scale and Hue shift for each pixel
        # blended_chroma_scale_p = sum_k (Normalized_W_pk * Target_Chroma_Scale_k)
        # blended_hue_shift_p    = sum_k (Normalized_W_pk * Target_Hue_Shift_k)
        
        # (N, K) @ (K,) -> (N,) if using matmul, or (N,K) * (1,K) -> sum(axis=1)
        blended_chroma_scales_per_pixel = np.sum(normalized_pixel_cluster_weights * target_cluster_chroma_scales[np.newaxis, :], axis=1) # Shape: (N,)
        blended_hue_shifts_rad_per_pixel = np.sum(normalized_pixel_cluster_weights * target_cluster_hue_shifts_rad[np.newaxis, :], axis=1) # Shape: (N,)

        # 5. Apply blended adjustments to original LCh values of pixels
        adjusted_pixels_lch_np_flat = np.copy(pixels_lch_np_flat) # Start with original LCh

        # Apply blended Chroma scaling (L is preserved from original)
        # Chroma scales are for the original C, not lerped from 1.0
        adjusted_pixels_lch_np_flat[:, 1] = pixels_lch_np_flat[:, 1] * blended_chroma_scales_per_pixel
        adjusted_pixels_lch_np_flat[:, 1] = np.maximum(0, adjusted_pixels_lch_np_flat[:, 1]) # Chroma >= 0

        # Apply blended Hue shift (original hue in rad + blended_shift_rad)
        original_hue_rad_per_pixel = np.deg2rad(pixels_lch_np_flat[:, 2])
        adjusted_hue_rad_per_pixel = original_hue_rad_per_pixel + blended_hue_shifts_rad_per_pixel
        adjusted_pixels_lch_np_flat[:, 2] = (np.rad2deg(adjusted_hue_rad_per_pixel) + 360.0) % 360.0
        
        # 6. Convert fully adjusted LCh pixels back to L*a*b*
        adjusted_pixels_lab_np_flat = Advanced_Color_Correction._shs_lch_to_lab_numpy(adjusted_pixels_lch_np_flat)

        # Reshape and convert to tensor
        adjusted_pixels_lab_np_hw_c = adjusted_pixels_lab_np_flat.reshape(H, W, C_channels)
        image_fully_adjusted_lab_chw = torch.from_numpy(adjusted_pixels_lab_np_hw_c).permute(2, 0, 1).to(original_device)
        
        # 7. Apply the step's 'strength' by lerping
        output_image_lab_chw = torch.lerp(image_tensor_lab_chw, image_fully_adjusted_lab_chw, strength)
        return output_image_lab_chw

    def _step4_clahe(self, image_tensor_lab_chw: torch.Tensor, strength: float, selected_device: torch.device) -> torch.Tensor:
        if strength == 0: return image_tensor_lab_chw.clone()
        l_channel_orig = image_tensor_lab_chw[0, :, :]
        l_numpy_u8 = (l_channel_orig.cpu().numpy().clip(0, 100) * 2.55).round().astype(np.uint8)
        clahe_filter = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        l_clahe_numpy_u8 = clahe_filter.apply(l_numpy_u8)
        l_clahe_float_0_100 = torch.from_numpy(l_clahe_numpy_u8.astype(np.float32) / 2.55).to(selected_device)
        l_channel_final = torch.lerp(l_channel_orig, l_clahe_float_0_100, strength)
        return torch.stack([l_channel_final, image_tensor_lab_chw[1], image_tensor_lab_chw[2]], dim=0)

    def _step5_gamut_management(self, image_tensor_rgb_chw: torch.Tensor, enabled: bool) -> torch.Tensor:
        assert image_tensor_rgb_chw.ndim == 3 and image_tensor_rgb_chw.shape[0] == 3, "RGB tensor must be (3, H, W)"

        if not enabled: # If clipping is disabled, pass through
            return image_tensor_rgb_chw
        
        # Apply simple clipping for gamut management.
        gm_image_tensor_rgb = torch.clamp(image_tensor_rgb_chw, 0.0, 1.0)
            
        assert gm_image_tensor_rgb.shape == image_tensor_rgb_chw.shape, "Output shape mismatch in Gamut"
        return gm_image_tensor_rgb

    def _rgb_to_lab_cv(self, image_tensor_rgb_chw_0_1: torch.Tensor, device_for_output: torch.device) -> torch.Tensor:
        img_hwc_numpy_rgb_f32 = image_tensor_rgb_chw_0_1.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        img_hwc_numpy_lab_cv = cv2.cvtColor(img_hwc_numpy_rgb_f32, cv2.COLOR_RGB2LAB)
        return torch.from_numpy(img_hwc_numpy_lab_cv).permute(2, 0, 1).to(device_for_output)

    def _lab_to_rgb_cv(self, image_tensor_lab_chw_standard: torch.Tensor, device_for_output: torch.device) -> torch.Tensor:
        img_hwc_numpy_lab_f32 = image_tensor_lab_chw_standard.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        img_hwc_numpy_lab_f32[:,:,0] = np.clip(img_hwc_numpy_lab_f32[:,:,0], 0.0, 100.0)
        img_hwc_numpy_lab_f32[:,:,1] = np.clip(img_hwc_numpy_lab_f32[:,:,1], -127.0, 127.0)
        img_hwc_numpy_lab_f32[:,:,2] = np.clip(img_hwc_numpy_lab_f32[:,:,2], -127.0, 127.0)
        img_hwc_numpy_rgb_f32 = cv2.cvtColor(img_hwc_numpy_lab_f32, cv2.COLOR_LAB2RGB)
        img_hwc_numpy_rgb_f32_clipped = np.clip(img_hwc_numpy_rgb_f32, 0.0, 1.0)
        return torch.from_numpy(img_hwc_numpy_rgb_f32_clipped).permute(2, 0, 1).to(device_for_output)

    def exec(self, image: torch.Tensor, 
                         brightness_mode: str, color_style: str, 
                         overall_strength: float,
                         strength_white_balance: float, strength_stat_transfer: float,
                         strength_selective_hue: float, strength_clahe: float,
                         gamut_clipping: bool, 
                         debug_prints: str, device: str):

        if overall_strength == 0.0:
            return (image,)

        debug_enabled = (debug_prints == "Enabled")
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        amp_enabled = supports_amp() and selected_device.type == 'cuda'
        original_image_batch_tensor_hwc = image.to(selected_device)
        processed_image_batch_list = []

        for i in range(original_image_batch_tensor_hwc.shape[0]):
            img_tensor_chw_rgb_0_1 = original_image_batch_tensor_hwc[i].permute(2, 0, 1).float()
            if original_image_batch_tensor_hwc.dtype == torch.uint8:
                img_tensor_chw_rgb_0_1 /= 255.0
            
            current_processing_image_rgb = img_tensor_chw_rgb_0_1.clone()

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                temp_lab_for_analysis = self._rgb_to_lab_cv(current_processing_image_rgb, selected_device)
                initial_L_mean, initial_L_std, _, _, detected_brightness_cat, is_bw_pipeline = \
                    self._initial_analysis(temp_lab_for_analysis, brightness_mode, color_style, debug_enabled)
                del temp_lab_for_analysis

                # Pass debug_enabled to white balance step
                wb_image_rgb = self._step1_white_balance(current_processing_image_rgb, strength_white_balance, 
                                                         is_bw_pipeline, debug_enabled) 
                current_processing_image_lab = self._rgb_to_lab_cv(wb_image_rgb, selected_device)

                st_image_lab = self._step2_statistical_transfer(
                    current_processing_image_lab, strength_stat_transfer,
                    brightness_mode, color_style,
                    initial_L_mean, initial_L_std,
                    detected_brightness_cat, is_bw_pipeline
                )
                current_processing_image_lab = st_image_lab

                sh_image_lab = self._step3_selective_hue_saturation(current_processing_image_lab, strength_selective_hue, is_bw_pipeline)
                current_processing_image_lab = sh_image_lab

                clahe_image_lab = self._step4_clahe(current_processing_image_lab, strength_clahe, selected_device)
                current_processing_image_lab = clahe_image_lab
                
                processed_image_rgb_before_gamut = self._lab_to_rgb_cv(current_processing_image_lab, selected_device)
                final_image_rgb_chw_processed = self._step5_gamut_management(processed_image_rgb_before_gamut, gamut_clipping)

                final_image_rgb_chw = torch.lerp(img_tensor_chw_rgb_0_1, final_image_rgb_chw_processed, overall_strength)
                final_image_rgb_chw = torch.clamp(final_image_rgb_chw, 0.0, 1.0)
                processed_image_batch_list.append(final_image_rgb_chw.permute(1, 2, 0))

        result_tensor_hwc = torch.stack(processed_image_batch_list, dim=0)
        if image.dtype == torch.uint8:
            result_tensor_hwc = (result_tensor_hwc.cpu() * 255.0).round().byte()
        else:
            result_tensor_hwc = result_tensor_hwc.cpu()
        return (result_tensor_hwc,)

NODE_CLASS_MAPPINGS = {"Advanced_Color_Correction": Advanced_Color_Correction}
NODE_DISPLAY_NAME_MAPPINGS = {"Advanced_Color_Correction": "🐐 Advanced Color Correction"}