import comfy.utils  # type: ignore
import torch  # type: ignore
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel  # type: ignore
import math


class Advanced_Upscale_Image_Using_Model:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.025}),
                "rescale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "lanczos"}),
                "stage2_order": (["upscale_first", "downscale_first"], {"default": "downscale_first"}),
                "mixed_initial": ("BOOLEAN", {"default": True}),
                "tiled_upscale": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Image'
    DESCRIPTION = '''
    Upscales the image using an upscale model.\n
    ‣ upscale_by | Upscale factor for the final resolution of the output image.\n
    ‣ rescale_method | Rescale method for resizing the image in intermediary steps.\n
    ‣ stage2_order | Decides the order of upscaling and resizing when using very high upscale_by. downscale_first is a lot faster, but with a bit less detail. upscale_first is a lot slower but with a bit more detail.\n
    ‣ mixed_initial | Mixes the initial image with an upscaled and downscaled version before upscaling for very slight sharpness improvement.\n
    ‣ tiled_upscale | Upscales the image in 2x2 tiles, which might be helpful in certain use cases. (Not faster, but might be more stable at extreme resolutions.)
    '''


    @classmethod
    def exec(self, upscale_model, image, upscale_by,
                         rescale_method, stage2_order, mixed_initial, tiled_upscale):
        try:
            if mixed_initial:
                image = self.create_mixed_initial(image, rescale_method)
                
            if(upscale_model.scale == 1):
                print(f"🐐 Advanced Upscale: 1x upscale model detected.\n🐐 Advanced Upscale: Initializing quick upscale. (No tiling)")
                image = self.only_upscale(upscale_model, image, rescale_method, upscale_by)
                return (image, image.shape[1], image.shape[2],)

            # Process Stage 1 (always upscale first)
            stage1_result = self.process_stage1(upscale_model, image, upscale_by, rescale_method, tiled_upscale)

            # Check if Stage 2 is needed
            current_scale = stage1_result[1]
            if current_scale < upscale_by:
                # Process Stage 2
                final_image = self.process_stage2(upscale_model, stage1_result[0], upscale_by, current_scale, rescale_method, stage2_order == "upscale_first", tiled_upscale)
            else:
                final_image = stage1_result[0]

            return (final_image, final_image.shape[1], final_image.shape[2],)
        except Exception as e:
            print(f"🐐 Advanced Upscale: Returning input image. Error in exec: {str(e)}")
            return (image, image.shape[1], image.shape[2],)


    @staticmethod
    def create_mixed_initial(image, method):
        samples = image.movedim(-1, 1)
        upscaled = comfy.utils.common_upscale(samples, samples.shape[3] * 2, samples.shape[2] * 2, method, "disabled")
        downscaled = comfy.utils.common_upscale(upscaled, samples.shape[3], samples.shape[2], method, "disabled")
        mixed = samples * 0.5 + downscaled * 0.5
        return mixed.movedim(1, -1)


    @classmethod
    def only_upscale(self, upscale_model, image, rescale_method, upscale_by):
        samples = image.movedim(-1, 1)
        
        original_width, original_height = samples.shape[3], samples.shape[2]
        
        target_width = round(original_width * upscale_by)
        target_height = round(original_height * upscale_by)
        
        upscaled = ImageUpscaleWithModel().upscale(upscale_model, image)[0].movedim(-1, 1)
        
        samples = comfy.utils.common_upscale(upscaled, target_width, target_height, rescale_method, "disabled")
        
        return samples.movedim(1, -1)


    @classmethod
    def process_stage1(self, upscale_model, image, upscale_by, rescale_method, tiled_upscale):
        print(f"🐐 Advanced Upscale: Initializing Stage 1 of upscaling.")
        samples = image.movedim(-1, 1)
        original_width, original_height = samples.shape[3], samples.shape[2]

        # Upscale using the model
        if tiled_upscale:
            print(f"🐐 Advanced Upscale: Using tiled upscaling.")
            upscaled = self.tiled_upscaling(upscale_model, image)
        else:
            upscaled = ImageUpscaleWithModel().upscale(upscale_model, image)[0].movedim(-1, 1)
        
        achieved_scale = upscaled.shape[3] / original_width

        # If the achieved scale is more than needed, downscale
        if achieved_scale > upscale_by:
            target_width = round(original_width * upscale_by)
            target_height = round(original_height * upscale_by)
            samples = comfy.utils.common_upscale(upscaled, target_width, target_height, rescale_method, "disabled")
            return (samples.movedim(1, -1), upscale_by)
        else:
            return (upscaled.movedim(1, -1), achieved_scale)


    @classmethod
    def process_stage2(self, upscale_model, image, upscale_by, current_scale, rescale_method, upscale_first, tiled_upscale):
        print(f"🐐 Advanced Upscale: Initializing Stage 2 of upscaling.")
        samples = image.movedim(-1, 1)
        original_width, original_height = samples.shape[3], samples.shape[2]
        target_width = round(original_width * (upscale_by / current_scale))
        target_height = round(original_height * (upscale_by / current_scale))

        if upscale_first:
            if tiled_upscale:
                upscaled = self.tiled_upscaling(upscale_model, image)
            else:
                upscaled = ImageUpscaleWithModel().upscale(upscale_model, image)[0].movedim(-1, 1)

            if upscaled.shape[3] != target_width or upscaled.shape[2] != target_height:
                samples = comfy.utils.common_upscale(upscaled, target_width, target_height, rescale_method, "disabled")
        else:
            interim_width = round(original_width * (upscale_by / current_scale) / upscale_model.scale)
            interim_height = round(original_height * (upscale_by / current_scale) / upscale_model.scale)

            downscaled = comfy.utils.common_upscale(samples, interim_width, interim_height, rescale_method, "disabled")

            if tiled_upscale:
                upscaled = self.tiled_upscaling(upscale_model, downscaled.movedim(1, -1))
            else:
                upscaled = ImageUpscaleWithModel().upscale(upscale_model, downscaled.movedim(1, -1))[0].movedim(-1, 1)

            if upscaled.shape[3] != target_width or upscaled.shape[2] != target_height:
                samples = comfy.utils.common_upscale(upscaled, target_width, target_height, rescale_method, "disabled")
            else:
                samples = upscaled

        return samples.movedim(1, -1)


    @classmethod
    def tiled_upscaling(self, upscale_model, image):
        # Get image dimensions
        _, height, width, channels = image.shape
        
        scale_factor = upscale_model.scale

        # Calculate tile size and overlap for both dimensions
        tile_width = width // 2
        tile_height = height // 2
        overlap_w = max(min(tile_width // 8, 64), 8)
        overlap_h = max(min(tile_height // 8, 64), 8)

        # Ensure overlap is at least 1 pixel less than tile dimensions
        overlap_w = max(1, min(overlap_w, tile_width - 1))
        overlap_h = max(1, min(overlap_h, tile_height - 1))
        
        print(f"🐐 Advanced Upscale: Tile size: {tile_width}x{tile_height} | Overlap: {overlap_w}x{overlap_h}")

        # Calculate coordinates for tiles
        coords = [
            (0, 0, tile_width + overlap_w, tile_height + overlap_h),
            (width - tile_width - overlap_w, 0, width, tile_height + overlap_h),
            (0, height - tile_height - overlap_h, tile_width + overlap_w, height),
            (width - tile_width - overlap_w, height - tile_height - overlap_h, width, height)
        ]

        # Create and upscale tiles
        upscaled_tiles = []
        for i, (x1, y1, x2, y2) in enumerate(coords):
            print(f"🐐 Advanced Upscale: Processing tile {i+1}/{len(coords)} - Coordinates: ({x1}, {y1}, {x2}, {y2})")
            
            tile = image[:, y1:y2, x1:x2, :]
            upscaled_tile = ImageUpscaleWithModel().upscale(upscale_model, tile)[0]
            upscaled_tiles.append(upscaled_tile)

        # Calculate dimensions of upscaled image
        upscaled_height = int(height * scale_factor)
        upscaled_width = int(width * scale_factor)

        # Create empty tensor for final upscaled image
        upscaled_image = torch.zeros((1, upscaled_height, upscaled_width, channels), dtype=upscaled_tiles[0].dtype)
        weight_sum = torch.zeros((1, upscaled_height, upscaled_width, channels), dtype=upscaled_tiles[0].dtype)

        # Define the sine blending function
        def sine_blend(t):
            return 0.5 - 0.5 * math.cos(math.pi * t)

        # Place and blend upscaled tiles
        for i, (x1, y1, x2, y2) in enumerate(coords):
            ux1, uy1, ux2, uy2 = [int(coord * scale_factor) for coord in (x1, y1, x2, y2)]
            tile = upscaled_tiles[i]
            
            # Ensure we don't overshoot the final dimensions
            ux2 = min(ux2, upscaled_width)
            uy2 = min(uy2, upscaled_height)
            
            # Calculate the actual dimensions of the upscaled tile
            actual_tile_height = uy2 - uy1
            actual_tile_width = ux2 - ux1
            
            # Crop the tile to match the target dimensions
            tile = tile[0, :actual_tile_height, :actual_tile_width, :]

            # Calculate blending weights for each tile
            weight_matrix = torch.ones((actual_tile_height, actual_tile_width, channels), dtype=tile.dtype)

            # Apply blending based on overlap regions
            for j in range(overlap_w * scale_factor):
                blend_weight = sine_blend(float(j) / (overlap_w * scale_factor))
                if x1 > 0:
                    weight_matrix[:, j, :] *= blend_weight  # Left edge
                if x2 < width:
                    weight_matrix[:, -(j + 1), :] *= blend_weight  # Right edge
            
            for j in range(overlap_h * scale_factor):
                blend_weight = sine_blend(float(j) / (overlap_h * scale_factor))
                if y1 > 0:
                    weight_matrix[j, :, :] *= blend_weight  # Top edge
                if y2 < height:
                    weight_matrix[-(j + 1), :, :] *= blend_weight  # Bottom edge
            
            # Add the weighted tile to the final image
            upscaled_image[0, uy1:uy2, ux1:ux2, :] += tile * weight_matrix
            weight_sum[0, uy1:uy2, ux1:ux2, :] += weight_matrix

        # Normalize by the sum of the weights to avoid artifacts
        upscaled_image = torch.where(weight_sum > 0, upscaled_image / weight_sum, upscaled_image)
        return upscaled_image.movedim(-1, 1)


NODE_CLASS_MAPPINGS = {
    "Advanced_Upscale_Image_Using_Model": Advanced_Upscale_Image_Using_Model
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Advanced_Upscale_Image_Using_Model": "🐐 Advanced Upscale Image (using Model)"
}