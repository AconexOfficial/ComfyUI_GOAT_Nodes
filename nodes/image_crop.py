import torch
import math
import comfy.utils  # Import ComfyUI utilities

class Image_Crop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "padding": ("INT", {"default": 32, "min": 0, "max": 512}),
                "context_size": ("INT", {"default": 256, "min": 256, "max": 2048}),
                "upscale_images": ("BOOLEAN", {"default": False}),
                "rescale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "lanczos"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "max_upscale_size": ("INT", {"default": 1024, "min": 256, "max": 8192}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA")
    RETURN_NAMES = ("CROPPED_IMAGES", "CROPPED_MASKS", "CROP_DATA")
    FUNCTION = "exec"
    CATEGORY = "🐐 GOAT Nodes/Image"
    DESCRIPTION = '''
    Creates cropped images from an image and a batch of masks.\n
    ‣ image | The original image to be cropped from.\n
    ‣ masks | A batch of masks to be used to crop specific areas of the original image.\n
    ‣ padding | Adds optional padding to the mask dimensions, if it is close to the edge of the cropped image.\n
    ‣ context_size | Specifies the minimum size of the cropped image.\n
    ‣ upscale_images | Turns on automatic upscaling of the cropped images (f.e. to detail the cropped images with a sampler).\n
    ‣ rescale_method | Rescale method for resizing the image in intermediary steps.\n
    ‣ upscale_factor | Specifies the additional upscale factor of the cropped images.\n
    ‣ max_upscale_size | Specifies the maximum size to which the cropped images should be upscaled.
    '''

    def round_to_multiple(self, value, multiple):
        """Round a value to the nearest multiple."""
        return multiple * ((value + multiple - 1) // multiple)

    def exec(self, image, masks, padding, context_size, upscale_images, rescale_method, upscale_factor, max_upscale_size):
        # Ensure masks is a batch of masks
        if masks.dim() != 3:
            raise ValueError("Masks should be a 3D tensor (batch of masks).")

        # Get the height and width of the original image
        image_height, image_width = image.shape[1], image.shape[2]

        # Round down the minimum dimension of the original image to the nearest multiple of 8
        min_image_dim = min(image_height, image_width)
        min_image_dim = self.round_to_multiple(min_image_dim, 8) // 8 * 8  # Ensure it's a multiple of 8

        # Initialize lists to store the results
        cropped_images = []
        cropped_masks = []
        crop_data = []

        # Find the largest crop_size in the batch
        max_crop_size = 0

        # First pass: Calculate the largest crop_size
        for i in range(masks.shape[0]):
            mask = masks[i]

            # Step 1: Calculate the center of the mask
            non_zero_indices = torch.nonzero(mask, as_tuple=False)
            if non_zero_indices.size(0) == 0:
                # If the mask is empty, skip it
                continue

            # Step 2: Determine the dimensions of the mask with padding
            min_y, min_x = non_zero_indices.min(dim=0)[0]
            max_y, max_x = non_zero_indices.max(dim=0)[0]

            # Add padding
            min_y = max(min_y - padding, 0)
            min_x = max(min_x - padding, 0)
            max_y = min(max_y + padding, mask.shape[0] - 1)
            max_x = min(max_x + padding, mask.shape[1] - 1)

            # Calculate the width and height of the bounding box
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Step 3: Ensure the cropped image is a square with at least context_size
            crop_size = max(width, height, context_size)

            # Round up the crop_size to the nearest multiple of 8
            crop_size = self.round_to_multiple(crop_size, 8)

            # Cap the crop_size at the minimum of the image dimensions
            crop_size = min(crop_size, min_image_dim)

            # Update the largest crop_size
            if upscale_images:
                if crop_size < max_upscale_size:
                    crop_size = self.round_to_multiple(min(int(crop_size * upscale_factor), max_upscale_size), 8) // 8 * 8  # Ensure it's a multiple of 8

            if crop_size > max_crop_size:
                max_crop_size = crop_size

        # Second pass: Crop and upscale images to the largest crop_size
        for i in range(masks.shape[0]):
            mask = masks[i]

            # Step 1: Calculate the center of the mask
            non_zero_indices = torch.nonzero(mask, as_tuple=False)
            if non_zero_indices.size(0) == 0:
                # If the mask is empty, skip it
                continue

            center = non_zero_indices.float().mean(dim=0)
            center_y, center_x = center[0].item(), center[1].item()

            # Step 2: Determine the dimensions of the mask with padding
            min_y, min_x = non_zero_indices.min(dim=0)[0]
            max_y, max_x = non_zero_indices.max(dim=0)[0]

            # Add padding
            min_y = max(min_y - padding, 0)
            min_x = max(min_x - padding, 0)
            max_y = min(max_y + padding, mask.shape[0] - 1)
            max_x = min(max_x + padding, mask.shape[1] - 1)

            # Calculate the width and height of the bounding box
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Step 3: Ensure the cropped image is a square with at least context_size
            crop_size = max(width, height, context_size)

            # Round up the crop_size to the nearest multiple of 8
            crop_size = self.round_to_multiple(crop_size, 8)

            # Cap the crop_size at the minimum of the image dimensions
            crop_size = min(crop_size, min_image_dim)

            # Calculate the new bounding box to make it square
            # Center the bounding box around the original center
            half_size = crop_size // 2
            new_min_x = int(center_x) - half_size
            new_min_y = int(center_y) - half_size
            new_max_x = new_min_x + crop_size
            new_max_y = new_min_y + crop_size

            # Step 4: Offset the bounding box if it exceeds image dimensions
            if new_min_x < 0:
                new_max_x -= new_min_x  # Shift right
                new_min_x = 0
            if new_min_y < 0:
                new_max_y -= new_min_y  # Shift down
                new_min_y = 0
            if new_max_x > image_width:
                new_min_x -= (new_max_x - image_width)  # Shift left
                new_max_x = image_width
            if new_max_y > image_height:
                new_min_y -= (new_max_y - image_height)  # Shift up
                new_max_y = image_height

            # Step 5: Crop the image and mask
            cropped_image = image[:, new_min_y:new_max_y, new_min_x:new_max_x, :]
            cropped_mask = mask[new_min_y:new_max_y, new_min_x:new_max_x]

            # Step 6: Upscale the cropped image and mask to the largest crop_size
            init_upscale_factor = max_crop_size / crop_size
            if init_upscale_factor != 1:
                
                print(f"max_crop_size: {max_crop_size}")
                
                # Upscale the image using ComfyUI's common_upscale
                cropped_image = comfy.utils.common_upscale(
                    cropped_image.permute(0, 3, 1, 2),  # Change to (B, C, H, W)
                    max_crop_size,
                    max_crop_size,
                    rescale_method,
                    "disabled",
                ).permute(0, 2, 3, 1)  # Change back to (B, H, W, C)

                # Upscale the mask using ComfyUI's common_upscale
                cropped_mask = comfy.utils.common_upscale(
                    cropped_mask.unsqueeze(0).unsqueeze(0).float(),  # Change to (1, 1, H, W)
                    max_crop_size,
                    max_crop_size,
                    "bilinear",
                    "disabled",
                ).squeeze(0).squeeze(0).float()  # Change back to (H, W)

            print(f"🐐 Created Cropped Image at location: {center_y, center_x}. Dimensions are: {int(crop_size)}x{int(crop_size)}. Upscale factor is: {init_upscale_factor}")

            # Step 7: Store the crop data (center, bounding box, etc.)
            crop_data.append({
                "center": (center_y, center_x),
                "original_bbox": (min_y, min_x, max_y, max_x),
                "square_bbox": (new_min_y, new_min_x, new_max_y, new_max_x),
                "crop_size": crop_size,
            })

            # Append to the results
            cropped_images.append(cropped_image)
            cropped_masks.append(cropped_mask)

        # Stack the results into batches
        if cropped_images:
            cropped_images = torch.stack(cropped_images).squeeze(1)  # Stack and remove extra dimension
            cropped_masks = torch.stack(cropped_masks)
        else:
            # If no masks were valid, return empty tensors
            cropped_images = torch.zeros((0, *image.shape[1:]), dtype=image.dtype)
            cropped_masks = torch.zeros((0, *masks.shape[1:]), dtype=masks.dtype)

        return (cropped_images, cropped_masks, crop_data)
        

class Image_Stitch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "blend_range": ("INT", {"default": 32, "min": 0, "max": 512}),
                "blend_mode": (["linear", "sine", "cubic", "quadratic", "hermite", "sine_quadratic_mix", "quadratic_sine_mix"], {"default": "sine"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "exec"
    CATEGORY = "🐐 GOAT Nodes/Image"
    DESCRIPTION = '''
    Stitches cropped images back together with an image.\n
    ‣ original_image | The original image to stitch the cropped images onto.\n
    ‣ cropped_images | A batch of cropped_images to stitch onto the original image.\n
    ‣ crop_data | Includes the target size as well as the coordinates of the cropped images.\n
    ‣ blend_range | Specifies the size of the blending region around the edges of each cropped image.\n
    ‣ blend_mode | Specifies the blending function to use when merging the cropped images with the original image.
    '''

    def exec(self, original_image, cropped_images, crop_data, blend_range, blend_mode):
        # Ensure the original image is in the correct format
        if original_image.dim() != 4:
            raise ValueError("Original image should be a 4D tensor (batch, height, width, channels).")

        # Ensure the cropped images are in the correct format
        if cropped_images.dim() != 4:
            raise ValueError("Cropped images should be a 4D tensor (batch, height, width, channels).")

        # Ensure the number of cropped images matches the crop_data
        if len(crop_data) != cropped_images.shape[0]:
            raise ValueError("The number of cropped images does not match the crop_data.")

        # Clone the original image to avoid modifying it directly
        stitched_image = original_image.clone()

        # Iterate over each cropped image and its corresponding crop_data
        for i, (cropped_image, crop_info) in enumerate(zip(cropped_images, crop_data)):
            # Ensure the cropped image is a 4D tensor (batch, height, width, channels)
            if cropped_image.dim() == 3:
                cropped_image = cropped_image.unsqueeze(0)  # Add batch dimension if missing

            # Get the crop coordinates and dimensions
            min_y, min_x = crop_info["square_bbox"][0], crop_info["square_bbox"][1]
            max_y, max_x = crop_info["square_bbox"][2], crop_info["square_bbox"][3]
            crop_height, crop_width = max_y - min_y, max_x - min_x

            # Use the original crop_size from crop_data to determine the target dimensions
            target_height = crop_info["crop_size"]
            target_width = crop_info["crop_size"]

            # Downscale the cropped image to the original crop_size
            if cropped_image.shape[1] != target_height or cropped_image.shape[2] != target_width:
                cropped_image = comfy.utils.common_upscale(
                    cropped_image.permute(0, 3, 1, 2),  # Change to (B, C, H, W)
                    target_width,
                    target_height,
                    "bilinear",
                    "disabled",
                ).permute(0, 2, 3, 1)  # Change back to (B, H, W, C)

            # Ensure the cropped image matches the expected dimensions
            if cropped_image.shape[1] != crop_height or cropped_image.shape[2] != crop_width:
                raise ValueError(
                    f"Cropped image {i} does not match the expected dimensions in crop_data. "
                    f"Expected: ({crop_height}, {crop_width}), Got: ({cropped_image.shape[1]}, {cropped_image.shape[2]})"
                )

            # Create a weight matrix for blending
            weight_matrix = self.create_weight_matrix(crop_height, crop_width, blend_range, blend_mode)

            # Blend the cropped image into the original image
            stitched_image[:, min_y:max_y, min_x:max_x, :] = (
                weight_matrix * cropped_image +
                (1 - weight_matrix) * stitched_image[:, min_y:max_y, min_x:max_x, :]
            )

        return (stitched_image,)

    def create_weight_matrix(self, height, width, blend_range, blend_mode):
        # Create a weight matrix with shape (height, width, 1)
        weight_matrix = torch.ones((height, width, 1), dtype=torch.float32)

        # Define the blending function based on the blend_type
        if blend_mode == "linear":
            blend_func = lambda x: x
        elif blend_mode == "sine":
            blend_func = lambda x: 0.5 - 0.5 * torch.cos(torch.tensor(math.pi * x))
        elif blend_mode == "cubic":
            blend_func = lambda x: -2 * x**3 + 3 * x**2
        elif blend_mode == "quadratic":
            blend_func = lambda x: x * (2 - x)
        elif blend_mode == "hermite":
            blend_func = lambda x: 3 * x**2 - 2 * x**3
        elif blend_mode == "sine_quadratic_mix":
            sine_func = lambda t: 0.5 - 0.5 * math.cos(math.pi * t)
            quadratic_func = lambda t: t * (2 - t)
            # More weight to Quadratic near the center (t ≈ 0.5), more to sine near the edges (t ≈ 0 or t ≈ 1)
            return lambda t: (1 - (2 * t - 1)**2) * quadratic_func(t) + (2 * t - 1)**2 * sine_func(t)
        elif blend_mode == "quadratic_sine_mix":
            sine_func = lambda t: 0.5 - 0.5 * math.cos(math.pi * t)
            quadratic_func = lambda t: t * (2 - t)
            # More weight to Sine near the center (t ≈ 0.5), more to Quadratic near the edges (t ≈ 0 or t ≈ 1)
            return lambda t: (1 - (2 * t - 1)**2) * sine_func(t) + (2 * t - 1)**2 * quadratic_func(t)
        else:
            raise ValueError(f"Unknown blend_type: {blend_mode}")

        # Apply weights to the edges
        for i in range(blend_range):
            weight = blend_func(float(i) / blend_range)
            # Top edge
            weight_matrix[i, :, :] *= weight
            # Bottom edge
            weight_matrix[-(i + 1), :, :] *= weight
            # Left edge
            weight_matrix[:, i, :] *= weight
            # Right edge
            weight_matrix[:, -(i + 1), :] *= weight

        return weight_matrix


NODE_CLASS_MAPPINGS = {
    "Image_Crop": Image_Crop,
    "Image_Stitch": Image_Stitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_Crop": "🐐 Image Crop",
    "Image_Stitch": "🐐 Image Stitch",
}