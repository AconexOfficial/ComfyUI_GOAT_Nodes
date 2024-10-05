import torch


def supports_amp():
    # Check if the GPU supports AMP (requires compute capability >= 7.0)
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return capability[0] >= 7  # AMP is supported on compute capability 7.0 and above (Volta and newer)
    return False


class Fast_Color_Match:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                "adaptive_matching": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Postprocessing'
    DESCRIPTION = '''
    Matches the colors of the image to those of a reference image.
    '''


    def match_channel(self, input_channel, ref_channel, strength, adaptive_enabled):
        # Fixed base strength
        base_strength = 1.0

        # Calculate means and standard deviations
        input_mean = input_channel.float().mean()
        input_std = input_channel.float().std()
        ref_mean = ref_channel.float().mean()
        ref_std = ref_channel.float().std()

        # Normalize the input channel
        normalized = (input_channel - input_mean) / (input_std + 1e-8)

        # Create the matched channel based on the reference stats
        matched_channel = normalized * ref_std + ref_mean

        # Compute adaptive strength if enabled
        if adaptive_enabled == True:
            # Calculate differences for adaptive strength
            mean_diff = torch.abs(input_mean - ref_mean)
            std_diff = torch.abs(input_std - ref_std)

            # Use weights for differences
            weight_mean = 2.0  # Weight for mean difference
            weight_std = 1.0   # Weight for standard deviation difference

            # Compute the adjustment factor based on differences
            adjustment_factor = (weight_mean * mean_diff + weight_std * std_diff) / (weight_mean + weight_std + 1e-5)

            # Compute adaptive strength based on the base strength and adjustment factor
            adaptive_strength = base_strength * (1 - adjustment_factor)

            # Scale adaptive strength by the input strength parameter
            final_strength = strength * adaptive_strength
            
            # Clamp final strength between 0 and 1
            final_strength = torch.clamp(final_strength, 0, 1)

        else:
            # If adaptive strength is disabled, just use fixed strength
            final_strength = strength

        # Perform linear interpolation with final strength
        return torch.lerp(input_channel, matched_channel, final_strength)


    def exec(self, image, reference, strength, adaptive_matching, device):
        if strength == 0:
            return (image,)

        amp_enabled = supports_amp()
        if amp_enabled == False and device == "cuda":
            amp_enabled == True
        elif amp_enabled == True and device == "cpu":
            amp_enabled == False

        if amp_enabled:
            image = image.cuda()
            reference = reference.cuda()

        # Ensure inputs are float tensors in range [0, 1]
        image = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()
        reference = reference.float() / 255.0 if reference.dtype == torch.uint8 else reference.float()

        matched_images = []
        for i in range(image.shape[0]):
            img = image[i].permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            ref = reference[i].permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

            # Use autocast to enable mixed precision
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # Match each channel separately with fixed base strength and input strength
                matched_r = self.match_channel(img[0], ref[0], strength, adaptive_matching)
                matched_g = self.match_channel(img[1], ref[1], strength, adaptive_matching)
                matched_b = self.match_channel(img[2], ref[2], strength, adaptive_matching)

                # Reconstruct the image by stacking the matched channels
                matched_image = torch.stack([matched_r, matched_g, matched_b], dim=0)  # Shape (C, H, W)
                matched_image = matched_image.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

            matched_images.append(matched_image)

        # Stack the processed images back into a batch
        result = torch.stack(matched_images, dim=0)

        # Ensure the output is in the range [0, 1]
        result = torch.clamp(result, 0, 1).cpu()

        # Convert back to the original format (uint8 if it was originally uint8)
        if image.dtype == torch.uint8:
            result = (result * 255.0).round().byte()

        return (result,)


NODE_CLASS_MAPPINGS = {
    "Fast_Color_Match": Fast_Color_Match
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Fast_Color_Match": "🐐 Fast Color Match"
}
