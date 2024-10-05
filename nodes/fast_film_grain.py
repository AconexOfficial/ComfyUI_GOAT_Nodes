import torch


def supports_amp():
    # Check if the GPU supports AMP (requires compute capability >= 7.0)
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return capability[0] >= 7  # AMP is supported on compute capability 7.0 and above (Volta and newer)
    return False


def apply_gaussian_grain_(image, strength, seed, multiplier):
    mul_strength = strength * multiplier
    torch.manual_seed(seed)
    noise = torch.randn_like(image) * mul_strength
    return torch.clamp(image + noise, 0, 1)


def apply_gaussian_grain_cuda(image, strength, seed, multiplier):
    mul_strength = strength * multiplier
    # Check if AMP is supported
    amp_enabled = supports_amp()

    # Use autocast only if AMP is supported
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        image = image.cuda()
        torch.manual_seed(seed)
        noise = torch.randn_like(image) * mul_strength
        return torch.clamp(image + noise, 0, 1).cpu()


def apply_fft_grain(image, strength, seed, multiplier):
    torch.manual_seed(seed)
    noise = torch.randn_like(image)
    
    mul_strength = strength * multiplier

    # FFT of noise
    noise_fft = torch.fft.fft2(noise, dim=(1, 2))
    
    # Create a proper low-pass filter
    height, width = image.shape[1], image.shape[2]
    y_freq = torch.fft.fftfreq(height)[:, None]
    x_freq = torch.fft.fftfreq(width)[None, :]
    dist = torch.sqrt(x_freq**2 + y_freq**2)
    low_pass_filter = torch.exp(-dist**2 / (2 * 0.01**2))  # Adjust 0.01 to control filter strength
    
    # Expand dimensions to match noise_fft
    low_pass_filter = low_pass_filter.unsqueeze(0).unsqueeze(-1).expand_as(noise_fft)

    # Apply low-pass filter
    filtered_noise_fft = noise_fft * low_pass_filter
    filtered_noise = torch.fft.ifft2(filtered_noise_fft, dim=(1, 2)).real

    grain = (filtered_noise - filtered_noise.mean()) * mul_strength
    return torch.clamp(image + grain, 0, 1)


def apply_fft_grain_cuda(image, strength, seed, multiplier):
    # Check if AMP is supported
    amp_enabled = supports_amp()

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        image = image.cuda()
        torch.manual_seed(seed)
        noise = torch.randn_like(image)

        mul_strength = strength * multiplier

        noise_fft = torch.fft.fft2(noise, dim=(1, 2))

        height, width = image.shape[1], image.shape[2]
        y_freq = torch.fft.fftfreq(height).cuda()[:, None]
        x_freq = torch.fft.fftfreq(width).cuda()[None, :]
        dist = torch.sqrt(x_freq**2 + y_freq**2)
        low_pass_filter = torch.exp(-dist**2 / (2 * 0.01**2))

        low_pass_filter = low_pass_filter.unsqueeze(0).unsqueeze(-1).expand_as(noise_fft)

        filtered_noise_fft = noise_fft * low_pass_filter
        filtered_noise = torch.fft.ifft2(filtered_noise_fft, dim=(1, 2)).real

        grain = (filtered_noise - filtered_noise.mean()) * mul_strength
        return torch.clamp(image + grain, 0, 1).cpu()


def apply_mixed_grain(image, strength, seed, multiplier):
    torch.manual_seed(seed)
    noise = torch.randn_like(image)

    # FFT of noise
    noise_fft = torch.fft.fft2(noise, dim=(1, 2))
    
    mul_strength = strength * multiplier
    noise_mix = 0.2
    
    # Create a proper low-pass filter
    height, width = image.shape[1], image.shape[2]
    y_freq = torch.fft.fftfreq(height)[:, None]
    x_freq = torch.fft.fftfreq(width)[None, :]
    dist = torch.sqrt(x_freq**2 + y_freq**2)
    low_pass_filter = torch.exp(-dist**2 / (2 * 0.01**2))  # Adjust 0.01 to control filter strength
    
    # Expand dimensions to match noise_fft
    low_pass_filter = low_pass_filter.unsqueeze(0).unsqueeze(-1).expand_as(noise_fft)

    # Apply low-pass filter
    filtered_noise_fft = noise_fft * low_pass_filter
    filtered_noise = torch.fft.ifft2(filtered_noise_fft, dim=(1, 2)).real

    # Mix random noise with filtered noise
    mixed_noise = noise_mix * noise + (1 - noise_mix) * filtered_noise
    grain = (mixed_noise - mixed_noise.mean()) * mul_strength
    return torch.clamp(image + grain, 0, 1)


def apply_mixed_grain_cuda(image, strength, seed, multiplier):
    # Check if AMP is supported
    amp_enabled = supports_amp()
    if amp_enabled == True: print("AMP is enabled")

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        image = image.cuda()
        torch.manual_seed(seed)
        noise = torch.randn_like(image)

        noise_fft = torch.fft.fft2(noise, dim=(1, 2))

        noise_mix = 0.2
        mul_strength = strength * multiplier

        height, width = image.shape[1], image.shape[2]
        y_freq = torch.fft.fftfreq(height).cuda()[:, None]
        x_freq = torch.fft.fftfreq(width).cuda()[None, :]
        dist = torch.sqrt(x_freq**2 + y_freq**2)
        low_pass_filter = torch.exp(-dist**2 / (2 * 0.01**2))

        low_pass_filter = low_pass_filter.unsqueeze(0).unsqueeze(-1).expand_as(noise_fft)

        filtered_noise_fft = noise_fft * low_pass_filter
        filtered_noise = torch.fft.ifft2(filtered_noise_fft, dim=(1, 2)).real

        mixed_noise = noise_mix * noise + (1 - noise_mix) * filtered_noise
        grain = (mixed_noise - mixed_noise.mean()) * mul_strength
        return torch.clamp(image + grain, 0, 1).cpu()


class Fast_Film_Grain:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.1, "min": 0, "max": 1.0, "step": 0.01}),
                "grain": (["gaussian", "gaussian_cuda", "fft", "fft_cuda", "mixed", "mixed_cuda"], {"default": "mixed_cuda"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Postprocessing'
    DESCRIPTION = '''
    Applies film grain to the image with a strength between 0 and 1. \n
    ‣ gaussian | Simple and fast noise generation.\n
    ‣ gaussian_cuda | Simplest and fastest noise generation. (GPU accelerated)\n
    ‣ fft | Random noise generation with low-pass filtering using Fast Fourier transform.\n
    ‣ fft_cuda | Random noise generation with low-pass filtering using Fast Fourier transform. (GPU accelerated)\n
    ‣ mixed | Mixed noise generation combining both gaussian and ftt.\n
    ‣ mixed_cuda | Mixed noise generation combining both gaussian and ftt. (GPU accelerated)\n
    '''


    def exec(self, image, strength, grain, seed):
        grained_image = image
        scaled_strength = strength * 0.25
        
        if grain == "gaussian":
            grained_image = apply_gaussian_grain_(image, scaled_strength, seed, 0.5)
        elif grain == "gaussian_cuda":
            grained_image = apply_gaussian_grain_cuda(image, scaled_strength, seed, 0.5)
        elif grain == "fft":
            grained_image = apply_fft_grain(image, scaled_strength, seed, 5)
        elif grain == "fft_cuda":
            grained_image = apply_fft_grain_cuda(image, scaled_strength, seed, 5)
        elif grain == "mixed":
            grained_image = apply_mixed_grain(image, scaled_strength, seed, 2.5)
        elif grain == "mixed_cuda":
            grained_image = apply_mixed_grain_cuda(image, scaled_strength, seed, 2.5)
            
        return (grained_image,)


NODE_CLASS_MAPPINGS = {
    "Fast_Film_Grain": Fast_Film_Grain
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Fast_Film_Grain": "🐐 Fast Film Grain"
}