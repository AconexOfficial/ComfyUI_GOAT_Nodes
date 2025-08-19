import torch
import comfy.samplers
import comfy.utils
import comfy.sample
import comfy.sampler_helpers
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.ldm.modules.diffusionmodules import openaimodel
import latent_preview

# --- Helper Classes for Tiled Context Management ---

class TiledContextOptions:
    def __init__(self, use_coordinates, attn_strength, adain_strength):
        self.use_coordinates = use_coordinates
        self.attn_strength = attn_strength
        self.adain_strength = adain_strength

class TiledContextManager:
    """
    Handles patching and context extraction.
    This final version uses ComfyUI's internal sampler helper functions
    for maximum compatibility and robustness.
    """
    def __init__(self, guider):
        self.guider = guider
        self.model_patcher = guider.model_patcher
        self.patched_layers = {}
        self.context_banks = {}
        self.current_tile_info = None
        self.mode = "IDLE"

    def _find_and_patch_layers(self):
        """Recursively find all compatible layers and patch their forward methods."""
        for name, module in self.model_patcher.model.diffusion_model.named_modules():
            if isinstance(module, BasicTransformerBlock):
                if name not in self.patched_layers:
                    original_forward = module.forward
                    self.patched_layers[id(module)] = original_forward
                    module.forward = self._patched_attn_forward_factory(original_forward).__get__(module, module.__class__)
            elif isinstance(module, openaimodel.TimestepEmbedSequential) and hasattr(module, 'op') and isinstance(module.op, torch.nn.Conv2d):
                if name not in self.patched_layers:
                    original_forward = module.forward
                    self.patched_layers[id(module)] = original_forward
                    module.forward = self._patched_adain_forward_factory(original_forward).__get__(module, module.__class__)

    def prepare_global_context(self, global_context_latent, options: TiledContextOptions):
        """
        Performs the 'WRITE' pass by correctly preparing the model for sampling and
        using calc_cond_batch to run the forward pass.
        """
        print("\n[Tiled Context DEBUG] Starting global context preparation...")
        self._find_and_patch_layers()
        self.options = options
        self.mode = "WRITE"

        # This is the correct, robust way to prepare the model for a single forward pass.
        # It mirrors the setup process of the native ComfyUI samplers.
        inner_model, conds, loaded_models = comfy.sampler_helpers.prepare_sampling(
            self.model_patcher,
            global_context_latent['samples'].shape,
            self.guider.original_conds,
            self.guider.model_options
        )

        try:
            # pre_run is necessary to set up model.current_patcher, preventing the NoneType error.
            self.model_patcher.pre_run()

            positive_conds = conds.get("positive", None)
            # Add custom assertion for better error message
            assert positive_conds is not None, "[Tiled Context ERROR] Positive conditioning is missing from the guider. This should not happen."
            
            print(f"[Tiled Context DEBUG] Global Latent Shape: {global_context_latent['samples'].shape}")
            print(f"[Tiled Context DEBUG] Running 'calc_cond_batch' for WRITE pass...")

            # Use calc_cond_batch, which handles all model-specific logic internally.
            # We only need to process the positive conditioning to get our features.
            comfy.samplers.calc_cond_batch(
                inner_model,
                [positive_conds], # Note: Pass it as a list of cond lists
                global_context_latent['samples'],
                torch.tensor([999.0]),
                self.guider.model_options
            )
            
            print("[Tiled Context DEBUG] WRITE pass completed.")

        finally:
            # CRITICAL: Always clean up the model state and loaded models.
            self.model_patcher.cleanup()
            comfy.sampler_helpers.cleanup_models(conds, loaded_models)
            print("[Tiled Context DEBUG] Cleanup after WRITE pass completed.")

        self.mode = "READ"
        print("[Tiled Context DEBUG] Global context prepared successfully. Manager is now in READ mode.\n")

    def set_current_tile(self, tile_coords, full_latent_size):
        self.current_tile_info = {"coords": tile_coords, "full_latent_size": full_latent_size}

    def _patched_attn_forward_factory(self, original_forward):
        def patched_forward(instance, x, context=None, **kwargs):
            if self.mode == "WRITE":
                self.context_banks[id(instance)] = {'context_tensor': x.detach().clone()}
                return original_forward(x, context, **kwargs)

            if self.mode == "READ" and self.options.attn_strength > 0 and id(instance) in self.context_banks:
                attn_context = x if context is None else context
                banked_context = self.context_banks[id(instance)]['context_tensor'].to(x.device, dtype=x.dtype)

                if self.options.use_coordinates and self.current_tile_info:
                    downsample_factor = round(self.current_tile_info["full_latent_size"][1] / banked_context.shape[2])
                    if downsample_factor == 0: downsample_factor = 1
                    h = self.current_tile_info["coords"][1] // downsample_factor
                    w = self.current_tile_info["coords"][0] // downsample_factor
                    th, tw = x.shape[2], x.shape[3]
                    banked_context = banked_context[:, :, h:h+th, w:w+tw]
                else:
                    banked_context = comfy.utils.common_upscale(banked_context, x.shape[3], x.shape[2], "bilinear", "center")

                final_context = torch.lerp(attn_context, banked_context, self.options.attn_strength)
                return original_forward(x, context=final_context, **kwargs)

            return original_forward(x, context, **kwargs)
        return patched_forward

    def _patched_adain_forward_factory(self, original_forward):
        def patched_forward(instance, x, emb, **kwargs):
            original_out = original_forward(x, emb, **kwargs)
            if self.mode == "WRITE":
                var, mean = torch.var_mean(original_out, dim=(2, 3), keepdim=True, correction=0)
                self.context_banks[id(instance)] = {'mean': mean.detach().clone(), 'var': var.detach().clone()}
                return original_out

            if self.mode == "READ" and self.options.adain_strength > 0 and id(instance) in self.context_banks:
                eps = 1e-6
                tile_var, tile_mean = torch.var_mean(original_out, dim=(2, 3), keepdim=True, correction=0)
                tile_std = torch.maximum(tile_var, torch.zeros_like(tile_var) + eps) ** 0.5
                banked_mean = self.context_banks[id(instance)]['mean'].to(x.device, dtype=x.dtype)
                banked_var = self.context_banks[id(instance)]['var'].to(x.device, dtype=x.dtype)
                banked_mean = torch.mean(banked_mean, dim=(2,3), keepdim=True)
                banked_var = torch.mean(banked_var, dim=(2,3), keepdim=True)
                banked_std = torch.maximum(banked_var, torch.zeros_like(banked_var) + eps) ** 0.5
                context_styled_out = (((original_out - tile_mean) / tile_std) * banked_std) + banked_mean
                return torch.lerp(original_out, context_styled_out, self.options.adain_strength)

            return original_out
        return patched_forward

# --- The ComfyUI Node ---

class SamplerTiledContextAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "noise": ("NOISE", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "guider": ("GUIDER", ),
                    "latent_image_batch": ("LATENT", ),
                    "global_context_latent": ("LATENT", ),
                    "tile_data": ("TILE_DATA",),
                    "use_coordinates": (("enable", "disable"),),
                    "attn_ctx_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "adain_ctx_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                }}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "🐐 GOAT Nodes/Sampling"
    DESCRIPTION = "Advanced Sampler that processes latent tiles with awareness of a global context, for use with custom schedulers, guiders, etc."

    def sample(self, noise, sampler, sigmas, guider, latent_image_batch, global_context_latent, tile_data, use_coordinates, attn_ctx_strength, adain_ctx_strength):
        
        # This is the correct attribute for the ModelPatcher in a guider.
        original_model_patcher = guider.model_patcher
        
        try:
            # Clone the ModelPatcher for a safe, isolated environment.
            local_model_patcher = original_model_patcher.clone()
            
            # Temporarily replace the model in the guider with our clone.
            guider.model_patcher = local_model_patcher

            context_options = TiledContextOptions(
                use_coordinates=(use_coordinates == "enable"),
                attn_strength=attn_ctx_strength,
                adain_strength=adain_ctx_strength
            )
            context_manager = TiledContextManager(guider)
            
            context_manager.prepare_global_context(global_context_latent, context_options)

            final_height, final_width, tile_coordinates = tile_data
            num_tiles = latent_image_batch["samples"].shape[0]
            output_tiles = []
            
            pbar = comfy.utils.ProgressBar(num_tiles)
            
            for i in range(num_tiles):
                tile_coords = tile_coordinates[i]
                print(f"Tiled Context: Sampling tile {i + 1}/{num_tiles} at {tile_coords}...")
                
                context_manager.set_current_tile(tile_coords, (final_width, final_height))
                
                current_tile_latent = {"samples": latent_image_batch["samples"][i:i+1]}
                tile_noise = noise.generate_noise(current_tile_latent)
                
                x0_output = {}
                callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

                denoised_samples = guider.sample(tile_noise, current_tile_latent["samples"], sampler, sigmas,
                                                 denoise_mask=None, callback=callback, disable_pbar=True, seed=noise.seed)
                
                output_tiles.append(denoised_samples)
                pbar.update(1)

            final_latent_batch = torch.cat(output_tiles, dim=0)
            out_latent = {"samples": final_latent_batch.to(comfy.model_management.intermediate_device())}
            
            return (out_latent, out_latent)

        finally:
            # CRITICAL: Always restore the original model patcher to the guider.
            print("Tiled Context: Restoring original model patcher to guider.")
            guider.model_patcher = original_model_patcher


NODE_CLASS_MAPPINGS = {
    "SamplerTiledContextAdvanced": SamplerTiledContextAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerTiledContextAdvanced": "🐐 Sampler (Tiled Context Advanced)"
}