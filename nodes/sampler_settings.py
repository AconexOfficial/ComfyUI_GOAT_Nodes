class Sampler_Settings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("steps", "cfg", "denoise",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Input'
    DESCRIPTION = '''
    Allows selection of sampler settings: steps, CFG, and denoise. \n
    Outputs the selected values.
    '''


    def exec(self, steps, cfg, denoise):
        return (steps, cfg, denoise,)


NODE_CLASS_MAPPINGS = {
    "Sampler_Settings": Sampler_Settings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sampler_Settings": "🐐 Sampler Settings"
}