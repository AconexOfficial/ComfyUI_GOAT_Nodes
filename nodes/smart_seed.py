import random
import json
import os
from server import PromptServer
from aiohttp import web

def new_random_seed():
    return random.randint(0, 1125899906842624)

class Smart_Seed:
    def __init__(self):
        self.previous_seeds = [0, 0, 0, 0]
        self.current_seed = -1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 1125899906842624}),
                "generate_new": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("current_seed",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Math'
    DESCRIPTION = '''
    Generate a new seed or keep the current one.
    Displays and allows selection of the 4 previously used seeds.
    '''

    def exec(self, seed, generate_new):
        if seed in self.previous_seeds:
            self.current_seed = seed
        elif generate_new or seed == -1:
            self.current_seed = new_random_seed()
        else:
            self.current_seed = seed

        # Update previous seeds
        if self.current_seed not in self.previous_seeds:
            self.previous_seeds = [self.current_seed] + self.previous_seeds[:3]

        return (self.current_seed,)

    @classmethod
    def EXTRA_OUTPUTS(s):
        return {
            "ui": {
                "seeds": ("STRING", {"multiline": True}),
            }
        }

    def ui(self, **kwargs):
        seeds_display = json.dumps(self.previous_seeds)
        return {"seeds": seeds_display}

NODE_CLASS_MAPPINGS = {
    "Smart_Seed": Smart_Seed
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Smart_Seed": "🐐 Smart Seed"
}

# This part ensures the JavaScript file is served
@PromptServer.instance.routes.get("/smart_seed/smart_seed.js")
async def get_custom_js(request):
    js_path = os.path.join(os.path.dirname(__file__), "smart_seed.js")
    with open(js_path, "r") as file:
        content = file.read()
    return web.Response(text=content, content_type="application/javascript")