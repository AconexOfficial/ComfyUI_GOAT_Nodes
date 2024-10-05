class Capped_Float_Positive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1125899906842624.0}),
                "cap": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1125899906842624.0}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Math'
    DESCRIPTION = '''
    Caps the (positive) float at a specific number. A cap of 0 equals uncapped. \n
    '''


    def exec(self, float, cap):
        if cap == 0:
            return (float,)
        
        return (min(float, cap),)


NODE_CLASS_MAPPINGS = {
    "Capped_Float_Positive": Capped_Float_Positive
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Capped_Float_Positive": "🐐 Capped Float (Positive)"
}