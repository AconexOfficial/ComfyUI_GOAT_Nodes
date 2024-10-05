class Capped_Int_Positive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "cap": ("INT", {"default": 10, "min": 0, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Math'
    DESCRIPTION = '''
    Caps the integer at a specific number. A cap of 0 equals uncapped. \n
    '''


    def exec(self, int, cap):
        if cap == 0:
            return (int,)
        
        return (min(int, cap),)


NODE_CLASS_MAPPINGS = {
    "Capped_Int_Positive": Capped_Int_Positive
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Capped_Int_Positive": "🐐 Capped Int (Positive)"
}