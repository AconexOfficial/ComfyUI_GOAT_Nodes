class Int_Divide_Rounded:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("INT",),
                "b": ("INT",),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Math'
    DESCRIPTION = '''
    Divides a with b and rounds the result to the nearest int. \n
    '''


    def exec(self, a, b):
        # Ensure inputs are treated as floats before division
        return (int(round(a / b)),)


NODE_CLASS_MAPPINGS = {
    "Int_Divide_Rounded": Int_Divide_Rounded
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Int_Divide_Rounded": "🐐 Int Divide (Rounded)"
}