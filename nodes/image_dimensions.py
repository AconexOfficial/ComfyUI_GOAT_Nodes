class Image_Dimensions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 1,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 1,
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Input'
    DESCRIPTION = '''
    Allows selection or manual input of image dimensions (width and height). \n
    Outputs the selected values.
    '''


    def exec(self, width, height):
        return (width, height,)


NODE_CLASS_MAPPINGS = {
    "Image_Dimensions": Image_Dimensions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_Dimensions": "🐐 Image Dimensions"
}