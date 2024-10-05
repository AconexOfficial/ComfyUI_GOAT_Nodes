class Get_Side_Length_Of_Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "side_length": ("BOOLEAN", {"default": True, "label_on": "max", "label_off": "min"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Image'
    DESCRIPTION = '''
    Returns the length of the images longest (max) or shortest (min) side.\n
    '''


    def exec(self, image, side_length):
        image_width = image.shape[2]
        image_height = image.shape[1]
        
        if side_length == True:
            return (max(image_width, image_height),)
        else:
            return (min(image_width, image_height),)
        
        
NODE_CLASS_MAPPINGS = {
    "Get_Side_Length_Of_Image": Get_Side_Length_Of_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Get_Side_Length_Of_Image": "🐐 Get Side Length Of Image"
}