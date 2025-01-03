class Triple_Prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text1": ("STRING", {"default": "", "multiline": True}),
                "text2": ("STRING", {"default": "", "multiline": True}),
                "text3": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text1", "text1", "concatenated_text",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Input'
    DESCRIPTION = '''
    Takes three text inputs and outputs them either separately or as a concatenated version with automatic comma separation handling.
    '''

    def exec(self, text1, text2, text3):
        # Ensure each non-empty text ends with a comma if it doesn't already
        if text1 and not text1.endswith(','):
            text1 += ','
        if text2 and not text2.endswith(','):
            text2 += ','
        if text3 and not text3.endswith(','):
            text3 += ','
        
        # Concatenate the texts, skipping empty ones
        concatenated_text = ''.join([text for text in [text1, text2, text3] if text])
        
        return (text1, text2, text3, concatenated_text,)


NODE_CLASS_MAPPINGS = {
    "Triple_Prompt": Triple_Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Triple_Prompt": "🐐 Triple Prompt"
}