from pathlib import Path
import folder_paths

class Embedding_Selector:
    @classmethod
    def INPUT_TYPES(s):
        # Get the list of available embeddings
        embeddings = folder_paths.get_filename_list("embeddings")
        
        return {
            "required": {
                "embedding": ((embeddings),),  # Dropdown for selecting embeddings
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Input'
    DESCRIPTION = '''
    Picks an embedding and returns it as a string with the chosen strength.
    '''


    def exec(self, embedding, strength):
        # If emphasis is too low, return the original text
        if strength < 0.05:
            return ("",)
        
        else:
            # Extract the embedding name (without file extension)
            embedding_name = Path(embedding).stem
            
            # Format the embedding with strength if not 1.0
            if strength != 1.0:
                embedding_formatted = f"({embedding_name}:{strength:.3f})"
            else:
                embedding_formatted = embedding_name
            
            # Combine the embedding and text
            embedding_string = f"{embedding_formatted}, "
            
            return (embedding_string,)


NODE_CLASS_MAPPINGS = {
    "Embedding_Selector": Embedding_Selector
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Embedding_Selector": "🐐 Embedding Selector"
}