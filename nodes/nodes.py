from ComfyUI_GOAT_Nodes.nodes.image_tiler import Image_Tiler, Image_Untiler # type: ignore
from ComfyUI_GOAT_Nodes.nodes.get_side_length_of_image import Get_Side_Length_Of_Image # type: ignore
from ComfyUI_GOAT_Nodes.nodes.capped_int_positive import Capped_Int_Positive # type: ignore
from ComfyUI_GOAT_Nodes.nodes.capped_float_positive import Capped_Float_Positive # type: ignore
from ComfyUI_GOAT_Nodes.nodes.int_divide_rounded import Int_Divide_Rounded # type: ignore
from ComfyUI_GOAT_Nodes.nodes.fast_film_grain import Fast_Film_Grain # type: ignore
from ComfyUI_GOAT_Nodes.nodes.advanced_upscale_image_using_model import Advanced_Upscale_Image_Using_Model # type: ignore
from ComfyUI_GOAT_Nodes.nodes.fast_color_match import Fast_Color_Match # type: ignore
from ComfyUI_GOAT_Nodes.nodes.triple_prompt import Triple_Prompt # type: ignore
from ComfyUI_GOAT_Nodes.nodes.embedding_selector import Embedding_Selector # type: ignore
from ComfyUI_GOAT_Nodes.nodes.sampler_settings import Sampler_Settings # type: ignore
from ComfyUI_GOAT_Nodes.nodes.image_dimensions import Image_Dimensions # type: ignore
from ComfyUI_GOAT_Nodes.nodes.image_crop import Image_Crop # type: ignore
from ComfyUI_GOAT_Nodes.nodes.image_crop import Image_Stitch # type: ignore


NODE_CLASS_MAPPINGS = {
    "Image_Tiler": Image_Tiler,
    "Image_Untiler": Image_Untiler,
    "Get_Side_Length_Of_Image": Get_Side_Length_Of_Image,
    "Capped_Int_Positive": Capped_Int_Positive,
    "Capped_Float_Positive": Capped_Float_Positive,
    "Int_Divide_Rounded": Int_Divide_Rounded,
    "Fast_Film_Grain": Fast_Film_Grain,
    "Advanced_Upscale_Image_Using_Model": Advanced_Upscale_Image_Using_Model,
    "Fast_Color_Match": Fast_Color_Match,
    "Triple_Prompt": Triple_Prompt,
    "Embedding_Selector": Embedding_Selector,
    "Sampler_Settings": Sampler_Settings,
    "Image_Dimensions": Image_Dimensions,
    "Image_Crop": Image_Crop,
    "Image_Stitch": Image_Stitch,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_Tiler": "🐐 Image Tiler",
    "Image_Untiler": "🐐 Image Untiler",
    "Get_Side_Length_Of_Image": "🐐 Get Side Length Of Image",
    "Capped_Int_Positive": "🐐 Capped Int (Positive)",
    "Capped_Float_Positive": "🐐 Capped Float (Positive)",
    "Int_Divide_Rounded": "🐐 Int Divide (Rounded)",
    "Fast_Film_Grain": "🐐 Fast Film Grain",
    "Advanced_Upscale_Image_Using_Model": "🐐 Advanced Upscale Image (using Model)",
    "Fast_Color_Match": "🐐 Fast Color Match",
    "Triple_Prompt": "🐐 Triple Prompt",
    "Embedding_Selector": "🐐 Embedding Selector",
    "Sampler_Settings": "🐐 Sampler Settings",
    "Image_Dimensions": "🐐 Image Dimensions",
    "Image_Crop": "🐐 Image Crop",
    "Image_Stitch": "🐐 Image Stitch",
}
