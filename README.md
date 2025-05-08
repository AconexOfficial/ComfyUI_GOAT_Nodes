# 🐐 ComfyUI GOAT Nodes 🐐

(README still under construction!)

Welcome to the **ComfyUI GOAT Nodes**! 
These custom nodes both cover things, which had to be done through the use of multiple nodes or simply things, that were not performant enough and had other problems. These nodes currently are categorized into Image manipulation, Math utilities, and Postprocessing effects and Input.

## Table of Contents
- [Node Overview](#node-overview)
- [Custom Nodes](#custom-nodes)
  - [Image](#image)
    - [🐐 Image Tiler](#-image-tiler)
    - [🐐 Image Untiler](#-image-untiler)
    - [🐐 Get Side Length Of Image](#-get-side-length-of-image)
    - [🐐 Advanced Upscale Image (using Model)](#-advanced-upscale-image-using-model)
    - [🐐 Image Crop](#-image-crop)
    - [🐐 Image Stitch](#-image-stitch)
  - [Math](#math)
    - [🐐 Capped Int (Positive)](#-capped-int-positive)
    - [🐐 Capped Float (Positive)](#-capped-float-positive)
    - [🐐 Int Divide (Rounded)](#-int-divide-rounded)
  - [Postprocessing](#postprocessing)
    - [🐐 Fast Film Grain](#-fast-film-grain)
    - [🐐 Fast Color Match](#-fast-color-match)
  - [Input](#input)
    - [🐐 Triple Prompt](#-triple-prompt)
    - [🐐 Sampler Settings](#-sampler-settings)
	- [🐐 Image Dimensions](#-image-dimensions)
    - [🐐 Embedding Selector](#-embedding-picker)
- [Installation](#installation)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

## Node Overview

| Category          | Node Name                           | Description                                                   |
|-------------------|-------------------------------------|---------------------------------------------------------------|
| **Image**         | 🐐 **Image Tiler**                  | Breaks an image into smaller tiles for processing.             |
|                   | 🐐 **Image Untiler**                | Seamlessly reassembles tiles into a full image.               |
|                   | 🐐 **Get Side Length Of Image**     | Retrieves the wanted side length of an input image.           |
|                   | 🐐 **Advanced Upscale Image**       | Upscales an image using an upscale model.                     |
|                   | 🐐 **Image Crop**    		  | Crops an image based on a batch of proviced masks.             |
|                   | 🐐 **Image Stitch**	          | Stitches previously cropped images back onto the original image.|
| **Math**          | 🐐 **Capped Int (Positive)**        | Restricts an integer input within a positive range.            |
|                   | 🐐 **Capped Float (Positive)**      | Restricts a float input within a positive range.               |
|                   | 🐐 **Int Divide (Rounded)**         | Divides two integers and rounds the result to an integer.      |
| **Postprocessing**| 🐐 **Fast Film Grain**              | Quickly adds realistic film grain to an image.                 |
|                   | 🐐 **Fast Color Match**             | Quickly matches colors between two images.                     |
| **Input**         | 🐐 **Triple Prompt**                | Takes three text inputs and concatenates them. |
|				    | 🐐 **Sampler Settings**             | Allows selection of sampler settings: steps, CFG, and denoise. |
|                   | 🐐 **Image Dimensions**             | Allows selection or manual input of image dimensions (width and height). |
| 					| 🐐 **Embedding Selector**             | Allows the selection of an embedding and it's strength. |

## Custom Nodes

### Image

#### 🐐 Image Tiler
**Description**:  
_Add a detailed description of how the Tiler node works here._

**Example workflow**:  
![Example workflow for the Image Tiler and Image Untiler nodes](https://raw.githubusercontent.com/AconexOfficial/ComfyUI_GOAT_Nodes/refs/heads/main/workflows/image/image_tiler_AND_image_untiler.png)

---

#### 🐐 Image Untiler
**Description**:  
_Add a detailed description of how the Untiler node works here._

**Example workflow**:  
See the included workflow for 🐐 Image Tiler

---

#### 🐐 Get Side Length Of Image
**Description**:  
_Add a detailed description of how this node retrieves the side length of an image._

**Example workflow**:  
![Example workflow for the Get Side Length Of Image node](https://raw.githubusercontent.com/AconexOfficial/ComfyUI_GOAT_Nodes/refs/heads/main/workflows/image/get_side_length_of_image.png)

---

#### 🐐 Advanced Upscale Image (using Model)
**Description**:  
_Add a detailed description of how this node performs advanced upscaling using a machine learning model._

**Example workflow**:  
![Example workflow for the Advanced Upscale image (using Model) node](https://raw.githubusercontent.com/AconexOfficial/ComfyUI_GOAT_Nodes/refs/heads/main/workflows/image/advanced_upscale_image_using_model.png)

---

#### 🐐 Image Crop
**Description**:  
_Add a detailed description of how this node works._

**Example workflow**:  

---

#### 🐐 Image Stitch
**Description**:  
_Add a detailed description of how this node works._

**Example workflow**:  

---


### Math

#### 🐐 Capped Int (Positive)
**Description**:  
_Add a detailed description of the Capped Int (Positive) node, which restricts integers to a positive range._

**Example workflow**:  
![Example workflow for the Capped Int node](https://raw.githubusercontent.com/AconexOfficial/ComfyUI_GOAT_Nodes/refs/heads/main/workflows/math/capped_int.png)

---

#### 🐐 Capped Float (Positive)
**Description**:  
_Add a detailed description of the Capped Float (Positive) node, which restricts float values to a positive range._

**Example workflow**:  
![Example workflow for the Capped Int node](https://raw.githubusercontent.com/AconexOfficial/ComfyUI_GOAT_Nodes/refs/heads/main/workflows/math/capped_float.png)

---

#### 🐐 Int Divide (Rounded)
**Description**:  
_Add a detailed description of how the Int Divide (Rounded) node divides integers and rounds the result._

**Example workflow**:  
![Example workflow for the Int Divide (Rounded) node](https://raw.githubusercontent.com/AconexOfficial/ComfyUI_GOAT_Nodes/refs/heads/main/workflows/math/int_divide_rounded.png)

---

### Postprocessing

#### 🐐 Fast Film Grain
**Description**:  
_Add a detailed description of how this node adds fast, realistic film grain to an image._

**Example workflow**:  
![Example workflow for the Fast Film Grain node](https://raw.githubusercontent.com/AconexOfficial/ComfyUI_GOAT_Nodes/refs/heads/main/workflows/postprocessing/fast_film_grain.png)

---

#### 🐐 Fast Color Match
**Description**:  
_Add a detailed description of how this node quickly matches the color profile of one image to another._

**Example workflow**:  
![Example workflow for the Fast Color Match node](https://raw.githubusercontent.com/AconexOfficial/ComfyUI_GOAT_Nodes/refs/heads/main/workflows/postprocessing/fast_color_match.png)

---

### Input

#### 🐐 Triple Prompt
**Description**:  
Takes three text inputs and outputs them separately along with a concatenated version. Adds a comma as separation if there isn't already one at the end of a text. If a text field is empty, no unnecessary comma is added.

**Example workflow**:  
_Add an example workflow image or description here._

---

#### 🐐 Sampler Settings
**Description**:  
Allows selection of sampler settings: steps (0-100), CFG (0.0-100.0), and denoise (0.0-1.0). Outputs the selected values for use in other nodes.

**Example workflow**:  
_Add an example workflow image or description here._

---

#### 🐐 Image Dimensions
**Description**:  
Allows selection or manual input of image dimensions (width and height).

**Example workflow**:  
_Add an example workflow image or description here._

---

#### 🐐 Embedding Selector
**Description**:  
Picks an embedding from a dropdown list and applies returns it as a string with its selected strength. If the strength is too low, the embedding is ignored.

**Example workflow**:  
_Add an example workflow image or description here._

---

## Installation

To install these custom nodes, do one of the following things below:

<em>Method 1: Clone the repository into your custom nodes folder</em>

```bash
git clone https://github.com/AconexOfficial/ComfyUI_GOAT_Nodes.git
```

Or use ComfyUI-Manager to install

## Changelog

Version 1.1.1
- Added 2 new nodes
   - Image Crop
   - Image Stitch
- Fixed output naming in Triple Prompt

Version 1.1
- Added 4 new nodes
  - Triple Prompt
  - Sampler Settings
  - Image Dimensions
  - Embedding Selector
