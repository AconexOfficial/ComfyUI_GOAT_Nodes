import sys
import os
import torch
import math


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


def radial_order_tiling(tiles, image_width, image_height, tile_width, tile_height):
    # Get the center of the image
    center_x = image_width // 2
    center_y = image_height // 2
    
    # Function to calculate the Euclidean distance from the tile center to the image center
    def tile_distance_from_center(tile):
        tile_center_x = tile[0] + tile_width // 2
        tile_center_y = tile[1] + tile_height // 2
        return math.sqrt((tile_center_x - center_x) ** 2 + (tile_center_y - center_y) ** 2)
    
    # Sort the tiles by radial distance (Euclidean distance) from the image center
    tiles = sorted(tiles, key=tile_distance_from_center)
    
    # Reverse the order so that the center is processed last
    tiles = tiles[::-1]
    
    return tiles


def checkerboard_order_tiling(tiles):
    # Separate tiles into black and white groups for odd and even tiles
    white_tiles = tiles[1::2]
    black_tiles = tiles[0::2]

    
    return white_tiles + black_tiles


def spiral_order_tiling(tiles):
    # Create a set to hold tile coordinates for easy lookup
    tile_set = set(tiles)
    
    # Initialize the spiral order list
    spiral_tiles = []
    
    # Determine the bounds of the grid based on tile coordinates
    min_x = min(tile[0] for tile in tiles)
    max_x = max(tile[0] for tile in tiles)
    min_y = min(tile[1] for tile in tiles)
    max_y = max(tile[1] for tile in tiles)

    left, right = min_x, max_x
    top, bottom = min_y, max_y

    while left <= right and top <= bottom:
        # Traverse from left to right along the top row
        for i in range(left, right + 1):
            tile = (i, top)
            if tile in tile_set:
                spiral_tiles.append(tile)
        top += 1

        # Traverse from top to bottom along the right column
        for i in range(top, bottom + 1):
            tile = (right, i)
            if tile in tile_set:
                spiral_tiles.append(tile)
        right -= 1

        # Traverse from right to left along the bottom row, if applicable
        if top <= bottom:
            for i in range(right, left - 1, -1):
                tile = (i, bottom)
                if tile in tile_set:
                    spiral_tiles.append(tile)
            bottom -= 1

        # Traverse from bottom to top along the left column, if applicable
        if left <= right:
            for i in range(bottom, top - 1, -1):
                tile = (left, i)
                if tile in tile_set:
                    spiral_tiles.append(tile)
            left += 1

    return spiral_tiles


def row_order_tiling(tiles):
    # Sort tiles by their y-coordinate, then x-coordinate
    return sorted(tiles, key=lambda tile: (tile[1], tile[0]))


def column_order_tiling(tiles):
    # Sort tiles by their x-coordinate, then y-coordinate
    return sorted(tiles, key=lambda tile: (tile[0], tile[1]))


def diagonal_order_tiling(tiles):
    # Sort tiles by the sum of their x and y coordinates
    return sorted(tiles, key=lambda tile: (tile[0] + tile[1]))


def generate_tiles(
    image_width, image_height, tile_width, tile_height, row_overlap, col_overlap, row_offset=0, col_offset=0, tiling_mode="radial"
):
    tiles = []

    y = 0
    row_count = 0  # To track which row we're on
    while y < image_height:
        if y == 0:
            next_y = y + tile_height - col_overlap
        else:
            next_y = y + tile_height - col_overlap

        if y + tile_height >= image_height:
            y = max(image_height - tile_height, 0)
            next_y = image_height

        x = 0
        col_count = 0  # To track which column we're on
        while x < image_width:
            if x == 0:
                next_x = x + tile_width - row_overlap
            else:
                next_x = x + tile_width - row_overlap

            if x + tile_width >= image_width:
                x = max(image_width - tile_width, 0)
                next_x = image_width

            # Apply row/col offset: Shift each row/column by the specified amount
            x_offset = col_count * row_offset  # Offset for each column
            y_offset = row_count * col_offset  # Offset for each row

            tile_x = x + x_offset
            tile_y = y + y_offset

            # Ensure tiles stay within the image bounds
            if tile_x + tile_width > image_width:
                tile_x = max(image_width - tile_width, 0)
            if tile_y + tile_height > image_height:
                tile_y = max(image_height - tile_height, 0)

            # Append tile only if it's within the image
            if tile_x < image_width and tile_y < image_height:
                tiles.append((tile_x, tile_y))

            if next_x > image_width:
                break
            x = next_x
            col_count += 1  # Increment column count after placing tile

        if next_y > image_height:
            break
        y = next_y
        row_count += 1  # Increment row count after processing each row

    # Choose the tile ordering based on the tiling mode
    if tiling_mode == "radial":
        return radial_order_tiling(tiles, image_width, image_height, tile_width, tile_height)
    elif tiling_mode == "checkerboard":
        return checkerboard_order_tiling(tiles)
    elif tiling_mode == "spiral":
        return spiral_order_tiling(tiles)
    elif tiling_mode == "row":
        return row_order_tiling(tiles)
    elif tiling_mode == "column":
        return column_order_tiling(tiles)
    elif tiling_mode == "diagonal":
        return diagonal_order_tiling(tiles)
    else:
        raise ValueError(f"Unknown tiling mode: {tiling_mode}")


class Image_Tiler:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "tile_height": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "row_overlap": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "col_overlap": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "row_offset": ("INT", {"default": 0, "min": -8192, "max": 8192}),  # New row offset
                "col_offset": ("INT", {"default": 0, "min": -8192, "max": 8192}),  # New column offset
                "tiling_mode": (["radial", "checkerboard", "spiral", "row", "column", "diagonal"], {"default": "radial"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_DATA", "INT",)
    RETURN_NAMES = ("IMAGES", "TILE_DATA", "TILE_COUNT",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Image'
    DESCRIPTION = '''
    Splits an image into tiles.\n
    ‣ tile_width | Sets the width of the tiles to the specified value.\n
    ‣ tile_height | Sets the height of the tiles to the specified value.\n
    ‣ row_overlap | Specifies the amount of overlap of information between horizontal adjacent tiles.\n
    ‣ col_overlap | Specifies the amount of overlap of information between vertical adjacent tiles.\n
    ‣ row_offset | Shifts each row by the specified amount (f.e. to remove inconvenient horizontal tile edges).\n
    ‣ col_offset | Shifts each column by the specified amount (f.e. to remove inconvenient vertical tile edges).\n
    ‣ tiling_mode | Specifies the order in which the tiles are processed.
    '''


    def exec(self, image, tile_width, tile_height, row_overlap, col_overlap, row_offset, col_offset, tiling_mode):
        image_height = image.shape[1]
        image_width = image.shape[2]

        self.debugger(image_width, image_height, tile_width, tile_height, row_overlap, col_overlap, row_offset, col_offset)

        tile_coordinates = generate_tiles(image_width, image_height, tile_width, tile_height, row_overlap, col_overlap, row_offset, col_offset, tiling_mode)

        print("🐐 Image Tiler: Tile coordinates: {}".format(tile_coordinates))

        image_tiles = []
        image_count = 0
        for tile_coordinate in tile_coordinates:
            image_count += 1
            image_tile = image[
                :,
                tile_coordinate[1] : tile_coordinate[1] + tile_height,
                tile_coordinate[0] : tile_coordinate[0] + tile_width,
                :,
            ]

            image_tiles.append(image_tile)

        tiles_tensor = torch.stack(image_tiles).squeeze(1)
        tile_data = (image_height, image_width, tile_coordinates)

        return (tiles_tensor, tile_data, image_count)
    
    
    @staticmethod
    def debugger(image_width, image_height, tile_width, tile_height, row_overlap, col_overlap, row_offset, col_offset):
        if image_width < tile_width:
            raise ValueError(f"tile_width: {tile_width} is greater than image_width: {image_width}")
        if image_height < tile_height:
            raise ValueError(f"tile_height: {tile_height} is greater than image_height: {image_height}")
        if row_overlap >= tile_width:
            raise ValueError(f"row_overlap: {row_overlap} is greater than or equal to tile_width: {tile_width}")
        if col_overlap >= tile_height:
            raise ValueError(f"col_overlap: {col_overlap} is greater than or equal to tile_height: {tile_height}")
        if abs(row_offset) > tile_width - row_overlap - 2:
            raise ValueError(f"row_offset: {row_offset} is greater than the remaining unused overlap space: {tile_width - row_overlap - 2}")
        if abs(col_offset) > tile_height - col_overlap - 2:
            raise ValueError(f"col_offset: {col_offset} is greater than the remaining unused overlap space: {tile_height - col_overlap - 2}")


class Image_Untiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_data": ("TILE_DATA",),
                "blend_mode": (["linear", "sine", "cubic", "quadratic", "hermite", "sine_quadratic_mix", "quadratic_sine_mix"], {"default": "sine"}),
                "blend_range": ("INT", {"default": 128, "min": 0, "max": 8192}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "exec"
    CATEGORY = '🐐 GOAT Nodes/Image'
    DESCRIPTION = '''
    Merges tiles into an image.\n
    ‣ blend_range | Specifies the size of the blending region around the edges of each tile.\n
    ‣ blend_mode | Specifies the blending function to use when merging the tiles.
    '''
    
    def exec(self, images, blend_range, tile_data, blend_mode="sine"):
        final_height, final_width, tile_coordinates = tile_data
        tile_height, tile_width = images.shape[1], images.shape[2]

        original_shape = (1, final_height, final_width, 3)
        output = torch.zeros(original_shape, dtype=images.dtype)
        weight_sum = torch.zeros(original_shape, dtype=images.dtype)

        weight_func = self.get_weight_function(blend_mode)

        for index, (x, y) in enumerate(tile_coordinates):
            image_tile = images[index]
            
            # Calculate blending weights
            weight_matrix = torch.ones((tile_height, tile_width, 3), dtype=images.dtype)
            
            for i in range(blend_range):
                weight = weight_func(float(i) / blend_range)
                if x > 0:
                    weight_matrix[:, i, :] *= weight  # Left edge
                if y > 0:
                    weight_matrix[i, :, :] *= weight  # Top edge
                if x + tile_width < final_width:
                    weight_matrix[:, -(i + 1), :] *= weight  # Right edge
                if y + tile_height < final_height:
                    weight_matrix[-(i + 1), :, :] *= weight  # Bottom edge

            # Apply the weight matrix to the tile
            weighted_tile = image_tile * weight_matrix

            # Add the weighted tile to the output
            output[:, y:y + tile_height, x:x + tile_width, :] += weighted_tile
            weight_sum[:, y:y + tile_height, x:x + tile_width, :] += weight_matrix

        # Normalize the output by dividing by the sum of weights
        output = torch.where(weight_sum > 0, output / weight_sum, output)

        return [output]
    
    
    @staticmethod
    def get_weight_function(blend_mode):
        if blend_mode == "linear":
            return lambda t: t
        elif blend_mode == "sine":
            return lambda t: 0.5 - 0.5 * math.cos(math.pi * t)
        elif blend_mode == "cubic":
            return lambda t: -2 * t**3 + 3 * t**2
        elif blend_mode == "quadratic":
            return lambda t: t * (2 - t)
        elif blend_mode == "hermite":
            return lambda t: 3 * t**2 - 2 * t**3
        elif blend_mode == "sine_quadratic_mix":
            sine_func = lambda t: 0.5 - 0.5 * math.cos(math.pi * t)
            quadratic_func = lambda t: t * (2 - t)
            # More weight to Quadratic near the center (t ≈ 0.5), more to sine near the edges (t ≈ 0 or t ≈ 1)
            return lambda t: (1 - (2 * t - 1)**2) * quadratic_func(t) + (2 * t - 1)**2 * sine_func(t)
        elif blend_mode == "quadratic_sine_mix":
            sine_func = lambda t: 0.5 - 0.5 * math.cos(math.pi * t)
            quadratic_func = lambda t: t * (2 - t)
            # More weight to Sine near the center (t ≈ 0.5), more to Quadratic near the edges (t ≈ 0 or t ≈ 1)
            return lambda t: (1 - (2 * t - 1)**2) * sine_func(t) + (2 * t - 1)**2 * quadratic_func(t)
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")


NODE_CLASS_MAPPINGS = {
    "Image_Tiler": Image_Tiler,
    "Image_Untiler": Image_Untiler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_Tiler": "🐐 Image Tiler",
    "Image_Untiler": "🐐 Image Untiler",
}
