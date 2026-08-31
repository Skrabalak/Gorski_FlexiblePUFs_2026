#!/bin/python3/

##########
# Author: Joey Gorski
# Date: Aug 2024
##########

##########
# IMPORTS
##########

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from scipy.ndimage import binary_dilation, gaussian_filter, grey_opening, median_filter
from skimage.filters import threshold_local

if TYPE_CHECKING:
    from progress import ProgressTracker

from helper_functions import (
    is_black,
    is_green,
    is_red,
    save_image,
    write_to_file,
)

##########
# GLOBAL VARIABLES
##########

# pixel colors for testing
green_pixel = [0, 255, 0]
red_pixel = [255, 0, 0]
blue_pixel = [0, 0, 255]
black_pixel = [0, 0, 0]


##########
# FUNCTIONS
##########

# TODO: track particle movement
# steps:
# 1. find the particles within the red square
#     a. find the red square and create a new image of ONLY what it contains
#     b. find each grouping of pixels (each particle) that are not black
#     c. handle potential image grime (remaining pixel groups that are not particles)
# 2. assign each particle an id
# 3. find each particle in the successor picture
# 4. check distances between particles
# 5. give a final value for how much stretching was involved


##
# takes a large image and finds all red squares
#  returns a list of ndarrays, one for each red square found
#  if the image doesn't have any red squares, don't call this function because no error checking :)
##
def get_red_square_image(image: np.ndarray, save_dir: str) -> list[np.ndarray]:
    """Extract every red-bordered interior region from an image.

    Args:
        image: RGB image array or list-like image data.
        save_dir: Unused for output here; retained for API compatibility with
            the rest of the pipeline.

    Returns:
        A list of cropped ``np.uint8`` arrays, one for each detected red square.
        The red border itself is excluded from the returned content and any red
        pixels still encountered inside the crop are replaced with black.
    """
    # Convert to numpy array if it's a list
    if isinstance(image, list):
        # write_to_file("process_image.log", "Converting list to numpy array\n")
        image = np.array(image)

    image_backup = image.copy()

    # Track which pixels have been visited to avoid finding the same box multiple times
    visited = np.zeros((len(image), len(image[0])), dtype=bool)
    red_squares = []

    # write_to_file("process_image.log", f"Starting red square detection on image of size {len(image)}x{len(image[0])}\n")

    # Count red pixels for diagnostics
    red_pixel_count = 0
    sample_red_pixels = []
    potential_corners_checked = 0
    corner_check_failures = []

    # find all red squares (upper left and bottom right corners)
    for row_index, row in enumerate(image):
        for col_index, pixel in enumerate(row):
            # Check if pixel is red
            if pixel[0] >= 240 and pixel[1] <= 5 and pixel[2] <= 5:
                red_pixel_count += 1
                if len(sample_red_pixels) < 5:
                    sample_red_pixels.append(f"({row_index},{col_index}): {pixel}")

            # Skip if already visited or not red
            if visited[row_index][col_index] or not (pixel[0] >= 240 and pixel[1] <= 5 and pixel[2] <= 5):
                continue

            potential_corners_checked += 1

            # check if current pixel is upper left corner of a red box
            is_upper_left = True
            failure_reason = ""
            for modifier in range(10):
                if row_index + modifier >= len(image) or col_index + modifier >= len(image[0]):
                    is_upper_left = False
                    failure_reason = f"boundary check failed at modifier={modifier}"
                    break
                if not is_red(image[row_index + modifier][col_index]):
                    is_upper_left = False
                    pixel_below = image[row_index + modifier][col_index]
                    failure_reason = f"pixel below at ({row_index + modifier},{col_index}) is {pixel_below}, not red"
                    break
                if not is_red(image[row_index][col_index + modifier]):
                    is_upper_left = False
                    pixel_right = image[row_index][col_index + modifier]
                    failure_reason = f"pixel right at ({row_index},{col_index + modifier}) is {pixel_right}, not red"
                    break

            if not is_upper_left:
                if len(corner_check_failures) < 10:
                    corner_check_failures.append(f"Red pixel at ({row_index},{col_index}): {failure_reason}")
                continue

            # write_to_file("process_image.log", f"Found potential red square upper left at ({row_index}, {col_index})\n")
            upper_left_pixel_index = [row_index, col_index]

            # Find the bottom right corner of this box
            # Strategy: Find the first pixel going right that's not red (right edge)
            # Then find the first pixel going down from upper-left that's not red (bottom edge)
            # The bottom-right corner is where these edges meet
            bottom_right_pixel_index = None

            # Find right edge: scan right from upper-left until we hit a non-red pixel
            right_edge_col = col_index
            for scan_col in range(col_index, len(image[0])):
                if not is_red(image[row_index][scan_col]):
                    right_edge_col = scan_col - 1  # Last red pixel
                    break
            else:
                # Hit the edge of the image
                right_edge_col = len(image[0]) - 1

            # Find bottom edge: scan down from upper-left until we hit a non-red pixel
            bottom_edge_row = row_index
            for scan_row in range(row_index, len(image)):
                if not is_red(image[scan_row][col_index]):
                    bottom_edge_row = scan_row - 1  # Last red pixel
                    break
            else:
                # Hit the edge of the image
                bottom_edge_row = len(image) - 1

            # Bottom-right corner should be at the intersection
            potential_br_row = bottom_edge_row
            potential_br_col = right_edge_col

            # Verify this is actually a bottom-right corner (10 pixels up and left should be red)
            is_valid_bottom_right = True
            for modifier in range(10):
                if potential_br_row - modifier < row_index or potential_br_col - modifier < col_index:
                    is_valid_bottom_right = False
                    break
                if not is_red(image[potential_br_row - modifier][potential_br_col]):
                    is_valid_bottom_right = False
                    break
                if not is_red(image[potential_br_row][potential_br_col - modifier]):
                    is_valid_bottom_right = False
                    break

            if is_valid_bottom_right:
                bottom_right_pixel_index = [potential_br_row, potential_br_col]

            if bottom_right_pixel_index is None:
                # write_to_file(
                #     "process_image.log",
                #     f"Warning: No bottom right corner found for upper left at ({row_index}, {col_index})\n",
                # )
                continue

            # write_to_file(
            #     "process_image.log",
            #     f"Found red square bottom right at ({bottom_right_pixel_index[0]}, {bottom_right_pixel_index[1]})\n",
            # )

            # Mark all pixels in this red box as visited
            ulr, ulc = upper_left_pixel_index
            brr, brc = bottom_right_pixel_index
            for r, row_arr in enumerate(image[ulr : brr + 1], start=ulr):
                for c, px in enumerate(row_arr[ulc : brc + 1], start=ulc):
                    if is_red(px):
                        visited[r][c] = True

            # final upper left and bottom right modifications
            upper_left_pixel_index[0] += 3
            upper_left_pixel_index[1] += 3
            bottom_right_pixel_index[0] -= 1
            bottom_right_pixel_index[1] -= 1

            # create a copy of what is inside this red square
            square_image = []
            height = abs(upper_left_pixel_index[0] - bottom_right_pixel_index[0])
            width = abs(upper_left_pixel_index[1] - bottom_right_pixel_index[1])
            # write_to_file("process_image.log", f"Extracting red square content: {height}x{width} pixels\n")

            for row_mod in range(height):
                square_image.append([])
                for col_mod in range(width):
                    r_idx = upper_left_pixel_index[0] + row_mod
                    c_idx = upper_left_pixel_index[1] + col_mod
                    # append pixel to new image if the pixel is NOT red. If it IS red, append black
                    if not is_red(image[r_idx][c_idx]):
                        square_image[row_mod].append(image[r_idx][c_idx])
                    else:
                        square_image[row_mod].append(black_pixel)

            # Add this square to our list
            red_squares.append(np.array(square_image, dtype=np.uint8))
            # write_to_file("process_image.log", f"Successfully extracted red square #{len(red_squares)}\n")

    # restore the original image in case it was modified during processing
    image = image_backup.copy()

    write_to_file("process_image.log", f"Red square detection complete. Found {len(red_squares)} red squares\n")

    # return all found red squares
    return red_squares


##
# takes a large image and processes it to remove everything outside of the green square
##
def get_green_square_image(image: np.ndarray, save_dir: str) -> np.ndarray | None:
    """
    Extract the interior of the green-bordered region from an image.

    Args:
        image: RGB image array or list-like image data.
        save_dir: Unused for image output here; kept so the function matches the
            rest of the processing pipeline.

    Returns:
        The cropped interior as a ``np.uint8`` array, or ``None`` if no valid
        green border is detected.
    """
    # Convert to numpy array if it's a list
    if isinstance(image, list):
        image = np.array(image)

    # Track visited pixels to handle multiple potential green regions
    visited = np.zeros((len(image), len(image[0])), dtype=bool)

    # Find all green pixels and track extremes
    green_pixels_found = 0
    min_row = len(image)
    min_col = len(image[0])
    max_row = -1
    max_col = -1

    # First pass: find bounding box of all green pixels
    for row_index in range(len(image)):
        for col_index in range(len(image[0])):
            pixel = image[row_index][col_index]
            if is_green(pixel):
                green_pixels_found += 1
                min_row = min(min_row, row_index)
                min_col = min(min_col, col_index)
                max_row = max(max_row, row_index)
                max_col = max(max_col, col_index)

    # Validation: Did we find any green pixels?
    if green_pixels_found == 0:
        write_to_file("process_image.log", "ERROR: No green pixels found in image\n")
        print("Green box not found! No green pixels detected.")
        return None

    write_to_file(
        "process_image.log",
        f"Found {green_pixels_found} green pixels. Bounding box: ({min_row}, {min_col}) to ({max_row}, {max_col})\n",
    )

    # Validation: Check that we have a reasonable rectangle
    height = max_row - min_row
    width = max_col - min_col

    if height < 10 or width < 10:
        write_to_file(
            "process_image.log", f"ERROR: Green region too small ({height}x{width}). Likely not a valid border.\n"
        )
        print(f"Green box not found! Region too small: {height}x{width}")
        return None

    # Validation: Verify top and bottom edges have continuous green pixels
    # Check top edge
    top_edge_green_count = sum(1 for c in range(min_col, max_col + 1) if is_green(image[min_row][c]))
    # Check bottom edge
    bottom_edge_green_count = sum(1 for c in range(min_col, max_col + 1) if is_green(image[max_row][c]))
    # Check left edge
    left_edge_green_count = sum(1 for r in range(min_row, max_row + 1) if is_green(image[r][min_col]))
    # Check right edge
    right_edge_green_count = sum(1 for r in range(min_row, max_row + 1) if is_green(image[r][max_col]))

    # At least 80% of each edge should be green (allowing for some gaps)
    edge_threshold = 0.8
    if (
        top_edge_green_count < width * edge_threshold
        or bottom_edge_green_count < width * edge_threshold
        or left_edge_green_count < height * edge_threshold
        or right_edge_green_count < height * edge_threshold
    ):
        write_to_file(
            "process_image.log",
            f"WARNING: Green border validation failed. "
            f"Edge coverage: top={top_edge_green_count}/{width}, "
            f"bottom={bottom_edge_green_count}/{width}, "
            f"left={left_edge_green_count}/{height}, "
            f"right={right_edge_green_count}/{height}\n",
        )

    # Determine border thickness by scanning inward from edges
    # Scan from top edge downward
    border_thickness_top = 0
    for offset in range(min(10, height // 2)):
        if is_green(image[min_row + offset][min_col + width // 2]):
            border_thickness_top = offset + 1
        else:
            break

    # Scan from left edge rightward
    border_thickness_left = 0
    for offset in range(min(10, width // 2)):
        if is_green(image[min_row + height // 2][min_col + offset]):
            border_thickness_left = offset + 1
        else:
            break

    # Use the maximum detected border thickness, default to 1 if detection fails
    border_thickness = max(border_thickness_top, border_thickness_left, 1)

    write_to_file("process_image.log", f"Detected border thickness: {border_thickness} pixels\n")

    # Apply border offset to get interior region
    # Add extra pixel for safety margin
    upper_left_pixel_index = [min_row + border_thickness, min_col + border_thickness]
    bottom_right_pixel_index = [max_row - border_thickness, max_col - border_thickness]

    # Validation: Make sure we still have content after removing borders
    final_height = bottom_right_pixel_index[0] - upper_left_pixel_index[0]
    final_width = bottom_right_pixel_index[1] - upper_left_pixel_index[1]

    if final_height <= 0 or final_width <= 0:
        write_to_file(
            "process_image.log",
            f"ERROR: No content remaining after border removal. Final dimensions: {final_height}x{final_width}\n",
        )
        print("Green box not found! Border too thick for image size.")
        return None

    write_to_file(
        "process_image.log",
        f"Extracting green square content: {final_height}x{final_width} pixels "
        f"from ({upper_left_pixel_index[0]}, {upper_left_pixel_index[1]}) "
        f"to ({bottom_right_pixel_index[0]}, {bottom_right_pixel_index[1]})\n",
    )

    # Extract the region (inclusive of both corners)
    row_start, col_start = upper_left_pixel_index
    row_end, col_end = bottom_right_pixel_index
    square_image = image[row_start : row_end + 1, col_start : col_end + 1]

    return np.array(square_image, dtype=np.uint8)


##
# takes a row and column and an image and returns a unique identifier for that pixel
##
def coord_to_identifier(row: int, col: int, image: np.ndarray) -> int:
    """Convert a 2D image coordinate into a unique linear union-find index.

    Args:
        row: Pixel row index.
        col: Pixel column index.
        image: Image whose width defines the row stride.

    Returns:
        A stable integer ID for the coordinate, or ``0`` for an empty image.
    """
    if len(image) == 0:
        return 0
    width = len(image[0])
    return (row * width) + col


##
# takes a grayscale image and replaces groupings of white with black
##
def remove_large_aggregations_grayscale(image: np.ndarray, save_dir: str) -> np.ndarray:
    """Remove large connected bright regions from a grayscale image.

    Pixels above ``mean(image) * 1.5`` are treated as white, grouped with
    four-neighbour connectivity, and any connected component larger than the
    hardcoded maximum size is replaced with black.

    Args:
        image: Grayscale image array modified in place.
        save_dir: Unused; retained for API compatibility.

    Returns:
        The same array instance after large bright components have been zeroed.
    """
    # variables
    MAX_ACCEPTABLE_SIZE = 100
    current_pixel = 0

    uf_size = len(image) * len(image[0]) + 1
    # print("uf_size:", uf_size)

    union_find = DisjointSet(range(uf_size))

    # Define white threshold for grayscale
    # Pixels above this threshold are considered "white"
    WHITE_THRESHOLD = np.mean(image) * 1.5

    # group all WHITE pixels together into clusters
    # print("grouping pixels")
    for row_idx, row in enumerate(image):
        for col_idx, pixel in enumerate(row):
            current_pixel += 1
            pixel_id = coord_to_identifier(row_idx, col_idx, image)

            # For grayscale, pixel is just a number, not an array
            pixel_is_white = pixel > WHITE_THRESHOLD

            # Only merge white pixels with adjacent white pixels
            # This creates clusters of connected white pixels
            if pixel_is_white:
                # above
                if row_idx - 1 >= 0:
                    above_is_white = image[row_idx - 1][col_idx] > WHITE_THRESHOLD
                    if above_is_white:
                        union_find.merge(pixel_id, coord_to_identifier(row_idx - 1, col_idx, image))
                # below
                if row_idx + 1 < len(image):
                    below_is_white = image[row_idx + 1][col_idx] > WHITE_THRESHOLD
                    if below_is_white:
                        union_find.merge(pixel_id, coord_to_identifier(row_idx + 1, col_idx, image))
                # left
                if col_idx - 1 >= 0:
                    left_is_white = image[row_idx][col_idx - 1] > WHITE_THRESHOLD
                    if left_is_white:
                        union_find.merge(pixel_id, coord_to_identifier(row_idx, col_idx - 1, image))
                # right
                if col_idx + 1 < len(row):
                    right_is_white = image[row_idx][col_idx + 1] > WHITE_THRESHOLD
                    if right_is_white:
                        union_find.merge(pixel_id, coord_to_identifier(row_idx, col_idx + 1, image))

    # print uf
    # print("union_find:")
    # print(union_find.subsets())

    # if a WHITE cluster size is greater than MAX_ACCEPTABLE_SIZE then make every pixel in that cluster black
    # print("removing large white clusters")
    current_pixel = 0
    for row_idx, row in enumerate(image):
        for col_idx, pixel in enumerate(row):
            # if current_pixel % 1000 == 0:
            #     print("current pixel: ", current_pixel)
            current_pixel += 1
            pixel_id = coord_to_identifier(row_idx, col_idx, image)

            # Only replace pixels that are white AND part of a large cluster
            pixel_is_white = pixel > WHITE_THRESHOLD
            if pixel_is_white and union_find.subset_size(pixel_id) > MAX_ACCEPTABLE_SIZE:
                image[row_idx][col_idx] = 0  # Black for grayscale

    return image


##
# takes an image and replaces groupings of color with black
##
def remove_large_aggregations(image: np.ndarray, save_dir: str) -> np.ndarray:
    """Remove large connected components from an RGB image.

    Pixels are grouped by whether they are classified as black or non-black,
    then any component larger than the hardcoded maximum size is replaced with
    black.

    Args:
        image: RGB image array modified in place.
        save_dir: Unused; retained for API compatibility.

    Returns:
        The same image array after oversized components have been painted black.
    """
    # variables
    MAX_ACCEPTABLE_SIZE = 20
    current_pixel = 0

    uf_size = len(image) * len(image[0]) + 1
    # print("uf_size:", uf_size)

    union_find = DisjointSet(range(uf_size))

    # group all pixels as "color" or "black"
    # print("grouping pixels")
    for row_idx, row in enumerate(image):
        for col_idx, pixel in enumerate(row):
            # if current_pixel % 1000 == 0:
            #     print("current pixel: ", current_pixel)
            current_pixel += 1
            pixel_id = coord_to_identifier(row_idx, col_idx, image)

            # above
            if row_idx - 1 >= 0 and is_black(pixel) == is_black(image[row_idx - 1][col_idx]):
                union_find.merge(pixel_id, coord_to_identifier(row_idx - 1, col_idx, image))
            # below
            if row_idx + 1 < len(image) and is_black(pixel) == is_black(image[row_idx + 1][col_idx]):
                union_find.merge(pixel_id, coord_to_identifier(row_idx + 1, col_idx, image))
            # left
            if col_idx - 1 >= 0 and is_black(pixel) == is_black(image[row_idx][col_idx - 1]):
                union_find.merge(pixel_id, coord_to_identifier(row_idx, col_idx - 1, image))
            # right
            if col_idx + 1 < len(row) and is_black(pixel) == is_black(image[row_idx][col_idx + 1]):
                union_find.merge(pixel_id, coord_to_identifier(row_idx, col_idx + 1, image))

    # print uf
    # print("union_find:")
    # print(union_find.subsets())

    # if group size is greater than MAX_ACCEPTABLE_SIZE then make every pixel black
    # print("removing large groups of pixels")
    current_pixel = 0
    for row_idx, row in enumerate(image):
        for col_idx, pixel in enumerate(row):
            # if current_pixel % 1000 == 0:
            #     print("current pixel: ", current_pixel)
            current_pixel += 1
            pixel_id = coord_to_identifier(row_idx, col_idx, image)

            if union_find.subset_size(pixel_id) > MAX_ACCEPTABLE_SIZE:
                image[row_idx][col_idx] = black_pixel

    return image


def flatten_image(image: np.ndarray) -> np.ndarray:
    """Apply a 3x3 median filter while preserving the input channel layout.

    Args:
        image: Grayscale, RGB, or RGBA image array.

    Returns:
        A filtered image with the same shape as the input. For RGBA images the
        alpha channel is copied through unchanged.
    """
    # Apply median filter to each color channel separately

    # Handle RGBA images - only process RGB channels
    if len(image.shape) == 3 and image.shape[2] == 4:
        # RGBA image - process only RGB, preserve alpha
        processed_image = np.zeros_like(image)
        for channel in range(3):  # Only RGB
            processed_image[:, :, channel] = median_filter(image[:, :, channel], size=3)
        processed_image[:, :, 3] = image[:, :, 3]  # Copy alpha channel as-is
    elif len(image.shape) == 3:
        # RGB image
        processed_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            processed_image[:, :, channel] = median_filter(image[:, :, channel], size=3)
    else:
        # Grayscale image
        processed_image = median_filter(image, size=3)

    return processed_image


# Process and capture intermediate steps
def process_image(
    image: np.ndarray,
    save_dir: str,
    save_step_images: bool = False,
    progress: "ProgressTracker | None" = None,
    progress_prefix: str = "",
) -> np.ndarray:
    """Run the preprocessing pipeline that isolates likely particle pixels.

    The pipeline applies median filtering, grayscale conversion, large-cluster
    removal, halo smoothing, background estimation, adaptive thresholding,
    RGB masking, another aggregation-removal pass, and edge cleanup.

    Args:
        image: Cropped RGB image to process.
        save_dir: Directory used for optional step-image output.
        save_step_images: When ``True``, save a PNG for each processing step.
        progress: Optional progress tracker updated for each pipeline stage.
        progress_prefix: Prefix prepended to each progress sub-phase label.

    Returns:
        A processed RGB ``np.ndarray`` in which non-particle content has been
        driven toward black.
    """
    _TOTAL_STEPS = 11  # steps 0–10

    def _step(n: int, label: str) -> None:
        """Update the optional sub-phase progress display for one pipeline step."""
        if progress is not None:
            progress.set_sub_phase(f"{progress_prefix}{label}", n + 1, _TOTAL_STEPS)

    # Approximate size of particles in pixels
    # NOTE: this must be an odd number for the adaptive thresholding block size to work properly
    PARTICLE_SIZE = 15
    BRIGHTNESS_OFFSET = -3  # Adjust as needed for sensitivity of adaptive thresholding
    PRINT_ALL_STEPS_DEBUG = save_step_images

    fig, ax = plt.subplots(nrows=1, ncols=1)

    try:
        # Original
        _step(0, "original")
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_0_original", save_dir, fig=fig)

        # Step 1: Apply median filter to reduce noise while preserving edges
        _step(1, "median filter")
        filtered = flatten_image(image)
        ax.imshow(filtered, cmap="gray")
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_1_median_filter", save_dir, fig=fig)

        # Step 2: Convert to grayscale for easier processing
        # Use standard luminance conversion: 0.299*R + 0.587*G + 0.114*B
        # Convert RGB to grayscale using weighted average (better than min/max)
        _step(2, "grayscale")
        grayscale = (0.299 * filtered[:, :, 0] + 0.587 * filtered[:, :, 1] + 0.114 * filtered[:, :, 2]).astype(np.uint8)
        ax.imshow(grayscale, cmap="gray")
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_2_grayscale", save_dir, fig=fig)

        # Step 3: Remove large aggregations of white pixels that interfere with background estimation
        _step(3, "remove aggregations")
        grayscale_before_removal = grayscale.copy()
        grayscale = remove_large_aggregations_grayscale(grayscale, save_dir)
        ax.imshow(grayscale, cmap="gray")
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_3_remove_large_aggregations", save_dir, fig=fig)

        # Step 4: Smooth halo artifacts left by large aggregation removal
        _step(4, "halo smoothing")
        zeroed_mask = (grayscale_before_removal > 0) & (grayscale == 0)
        halo_mask = binary_dilation(zeroed_mask, iterations=6)
        blurred = gaussian_filter(grayscale.astype(np.float32), sigma=3).astype(np.uint8)
        grayscale = np.where(halo_mask, blurred, grayscale).astype(np.uint8)
        ax.imshow(grayscale, cmap="gray")
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_4_halo_smoothing", save_dir, fig=fig)

        # Step 5: Estimate background using morphological opening
        _step(5, "background estimate")
        # Create disk-like structuring element (circular kernel)
        # Using a larger kernel to capture background variations
        y, x = np.ogrid[-PARTICLE_SIZE : PARTICLE_SIZE + 1, -PARTICLE_SIZE : PARTICLE_SIZE + 1]
        kernel = x**2 + y**2 <= PARTICLE_SIZE**2

        # Estimate background using morphological opening
        background = grey_opening(grayscale, structure=kernel)
        ax.imshow(background, cmap="gray")
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_5_background_estimate", save_dir, fig=fig)

        # Step 6: Subtract background from grayscale image to get foreground
        _step(6, "background subtract")
        # Process foreground pixel by pixel based on background strength
        # Background strength determines what percentage of the original pixel to keep
        # If background = 255 (white/bright), keep 100% of pixel
        # If background = 127, keep ~50% of pixel
        # If background = 0 (black/dark), keep 0% of pixel

        # Calculate the percentage to keep for EACH pixel individually (0.0 to 1.0)
        # background_strength is a 2D array where each element is the percentage for that pixel
        background_strength = background.astype(np.float32) / 255.0

        # Apply the percentage to EACH pixel in the grayscale image
        # This multiplies grayscale[i,j] * background_strength[i,j] for every pixel (i,j)
        foreground = (grayscale.astype(np.float32) * background_strength).astype(np.uint8)

        ax.imshow(foreground, cmap="gray")
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_6_background_weighted_foreground", save_dir, fig=fig)

        # Step 7: Use adaptive thresholding to drop the background
        _step(7, "adaptive threshold")
        # Calculate local threshold for each pixel based on surrounding region
        # block_size: size of local neighborhood (must be odd)
        # method: 'gaussian' uses weighted average, 'mean' uses simple average
        # offset: constant subtracted from threshold (lower = more sensitive)
        local_thresh = threshold_local(
            foreground, block_size=PARTICLE_SIZE, method="gaussian", offset=BRIGHTNESS_OFFSET
        )

        # Apply adaptive threshold
        binary = (foreground > local_thresh).astype(np.uint8)
        binary_image = binary * 255  # Scale for visualization
        ax.imshow(binary_image, cmap="gray")
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_7_adaptive_thresholding", save_dir, fig=fig)

        # Step 8: Convert back to RGB for display (get colors from the original image, only where binary mask is 1)
        _step(8, "restore RGB")
        rgb_result = np.zeros_like(image)
        for c in range(3):  # For R, G, B channels
            rgb_result[:, :, c] = np.where(binary == 1, filtered[:, :, c], 0)
        ax.imshow(rgb_result)
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_8_rgb_result", save_dir, fig=fig)
        # Step 9: Remove large aggregations of color pixels that interfere with particle detection
        _step(9, "remove color aggregations")
        aggregationless_result = remove_large_aggregations(rgb_result, save_dir)
        ax.imshow(aggregationless_result)
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_9_remove_large_aggregations_color", save_dir, fig=fig)

        # Step 10: Sanitize the edges of the image
        _step(10, "sanitize edges")
        final_result = remove_outer_noise(aggregationless_result)
        ax.imshow(final_result)
        ax.axis("off")
        if PRINT_ALL_STEPS_DEBUG:
            save_image("step_10_final_result_edges_sanitized", save_dir, fig=fig)

        return final_result
    finally:
        plt.close(fig)


##
#  takes an image and removes noise from the outside edges
#  practically, this just means making the outer n pixels on each edge black
##
def remove_outer_noise(image: np.ndarray) -> np.ndarray:
    """Black out a fixed-width border around the image.

    Args:
        image: RGB image array.

    Returns:
        A copy of ``image`` with the outer five pixels on every edge replaced
        with black.
    """
    OUTER_PIXELS_TO_REMOVE = 5

    cleaned_image = image.copy()
    # Top edge
    cleaned_image[:OUTER_PIXELS_TO_REMOVE, :] = black_pixel
    # Bottom edge
    cleaned_image[-OUTER_PIXELS_TO_REMOVE:, :] = black_pixel
    # Left edge
    cleaned_image[:, :OUTER_PIXELS_TO_REMOVE] = black_pixel
    # Right edge
    cleaned_image[:, -OUTER_PIXELS_TO_REMOVE:] = black_pixel

    return cleaned_image
