import json
import os
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

##########
# GLOBAL VARIABLES
##########

white_minimum_pixel_value = 58  # pixels with all channels above this value are considered "white"
white_pixel_cutoff = 10  # if a pixel is surrounded by at least this many white pixels, it can be dropped

COLOR_THRESHOLD = 50  # drops pixels with total brightness below this value

IMAGE_DPI = 2400  # duh

# scaling factor applied to the median K-th nearest-neighbour distance to derive the connection search radius
NN_RADIUS_MULTIPLIER = 1.5
# number of nearest neighbours (K) used to derive the connection search radius in get_nearest_neighbor_search_radius()
NN_K = 3
# fallback radius (pixels) used when fewer than K+1 particles are detected
NN_FALLBACK_RADIUS = 200

MIN_PARTICLE_SIZE = 1  # minimum number of pixels for a particle to be considered valid

# number of best percentages to display in the "best percentages" image per red box image
BEST_PERCENTAGES_TO_DISPLAY = 1

# whether to draw individual best percentage images for each red box, in addition to the conglomerate image with all annotations combined.
DRAW_INDIVIDUAL_BEST_PERCENTAGES = False

matplotlib.use("Agg")

DEBUG_LOG_FILENAMES = {"process_image.log", "timing.log"}
DEBUG_LOGS_ENABLED = os.environ.get("SIP_DEBUG_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}

date = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# Module-level lock and counter used by save_image and save_io_image to produce
# unique, monotonically-increasing filenames without racing on os.listdir().
_file_counter_lock = threading.Lock()
_file_counter: dict[str, int] = {}  # keyed by save_dir so each directory has its own counter

# matplotlib's global figure registry (Gcf) is not thread-safe; serialise all
# savefig + plt.close calls with this lock so concurrent directory threads
# don't race on shared matplotlib state.
_matplotlib_lock = threading.Lock()


def _next_file_index(save_dir: str) -> int:
    """Return a unique, monotonically-increasing integer for save_dir.

    Uses a per-directory counter protected by _file_counter_lock so that
    concurrent threads never receive the same index and thus never produce
    colliding filenames.  Replaces the previous os.listdir()-based approach
    which was non-atomic and caused files to be silently overwritten.
    """
    with _file_counter_lock:
        idx = _file_counter.get(save_dir, 0)
        _file_counter[save_dir] = idx + 1
    return idx


def save_image(name: str, save_dir: str, fig: Optional[plt.Figure] = None) -> None:
    """Save a matplotlib figure to a named PNG in save_dir.

    The filename is prefixed with a timestamp (module-level `date`) and a
    sequential counter that is unique per directory even under concurrent calls.

    NOTE: this function should only be called when the image is prepped and
    ready to be printed. For maximum safety, don't call it unless necessary
    because this helper performs minimal error checking--it assumes the
    figure is ready and everything is "on the up and up".

    Args:
        name (str): Short name for the image (used in the filename).
        save_dir (str): Directory to save the image into. The caller is
            responsible for ensuring the figure is ready to be saved.
        fig (Optional[plt.Figure]): Figure to save. If omitted, uses the current
            matplotlib figure.

    Side Effects:
        Creates ``save_dir`` if needed, writes a PNG file, and closes the saved
        figure under a module-level matplotlib lock.
    """
    # ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    num_images = _next_file_index(save_dir)
    plot_name = f"{date}_{name}_num{num_images:03d}.png"

    # Use provided figure or current figure
    fig_to_save = fig if fig is not None else plt.gcf()
    # If an Axes was passed in, get its Figure so savefig is available.
    if not hasattr(fig_to_save, "savefig") and hasattr(fig_to_save, "figure"):
        fig_to_save = fig_to_save.figure
    with _matplotlib_lock:
        try:
            fig_to_save.savefig(
                os.path.join(save_dir, plot_name),
                format="png",
                dpi=IMAGE_DPI,
                bbox_inches="tight",
                transparent=True,
                pad_inches=0,
            )
        except Exception as e:
            write_to_file("main.log", f"ERROR: failed to save figure for {name}\n{e}")
            raise
        try:
            plt.close(fig_to_save)
        except Exception as e:
            write_to_file("main.log", f"ERROR: failed to close figure for {name}\n{e}")
            raise


def save_io_image(name: str, image: np.ndarray, save_dir: str, best_percentages_image: bool = False) -> None:
    """Save a NumPy image array to disk as a PNG.

    This writes a uint8 PNG using imageio.v3 and names the file with a
    timestamped prefix and a sequential number that is unique per directory
    even under concurrent calls.

    This is intended for the specific pixel-format used by the project; note
    that some other image IO helpers (historically, SciPy's image writers)
    have been incompatible — so stick to this helper for project images.

    Args:
        name (str): Short name for the image (used in the filename).
        image (array-like): Image data (H x W x C) that will be converted to
            uint8 before writing.
        save_dir (str): Directory to write the image into; it will be
            created if it does not exist.
        best_percentages_image (bool): When ``True``, omit the numeric suffix so
            the caller gets a stable filename for aggregate best-percentage
            outputs.

    Returns:
        The filename that was written inside ``save_dir``.
    """
    # ensure that the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    num_images = _next_file_index(save_dir)
    if not best_percentages_image:
        plot_name = f"{date}_{name}_num{num_images}.png"
    else:
        plot_name = f"{date}_{name}.png"
        # this was originally different, but after getting towards the end of the project, I realized the num being first just makes the flow of images harder to follow.
        # plot_name = f"{date}_num{num_images}_{name}.png"
    output_path = os.path.join(save_dir, plot_name)
    iio.imwrite(output_path, np.array(image, dtype=np.uint8), dpi=(IMAGE_DPI, IMAGE_DPI))
    return plot_name


def write_to_file(logfilename: str, string: str, print_to_terminal: bool = False) -> None:
    """Append a string to a log file in the repository root.

    Args:
        logfilename (str): Filename (relative to the project root) to append to.
        string (str): Text to append to the file. A newline is not added
            automatically; callers should include one if required.
        print_to_terminal (bool): If True, also print the string to stdout.
            Defaults to False.
    """
    if logfilename in DEBUG_LOG_FILENAMES and not DEBUG_LOGS_ENABLED:
        return

    outfile = Path(__file__).parent / logfilename
    with outfile.open("a") as f:
        if print_to_terminal:
            print(string)
        f.write(string)


def load_io_image(image_path: str | Path) -> np.ndarray:
    """Load one image from disk and return it as a NumPy array.

    Args:
        image_path: Path to an image readable by ``skimage.io.imread``.

    Returns:
        numpy.ndarray: Loaded image array.
    """
    return io.imread(image_path)


def load_io_images(image_paths: list[str]) -> list[np.ndarray]:
    """Load images from a list of file paths and return them as NumPy arrays.

    Args:
        image_paths (list[str | Path]): Iterable of image file paths.

    Returns:
        list[numpy.ndarray]: Loaded images as NumPy arrays (dtype depends on
        the file contents; callers often convert to uint8).
    """
    images = []
    for path in image_paths:
        images.append(load_io_image(path))
    return images


def is_red(pixel: tuple[int, int, int]) -> bool:
    """Return True if a pixel is 'red' by simple thresholding.

    A pixel is considered red when its red channel is high and the green and
    blue channels are low. Anything outside of the range can kick sand and
    eat rocks :) — in other words, this is a deliberately aggressive test.

    Args:
        pixel (sequence): Sequence with at least three elements (R, G, B).

    Returns:
        bool: True if the pixel is red.
    """
    R_RANGE = 240
    G_RANGE = 5
    B_RANGE = 5
    return pixel[0] >= R_RANGE and pixel[1] <= G_RANGE and pixel[2] <= B_RANGE


def is_green(pixel: tuple[int, int, int]) -> bool:
    """Return True if a pixel is 'green' by simple thresholding.

    Args:
        pixel (sequence): Sequence with at least three elements (R, G, B).

    Returns:
        bool: True if the pixel is green.
    """
    R_RANGE = 5
    G_RANGE = 240
    B_RANGE = 5
    return pixel[0] <= R_RANGE and pixel[1] >= G_RANGE and pixel[2] <= B_RANGE


def is_black(pixel: tuple[int, int, int]) -> bool:
    """Return True if a pixel is 'black' using a low-intensity cutoff.

    Args:
        pixel (sequence): Sequence with at least three elements (R, G, B).

    Returns:
        bool: True if all channels are below the TOO_DARK_CUTOFF.
    """
    TOO_DARK_CUTOFF = 10
    r_value_black = pixel[0] < TOO_DARK_CUTOFF
    g_value_black = pixel[1] < TOO_DARK_CUTOFF
    b_value_black = pixel[2] < TOO_DARK_CUTOFF
    return r_value_black and g_value_black and b_value_black


def get_random_pixel_color() -> list[int]:
    """Return a random RGB triple with moderately high channel values.

    The values are chosen between 100 and 255 to avoid very dark colors which
    may be hard to see on the images. This keeps the annotation colors bright
    and readable.
    """
    return [
        random.randint(100, 255),
        random.randint(100, 255),
        random.randint(100, 255),
    ]


def write_sorted_horizontal_distance_csv(path: Path, rows: list[tuple[str, str, float]]) -> None:
    """Write horizontal_distance.csv rows sorted by green_square then red_square (both numeric).

    Args:
        path: Full path to the CSV file to write.
        rows: List of (red_square, green_square, horizontal_distance) tuples,
              where red_square and green_square are already-extracted numeric strings.
    """
    rows_sorted = sorted(rows, key=lambda r: (int(r[1]), int(r[0])))
    with path.open("w") as f:
        f.write("red_square,green_square,horizontal_distance\n")
        for rs_col, gs_col, horizontal_distance in rows_sorted:
            f.write(f"{rs_col},{gs_col},{horizontal_distance:03f}\n")


def write_sorted_percentage_data_csv(
    path: Path, rows: list[tuple[str, str, float, float, float, float, float, list, str]]
) -> None:
    """Write percentage_data.csv rows sorted by green_square then red_square (both numeric).

    Args:
        path: Full path to the CSV file to write.
        rows: List of (red_square, green_square, percentage, distance,
              base_score, completeness_ratio, geometry_consistency,
              individual_scores, particle_details) tuples, where red_square and
              green_square are already-extracted numeric strings.

    Side Effects:
        Overwrites ``path`` with a fully sorted CSV including the
        ``individual_scores`` JSON column and the serialized
        ``particle_details`` column.
    """
    rows_sorted = sorted(rows, key=lambda r: (int(r[1]), int(r[0])))
    with path.open("w") as f:
        f.write(
            "red_square,green_square,percentage,distance,base_score,completeness_ratio,geometry_consistency,individual_scores,particle_details\n"
        )
        for (
            rs_col,
            gs_col,
            percentage,
            distance,
            base_score,
            completeness_ratio,
            geometry_consistency,
            individual_scores,
            particle_details,
        ) in rows_sorted:
            scores_str = json.dumps([round(s, 6) for s in individual_scores])
            f.write(
                f'{rs_col},{gs_col},{percentage:03f},{distance:03f},{base_score:03f},{completeness_ratio:03f},{geometry_consistency:03f},"{scores_str}","{particle_details.replace(chr(34), chr(34) + chr(34))}"\n'
            )
