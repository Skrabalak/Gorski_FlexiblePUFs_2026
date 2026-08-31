"""
manual_approval.py

Interactive post-processing tool for manually reviewing and overriding the
"best percentage" particle matches produced by main.py.

Workflow (per sample directory):
  1. For each mm stretch, show the all_best_percentages image alongside the
     corresponding locations-fixed image, labelled with the mm stretch value.
  2. Ask YES / NO — does the composite match the expected layout?
  3. If YES → move on.
  4. If NO  → for each red-square sub-image that contributed to that stretch,
     show a labelled grid of individual per-particle candidates (9 at a time)
     and let the user pick one, or page forwards/backwards, marking the chosen
     row with percentage=999 (manual override) in the CSV.

Usage:
    python manual_approval.py [--root-dir <path>]

The script reads the percentage_data.csv produced by main.py and updates it
in-place when manual overrides are applied.
"""

import csv
import os
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as skio

from handle_prominent_features import draw_box_given_coords
from main import (
    GREEN_BOX_IMAGE_RE,
    RED_BOX_IMAGE_RE,
    get_sample_dirs,
    load_green_square_image_data,
    load_red_square_image_data,
)

matplotlib.use("TkAgg")  # interactive backend

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "red_square",
    "green_square",
    "percentage",
    "distance",
    "base_score",
    "completeness_ratio",
    "geometry_consistency",
    "individual_scores",
    "particle_details",
]

MANUAL_OVERRIDE_PERCENTAGE = 999.0


def load_percentage_csv(csv_path: Path) -> list[dict]:
    """Load ``percentage_data.csv`` into a list of row dictionaries."""
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def save_percentage_csv(csv_path: Path, rows: list[dict]) -> None:
    """Write row dictionaries back to ``percentage_data.csv``."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_best_row_for_rs_gs(rows: list[dict], rs_col: str, gs_col: str) -> dict | None:
    """Return the highest-percentage row for one red-square/green-square pair."""
    candidates = [r for r in rows if r["red_square"] == rs_col and r["green_square"] == gs_col]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r["percentage"]))


def get_all_rows_for_rs_gs(rows: list[dict], rs_col: str, gs_col: str) -> list[dict]:
    """Return all candidate rows for one pair, sorted by percentage descending."""
    candidates = [r for r in rows if r["red_square"] == rs_col and r["green_square"] == gs_col]
    return sorted(candidates, key=lambda r: float(r["percentage"]), reverse=True)


def apply_manual_override(rows: list[dict], rs_col: str, gs_col: str, chosen_row_idx_in_sorted: int) -> None:
    """
    Mark one candidate row as a manual override for a red/green pair.

    Args:
        rows: Full in-memory CSV row list to modify.
        rs_col: Red-square index column value.
        gs_col: Green-square index column value.
        chosen_row_idx_in_sorted: Index into the sorted candidate list returned
            by ``get_all_rows_for_rs_gs``.

    Side Effects:
        Sets the chosen row's ``percentage`` field to ``999.0``. Existing
        override markers on other rows are left unchanged because the original
        percentage values are not retained.
    """
    candidates_sorted = get_all_rows_for_rs_gs(rows, rs_col, gs_col)
    chosen_row = candidates_sorted[chosen_row_idx_in_sorted]

    # Find and update the matching row in the main list
    for row in rows:
        if row["red_square"] == rs_col and row["green_square"] == gs_col:
            if row is chosen_row or (
                row["particle_details"] == chosen_row["particle_details"] and row["distance"] == chosen_row["distance"]
            ):
                row["percentage"] = str(MANUAL_OVERRIDE_PERCENTAGE)
            else:
                # Restore a previously set override if present
                if float(row["percentage"]) == MANUAL_OVERRIDE_PERCENTAGE:
                    # We can't recover the original value, so just leave the
                    # numeric value as-is for non-chosen rows; they will still
                    # sort below 999.
                    pass


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def load_image_rgb(path: Path) -> np.ndarray | None:
    """Load an image file as an RGB ``uint8`` array, or ``None`` on failure."""
    try:
        img = skio.imread(str(path))
        if img is None:
            return None
        img = np.array(img, dtype=np.uint8)
        # If RGBA, drop alpha channel
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    except Exception as e:
        print(f"[WARNING] Could not load image {path}: {e}")
        return None


def find_latest_all_best_percentages_image(best_pct_dir: Path, gs_stem: str) -> Path | None:
    """
    Return the lexicographically latest ``all_best_percentages`` image for a
    green-square stem.

    Images are named ``{date}_{gs_stem}_all_best_percentages.png``, so sorting
    by filename also sorts by the timestamp prefix.
    """
    if not best_pct_dir.exists():
        return None
    pattern = f"*_{gs_stem}_all_best_percentages.png"
    matches = sorted(best_pct_dir.glob(pattern))
    if not matches:
        return None
    # Return the last one (latest timestamp — filenames are sortable by date prefix)
    return matches[-1]


def find_locations_fixed_image(samples_dir: Path) -> Path | None:
    """Return the first red-box source image path in a sample directory."""
    for p in sorted(samples_dir.glob("*.tif")) + sorted(samples_dir.glob("*.tiff")):
        if RED_BOX_IMAGE_RE.match(p.name):
            return p
    return None


def extract_mm_from_gs_stem(gs_stem: str) -> str:
    """Extract the mm stretch label from the green-square image stem.

    e.g. 'GB.AE1.DCpdms.50x.0mm.1' → '0mm'
         'GB.AE1.DCpdms.50x.12mm.1' → '12mm'
    """
    m = re.search(r"(\d+mm)", gs_stem, re.IGNORECASE)
    return m.group(1) if m else gs_stem


def build_labeled_particle_image(row: dict, index_label: int, image_size: int = 120) -> np.ndarray:
    """
    Construct a small labeled tile for one percentage-data row.

    The function tries to reconstruct the particle crop from the serialized
    ``full_particle`` JSON, then overlays the candidate number and displayed
    percentage. If reconstruction fails, it falls back to a grey placeholder
    tile.
    """
    import json as _json

    from PIL import Image, ImageDraw, ImageFont

    tile = np.full((image_size, image_size, 3), 128, dtype=np.uint8)

    # Attempt to extract the full_particle pixel data from particle_details JSON
    try:
        details_str = row.get("particle_details", "")
        if details_str:
            details = _json.loads(details_str)
            fp = details.get("full_particle")
            if fp:
                patch = np.array(fp, dtype=np.uint8)
                if patch.ndim == 3 and patch.shape[2] >= 3:
                    # Centre the patch in the tile
                    ph, pw = patch.shape[0], patch.shape[1]
                    sh = min(ph, image_size)
                    sw = min(pw, image_size)
                    r_off = (image_size - sh) // 2
                    c_off = (image_size - sw) // 2
                    tile[r_off : r_off + sh, c_off : c_off + sw] = patch[:sh, :sw, :3]
    except Exception:
        pass

    # Overlay text label using PIL
    try:
        pil_img = Image.fromarray(tile)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        label_text = str(index_label)
        # White shadow then black text for visibility on any background
        draw.text((3, 3), label_text, font=font, fill=(255, 255, 255))
        draw.text((2, 2), label_text, font=font, fill=(0, 0, 0))
        pct_text = f"{float(row['percentage']):.1f}%"
        draw.text((3, image_size - 20), pct_text, font=font, fill=(255, 255, 255))
        draw.text((2, image_size - 21), pct_text, font=font, fill=(0, 0, 0))
        tile = np.array(pil_img, dtype=np.uint8)
    except Exception:
        pass

    return tile


def make_grid_image(tiles: list[np.ndarray], cols: int = 3, tile_size: int = 120) -> np.ndarray:
    """Arrange equally sized image tiles into a simple grid canvas."""
    rows_needed = (len(tiles) + cols - 1) // cols
    grid_h = rows_needed * tile_size
    grid_w = cols * tile_size
    grid = np.full((grid_h, grid_w, 3), 64, dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        r0 = row * tile_size
        c0 = col * tile_size
        th = min(tile.shape[0], tile_size)
        tw = min(tile.shape[1], tile_size)
        grid[r0 : r0 + th, c0 : c0 + tw] = tile[:th, :tw]

    return grid


# ---------------------------------------------------------------------------
# Per-RS annotated image builder
# ---------------------------------------------------------------------------


def build_rs_best_match_image(samples_dir: Path, gs_stem: str, rs_col: str, best_row: dict) -> np.ndarray | None:
    """
    Reconstruct an annotated green-square image for one red-square contributor.

    The function loads cached processed-data metadata and the referenced cached
    image arrays for the selected green-square image and red-square template,
    resolves the anchor particle referenced by ``best_row``, draws bounding
    boxes for all green particles, and then draws the red-square relation
    template onto a copy of the green image.
    """
    import json as _json

    # Locate the green-square image file (real .tif) so the loader can derive
    # the correct data path from its stem.
    gs_image_path = None
    for p in sorted(samples_dir.glob("*.tif")) + sorted(samples_dir.glob("*.tiff")):
        if p.stem == gs_stem:
            gs_image_path = p
            break
    if gs_image_path is None:
        return None

    # Find the red-square data file for this rs_col index.
    data_dir = samples_dir / "data"
    rs_data_files = sorted(data_dir.glob(f"*redsquare{rs_col}_processed_data.json"))
    if not rs_data_files:
        # Single red-square case (no redsquare suffix)
        rs_data_files = sorted(data_dir.glob("*locations fixed_processed_data.json"))
    if not rs_data_files:
        return None
    # Build the virtual image path that load_red_square_image_data expects:
    # it only uses the path to derive {parent}/data/{stem}_processed_data.json
    rs_stem = rs_data_files[0].name.replace("_processed_data.json", "")
    rs_virtual_path = str(samples_dir / f"{rs_stem}.tif")

    # Load green-square particles and image from the cache metadata.
    green_particles, green_image = load_green_square_image_data(str(gs_image_path))
    if green_particles is None or green_image is None:
        return None

    # Load red-square relations from the cache metadata.
    red_relations, _ = load_red_square_image_data(rs_virtual_path)
    if red_relations is None:
        return None

    # Find the best anchor particle in the green-square data using the particle
    # ID stored in the CSV row's particle_details JSON.
    best_particle = None
    try:
        details = _json.loads(best_row["particle_details"])
        target_id = details.get("id")
        if target_id is not None:
            for p in green_particles:
                if getattr(p, "id", None) == int(target_id):
                    best_particle = p
                    break
    except Exception:
        pass

    if best_particle is None:
        return None

    annotated = green_image.copy()

    # Draw a bounding box around every detected green particle so all particles
    # are visible, mirroring what display_best_percentages does.
    for p in green_particles:
        draw_box_given_coords(p.corner_coordinates, [255, 0, 0], annotated)

    # Draw the relations for this RS's best match onto the green-square image.
    red_relations.draw_on_image(best_particle, green_particles, annotated)

    return annotated


# ---------------------------------------------------------------------------
# Interactive display helpers
# ---------------------------------------------------------------------------


def show_two_images_and_ask(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_title: str,
    right_title: str,
    window_title: str,
    question: str,
) -> bool:
    """
    Display two images side-by-side and ask a YES/NO question.

    Returns ``True`` for YES and ``False`` for NO. The window closes before the
    function returns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.manager.set_window_title(window_title)

    axes[0].imshow(left_image)
    axes[0].set_title(left_title, fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(right_image)
    axes[1].set_title(right_title, fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(question, fontsize=13, color="darkblue", fontweight="bold")

    result = {"answer": None}

    def on_key(event):
        if event.key in ("y", "Y"):
            result["answer"] = True
            plt.close(fig)
        elif event.key in ("n", "N"):
            result["answer"] = False
            plt.close(fig)

    yes_ax = fig.add_axes([0.35, 0.02, 0.12, 0.05])
    no_ax = fig.add_axes([0.53, 0.02, 0.12, 0.05])

    from matplotlib.widgets import Button

    yes_btn = Button(yes_ax, "YES  (Y)", color="lightgreen", hovercolor="green")
    no_btn = Button(no_ax, "NO  (N)", color="lightsalmon", hovercolor="red")

    def on_yes(_):
        result["answer"] = True
        plt.close(fig)

    def on_no(_):
        result["answer"] = False
        plt.close(fig)

    yes_btn.on_clicked(on_yes)
    no_btn.on_clicked(on_no)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.subplots_adjust(bottom=0.10)
    plt.show(block=True)

    return result["answer"] if result["answer"] is not None else True


def show_labeled_percentage_images_and_ask(
    all_rows: list[dict],
    page: int,
    rs_col: str,
    gs_col: str,
    mm_label: str,
    samples_dir: Path,
    gs_stem: str,
    loc_fixed_image: np.ndarray | None = None,
    page_size: int = 9,
) -> tuple[int | None, str]:
    """
    Display one paged grid of candidate images for a manual-override choice.

    Returns:
        (selected_index_in_all_rows, action) where action is one of:
          'select'   — user picked a tile; selected_index is the index in all_rows
          'next'     — user wants the next page
          'prev'     — user wants the previous page
          'skip'     — user chose none (pressed S)
    """
    start = page * page_size
    end = min(start + page_size, len(all_rows))
    page_rows = all_rows[start:end]

    n = len(page_rows)
    candidate_cols = 3
    rows_needed = (n + candidate_cols - 1) // candidate_cols

    total_pages = (len(all_rows) + page_size - 1) // page_size
    page_info = f"Page {page + 1}/{total_pages}  |  Candidates {start + 1}–{end} of {len(all_rows)}"
    sup_title = (
        f"RS {rs_col} → {mm_label}  —  {page_info}\n"
        f"Press a number key (1–{min(page_size, n)}) to select, "
        f"→/N for next page, ←/P for previous page, S to skip / select none"
    )

    # Total figure columns: 1 (locations-fixed reference) + 3 (candidates)
    total_cols = candidate_cols + 1
    fig = plt.figure(figsize=(6 * total_cols, 5 * rows_needed + 1))
    fig.canvas.manager.set_window_title(f"Manual Override: RS {rs_col} | {mm_label}")

    gs_layout = matplotlib.gridspec.GridSpec(rows_needed, total_cols, figure=fig, wspace=0.05, hspace=0.3)

    # Left panel: locations-fixed reference image spanning all rows
    ax_ref = fig.add_subplot(gs_layout[:, 0])
    if loc_fixed_image is not None:
        ax_ref.imshow(loc_fixed_image)
        ax_ref.set_title("Locations Fixed\n(reference)", fontsize=10, fontweight="bold")
    else:
        ax_ref.set_visible(False)
    ax_ref.axis("off")

    # Candidate panels in the remaining 3 columns
    axes_flat = [fig.add_subplot(gs_layout[r, c + 1]) for r in range(rows_needed) for c in range(candidate_cols)]

    for offset, row in enumerate(page_rows):
        ax = axes_flat[offset]
        label_num = start + offset + 1  # 1-based human label

        annotated = build_rs_best_match_image(samples_dir, gs_stem, rs_col, row)
        if annotated is not None:
            ax.imshow(annotated)
        else:
            tile = build_labeled_particle_image(row, label_num, image_size=200)
            ax.imshow(tile)

        pct_val = float(row["percentage"])
        pct_label = "OVERRIDE" if pct_val == MANUAL_OVERRIDE_PERCENTAGE else f"{pct_val:.2f}%"
        ax.set_title(f"[{label_num}]  {pct_label}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # Hide any unused candidate axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(sup_title, fontsize=10)

    # Add navigation buttons
    prev_ax = fig.add_axes([0.05, 0.01, 0.18, 0.04])
    skip_ax = fig.add_axes([0.41, 0.01, 0.18, 0.04])
    next_ax = fig.add_axes([0.77, 0.01, 0.18, 0.04])

    from matplotlib.widgets import Button

    prev_btn = Button(prev_ax, "◀ Prev (P)", color="lightblue", hovercolor="steelblue")
    skip_btn = Button(skip_ax, "None (S)", color="lightyellow", hovercolor="gold")
    next_btn = Button(next_ax, "Next ▶ (N)", color="lightblue", hovercolor="steelblue")

    result = {"action": None, "index": None}

    def on_key(event):
        key = event.key
        if key and key.isdigit() and key != "0":
            num = int(key)
            offset = num - 1
            abs_idx = start + offset
            if 0 <= offset < n and abs_idx < len(all_rows):
                result["action"] = "select"
                result["index"] = abs_idx
                plt.close(fig)
        elif key in ("right", "n", "N"):
            result["action"] = "next"
            plt.close(fig)
        elif key in ("left", "p", "P"):
            result["action"] = "prev"
            plt.close(fig)
        elif key in ("s", "S"):
            result["action"] = "skip"
            plt.close(fig)

    def on_prev(_):
        result["action"] = "prev"
        plt.close(fig)

    def on_skip(_):
        result["action"] = "skip"
        plt.close(fig)

    def on_next(_):
        result["action"] = "next"
        plt.close(fig)

    prev_btn.on_clicked(on_prev)
    skip_btn.on_clicked(on_skip)
    next_btn.on_clicked(on_next)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.subplots_adjust(bottom=0.08)
    plt.show(block=True)

    action = result["action"] if result["action"] is not None else "skip"
    return result["index"], action


def show_incorrect_candidates_and_ask(
    all_rows: list[dict],
    rs_col: str,
    gs_col: str,
    mm_label: str,
    samples_dir: Path,
    gs_stem: str,
    loc_fixed_image: np.ndarray | None = None,
) -> int | None:
    """
    Repeatedly show candidate pages until the user selects one or skips.

    Returns the absolute index in `all_rows` of the chosen candidate, or None
    if the user chose to skip everything.
    """
    page_size = 9
    total_pages = max(1, (len(all_rows) + page_size - 1) // page_size)
    page = 0

    while True:
        selected_idx, action = show_labeled_percentage_images_and_ask(
            all_rows, page, rs_col, gs_col, mm_label, samples_dir, gs_stem, loc_fixed_image, page_size
        )
        if action == "select" and selected_idx is not None:
            return selected_idx
        elif action == "next":
            page = min(page + 1, total_pages - 1)
        elif action == "prev":
            page = max(page - 1, 0)
        elif action == "skip":
            return None


def show_all_best_labeled_and_ask_incorrect(
    combined_image: np.ndarray | None,
    best_rows_by_rs: dict[str, dict],
    gs_stem: str,
    mm_label: str,
    samples_dir: Path,
) -> list[str]:
    """
    Show the current best match for each red-square contributor and ask which
    ones are wrong.

    Returns a list of rs_col strings that are marked incorrect.
    """
    n = len(best_rows_by_rs)
    if n == 0:
        return []

    rs_keys = sorted(best_rows_by_rs.keys(), key=lambda x: int(x))

    # Build a figure with one subplot per RS, plus a column for the combined image
    # if available.
    ncols = n + (1 if combined_image is not None else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    ax_offset = 0
    if combined_image is not None:
        axes[0].imshow(combined_image)
        axes[0].set_title(f"Combined\n{mm_label}", fontsize=10, fontweight="bold")
        axes[0].axis("off")
        ax_offset = 1

    for panel_idx, rs_col in enumerate(rs_keys):
        ax = axes[panel_idx + ax_offset]
        row = best_rows_by_rs[rs_col]

        # Reconstruct the full annotated green-square image showing only this
        # RS's best match. Falls back to a small particle tile if loading fails.
        panel_img = build_rs_best_match_image(samples_dir, gs_stem, rs_col, row)
        if panel_img is not None:
            ax.imshow(panel_img)
        else:
            tile = build_labeled_particle_image(row, panel_idx + 1, image_size=200)
            ax.imshow(tile)

        pct_val = float(row["percentage"])
        pct_label = "OVERRIDE" if pct_val == MANUAL_OVERRIDE_PERCENTAGE else f"{pct_val:.2f}%"
        ax.set_title(f"RS {rs_col}  [{panel_idx + 1}]\n{pct_label}", fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"{mm_label} — Which red-square matches are INCORRECT?\n"
        f"Enter number(s) separated by commas and press Enter,\n"
        f"or press Enter with no input if all are correct.",
        fontsize=11,
        color="darkred",
        fontweight="bold",
    )

    result = {"incorrect_rs": None}

    def on_key(event):
        # Handled via text box submission
        pass

    from matplotlib.widgets import Button, TextBox

    text_ax = fig.add_axes([0.25, 0.02, 0.35, 0.05])
    confirm_ax = fig.add_axes([0.63, 0.02, 0.12, 0.05])

    text_box = TextBox(text_ax, "Incorrect RS # (e.g. 1,3):", initial="")
    confirm_btn = Button(confirm_ax, "Confirm", color="lightgreen", hovercolor="green")

    def on_submit(text):
        result["incorrect_rs"] = text
        plt.close(fig)

    def on_confirm(_):
        result["incorrect_rs"] = text_box.text
        plt.close(fig)

    text_box.on_submit(on_submit)
    confirm_btn.on_clicked(on_confirm)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.subplots_adjust(bottom=0.12)
    plt.show(block=True)

    raw = result["incorrect_rs"] if result["incorrect_rs"] is not None else ""
    raw = raw.strip()
    if not raw:
        return []

    # Parse the comma-separated panel numbers (1-based) back to rs_col strings
    incorrect_rs_cols = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit():
            panel_num = int(tok)
            idx = panel_num - 1
            if 0 <= idx < len(rs_keys):
                incorrect_rs_cols.append(rs_keys[idx])

    return incorrect_rs_cols


# ---------------------------------------------------------------------------
# Core per–sample-dir workflow
# ---------------------------------------------------------------------------


def process_sample_dir_approval(samples_dir: Path) -> None:
    """
    Run the full manual-review workflow for one sample directory.

    For each unique green_square (mm stretch) found in the percentage CSV:
      1. Show all_best_percentages image + locations-fixed image side-by-side.
      2. Ask YES/NO.
      3. If YES → done for this stretch.
      4. If NO  → identify which RS matches are incorrect and let the user
                  pick an override from the candidate list.
    """
    csv_path = samples_dir / "data" / "percentage_data.csv"
    if not csv_path.exists():
        print(f"  [SKIP] No percentage_data.csv found in {samples_dir}")
        return

    rows = load_percentage_csv(csv_path)
    if not rows:
        print(f"  [SKIP] percentage_data.csv is empty in {samples_dir}")
        return

    # Find the locations-fixed image (used for every stretch in this dir)
    loc_fixed_path = find_locations_fixed_image(samples_dir)
    loc_fixed_image = None
    if loc_fixed_path:
        loc_fixed_image = load_image_rgb(loc_fixed_path)
    if loc_fixed_image is None:
        print(f"  [WARNING] Could not load locations-fixed image for {samples_dir.name}")
        loc_fixed_image = np.full((200, 200, 3), 200, dtype=np.uint8)

    best_pct_dir = samples_dir / "best_percentage_images"

    # Collect unique gs_col values from the CSV (each represents one mm stretch)
    unique_gs_cols = sorted(set(r["green_square"] for r in rows), key=lambda x: int(x) if x.isdigit() else 0)

    # Map gs_col → green-square image stem by scanning directory.
    # GREEN_BOX_IMAGE_RE (imported from main.py) has no capture group, so
    # extract the mm digit separately with a dedicated search.
    gs_col_to_stem: dict[str, str] = {}
    for p in sorted(samples_dir.glob("*.tif")) + sorted(samples_dir.glob("*.tiff")):
        if GREEN_BOX_IMAGE_RE.match(p.name):
            mm_match = re.search(r"\.(\d+)mm", p.name, re.IGNORECASE)
            if mm_match:
                gs_col_to_stem[mm_match.group(1)] = p.stem

    print(f"\n{'=' * 60}")
    print(f"  Sample dir: {samples_dir.name}")
    print(f"  Green-square stretches to review: {unique_gs_cols}")
    print(f"{'=' * 60}")

    for gs_col in unique_gs_cols:
        gs_stem = gs_col_to_stem.get(gs_col, f"gs_{gs_col}mm")
        mm_label = f"{gs_col}mm"

        # Find the latest all_best_percentages composite image for this stretch
        all_best_img_path = find_latest_all_best_percentages_image(best_pct_dir, gs_stem)
        if all_best_img_path:
            all_best_image = load_image_rgb(all_best_img_path)
        else:
            print(f"  [WARNING] No all_best_percentages image found for {gs_stem}")
            all_best_image = np.full((200, 200, 3), 150, dtype=np.uint8)

        print(f"\n  Reviewing mm stretch: {mm_label}")

        # --- Step 3: Show composite + locations-fixed, ask YES/NO ---
        answer = show_two_images_and_ask(
            left_image=all_best_image if all_best_image is not None else np.full((200, 200, 3), 150, dtype=np.uint8),
            right_image=loc_fixed_image,
            left_title=f"Best Percentages  ({mm_label})",
            right_title="Locations Fixed (expected)",
            window_title=f"{samples_dir.name}  |  {mm_label}  —  Does the match look correct?",
            question=f"[{samples_dir.name}  |  {mm_label}]  Does the best-percentage image match the expected layout?  (Y = Yes / N = No)",
        )

        if answer:
            print(f"    ✓ Approved: {mm_label}")
            continue

        # --- Step 4+: User said NO — figure out which RS matches are wrong ---
        # Gather the best row for each RS that contributes to this gs_col
        unique_rs_cols = sorted(
            set(r["red_square"] for r in rows if r["green_square"] == gs_col),
            key=lambda x: int(x) if x.isdigit() else 0,
        )
        best_rows_by_rs: dict[str, dict] = {}
        for rs_col in unique_rs_cols:
            best_row = get_best_row_for_rs_gs(rows, rs_col, gs_col)
            if best_row is not None:
                best_rows_by_rs[rs_col] = best_row

        # Show labeled panel per RS and ask which are wrong
        incorrect_rs_cols = show_all_best_labeled_and_ask_incorrect(
            combined_image=all_best_image,
            best_rows_by_rs=best_rows_by_rs,
            gs_stem=gs_stem,
            mm_label=mm_label,
            samples_dir=samples_dir,
        )

        if not incorrect_rs_cols:
            print(f"    ✓ No RS marked incorrect for {mm_label} — skipping overrides.")
            continue

        print(f"    Incorrect RS cols for {mm_label}: {incorrect_rs_cols}")

        # --- Step 7: For each incorrect RS, let user browse candidates and pick ---
        for rs_col in incorrect_rs_cols:
            all_candidates = get_all_rows_for_rs_gs(rows, rs_col, gs_col)
            if not all_candidates:
                print(f"    [WARNING] No candidate rows found for RS {rs_col} / {mm_label}")
                continue

            print(f"    Browsing candidates for RS {rs_col} / {mm_label} ({len(all_candidates)} total)...")
            chosen_idx = show_incorrect_candidates_and_ask(
                all_candidates, rs_col, gs_col, mm_label, samples_dir, gs_stem, loc_fixed_image
            )

            if chosen_idx is not None:
                apply_manual_override(rows, rs_col, gs_col, chosen_idx)
                print(f"    ✓ RS {rs_col} override applied: candidate #{chosen_idx + 1} marked as 999%")
            else:
                print(f"    – RS {rs_col}: no override selected, original match kept.")

        # Persist changes to CSV after processing all incorrect RS for this stretch
        save_percentage_csv(csv_path, rows)
        print(f"    CSV updated: {csv_path}")

    print(f"\n  Done with {samples_dir.name}.")


# ---------------------------------------------------------------------------
# Master CSV
# ---------------------------------------------------------------------------

MASTER_CSV_FIELDNAMES = ["sample_dir"] + CSV_FIELDNAMES


def write_master_csv(root_dir: Path, sample_dirs: list[Path]) -> Path:
    """Combine all percentage_data.csv files into a single master CSV.

    Each row from every sample directory's percentage_data.csv is written to
    the master CSV with an additional leading ``sample_dir`` column containing
    the name of the directory that the row came from.

    The master CSV is written to ``root_dir/master_percentage_data.csv``.

    Args:
        root_dir: The root directory; the master CSV is saved here.
        sample_dirs: Ordered list of sample directories to collect data from.

    Returns:
        The path to the written master CSV.
    """
    master_path = root_dir / "master_percentage_data.csv"
    with master_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_CSV_FIELDNAMES)
        writer.writeheader()
        for samples_dir in sample_dirs:
            csv_path = samples_dir / "data" / "percentage_data.csv"
            if not csv_path.exists():
                continue
            rows = load_percentage_csv(csv_path)
            for row in rows:
                master_row = {"sample_dir": samples_dir.name}
                master_row.update(row)
                writer.writerow(master_row)
    return master_path


def write_master_best_csv(root_dir: Path, sample_dirs: list[Path]) -> Path:
    """Combine all percentage_data.csv files, keeping only the best row per
    (sample_dir, red_square, green_square) combination.

    "Best" is defined as the row with the highest ``percentage`` value for that
    combination. Manual overrides (percentage=999) therefore sort to the top
    and are treated as the best row.

    The result is written to ``root_dir/master_percentage_data_best.csv``.

    Args:
        root_dir: The root directory; the master CSV is saved here.
        sample_dirs: Ordered list of sample directories to collect data from.

    Returns:
        The path to the written master CSV.
    """
    # Collect the best row per (sample_dir, red_square, green_square) key.
    # Using an ordered dict preserves insertion order (directory order) for
    # deterministic output.
    best: dict[tuple[str, str, str], dict] = {}
    for samples_dir in sample_dirs:
        csv_path = samples_dir / "data" / "percentage_data.csv"
        if not csv_path.exists():
            continue
        rows = load_percentage_csv(csv_path)
        for row in rows:
            key = (samples_dir.name, row["red_square"], row["green_square"])
            if key not in best or float(row["percentage"]) > float(best[key]["percentage"]):
                master_row = {"sample_dir": samples_dir.name}
                master_row.update(row)
                best[key] = master_row

    master_path = root_dir / "master_percentage_data_best.csv"
    with master_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_CSV_FIELDNAMES)
        writer.writeheader()
        for row in best.values():
            writer.writerow(row)
    return master_path


# ---------------------------------------------------------------------------
# Root-dir traversal
# ---------------------------------------------------------------------------


def main() -> None:
    """Run manual approval for every discovered sample directory and write master CSVs."""
    args = sys.argv[1:]

    current_dir = Path(__file__).parent

    variable = os.environ.get("COMPUTER")
    debug = variable != "SARAH"

    root_dir_arg = next((args[i + 1] for i, a in enumerate(args) if a == "--root-dir"), None)
    if root_dir_arg:
        root_dir = Path(root_dir_arg)
    elif debug:
        root_dir = current_dir / "Image_Samples" / "root_dir"
    else:
        root_dir = Path(r"D:\Codes!!\Supercomputer_calc\Samples")

    print("Manual Approval Tool")
    print(f"Root dir: {root_dir}")

    sample_dirs = get_sample_dirs(root_dir)
    if not sample_dirs:
        print(f"No sample directories found under {root_dir}")
        return

    print(f"Found {len(sample_dirs)} sample director{'y' if len(sample_dirs) == 1 else 'ies'}:")
    for d in sample_dirs:
        print(f"  {d}")

    for samples_dir in sample_dirs:
        process_sample_dir_approval(samples_dir)

    master_path = write_master_csv(root_dir, sample_dirs)
    print(f"\nMaster CSV written: {master_path}")
    master_best_path = write_master_best_csv(root_dir, sample_dirs)
    print(f"Master best CSV written: {master_best_path}")
    print("\nManual approval complete.")


if __name__ == "__main__":
    main()
