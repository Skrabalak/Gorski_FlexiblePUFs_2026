# Bug Report
**Date:** February 25, 2026
**Reviewed by:** GitHub Copilot
**Scope:** Full codebase audit of `main.py`, `handle_prominent_features.py`, `process_image.py`, `helper_functions.py`

---

## ~~Bug 14 — `draw_box_given_coords`: Box drawn with empty horizontal lines and vertical lines inside the particle~~ ✅ FIXED
**File:** `handle_prominent_features.py`
**Fixed:** March 11, 2026

`draw_box_given_coords` was originally written to match the **pre-Bug-1 inverted corner layout** where `corner_coordinates[0][1]` was `max_col` and `corner_coordinates[1][1]` was `min_col`. After Bug 1 was fixed (corners are now `[[min_row, min_col], [max_row, max_col]]`), the function became incorrect in two ways:

1. **Horizontal lines not drawn at all.** The column range `range(corner_coordinates[1][1], corner_coordinates[0][1] + 1)` became `range(max_col, min_col + 1)`, which is always empty since `max_col >= min_col`. Neither the top nor the bottom line of the box was drawn.

2. **Vertical lines drawn one pixel inside the particle.** The left line was drawn at `corner_coordinates[0][1] + 1 = min_col + 1` (inside), and the right line at `corner_coordinates[1][1] - 1 = max_col - 1` (inside). Both lines should be drawn *outside* the particle boundary.

The net result was that every "box" appeared as two vertical stripes painted over the interior of the particle with no surrounding rectangle — making it look like no box had been drawn at all.

**Fix:** Rewrote the function using explicit named variables (`min_row`, `min_col`, `max_row`, `max_col`) extracted from the corrected corner layout. Each line is now drawn one pixel *outside* the particle boundary with the correct direction:
- Top/bottom: `range(min_col, max_col + 1)` at rows `min_row - 1` / `max_row + 1`
- Left/right: `range(min_row, max_row + 1)` at cols `min_col - 1` / `max_col + 1`
- Bounds guards updated to match (`min_col - 1 >= 0`, `max_col + 1 < len(image[0])`, etc.)

---

## ~~Bug 13 — `display_best_percentages`: No bounding boxes drawn on conglomerate or individual best-percentage images~~ ✅ FIXED
**File:** `handle_prominent_features.py`
**Fixed:** March 11, 2026

The `conglomerate_image` (and individual per-match `image_array` copies) started as plain pixel copies of `green_square_image` with no annotations. `draw_on_image` only draws connection lines, search-radius circles, and repaints particle pixels via `clean_particle_in_image` — it never calls `draw_box_given_coords`. As a result, every particle in the green square image was invisible as a detected object on the saved output images: no box surrounded it, making it impossible to visually verify which particles were found.

**Fix:** In `display_best_percentages`, before `draw_on_image` is called, iterate over all `particle_details` and call `draw_box_given_coords` with a white box onto the `conglomerate_image` and onto each `image_array[idx]` copy. The boxes are drawn first so that the relation-line annotations from `draw_on_image` are layered on top.

---

## ~~Bug 9 — `detect_particles` / `get_particle_corners`: Race condition on shared global `checking_pixels_array`~~ ✅ FIXED
**File:** `handle_prominent_features.py`
**Fixed:** March 9, 2026

`checking_pixels_array` was declared as a **module-level global** and written to by both `detect_particles` (reset + population) and `get_particle_corners` (BFS visited marking). `main.py` processes images in parallel using `ThreadPoolExecutor(max_workers=4)`. When two or more threads call `detect_particles` concurrently, the following race occurs:

1. Thread A resets `checking_pixels_array = []` and populates it for its image.
2. Thread B calls `detect_particles`, **overwrites `checking_pixels_array`** with a new array sized for its own image.
3. Thread A's BFS continues calling `checking_pixels_array[delta_x][delta_y] = 1` — now writing into Thread B's array.
4. Thread B's scan loop reads `checking_pixels_array[row][col] == 0` — finds cells already set to `1` by Thread A's BFS — and **skips pixels that should seed new particles**.

**Fix:** Removed the module-level global. `detect_particles` creates `checking_pixels_array` as a local variable and passes it into `get_particle_corners` as an explicit parameter.

---

## ~~Bug 10 — `Particle_Details.largest_particle_length/width`: Unsynchronised class-level read-modify-write~~ ✅ FIXED
**File:** `handle_prominent_features.py`
**Fixed:** March 9, 2026

`largest_particle_length` and `largest_particle_width` are **class-level variables** shared across every instance and every thread. `set_full_particle` contains an unsynchronised check-then-set:

```python
if current_particle_length > Particle_Details.largest_particle_length:
    Particle_Details.largest_particle_length = current_particle_length
```

Two threads can both pass the `>` check with the same stale value and both write, producing a non-deterministically incorrect "largest" value. This corrupts `save_particles`, which uses these values to pad `full_particle` arrays to a uniform size before saving images — causing some particles to be under-padded and others over-padded, resulting in visually wrong or black particle images.

**Fix:** Added `Particle_Details._largest_lock = threading.Lock()` as a class variable. The read-modify-write in `set_full_particle` is now wrapped in `with Particle_Details._largest_lock:`. `save_particles` also reads both values under the lock to get a consistent snapshot.

---

## ~~Bug 11 — `save_particles`: `full_particle` mutated in-place, permanently corrupting particle data~~ ✅ FIXED
**File:** `handle_prominent_features.py`
**Fixed:** March 9, 2026

`save_particles` assigned `cur_particle_array = particle.full_particle` (a bare reference) and then appended black padding pixels directly to the rows. This permanently mutated the live `Particle_Details` object. Any subsequent call to `clean_particle_in_image` or `show_particle_details` would then read the now-bloated `full_particle` and index out of the particle's actual bounding box, drawing wrong pixels or overrunning array bounds.

**Fix:** Changed to a shallow row-copy: `cur_particle_array = [list(row) for row in particle.full_particle]`. The padding is applied only to the copy; the original particle object is untouched.

---

## ~~Bug 12 — `save_particles`: Directory cleared between threads, and non-atomic `os.listdir()` file counter~~ ✅ FIXED
**Files:** `handle_prominent_features.py`, `helper_functions.py`
**Fixed:** March 9, 2026

Two separate issues in the file-saving path:

1. `save_particles` deleted all files in `particles_dir` before writing, with no lock. A second thread writing to the same directory would have its freshly-written files deleted by the first thread's cleanup pass.

2. Both `save_image` and `save_io_image` used `len(os.listdir(save_dir))` to derive a sequential filename. This is non-atomic: two threads calling it simultaneously receive the same count and produce the **same filename**, causing one image to silently overwrite the other.

**Fix (1):** Removed the directory-clearing loop from `save_particles`. Each call now only truncates its own CSV and writes new files without touching previously written ones.

**Fix (2):** Added `_file_counter: dict[str, int]` and `_file_counter_lock: threading.Lock()` to `helper_functions`. The new `_next_file_index(save_dir)` helper increments the per-directory counter atomically under the lock. Both `save_image` and `save_io_image` now call `_next_file_index` instead of `len(os.listdir())`. `save_io_image` also returns the generated filename so callers (e.g. `save_particles`) can record it.

---

## ~~Bug 1 — `get_particle_corners`: Corner coordinates are inverted~~ ✅ FIXED
**File:** `handle_prominent_features.py` (~line 1033)
**Fixed:** March 9, 2026

The BFS-based corner tracker **swapped what "top-left" and "bottom-right" mean** for the column axis:

```python
# BEFORE (buggy):
if pos[1] < bottom_right_pos[1]:   # ← min col stored on bottom_right
    bottom_right_pos[1] = pos[1]
if pos[1] > top_left_pos[1]:       # ← max col stored on top_left
    top_left_pos[1] = pos[1]

# AFTER (fixed):
if pos[1] < top_left_pos[1]:       # min col → top_left  ✓
    top_left_pos[1] = pos[1]
if pos[1] > bottom_right_pos[1]:   # max col → bottom_right  ✓
    bottom_right_pos[1] = pos[1]
```

The minimum column (left edge) was tracked in `bottom_right_pos[1]`, and the maximum column (right edge) was tracked in `top_left_pos[1]`. Every `Particle_Details` object was created with `corner_coordinates = [[min_row, max_col], [max_row, min_col]]` instead of the correct `[[min_row, min_col], [max_row, max_col]]`.

**Root effect on box drawing:** `draw_box_given_coords` draws its vertical lines at `corner_coordinates[0][1] + 1` (left line) and `corner_coordinates[1][1] - 1` (right line). With inverted columns, these resolved to `max_col + 1` and `min_col - 1` — placing both vertical lines **outside** the particle bounding box. Particles near image edges had one or both vertical lines silently skipped by the bounds guards, meaning no box appeared around them at all. This caused valid particles to go unrecognised by the Relations detection pipeline.

This malformed bounding box also corrupted every other downstream function: `set_full_particle`, `size`, `clean_particle_in_image`, and `show_particle_details`.

---

## ~~Bug 2 — `set_full_particle` / `clean_particle_in_image` / `show_particle_details`: Column range and delta were tied to inverted corners~~ ✅ FIXED
**File:** `handle_prominent_features.py`
**Fixed:** March 9, 2026

Three functions contained column iteration ranges and `col_delta` values that were written to match the **inverted** corner layout produced by Bug 1. Once Bug 1 was corrected (corners are now `[[min_row, min_col], [max_row, max_col]]`), these functions needed matching corrections:

- **`set_full_particle`** — inner loop was `range(particle_position[1][1], particle_position[0][1] + 1)`, which was `range(min_col, max_col+1)` only because Bug 1 had the values swapped. Fixed to `range(particle_position[0][1], particle_position[1][1] + 1)`.
- **`clean_particle_in_image`** — `col_delta` was `coords[1][1]` (which was `min_col` under the old inversion). Fixed to `coords[0][1]` (now the true `min_col`). Column loop fixed from `range(coords[1][1], coords[0][1]+1)` to `range(coords[0][1], coords[1][1]+1)`.
- **`show_particle_details`** — same `col_delta` and column range pattern as `clean_particle_in_image`; corrected identically.

---

## ~~Bug 3 — `get_best_particle_collection_distance`: Distance calculation is outside the inner loop~~ ✅ REMOVED
**File:** `handle_prominent_features.py` (removed)

This function was superseded by `evaluate_with_pruning` and was never called in the active pipeline. It has been deleted from the codebase along with all other legacy matching functions.

---

## ~~Bug 4 — `color_closeness_evaluation` returns a lower score for *more similar* colors~~ ✅ REMOVED
**File:** `handle_prominent_features.py` (removed)

`color_closeness_evaluation`, `is_close_match`, and the entire legacy color/size/neighbor-closeness matching family have been deleted from the codebase. They were not part of the active pipeline.

---

## ~~Bug 5 — `get_relations_object` adds all particles to `relation_array` with no proximity filter~~ ✅ FIXED
**File:** `handle_prominent_features.py` (~line 1536)

```python
def get_relations_object(red_square_particle_details):
    largest_particle = ...
    relations_object = Relations(largest_particle)
    for particle in red_square_particle_details:   # ← iterates ALL particles
        relations_object.add_relation(particle)    # ← no proximity/cluster filter
```

`add_relation` guards against adding the primary particle to itself, so no crash occurs. However, the `relation_array` ends up containing **all** particles in the image with no filtering by proximity or cluster membership. This gives the relations object very low signal-to-noise when the image contains many unrelated particles spread across the field, undermining the entire matching pipeline.

---

## Bug 6 — `remove_large_aggregations` runs twice on the green square path
**Files:** `main.py` and `process_image.py`

In `main.py`, the green square processing path explicitly calls:
```python
green_square_image = remove_large_aggregations(green_processed_image, save_dir)  # call 1
```

And `process_image` itself already performs this step internally (step 8):
```python
aggregationless_result = remove_large_aggregations(rgb_result, save_dir)  # call 2
```

So aggregation removal runs **twice** on the green square image. Since `MAX_ACCEPTABLE_SIZE = 20` is quite strict, double-passing could erroneously eliminate valid particles that survived the first pass due to cluster merging order, reducing the detected particle count.

---

## ~~Bug 7 — `all_percentages.extend(full_image_top_percentages[0])` indexes a tuple, not a list~~ ✅ FIXED
**File:** `main.py` (~line 499)

```python
all_percentages.extend(full_image_top_percentages[0])
```

`get_percentages` returns a `list[tuple[float, Particle_Details]]`. Indexing `[0]` gives **the first tuple** `(float, Particle_Details)`, not the first list element. `extend` then iterates over the two elements of that tuple, adding the raw float and the particle object individually instead of adding the whole tuple. The `all_percentages` list thus contains a mix of floats and `Particle_Details` objects. Any code that attempts to unpack these as `(percentage, particle_details)` pairs would crash.

**Fix:** Change to `all_percentages.extend(full_image_top_percentages)` or `all_percentages.append(full_image_top_percentages[0])` depending on intent.

---

## ~~Bug 8 — `coord_to_identifier` uses `len(image)` (rows) instead of `len(image[0])` (cols) as the stride~~ ✅ REMOVED
**File:** `handle_prominent_features.py` (removed)

The incorrect version of `coord_to_identifier` in `handle_prominent_features.py` was only used by legacy matching functions (`get_close_matches`, `find_potential_groups`) that were not part of the active pipeline. Those legacy functions, and this function, have all been deleted. The correct version of `coord_to_identifier` (using `len(image[0])` as the column stride) remains in `process_image.py`.

---

## ~~Bug 9 — `cutoff_white` condition is inverted: replaces pixels *not* surrounded by white~~ ✅ REMOVED
**File:** `process_image.py` (removed)

`cutoff_white` was only called by the legacy `process_image_original()` pipeline. Both functions have been deleted from the codebase. The active `process_image()` pipeline does not use `cutoff_white`.

---

## Summary Table

| # | File | Function | Description |
|---|---|---|---|
| 1 | `handle_prominent_features.py` | `get_particle_corners` | Min/max column tracked on wrong corner — bounding boxes are geometrically malformed |
| 2 | `handle_prominent_features.py` | `set_full_particle` | Downstream of Bug 1 — particle pixel arrays and sizes are unreliable |
| 3 | `handle_prominent_features.py` | `get_best_particle_collection_distance` | ✅ REMOVED — function deleted (was never called in active pipeline) |
| 4 | `handle_prominent_features.py` | `color_closeness_evaluation` | ✅ REMOVED — function deleted along with all legacy matching code |
| 5 | `handle_prominent_features.py` | `get_relations_object` | ✅ FIXED — now only adds particles within the primary's `nearby_particles` cluster |
| 6 | `main.py` + `process_image.py` | green square pipeline | `remove_large_aggregations` runs twice on green square images |
| 7 | `main.py` | `main` | ✅ FIXED — `.extend(full_image_top_percentages[0])` → `.extend(full_image_top_percentages)` |
| 8 | `handle_prominent_features.py` | `coord_to_identifier` | ✅ REMOVED — incorrect version deleted; correct version remains in `process_image.py` |
| 9 | `process_image.py` | `cutoff_white` | ✅ REMOVED — function deleted along with `process_image_original()` |
