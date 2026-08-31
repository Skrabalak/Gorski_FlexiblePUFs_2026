# Codebase Documentation

## Repository purpose

This codebase processes microscopy sample directories containing `GB.` images, extracts particle information from the bordered regions, matches red-box particle templates against green-box particle fields, and produces CSV and image outputs for later analysis and manual review.

The current top-level scripts are:

- `main.py`: automated processing and matching.
- `manual_approval.py`: interactive override workflow and master-CSV generation.
- `run_manifest.py`: append-only run ledger and summary helpers for resumable batch scheduling.

## Module inventory

### `main.py`

Current responsibilities:

- discover sample directories with `get_sample_dirs(root_dir)`,
- scan each directory for eligible green-box and red-box images,
- load cached processed-data metadata and referenced image caches when possible,
- process uncached images one at a time,
- emit parent-owned run-manifest events for discovery, queueing, starts, completions, failures, and skips,
- rebuild resume state from prior run manifests,
- score red/green combinations,
- write CSV outputs and best-match images,
- drive concurrent directory processing with `ThreadPoolExecutor`,
- report live progress through `MultiProgressTracker` and `DirState`.

Key public functions:

- `get_image_paths(samples_dir)`
- `get_sample_dirs(root_dir)`
- `is_image_processed(image_path)`
- `is_red_square_image(image_path)`
- `save_red_square_particle_data(...)`
- `save_green_square_particle_data(...)`
- `load_red_square_image_data(image_path)`
- `load_green_square_image_data(image_path)`
- `hash_image(image)`
- `calc_avg_horizontal_distance(...)`
- `process_sample_dir(...)`
- `main()`

Important behavior:

- `get_sample_dirs()` discovers directories recursively with `root_dir.rglob("GB*.tif")`.
- `get_image_paths()` accepts both `.tif` and `.tiff` files once a sample directory has been discovered.
- red-box images can yield multiple extracted red-square crops; each crop is saved with an indexed `_redsquareN` stem when necessary.
- green-box and red-box cache metadata is stored in sibling `data/` JSON files, while schema-v2 processed image arrays are stored in adjacent `.npy` files.
- `process_sample_dir()` now separates cache classification from image loading: cached outputs are reloaded immediately, while uncached source images are read, processed, and released one at a time rather than being preloaded as a full in-memory list.
- `main()` now writes run manifests under `run_state/<run_id>/`, keeps manifest writes in the parent process for both thread and process modes, and can resume from a previous run by skipping `completed` directories and optionally requeueing `failed` directories.
- cache readers support both legacy JSON-embedded `image_data` and schema-v2 split caches, and invalid caches are treated as reprocessable instead of fatal.

### `run_manifest.py`

Current responsibilities:

- create unique run IDs and per-run state directories,
- append scheduler events to `events.jsonl`,
- derive and write `summary.json`,
- load prior run state for resume decisions,
- preserve per-directory attempt counts across resumed runs.

Important functions and classes:

- `get_run_state_root(base_dir)`
- `get_latest_run_id(base_dir)`
- `get_run_dir(base_dir, run_id)`
- `load_run_state(base_dir, run_id)`
- `RunManifest`

### `process_image.py`

Current responsibilities:

- detect and crop red-box interiors,
- detect and crop the green-box interior,
- preprocess cropped images into a form suitable for particle detection.

Important functions:

- `get_red_square_image(image, save_dir)`
- `get_green_square_image(image, save_dir)`
- `coord_to_identifier(row, col, image)`
- `remove_large_aggregations_grayscale(image, save_dir)`
- `remove_large_aggregations(image, save_dir)`
- `flatten_image(image)`
- `process_image(image, save_dir, save_step_images=False, progress=None, progress_prefix="")`
- `remove_outer_noise(image)`

`process_image()` currently performs these stages, in order:

1. original image display,
2. median filter,
3. grayscale conversion,
4. large bright-cluster removal,
5. halo smoothing,
6. background estimation,
7. background-weighted foreground generation,
8. adaptive thresholding,
9. RGB masking,
10. large colour-cluster removal,
11. edge sanitization.

### `handle_prominent_features.py`

Current responsibilities:

- represent detected particles,
- represent red-box relation templates,
- detect connected particle regions,
- derive nearby-particle relationships,
- score green-box anchor particles,
- serialize particle and relation data,
- generate annotated output images.

#### `Particle_Details`

Current stored fields:

- `corner_coordinates`
- `center_position`
- `full_particle`
- `color`
- `nearby_particles`
- `id`

Important methods:

- `to_json()` and `from_json()`
- `distance_to(other)`
- `sum_distances()`
- `size()`
- `set_full_particle(...)`
- `set_color()`
- `set_nearby_particles(...)`
- `clean_particle_in_image(image)`
- `get_dummy_particle(...)`

#### `Relations`

Current responsibilities:

- hold the red-box primary particle and related particles,
- find candidate matches in a green-box image,
- compute a best one-to-one assignment,
- score match quality,
- draw template geometry and selected matches.

Important methods:

- `to_json()` and `from_json()`
- `add_relation(new_particle)`
- `get_search_distance(target_particle)`
- `draw_on_image(primary_particle, particle_details, image)`
- `find_equivalent_particles(...)`
- `get_equivalent_particles_dict(image_particle, particle_details)`
- `get_best_combination(...)`
- `evaluate_with_pruning(...)`
- `get_percent_score(...)`
- `percentage_match(...)`

#### Important module-level functions

- `detect_particles(image)`
- `sanitize_particle_array(particle_array)`
- `get_particle_size(particle)`
- `is_in_bounds(row, col, image)`
- `inbounds_line(rr, cc, image)`
- `draw_box_given_coords(corner_coordinates, color_pixel, image)`
- `get_particle_corners(row, col, image, checking_pixels_array)`
- `show_detected_particles(...)`
- `get_particle_details(particle_positions, image)`
- `show_particle_details(...)`
- `save_particles(...)`
- `get_nearest_neighbor_search_radius(particle_details, image)`
- `get_relations_object(red_square_particle_details)`
- `assign_particle_ids(...)`
- `serialize_particle_collection(...)`
- `save_particle_collection(...)`
- `load_particle_collection(...)`
- `_convert_to_serializable(value)`
- `get_percentages(...)`
- `display_best_percentages(...)`

Notable current behavior:

- particle descriptors returned by `detect_particles()` use the form `[top_left, bottom_right, pixel_count]`.
- `get_particle_size()` returns the stored `pixel_count`, not bounding-box area.
- `get_nearest_neighbor_search_radius()` uses the median K-th nearest-neighbour distance scaled by `NN_RADIUS_MULTIPLIER` and falls back to `NN_FALLBACK_RADIUS` when too few particles are present.
- `display_best_percentages()` currently draws only the top `BEST_PERCENTAGES_TO_DISPLAY` results, which is set to `1` in `helper_functions.py`.

### `helper_functions.py`

Current responsibilities:

- write Matplotlib figures and raw image arrays to disk,
- generate unique filenames safely under concurrent use,
- append text to log files,
- load images, including a single-image helper used by the streaming directory pipeline,
- classify pixels by simple colour thresholds,
- write sorted CSV outputs.

Important constants currently used by the code:

- `IMAGE_DPI = 2400`
- `NN_RADIUS_MULTIPLIER = 1.5`
- `NN_K = 3`
- `NN_FALLBACK_RADIUS = 200`
- `MIN_PARTICLE_SIZE = 1`
- `BEST_PERCENTAGES_TO_DISPLAY = 1`
- `DRAW_INDIVIDUAL_BEST_PERCENTAGES = False`

Important functions:

- `_next_file_index(save_dir)`
- `save_image(name, save_dir, fig=None)`
- `save_io_image(name, image, save_dir, best_percentages_image=False)`
- `write_to_file(logfilename, string, print_to_terminal=False)`
- `load_io_image(image_path)`
- `load_io_images(image_paths)`
- `is_red(pixel)`
- `is_green(pixel)`
- `is_black(pixel)`
- `get_random_pixel_color()`
- `write_sorted_horizontal_distance_csv(path, rows)`
- `write_sorted_percentage_data_csv(path, rows)`

### `progress.py`

Current public classes:

- `DirState`
- `MultiProgressTracker`

Current behavior:

- one `DirState` is assigned to each active directory worker,
- `MultiProgressTracker` renders an ANSI multi-row block in TTY mode,
- milestone messages scroll above that block,
- non-TTY mode suppresses progress-block rendering and prints milestones plainly.

There is no current `ProgressTracker` class in the repository.

### `manual_approval.py`

Current responsibilities:

- load `percentage_data.csv` rows,
- show current best-match output beside the `locations fixed` reference image,
- collect manual corrections,
- mark chosen rows with `percentage=999`,
- write root-level master CSV files.

Important functions:

- `load_percentage_csv(csv_path)`
- `save_percentage_csv(csv_path, rows)`
- `get_best_row_for_rs_gs(rows, rs_col, gs_col)`
- `get_all_rows_for_rs_gs(rows, rs_col, gs_col)`
- `apply_manual_override(rows, rs_col, gs_col, chosen_row_idx_in_sorted)`
- `load_image_rgb(path)`
- `find_latest_all_best_percentages_image(best_pct_dir, gs_stem)`
- `find_locations_fixed_image(samples_dir)`
- `extract_mm_from_gs_stem(gs_stem)`
- `build_labeled_particle_image(...)`
- `make_grid_image(tiles, cols=3, tile_size=120)`
- `build_rs_best_match_image(samples_dir, gs_stem, rs_col, best_row)`
- `show_two_images_and_ask(...)`
- `show_labeled_percentage_images_and_ask(...)`
- `show_incorrect_candidates_and_ask(...)`
- `show_all_best_labeled_and_ask_incorrect(...)`
- `process_sample_dir_approval(samples_dir)`
- `write_master_csv(root_dir, sample_dirs)`
- `write_master_best_csv(root_dir, sample_dirs)`
- `main()`

Important behavior:

- Matplotlib uses the `TkAgg` backend, so this script requires a GUI-capable session.
- manual overrides set the selected row's `percentage` to `999.0`.
- the code does not restore original percentage values for older override rows because those original values are not retained anywhere.

## Current scoring model

For one candidate anchor particle in a green-box image:

1. `Relations.get_equivalent_particles_dict()` creates the best one-to-one mapping from relation particles to green-box particles.
2. `Relations.get_percent_score()` scores each matched related particle using exponential decay on the distance from its expected dummy position.
3. `_selectivity_penalty()` downweights matches when the best and second-best candidates are too similar.
4. `_geometry_consistency_score()` evaluates how well pairwise distances are preserved.
5. `Relations.percentage_match()` combines the pieces into:

```text
percentage = base_score * completeness_ratio * geometry_consistency
```

Where:

- `base_score` is the mean of matched-only particle scores,
- `completeness_ratio` is `matched_relations / total_relations`,
- `geometry_consistency` is a value in `[0, 1]`.

## Output files

Per sample directory, the current pipeline writes:

- `data/*_processed_data.json`
- `data/*_processed_image.npy`
- `data/percentage_data.csv`
- `data/horizontal_distance.csv`
- `data/particles.csv`
- `pictures/`
- `best_percentage_images/`

`percentage_data.csv` columns currently written by `write_sorted_percentage_data_csv()`:

- `red_square`
- `green_square`
- `percentage`
- `distance`
- `base_score`
- `completeness_ratio`
- `geometry_consistency`
- `individual_scores`
- `particle_details`

`horizontal_distance.csv` columns:

- `red_square`
- `green_square`
- `horizontal_distance`

Root-level files written by `manual_approval.py`:

- `master_percentage_data.csv`
- `master_percentage_data_best.csv`

## Execution flow

### Batch mode (`main.py`)

1. Resolve root directory and command-line flags.
2. Discover sample directories recursively.
3. Start `MultiProgressTracker` for thread mode, or use `NullProgress` inside subprocess workers for process mode.
4. Process directories concurrently with either threads or processes.
5. For each directory, load caches or generate them.
6. If not in `--process-only` mode, score every red/green pair.
7. Save CSVs and annotated images.

### Manual mode (`manual_approval.py`)

1. Discover the same sample directories.
2. Read each `percentage_data.csv`.
3. Review each `green_square` group interactively.
4. Apply overrides where needed.
5. Save updated per-directory CSVs.
6. Rebuild both root-level master CSVs.

## Current practical constraints

- The repository does not contain a `requirements.txt` or other dependency manifest.
- `get_sample_dirs()` discovers directories using `GB*.tif`, not `GB*.tiff`, so directories containing only `.tiff` files are not discovered automatically.
- `manual_approval.py` depends on an interactive GUI backend.
- `--parallel-mode process` disables the live multi-row progress display because subprocesses cannot safely share the terminal tracker.
- `process_image.log` and `timing.log` are debug-heavy logs and are skipped unless `SIP_DEBUG_LOGS=1` is set.
