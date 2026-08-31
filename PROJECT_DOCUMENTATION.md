# Project Documentation

## Overview

This project processes microscopy sample directories, extracts particle information from bordered images, matches red-box particle layouts against green-box full-field images, and writes CSV and image outputs for later review.

The current entry points are:

- `main.py` for automated batch processing.
- `manual_approval.py` for interactive post-processing review.

## Current pipeline

For each discovered sample directory, `main.py` performs the following steps:

1. Find image files whose names match the current green-box and red-box regex patterns.
2. Create a per-run manifest directory under `run_state/` and emit `run_started`.
3. Reuse cached JSON if it already exists, unless `--always-process` is set.
4. Keep uncached image paths as paths only during the scan, then load and process each uncached image one at a time.
5. Emit discovery, queue, start, completion, failure, and skip events from the parent process as directory work advances.
6. When resume mode is enabled, skip directories that were already completed in the prior run and optionally requeue failed ones.
7. For red-box images:
   - extract one or more red-box interiors,
   - preprocess each extracted image,
   - detect particles,
   - build a `Relations` template from the largest particle and its nearby particles,
   - save schema-versioned metadata to `data/*_processed_data.json` and the processed image array to `data/*_processed_image.npy`.
8. For green-box images:
   - extract the green-box interior,
   - preprocess the image,
   - remove large aggregations,
   - detect particles,
   - save schema-versioned metadata to `data/*_processed_data.json` and the processed image array to `data/*_processed_image.npy`.
9. If not in `--process-only` mode, compare every red-box template against every green-box particle field.
10. Save sorted CSVs, annotated result images, and a derived manifest summary for the run.

## Main modules

### `main.py`

Responsibilities:

- Resolve the root directory.
- Discover sample directories recursively.
- Coordinate directory-level concurrency with either `ThreadPoolExecutor` or `ProcessPoolExecutor`.
- Load or generate processed particle/image caches.
- Write schema-v2 split caches while continuing to read legacy JSON-only caches.
- Stream uncached image loading within each directory worker so only one source image is read at a time.
- Persist run-state events and derived resume summaries under `run_state/`.
- Filter discovered directories for resume mode based on a prior manifest.
- Separate red-box relation templates from green-box particle fields.
- Score red/green combinations and write final outputs.

Important functions:

- `get_image_paths(samples_dir)`: returns eligible `.tif` and `.tiff` paths in one sample directory.
- `get_sample_dirs(root_dir)`: returns directories containing at least one `GB*.tif` file.
- `is_image_processed(image_path)`: checks for cached JSON outputs.
- `load_red_square_image_data(image_path)`: reloads a cached red-box template and image.
- `load_green_square_image_data(image_path)`: reloads cached green-box particles and image.
- `calc_avg_horizontal_distance(...)`: computes the mean horizontal dummy-vs-match offset for one candidate anchor.
- `process_sample_dir(...)`: processes one sample directory end to end.

Cache behavior:

- Writers now emit `schema_version = 2` metadata JSON plus adjacent `.npy` image caches.
- Readers first try the split-cache metadata fields and fall back to legacy `image_data` when `schema_version` is absent.
- If a cache is missing required metadata or its referenced `.npy` file is invalid, the loader returns `(None, None)` so the directory can be reprocessed instead of crashing the run.

Resume behavior:

- `--resume-last-run` resumes from the newest prior manifest.
- `--resume-run <run_id>` resumes from a specific prior manifest.
- `--retry-failed` requeues directories that previously ended in `failed`; otherwise they are skipped.
- resume decisions are based on manifest state, not only on existing output files.

### `process_image.py`

Responsibilities:

- Extract bordered regions from source images.
- Preprocess those crops to isolate likely particle pixels.

Current preprocessing stages in `process_image()`:

1. median filtering,
2. grayscale conversion,
3. large bright-cluster removal,
4. halo smoothing,
5. background estimation with morphological opening,
6. background-weighted foreground generation,
7. adaptive thresholding,
8. RGB masking,
9. large colour-cluster removal,
10. outer-edge cleanup.

### `handle_prominent_features.py`

Responsibilities:

- Detect connected particle regions.
- Build `Particle_Details` objects.
- Build and serialize `Relations` templates.
- Score every green-box anchor particle against a red-box template.
- Draw and save match annotations.

Key classes:

- `Particle_Details`: stores bounding-box geometry, centre, copied particle pixels, average colour, nearby-particle list, and optional ID.
- `Relations`: stores the red-box anchor particle plus related particles and provides matching/scoring helpers.

Important functions:

- `detect_particles(image)`
- `get_particle_details(particle_positions, image)`
- `get_relations_object(red_square_particle_details)`
- `get_percentages(relations_object, full_image_particle_details, image, samples_dir, progress=None)`
- `display_best_percentages(...)`

### `helper_functions.py`

Responsibilities:

- save Matplotlib figures and raw image arrays,
- append to log files,
- load images with `skimage.io.imread`,
- classify pixels as red, green, or black,
- write sorted CSV outputs.

Performance note:

- `process_image.log` and `timing.log` are disabled by default unless `SIP_DEBUG_LOGS=1` is set, which reduces disk I/O during large batch runs.

### `progress.py`

Responsibilities:

- Display concurrent directory progress in the terminal.

Current public classes:

- `DirState`: mutable progress state for one active directory slot.
- `MultiProgressTracker`: manages the multi-row terminal display and milestone messages.
- `NullProgress`: no-op progress sink used by subprocess directory workers.

There is no active `ProgressTracker` class in the current codebase.

### `manual_approval.py`

Responsibilities:

- Review the batch output interactively.
- Let the user override incorrect best-match rows.
- Write combined root-level summary CSV files.

Manual-approval behavior:

1. load `percentage_data.csv`,
2. group rows by `green_square`,
3. show the latest `all_best_percentages` image and the original `locations fixed` image,
4. identify incorrect red-square contributors,
5. browse candidates page by page,
6. mark the chosen row with `percentage=999`,
7. write `master_percentage_data.csv` and `master_percentage_data_best.csv`.

## Data written by the pipeline

Per sample directory:

- `data/*_processed_data.json`
- `data/*_processed_image.npy`
- `data/percentage_data.csv`
- `data/horizontal_distance.csv`
- `data/particles.csv`
- `pictures/`
- `best_percentage_images/`

Per top-level batch run:

- `run_state/<run_id>/events.jsonl`
- `run_state/<run_id>/summary.json`

Root directory after manual approval:

- `master_percentage_data.csv`
- `master_percentage_data_best.csv`

## Matching and scoring

For each candidate anchor particle in a green-box image:

1. `Relations.get_equivalent_particles_dict()` builds the best one-to-one assignment from red-box related particles to green-box particles.
2. `Relations.percentage_match()` scores that assignment.
3. `get_percentages()` sorts all anchors by score and keeps the top 16.

Current score composition:

```text
percentage = base_score * completeness_ratio * geometry_consistency
```

Supporting details:

- `base_score` comes from the matched-only per-particle scores.
- Each per-particle score uses exponential decay on the distance from the expected dummy position.
- `_selectivity_penalty()` lowers the score when the best candidate is not much better than the second-best candidate.
- `geometry_consistency` rewards layouts that preserve pairwise particle distances.

## Concurrency model

- Directory-level concurrency happens in `main.py`.
- Image processing inside one sample directory is sequential, and uncached source images are now loaded from disk one at a time instead of being preloaded for the whole directory.
- Manifest writes happen only in the parent process, including when `--parallel-mode process` is used.
- `MultiProgressTracker` provides one live terminal row per active directory worker.
- Helper image saving uses locks to avoid filename collisions and Matplotlib state races.

## Current limitations reflected by the code

- The repository does not include a dependency manifest.
- `manual_approval.py` requires a GUI-capable environment because it uses `TkAgg`.
- `get_sample_dirs()` discovers directories based on `GB*.tif`, so directories containing only `.tiff` files will not be discovered automatically.
