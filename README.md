# sarah_image_processing_project

Image-processing and particle-matching tooling for Sarah Gorski's microscopy workflow.

## What the code does

The repository processes microscopy images whose filenames begin with `GB.` and separates them into two roles:

- Green-box images: full-field images used as the search space.
- Red-box images: `locations fixed` images used as the reference template.

For each sample directory, the pipeline:

1. Finds eligible source images.
2. Loads uncached source images one at a time instead of preloading a full directory batch.
3. Crops the content inside the green or red border.
4. Runs the preprocessing pipeline to suppress background noise and isolate particles.
5. Detects connected non-black particle regions.
6. Builds `Particle_Details` objects and, for red-box images, a `Relations` template.
7. Scores every candidate anchor particle in each green-box image against each red-box template.
8. Writes CSV outputs and annotated images.

## Main scripts

- `main.py`: batch processing pipeline.
- `manual_approval.py`: interactive review and override tool for `percentage_data.csv`.
- `process_image.py`: border extraction and image preprocessing.
- `handle_prominent_features.py`: particle detection, relation matching, scoring, and annotation.
- `helper_functions.py`: image I/O, logging, CSV writing, and pixel helpers.
- `progress.py`: terminal progress display for concurrent directory processing.

## How sample directories are discovered

`main.py` searches the configured root directory recursively and processes every directory that contains at least one `GB*.tif` file. Within each discovered sample directory, only files matching the current filename regexes are treated as source images:

- Green-box images: `GB.` filenames that do not contain `locations fixed` and end in `.tif` or `.tiff`.
- Red-box images: `GB.` filenames that do contain `locations fixed` and end in `.tif` or `.tiff`.

## Running the batch pipeline

Run from the repository root:

```bash
python main.py
```

Command-line options currently supported by the code:

- `--root-dir PATH`: override the default root directory.
- `--process-only` or `-po`: process or load images and write cached data, but skip matching and CSV generation.
- `--single-threaded` or `-st`: limit directory processing to one worker.
- `--parallel-mode thread|process` or `-pm thread|process`: choose directory-level threading or multiprocessing. `process` is the better choice for CPU-heavy Windows runs because it bypasses the Python GIL.
- `--always-process` or `-ap`: ignore cached JSON and reprocess all images.
- `--save-step-images` or `-ssi`: save intermediate preprocessing-step images.
- `--max-dirs N` or `-md N`: cap directory-level concurrency at `N` workers for either parallel mode.
- `--resume-last-run`: create a new run that resumes from the most recent manifest under `run_state/`.
- `--resume-run RUN_ID`: create a new run that resumes from the specified prior manifest.
- `--retry-failed`: when resuming, requeue directories that were marked `failed` in the prior run; without this flag they are skipped.
- `--help`, `-h`, `/?`, `--h`, `-help`: show the built-in help text.

Environment overrides:

- `SIP_PARALLEL_MODE=thread|process`: default parallel mode when `--parallel-mode` is omitted.
- `SIP_DEBUG_LOGS=1`: re-enable the very chatty `process_image.log` and `timing.log` files. They are now disabled by default because they add noticeable disk I/O overhead during large runs.

For Windows 10, the highest-throughput invocation is usually:

```bash
python main.py --parallel-mode process --max-dirs 12
```

Replace `12` with the number of logical CPUs you actually want to dedicate to the batch run.

Root-directory selection logic:

- If `--root-dir` is supplied, that path is used.
- Otherwise, when the `COMPUTER` environment variable is anything other than `SARAH`, the default root is `Image_Samples/root_dir` under this repository.
- When `COMPUTER=SARAH`, the code uses the production Windows path hardcoded in `main.py`.

Resume semantics:

- Resuming creates a new run manifest directory and uses the selected prior run only as input state.
- Directories marked `completed` in the resumed-from run are skipped.
- Directories marked `failed` are skipped unless `--retry-failed` is supplied.
- Directories left in `discovered`, `queued`, or `running` state by an interrupted run are requeued.
- If `--always-process` is combined with a resume flag, only the requeued directories ignore per-image caches; previously completed directories are still skipped.

## Outputs written by `main.py`

For each sample directory, the batch pipeline writes:

- `data/*_processed_data.json`: schema-versioned cached metadata for processed particles and, for red-box images, relations.
- `data/*_processed_image.npy`: processed image arrays referenced by the adjacent metadata JSON.
- `data/percentage_data.csv`: scored match rows for every red/green comparison.
- `data/horizontal_distance.csv`: average horizontal offset for the best match of each red/green pair.
- `data/particles.csv`: per-anchor particle dump written during scoring.
- `pictures/`: particle and preprocessing diagnostic images.
- `best_percentage_images/`: best-match annotated images for each green-box image.

For each top-level batch run, the repository also writes:

- `run_state/<run_id>/events.jsonl`: append-only scheduler events for discovery, queueing, starts, completion, failures, skips, and run lifecycle.
- `run_state/<run_id>/summary.json`: derived latest state for each discovered directory plus attempt counts and run metadata.

Processing note:

- `process_sample_dir()` now keeps uncached source-image loading sequential within each directory worker. It classifies cached versus uncached inputs first, then reads, processes, and releases each uncached image one at a time. Directory-level parallelism is unchanged.
- Cache writers now emit schema-v2 split caches: JSON metadata stays the completion marker, while processed image arrays are stored in adjacent `.npy` files. Readers still accept legacy JSON-only caches that embed `image_data`.

When `--parallel-mode process` is used, the live multi-row terminal progress display is disabled because subprocess workers cannot safely update the shared TTY block. Completion messages are still printed as directories finish.

`percentage_data.csv` currently contains these columns:

- `red_square`
- `green_square`
- `percentage`
- `distance`
- `base_score`
- `completeness_ratio`
- `geometry_consistency`
- `individual_scores`
- `particle_details`

Scoring is currently:

```text
percentage = base_score * completeness_ratio * geometry_consistency
```

Where:

- `base_score` is the mean of the matched-only per-particle scores.
- `completeness_ratio` is `matched_relations / expected_relations`.
- `geometry_consistency` measures how well pairwise distances are preserved.

The `distance` column is the average absolute horizontal offset between each matched particle and its expected dummy position for that candidate anchor.

## Running manual approval

After `main.py` finishes, run:

```bash
python manual_approval.py
```

Optional argument:

- `--root-dir PATH`: override the root directory.

What the manual approval tool does:

1. Loads `data/percentage_data.csv` for each sample directory.
2. Groups rows by `green_square` value.
3. Shows the combined `all_best_percentages` image beside the `locations fixed` source image.
4. If you reject the current result, lets you mark one or more red-square contributors as incorrect.
5. For each incorrect contributor, shows a paged candidate browser and lets you choose an override.
6. Marks the selected row with `percentage=999`.
7. Writes two root-level summary CSV files:
   - `master_percentage_data.csv`
   - `master_percentage_data_best.csv`

`manual_approval.py` uses Matplotlib's `TkAgg` backend, so it requires an interactive desktop environment with Tk support.

## Python dependencies used by the current code

The repository imports these third-party packages directly:

- `numpy`
- `matplotlib`
- `imageio`
- `scipy`
- `scikit-image`
- `Pillow` (via `PIL` inside the manual-approval image tile builder)

There is no dependency manifest in the repository at the moment, so the environment must already provide those packages.
