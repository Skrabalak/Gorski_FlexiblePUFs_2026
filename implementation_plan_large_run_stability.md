# Implementation Plan: Large-Run Stability and Resume Safety

## Goal

Improve the batch pipeline so long or intensive runs, especially 1000+ sample directories, are less likely to slow to a crawl, exhaust memory, or lose progress after interruption.

This plan covers three changes only:

1. stream unprocessed images one at a time instead of loading a whole directory batch into memory,
2. add a persistent run manifest with resume and retry support,
3. replace JSON-embedded image arrays with compact binary image caches while keeping metadata structured and backward-compatible.

This plan explicitly does not change the matching algorithm itself.

---

## Guiding Principles

- Keep the current red-box / green-box workflow intact.
- Prefer staged changes that can be shipped independently.
- Preserve compatibility with existing cached outputs during migration.
- Ensure parent-process coordination is the single source of truth for long-run state.
- Make failures recoverable rather than trying to prevent every possible crash.

---

## Current Pain Points

### 1. Directory-level image loading is memory-heavy

`process_sample_dir()` currently collects all unprocessed image paths and then loads every one of them at once through `load_io_images()`. This increases RAM usage with directory size and can cause paging under sustained multi-directory concurrency.

### 2. Progress recovery is implicit rather than explicit

The system can reuse per-image processed data, but there is no durable run ledger that records which directories were queued, started, completed, failed, or should be retried.

### 3. Cached image arrays are stored inefficiently

Processed image arrays are currently serialized into JSON via `tolist()`. This creates unnecessary CPU cost, memory churn, and disk overhead during both writes and later reloads.

---

## Delivery Order

Recommended order:

1. streaming image processing,
2. run manifest and resume support,
3. split metadata/image caches with binary image storage.

Reasoning:

- streaming reduces memory pressure immediately and is the least disruptive change,
- the run manifest provides operational safety for long unattended runs,
- cache-format migration is the most invasive and is safer after execution flow is more stable.

---

## Phase 1: Stream Images Per Directory

### Objective

Ensure each active worker keeps only one source image in memory at a time unless a later bounded-prefetch optimization is intentionally introduced.

### Scope

Files primarily affected:

- `main.py`
- `helper_functions.py`

### Current Behavior

Inside `process_sample_dir()`:

1. image paths are discovered,
2. cached outputs are loaded when available,
3. uncached image paths are appended to `unprocessed_images`,
4. all uncached images are loaded at once,
5. the processing loop iterates over the in-memory list.

### Target Behavior

Inside `process_sample_dir()`:

1. image paths are discovered,
2. cached outputs are loaded when available,
3. uncached image paths are retained as paths only,
4. each uncached image is read, processed, saved, and released before the next one starts,
5. only the minimum processed objects required for the later matching stage remain in memory.

### Design Steps

#### Step 1 — Separate classification from loading

Keep the initial image-path scan, but make it produce:

- cached image results loaded immediately,
- a plain list of uncached file paths.

No image bytes should be loaded during this classification pass.

#### Step 2 — Replace batch loading with one-at-a-time loading

Remove the `load_io_images(unprocessed_images)` dependency from the main batch path.

Instead, for each uncached path:

1. load exactly one image,
2. call the existing `_process_single()` logic or equivalent extracted helper,
3. append only processed results needed for later directory matching,
4. allow the raw image object to go out of scope immediately.

#### Step 3 — Keep the worker contract stable

Avoid redesigning the full red-square / green-square logic in the same pass.

Preferred approach:

- keep `_process_single()` as the logical unit of per-image work,
- change its caller so it receives one loaded image at a time rather than a directory-wide preloaded list.

#### Step 4 — Minimize retained in-memory objects

After each image is processed:

- keep processed particle metadata,
- keep only the processed image arrays that are genuinely required for matching or output generation later in that same directory,
- avoid retaining original source image arrays.

#### Step 5 — Leave room for future bounded prefetch

Do not add prefetching in the first pass.

If later needed, add a tiny bounded queue such as 1 to 3 images per worker rather than restoring full-directory preloading.

### Risks

- refactoring may accidentally change ordering assumptions inside `process_sample_dir()`,
- code paths that currently assume loaded image lists may need minor interface cleanup,
- progress display should continue to report per-image advancement correctly after the loop changes.

### Validation

- compare outputs for a small sample directory before and after the refactor,
- run with `--always-process` to force the full uncached path,
- monitor peak memory usage on a directory with many TIFF files,
- confirm that image ordering and output naming remain unchanged.

### Success Criteria

- peak memory use per active worker becomes roughly bounded by one in-flight source image plus retained processed results,
- no change in output correctness,
- no directory-level batching of uncached image bytes remains in the main path.

---

## Phase 2: Add A Persistent Run Manifest

### Objective

Make long runs resumable and auditable by recording directory state explicitly rather than inferring progress only from output files.

### Scope

Files primarily affected:

- `main.py`
- potentially a new helper module such as `run_manifest.py`

### Core Design Decision

Use a parent-process-owned append-only event log rather than a shared writable state file updated by workers.

Reasoning:

- simpler concurrency model,
- avoids cross-process file locking issues,
- preserves a recoverable history of what happened,
- works in both thread and process modes.

### Manifest Layout

Recommended structure:

```text
run_state/
  2026-03-28--14-22-10/
    events.jsonl
    summary.json
```

`events.jsonl` is append-only and contains one JSON object per line.

`summary.json` is optional derived state written periodically or at shutdown for quicker resume startup.

### Event Schema

Each event should contain:

- `run_id`
- `timestamp`
- `event_type`
- `sample_dir`
- `attempt`
- optional `duration_seconds`
- optional `error_type`
- optional `error_message`
- optional `parallel_mode`

### Minimum Event Types

- `directory_discovered`
- `directory_queued`
- `directory_started`
- `directory_completed`
- `directory_failed`
- `directory_skipped`
- `run_started`
- `run_completed`
- `run_aborted`

### State Model Derived From Events

For each directory, the latest effective state should be one of:

- `discovered`
- `queued`
- `running`
- `completed`
- `failed`
- `skipped`

Optional future refinement:

- `processing_complete`
- `matching_complete`
- `artifacts_complete`

That split is useful if directory work is ever resumable inside the directory rather than only at the directory boundary.

### Design Steps

#### Step 1 — Create run identity at startup

At the start of `main()`:

- generate a `run_id`,
- create a per-run state directory,
- emit `run_started`.

#### Step 2 — Record discovery before scheduling

After `get_sample_dirs(root_dir)` returns, emit one `directory_discovered` event per directory in deterministic order.

#### Step 3 — Record scheduling lifecycle in the parent process

Before submitting a directory to the executor:

- emit `directory_queued`.

When the worker begins actual work, the parent should record `directory_started` based on submission time or on an explicit worker-start notification if that is later introduced.

When the future resolves:

- emit `directory_completed` on success,
- emit `directory_failed` on exception.

#### Step 4 — Add resume mode

Add a CLI flag such as:

- `--resume-last-run`
- or `--resume-run <run_id>`

Resume logic should:

1. read the prior manifest,
2. rebuild latest directory states,
3. skip directories already marked `completed`,
4. optionally retry `failed` directories depending on policy.

#### Step 5 — Add retry policy

A minimal first version:

- no automatic retry by default,
- optional `--retry-failed` to requeue failed directories from a previous run,
- record incrementing `attempt` counts.

#### Step 6 — Keep output-file heuristics as a secondary safety net

The manifest should become the primary run-state source.

Existing cache checks can remain useful, but they should not be the only mechanism for deciding what work is already done.

### Risks

- ambiguous behavior if manifest state and on-disk outputs disagree,
- restart semantics must be defined clearly for `--always-process`,
- process-mode workers should not attempt to write the manifest directly.

### Validation

- interrupt a run mid-flight and confirm that resume skips completed directories,
- simulate a failed directory and verify it is recorded as failed without losing the rest of the run,
- test both thread and process modes,
- verify deterministic ordering on resume.

### Success Criteria

- a killed run can be resumed without rescanning completion heuristics for every directory,
- failed directories are clearly listed with error summaries,
- completed directories are not redone unintentionally unless explicitly requested.

---

## Phase 3: Split Metadata And Image Caches

### Objective

Reduce CPU, memory, and disk overhead by storing processed image arrays in binary form while preserving human-readable structured metadata.

### Scope

Files primarily affected:

- `main.py`
- potentially `handle_prominent_features.py` if shared cache helpers are centralized

### Current Behavior

The cache writers serialize image arrays using `image.tolist()` into JSON files alongside particle metadata and relation data.

### Target Behavior

Store:

- metadata in JSON,
- processed image arrays in adjacent binary files such as `.npy`.

Recommended example layout:

```text
data/
  GB.sample_processed_data.json
  GB.sample_processed_image.npy
```

The JSON remains the authoritative metadata marker and includes:

- schema version,
- image file reference,
- particle metadata,
- relation metadata for red-square results,
- maybe image shape and dtype for validation.

### Backward-Compatibility Strategy

Readers must support both:

1. new split-cache format,
2. legacy JSON-only format.

Writers should emit only the new format after migration begins.

This allows old caches to continue working until they are naturally refreshed.

### Design Steps

#### Step 1 — Introduce cache schema versioning

Each metadata JSON should include a `schema_version` field.

Suggested values:

- `1` for current JSON-with-image-data format,
- `2` for split JSON + binary image cache.

#### Step 2 — Define new writer behavior

For both green-square and red-square cache saves:

1. write the processed image array to a temp `.npy` file,
2. write the metadata JSON to a temp `.json` file with references to the image cache,
3. rename the binary image file into place,
4. rename the JSON metadata file into place last.

The JSON file should be the completion marker. If JSON exists, the binary image file it references must already be in place.

#### Step 3 — Define new reader behavior

Reader order:

1. look for schema-versioned JSON metadata,
2. if metadata references a binary image cache, load the array from that file,
3. otherwise fall back to legacy `image_data` from JSON.

This keeps the migration safe and incremental.

#### Step 4 — Validate cache integrity

On load, confirm:

- referenced binary image file exists,
- dtype and shape are reasonable,
- required metadata fields are present.

If validation fails, treat the cache as invalid and allow reprocessing rather than crashing the full run.

#### Step 5 — Decide compression policy

Default recommendation:

- start with uncompressed `.npy` for simplicity and speed,
- revisit `.npz` or another compressed format only if disk usage remains problematic.

The first objective is stability and reduced conversion overhead, not maximum compression ratio.

### Risks

- split-file atomicity needs a clear completion marker,
- partial cache writes must not be mistaken for valid caches,
- migration logic must handle both green and red-square variants consistently.

### Validation

- write and reload both green and red-square caches in the new format,
- confirm legacy caches still load correctly,
- compare output correctness after reloading from both formats,
- benchmark cache save and load time on representative large directories.

### Success Criteria

- processed image arrays are no longer serialized through `tolist()` in the main path,
- cache write/read CPU time decreases,
- cache files consume less disk and produce less serialization overhead,
- old caches remain readable during the transition.

---

## Cross-Cutting Concerns

### Logging

During all three phases, keep log verbosity controlled. Long-run improvements are easier to verify if logs remain concise and structured.

### Progress Display

Streaming-image refactors must not break the current per-directory progress display. Image counters and phase percentages should continue to reflect one-at-a-time processing.

### Process Mode Compatibility

Any new helper introduced for manifests or cache IO must work cleanly in both:

- thread mode,
- process mode.

The parent process should remain the sole owner of scheduler-level state.

---

## Rollout Strategy

### Milestone A — Streaming only

- ship image streaming with no manifest or cache-format changes,
- verify memory reduction and output parity.

### Milestone B — Manifest and resume

- add append-only run tracking,
- add resume behavior and failed-directory reporting,
- verify restart safety under real interruption scenarios.

### Milestone C — New cache format

- add versioned split caches,
- retain legacy readers,
- benchmark cache operations and confirm migration safety.

---

## Test Plan

### Functional Tests

- process a small sample directory in the old and new execution models and compare outputs,
- process a directory with red-square images that produce multiple extracted squares,
- process a directory with only green-square images,
- process a directory with only red-square images.

### Recovery Tests

- interrupt a run after several completed directories and resume,
- simulate a corrupted cache file and verify that the directory can be reprocessed,
- simulate a failed directory and confirm the rest of the run completes.

### Performance Tests

- compare peak RAM before and after streaming-image refactor,
- compare cache load/save time before and after binary image caches,
- compare total runtime on a representative multi-directory batch.

### Scale Tests

- run a larger root set with process mode enabled,
- verify manifest growth remains manageable,
- confirm no unbounded accumulation of in-memory raw images.

---

## Recommended First Implementation Slice

If only one slice is implemented first, do this:

1. refactor `process_sample_dir()` to stream uncached images one at a time,
2. keep all current cache and output formats unchanged,
3. verify memory behavior on a large directory set.

That delivers the lowest-risk stability gain and makes later manifest and cache work easier to validate.

---

## Expected Outcome

After all three phases:

- long runs should use memory more predictably,
- interrupted runs should be resumable without guesswork,
- cache IO should be materially cheaper,
- large directory sets should be operationally safer even if some directories fail.