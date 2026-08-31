# Implementation Plan: Root Directory Loop + Clean Terminal Output

## Overview

Two goals:
1. Retarget `main.py` so it points at a `root_dir`, discovers all subdirectories
   containing `GB*.tif` files, and runs the full workflow on each one.
2. Overhaul terminal output across the whole script suite to always show a clean
   progress line and reduce clutter.

---

## Part 1 — Root Directory Loop

### Step 1 — Parameterise `get_image_paths()`

**File:** `main.py`

Change the signature from:
```python
def get_image_paths() -> list[str]:
```
to:
```python
def get_image_paths(samples_dir: Path) -> list[str]:
```

Remove all internal `if DEBUG / else` path-selection logic from the function body.
The function becomes purely about globbing; the caller decides which directory to use.
Keep the two filter lines unchanged:
- only files matching `GB*.tif`
- exclude files with `"Locations"` in the name

---

### Step 2 — Add `get_sample_dirs(root_dir: Path) -> list[Path]`

**File:** `main.py`

New function placed just below `get_image_paths()`:

```python
def get_sample_dirs(root_dir: Path) -> list[Path]:
    """Return every direct child of root_dir that contains at least one GB*.tif file."""
    return sorted(
        d for d in root_dir.iterdir()
        if d.is_dir() and any(d.glob("GB*.tif"))
    )
```

Sorting ensures a deterministic, reproducible processing order.

---

### Step 3 — Define `root_dir` in `main()`

**File:** `main.py`

Move the `DEBUG`/SARAH path-selection logic from `get_image_paths()` into `main()`:

```python
if DEBUG:
    root_dir = current_dir / "Image_Samples" / "root_dir"
else:
    root_dir = Path(r"D:\Codes!!\Supercomputer_calc\Testing_DPI\grey_AP3")
```

`root_dir` is now the single variable to change when retargeting the run.
The name `root_dir` can be overridden via CLI (see Step 7).

---

### Step 4 — Extract per-directory logic into `process_sample_dir()`

**File:** `main.py`

The large block in `main()` that runs from the `get_image_paths()` call through to the
final CSV writes all operates on one `samples_dir`. Extract it to:

```python
def process_sample_dir(
    samples_dir: Path,
    save_dir: str,
    process_only: bool,
    single_threaded: bool,
    always_process: bool,
    save_step_images: bool,
    progress: ProgressTracker,   # see Part 2 Step 1
) -> None:
```

Everything internal already derives paths from `samples_dir` (e.g. `samples_dir / "data"`,
`samples_dir / "best_percentage_images"`), so it works correctly with no further changes
to save/load functions.

---

### Step 5 — Replace `main()` body with a loop

**File:** `main.py`

After resolving `root_dir` and `save_dir`, the core of `main()` becomes:

```python
sample_dirs = get_sample_dirs(root_dir)
write_to_file("main.log", f"found {len(sample_dirs)} sample directories\n")

progress = ProgressTracker(total_dirs=len(sample_dirs))  # see Part 2

for dir_idx, samples_dir in enumerate(sample_dirs):
    progress.set_dir(dir_idx, samples_dir.name)
    write_to_file("main.log", f"--- processing directory: {samples_dir} ---\n")
    process_sample_dir(samples_dir, save_dir, process_only, single_threaded,
                       always_process, save_step_images, progress)

progress.done()
```

Each call to `process_sample_dir` is fully self-contained.

---

### Step 6 — Decide `save_dir` scope

`save_dir` (currently the repo-level `pictures/` folder) is used for intermediate
debug images. Two options:

| Option | Path | Trade-off |
|--------|------|-----------|
| **Global** (current behaviour) | `<repo>/pictures/` | Simple; filenames from different subdirs may collide |
| **Per-subdir** | `samples_dir / "pictures"` | Clean isolation; output stays with its source data |

**Decision:** use **per-subdir** (`samples_dir / "pictures"`). Each directory's debug
images are written alongside that directory's other output, keeping everything
self-contained.

---

### Step 7 (optional) — `--root-dir` CLI argument

**File:** `main.py`

Parse from `sys.argv` so the launcher scripts can drive the run without touching source:

```python
root_dir_arg = next((args[i + 1] for i, a in enumerate(args) if a == "--root-dir"), None)
if root_dir_arg:
    root_dir = Path(root_dir_arg)
```

---

## Part 2 — Clean Terminal Output

The current codebase has ~30+ active `print()` calls across four files, producing a
noisy, unstructured log. The goal is:  
- **One always-visible progress line** (overwritten in-place with `\r`).  
- **Periodic milestone lines** that scroll past (one per major phase).  
- **All fine-grained chatter** redirected to `main.log` only (already written there).

---

### Step 8 — Audit and classify all `print()` calls

Go through every active (non-commented) `print()` in the four core files and assign a
category:

| Category | Action | Examples |
|----------|--------|---------|
| **Progress** | Replace with `ProgressTracker` calls | `"Processing image X of Y"` |
| **Milestone** | Keep as a single terse line, de-duplicate | `"No red squares found"`, `"No valid best_combination found"` |
| **Noise** | Remove (info already in `main.log`) | All `process_image.py` step banners (`"step 0: original image"` … `"step 9: sanitize edges"`), `"printing {name}"`, `"saving {name}"`, `"data saved successfully"`, `"Saving processed data to …"`, `"getting equivalent particles dictionary"`, `"getting all particles within search distance"`, `"creating array …"`, `"building equivalent_particles_dict"`, `"populating checking_pixels_array"`, `"median particle size: …"`, `"small particle size: …"`, `"finding percentage for …"` |
| **Timing / debug** | Remove (or guard behind `DEBUG`) | `"unique_combinations time: …"` |

Files affected: `main.py`, `process_image.py`, `handle_prominent_features.py`,
`helper_functions.py`.

---

### Step 9 — Create a `ProgressTracker` class

**File:** New file `progress.py` in the project root.

```python
import sys

class ProgressTracker:
    def __init__(self, total_dirs: int):
        self.total_dirs = total_dirs
        self.dir_idx = 0
        self.dir_name = ""
        self.total_images = 0
        self.image_idx = 0
        self.phase = ""

    def set_dir(self, idx: int, name: str) -> None:
        self.dir_idx = idx
        self.dir_name = name
        self._render()

    def set_images(self, total: int) -> None:
        self.total_images = total
        self.image_idx = 0
        self._render()

    def set_image(self, idx: int) -> None:
        self.image_idx = idx
        self._render()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self._render()

    def milestone(self, msg: str) -> None:
        """Print a message that scrolls (not overwritten)."""
        sys.stdout.write("\r" + " " * 100 + "\r")  # clear progress line
        print(msg)

    def done(self) -> None:
        sys.stdout.write("\r" + " " * 100 + "\r")
        print("All directories processed.")

    def _render(self) -> None:
        dir_pct = int(100 * self.dir_idx / self.total_dirs) if self.total_dirs else 0
        img_part = (
            f"  Image {self.image_idx}/{self.total_images}"
            if self.total_images
            else ""
        )
        phase_part = f"  [{self.phase}]" if self.phase else ""
        line = (
            f"\rDir {self.dir_idx + 1}/{self.total_dirs} ({dir_pct}%)"
            f"  {self.dir_name}{img_part}{phase_part}"
        )
        sys.stdout.write(line.ljust(100))
        sys.stdout.flush()
```

Key design decisions:
- `\r` overwrites the same terminal line — the progress counter is always visible
  without scrolling.
- `milestone()` clears the progress line before printing, so milestones scroll past
  cleanly above it.
- `ProgressTracker` is passed into `process_sample_dir` (and optionally down into
  `_process_single`) so all layers can update it.

---

### Step 10 — Thread-safety note for progress output

`_process_single` is called from a `ThreadPoolExecutor`. Two threads writing to
`sys.stdout` simultaneously will interleave characters.

Simple fix: wrap the `sys.stdout.write` + `sys.stdout.flush` pair in a
`threading.Lock` stored on `ProgressTracker`:

```python
import threading

class ProgressTracker:
    def __init__(self, ...):
        ...
        self._lock = threading.Lock()

    def _render(self) -> None:
        with self._lock:
            # ... same as above
```

Milestone prints should also acquire the lock before clearing and printing.

---

### Step 11 — Wire `ProgressTracker` into `process_sample_dir` and `_process_single`

Replace the existing scattered `print()` calls with `progress` method calls:

| Old call | Replacement |
|----------|-------------|
| `print(f"Processing image {idx+1} of {len(io_images)} unprocessed images")` | `progress.set_image(idx + 1)` |
| Phase banners in `_process_single` (e.g. "processing red square image") | `progress.set_phase("red square")` |
| `print("Saving processed data to …")` / `"data saved successfully"` | remove |
| `print("No red squares found in image: …")` | `progress.milestone(f"WARNING: No red squares found: {image_path}")` |
| `print("No valid best_combination found")` | `progress.milestone("WARNING: No valid best combination found")` |

At the top of `process_sample_dir`, call:
```python
image_paths = get_image_paths(samples_dir)
progress.set_images(len(image_paths))
progress.set_phase("loading")
```

---

## Summary of Files Changed

| File | Changes |
|------|---------|
| `main.py` | New `get_sample_dirs()`, parameterise `get_image_paths()`, extract `process_sample_dir()`, loop in `main()`, integrate `ProgressTracker`, remove noisy prints, optional `--root-dir` flag |
| `process_image.py` | Remove all 9 `"step N: …"` print banners and the blank `print()` |
| `handle_prominent_features.py` | Remove ~10 internal chatter prints; guard `"unique_combinations time"` behind `DEBUG`; keep `"No valid best_combination found"` as a milestone routed through `ProgressTracker` |
| `helper_functions.py` | Remove `"printing {name}"` and `"saving {name}"` prints; change `write_to_file` to default `print_to_terminal=False` |

## What Does NOT Change

- All `write_to_file("main.log", …)` calls — the log file keeps full verbosity.
- Any logic in save/load functions.
- Threading model, CSV writing, image processing algorithms.
- `DEBUG` flag behaviour.
