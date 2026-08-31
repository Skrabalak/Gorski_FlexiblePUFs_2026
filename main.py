import hashlib
import json
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from handle_prominent_features import (
    Particle_Details,
    Relations,
    assign_particle_ids,
    detect_particles,
    display_best_percentages,
    get_particle_details,
    get_percentages,
    get_relations_object,
    save_particles,
    show_detected_particles,
    show_particle_details,
)
from helper_functions import (
    load_io_image,
    save_io_image,
    write_sorted_horizontal_distance_csv,
    write_sorted_percentage_data_csv,
    write_to_file,
)
from process_image import (
    get_green_square_image,
    get_red_square_image,
    process_image,
    remove_large_aggregations,
)
from progress import DirState, MultiProgressTracker, NullProgress
from run_manifest import RunManifest, get_latest_run_id, load_run_state

# GLOBAL VARIABLES

# Regex patterns that define valid image filenames.
#
# Both types share the base structure:
#   GB.<sample_id>.<material>.<magnification>x.<distance>mm[.<index>]
#
# Green box (reference field images):
#   e.g. GB.AE1.DCpdms.50x.0mm.1.tif
GREEN_BOX_IMAGE_RE = re.compile(
    r"^GB\.(?!.*locations fixed).*\.tiff?$",
    re.IGNORECASE,
)

# Red box (stretched / "locations fixed" images):
#   e.g. GB.AE1.DCpdms.50x.0mm.1.locations fixed.tif
RED_BOX_IMAGE_RE = re.compile(
    r"^GB\.(?=.*locations fixed).*\.tiff?$",
    re.IGNORECASE,
)


variable = os.environ.get("COMPUTER")
DEBUG = variable != "SARAH"
CACHE_SCHEMA_VERSION = 2


def get_flag_value(args: list[str], flags: tuple[str, ...]) -> str | None:
    """Return the value that immediately follows one of ``flags`` in ``args``."""
    for index, arg in enumerate(args):
        if arg in flags:
            if index + 1 >= len(args):
                raise ValueError(f"Missing value after {arg}")
            return args[index + 1]
    return None


def get_available_worker_count() -> int:
    """Return the affinity-aware CPU count when available, else fall back to ``os.cpu_count``."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is None:
        count = os.cpu_count()
    else:
        count = process_cpu_count()
    return max(1, count or 1)


def resolve_parallel_mode(args: list[str], single_threaded: bool) -> str:
    """Return ``thread`` or ``process`` based on flags and environment overrides."""
    if single_threaded:
        return "thread"

    raw_mode = get_flag_value(args, ("--parallel-mode", "-pm"))
    if raw_mode is None:
        raw_mode = os.environ.get("SIP_PARALLEL_MODE", "thread")

    parallel_mode = raw_mode.strip().lower()
    if parallel_mode not in {"thread", "process"}:
        raise ValueError("Invalid parallel mode. Use '--parallel-mode thread' or '--parallel-mode process'.")
    return parallel_mode


def process_sample_dir_in_subprocess(
    samples_dir: Path,
    process_only: bool,
    always_process: bool,
    save_step_images: bool,
) -> str:
    """Run one directory in a subprocess without live in-terminal progress updates."""
    process_sample_dir(samples_dir, process_only, always_process, save_step_images, NullProgress())
    return samples_dir.name


def resolve_resume_run_id(args: list[str], current_dir: Path) -> str | None:
    """Return the prior run ID requested for resume, if any."""
    resume_last_run = "--resume-last-run" in args
    resume_run_id = get_flag_value(args, ("--resume-run",))

    if resume_last_run and resume_run_id is not None:
        raise ValueError("Use either '--resume-last-run' or '--resume-run <run_id>', not both.")

    if resume_last_run:
        latest_run_id = get_latest_run_id(current_dir)
        if latest_run_id is None:
            raise ValueError("No previous run manifest exists to resume.")
        return latest_run_id

    return resume_run_id


def get_image_paths(samples_dir: Path) -> list[str]:
    """
    Return sorted image paths in a sample directory that match the project's
    green-box or red-box filename conventions.

    Args:
        samples_dir: Directory to scan for ``.tif`` and ``.tiff`` files.

    Returns:
        A list of string paths for files whose names match
        ``GREEN_BOX_IMAGE_RE`` or ``RED_BOX_IMAGE_RE``. Files that do not match
        either pattern are ignored.
    """
    # collect all .tif/.tiff files whose names match either the green-box or
    # red-box pattern; any other files in the directory are ignored
    image_paths = [
        str(p)
        for p in sorted(samples_dir.glob("*.tif")) + sorted(samples_dir.glob("*.tiff"))
        if GREEN_BOX_IMAGE_RE.match(p.name) or RED_BOX_IMAGE_RE.match(p.name)
    ]
    return image_paths


def get_sample_dirs(root_dir: Path) -> list[Path]:
    """
    Return every directory below ``root_dir`` that contains at least one
    ``GB*.tif`` file.

    Args:
        root_dir: Root directory searched recursively with ``Path.rglob``.

    Returns:
        A sorted list of unique parent directories for matching ``.tif`` files.
    """
    return sorted({p.parent for p in root_dir.rglob("GB*.tif")})


def is_image_processed(image_path: str) -> bool:
    """
    Return whether cached processed-data JSON already exists for an image.

    Args:
        image_path: Path to the original source image.

    Returns:
        ``True`` if the image has a matching ``*_processed_data.json`` file in
        its sibling ``data`` directory, or if any indexed red-square cache file
        such as ``*_redsquare0_processed_data.json`` exists there.
    """
    p = Path(image_path)
    data_dir = p.parent / "data"
    # Check the base name first
    if (data_dir / f"{p.stem}_processed_data.json").exists():
        return True
    # Check for any indexed red-square variants (redsquare0, redsquare1, ...)
    if any(data_dir.glob(f"{p.stem}_redsquare*_processed_data.json")):
        return True
    return False


def is_red_square_image(image_path: str) -> bool:
    """
    Return whether an image path matches the red-box filename pattern.

    Args:
        image_path: Path to an image file.

    Returns:
        ``True`` when the basename matches ``RED_BOX_IMAGE_RE``; otherwise
        ``False``.
    """
    return bool(RED_BOX_IMAGE_RE.match(Path(image_path).name))


def get_processed_data_path(image_path: str) -> Path:
    """Return the processed-data JSON path for one source or virtual image path."""
    path = Path(image_path)
    return path.parent / "data" / f"{path.stem}_processed_data.json"


def get_processed_image_cache_path(processed_data_path: Path) -> Path:
    """Return the binary image-cache path adjacent to one processed-data JSON file."""
    cache_name = processed_data_path.name.replace("_processed_data.json", "_processed_image.npy")
    if cache_name == processed_data_path.name:
        raise ValueError(f"Unexpected processed-data filename: {processed_data_path.name}")
    return processed_data_path.with_name(cache_name)


def write_split_processed_cache(processed_data_path: Path, metadata: dict, image: np.ndarray) -> None:
    """Write schema-v2 metadata JSON plus a binary image cache atomically enough for resume safety."""
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    processed_image_path = get_processed_image_cache_path(processed_data_path)
    tmp_image_path = processed_image_path.with_name(f"{processed_image_path.name}.tmp")
    tmp_data_path = processed_data_path.with_name(f"{processed_data_path.name}.tmp")

    image_array = np.asarray(image, dtype=np.uint8)
    metadata_to_write = dict(metadata)
    metadata_to_write["schema_version"] = CACHE_SCHEMA_VERSION
    metadata_to_write["image_cache_file"] = processed_image_path.name
    metadata_to_write["image_shape"] = list(image_array.shape)
    metadata_to_write["image_dtype"] = str(image_array.dtype)

    with tmp_image_path.open("wb") as outfile:
        np.save(outfile, image_array, allow_pickle=False)

    with tmp_data_path.open("w", encoding="utf-8") as outfile:
        json.dump(metadata_to_write, outfile, indent=2, ensure_ascii=False, allow_nan=False)

    tmp_image_path.replace(processed_image_path)
    tmp_data_path.replace(processed_data_path)


def load_cached_image_from_metadata(processed_data_path: Path, metadata: dict) -> np.ndarray:
    """Load and validate the processed image array for either cache schema version."""
    schema_version = metadata.get("schema_version", 1)
    if schema_version >= 2:
        image_cache_file = metadata.get("image_cache_file")
        image_shape = metadata.get("image_shape")
        image_dtype = metadata.get("image_dtype")
        if not image_cache_file:
            raise ValueError("Schema-v2 cache is missing 'image_cache_file'.")
        if not isinstance(image_shape, list) or not image_shape:
            raise ValueError("Schema-v2 cache is missing a valid 'image_shape'.")
        if not image_dtype:
            raise ValueError("Schema-v2 cache is missing 'image_dtype'.")

        image_cache_path = processed_data_path.parent / image_cache_file
        if not image_cache_path.exists():
            raise FileNotFoundError(f"Referenced image cache is missing: {image_cache_path}")

        image = np.load(image_cache_path, allow_pickle=False)
        if not isinstance(image, np.ndarray):
            raise ValueError("Binary image cache did not load as a NumPy array.")
        if image.ndim not in {2, 3}:
            raise ValueError(f"Unexpected cached image rank: {image.ndim}")
        if any(dim <= 0 for dim in image.shape):
            raise ValueError(f"Invalid cached image shape: {image.shape}")
        if image.dtype == np.dtype("O"):
            raise ValueError("Object-dtype image caches are not supported.")
        if list(image.shape) != image_shape:
            raise ValueError(f"Cached image shape mismatch: metadata={image_shape}, actual={list(image.shape)}")
        if str(image.dtype) != str(image_dtype):
            raise ValueError(f"Cached image dtype mismatch: metadata={image_dtype}, actual={image.dtype}")
        return np.asarray(image, dtype=np.uint8)

    image_data = metadata.get("image_data")
    if image_data is None:
        raise ValueError("Legacy cache is missing 'image_data'.")
    return np.asarray(image_data, dtype=np.uint8)


def save_red_square_particle_data(relations_object: Relations, red_square_image: np.ndarray, image_path: str) -> None:
    """
    Serialize a processed red-square result to the image's sibling ``data``
    directory.

    Args:
        relations_object: Matched relation template derived from the processed
            red-square image.
        red_square_image: Processed red-square image array.
        image_path: Source or virtual image path used to derive the output JSON
            filename.

    Side Effects:
        Creates the output directory if needed, assigns particle IDs to every
        particle referenced by ``relations_object``, writes the processed image
        to an adjacent binary cache file, and then writes schema-versioned JSON
        metadata containing particle data and serialized relations.
    """
    processed_data_path = get_processed_data_path(image_path)

    # gather unique particles referenced by the relations object
    unique_particles = []
    # include primary_particle and any in relation_array
    if getattr(relations_object, "primary_particle", None) is not None:
        unique_particles.append(relations_object.primary_particle)
    for p in relations_object.relation_array:
        if p not in unique_particles:
            unique_particles.append(p)

    # assign ids to particles (in-place)
    assign_particle_ids(unique_particles)

    # build collection dict using helper
    coll = {
        "cache_kind": "red_square",
        "particles": [p.to_json() for p in unique_particles],
        "relations": relations_object.to_json(),
    }
    write_split_processed_cache(processed_data_path, coll, red_square_image)


def save_green_square_particle_data(
    particle_details: list[Particle_Details], green_square_image: np.ndarray, image_path: str
) -> None:
    """
    Serialize processed green-square particle data to the image's sibling
    ``data`` directory.

    Args:
        particle_details: Detected particles from the processed green-square
            image.
        green_square_image: Processed green-square image array.
        image_path: Source image path used to derive the output JSON filename.

    Side Effects:
        Creates the output directory if needed, assigns stable particle IDs, and
        writes the processed image to an adjacent binary cache file before
        writing schema-versioned JSON metadata.
    """
    processed_data_path = get_processed_data_path(image_path)

    # assign IDs to all particles
    assign_particle_ids(particle_details)

    # build collection and save
    coll = {
        "cache_kind": "green_square",
        "particles": [p.to_json() for p in particle_details],
    }
    write_split_processed_cache(processed_data_path, coll, green_square_image)


def load_red_square_image_data(image_path: str) -> tuple[Relations | None, np.ndarray | None]:
    """
    Load cached red-square data and reconstruct object references.

    Args:
        image_path: Source or virtual image path whose stem maps to a
            ``*_processed_data.json`` file in the sibling ``data`` directory.

    Returns:
        A tuple ``(relations_obj, image)``. If the cache file does not exist,
        returns ``(None, None)``. When present, the loader rebuilds
        ``Particle_Details`` objects, restores nearby-particle relationships,
        resolves relation particle IDs back to objects, and loads the image
        from either a schema-v2 binary cache or the legacy JSON ``image_data``
        field.
    """
    processed_data_path = get_processed_data_path(image_path)
    if not processed_data_path.exists():
        return None, None

    try:
        with processed_data_path.open("r", encoding="utf-8") as infile:
            coll = json.load(infile)

        particles_data = coll.get("particles")
        if not isinstance(particles_data, list):
            raise ValueError("Red-square cache is missing a valid 'particles' list.")

        particles = [Particle_Details.from_json(pd) for pd in particles_data]
        id_map = {int(p.id): p for p in particles if getattr(p, "id", None) is not None}

        for pd, particle in zip(particles_data, particles):
            nid_list = pd.get("nearby_particle_ids")
            if nid_list is None:
                particle.nearby_particles = -1
            else:
                particle.nearby_particles = [id_map.get(int(n)) if n is not None else None for n in nid_list]

        rel_data = coll.get("relations")
        if rel_data is None:
            raise ValueError("Red-square cache is missing 'relations'.")

        relations_obj = Relations.from_json(rel_data)
        pid = getattr(relations_obj, "_primary_particle_id", None)
        relations_obj.primary_particle = id_map.get(int(pid)) if pid is not None else None
        rel_ids = getattr(relations_obj, "_relation_ids", [])
        relations_obj.relation_array = [id_map.get(int(i)) if i is not None else None for i in rel_ids]

        eq_pairs = getattr(relations_obj, "_equivalent_particles_id_pairs", [])
        eq_dict = {}
        for src_id, dest_id in eq_pairs:
            src_obj = id_map.get(int(src_id)) if src_id is not None else None
            dest_obj = id_map.get(int(dest_id)) if dest_id is not None else None
            if src_obj is not None:
                eq_dict[src_obj] = dest_obj
        relations_obj.equivalent_particles_dict = eq_dict

        image = load_cached_image_from_metadata(processed_data_path, coll)
        return relations_obj, image
    except (AttributeError, FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as e:
        write_to_file(
            "main.log",
            f"Invalid red-square cache for {image_path}: {e}\n{traceback.format_exc()}\n",
        )
        return None, None


def load_green_square_image_data(image_path: str) -> tuple[list[Particle_Details] | None, np.ndarray | None]:
    """
    Load cached green-square data and reconstruct nearby-particle references.

    Args:
        image_path: Source image path whose stem maps to a
            ``*_processed_data.json`` file in the sibling ``data`` directory.

    Returns:
        A tuple ``(particles, image)``. If the cache file is missing, returns
        ``(None, None)``. When present, particle IDs are resolved back into the
        ``nearby_particles`` lists and the image is loaded from either a
        schema-v2 binary cache or the legacy JSON ``image_data`` field.
    """
    processed_data_path = get_processed_data_path(image_path)
    if not processed_data_path.exists():
        return None, None

    try:
        with processed_data_path.open("r", encoding="utf-8") as infile:
            coll = json.load(infile)

        particles_data = coll.get("particles")
        if not isinstance(particles_data, list):
            raise ValueError("Green-square cache is missing a valid 'particles' list.")

        particles = [Particle_Details.from_json(pd) for pd in particles_data]
        id_map = {int(p.id): p for p in particles if getattr(p, "id", None) is not None}

        for pd, particle in zip(particles_data, particles):
            nid_list = pd.get("nearby_particle_ids")
            if nid_list is None:
                particle.nearby_particles = -1
            else:
                particle.nearby_particles = [id_map.get(int(n)) if n is not None else None for n in nid_list]

        image = load_cached_image_from_metadata(processed_data_path, coll)
        return particles, image
    except (AttributeError, FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as e:
        write_to_file(
            "main.log",
            f"Invalid green-square cache for {image_path}: {e}\n{traceback.format_exc()}\n",
        )
        return None, None


def hash_image(image: np.ndarray) -> str:
    """Return a deterministic BLAKE2b hash for an image array.

    The hash includes the array shape, dtype, and raw bytes so identical image
    contents produce the same digest across the run.

    Args:
        image: NumPy image array or ``None``.

    Returns:
        A short hex digest string, or ``"null"`` when ``image`` is ``None``.
    """
    if image is None:
        return "null"
    arr = np.asarray(image)
    h = hashlib.blake2b(digest_size=16)
    # include shape and dtype for extra uniqueness
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    # hash full buffer (exact)
    h.update(arr.tobytes())
    return h.hexdigest()


def calc_avg_horizontal_distance(
    relations: "Relations",
    anchor_particle: "Particle_Details",
    full_image_particle_details: list["Particle_Details"],
) -> float:
    """Calculate the mean absolute horizontal offset for one candidate match.

    Args:
        relations: Red-square relation template used to derive expected
            positions.
        anchor_particle: Candidate anchor particle in the green-square image.
        full_image_particle_details: All detected particles in that
            green-square image.

    Returns:
        The mean of ``abs(expected_col - matched_col)`` across every related
        particle that receives a non-``None`` match from
        ``relations.get_equivalent_particles_dict``. Returns ``0.0`` when no
        related particles are matched.
    """
    equiv_dict = relations.get_equivalent_particles_dict(anchor_particle, full_image_particle_details)
    if not equiv_dict:
        return 0.0
    h_dists = []
    for related_particle in relations.relation_array:
        matched = equiv_dict.get(related_particle)
        if matched is not None:
            dummy = relations.primary_particle.get_dummy_particle(related_particle, anchor_particle)
            h_dists.append(abs(dummy.center_position[1] - matched.center_position[1]))
    return sum(h_dists) / len(h_dists) if h_dists else 0.0


def process_sample_dir(
    samples_dir: Path,
    process_only: bool,
    always_process: bool,
    save_step_images: bool,
    progress: DirState,
) -> None:
    """
    Run the full processing and matching pipeline for one sample directory.

    Args:
        samples_dir: Directory containing source microscopy images.
        process_only: When ``True``, stop after loading or generating cached
            processed-data cache metadata and diagnostic images.
        always_process: When ``True``, ignore caches and regenerate processed
            data for every image.
        save_step_images: When ``True``, save the intermediate image-processing
            steps generated by ``process_image.process_image``.
        progress: Mutable progress-display state for this directory.

    Side Effects:
        Reads and writes files under ``samples_dir``, updates terminal progress,
        appends log entries, saves processed-data metadata JSON plus adjacent
        binary image caches, writes ``percentage_data.csv`` and
        ``horizontal_distance.csv``, and saves annotated output images.
    """
    save_dir = str(samples_dir / "pictures")

    # get image paths
    write_to_file("main.log", f"getting images in {samples_dir}\n")
    image_paths = get_image_paths(samples_dir)
    write_to_file("main.log", f"found {len(image_paths)} images to process\n")
    progress.set_images(len(image_paths))
    progress.set_phase("loading")

    # load processed images and collect unprocessed images
    processed_image_data = []
    unprocessed_images = []
    image_to_image_path = {}
    for image_path in image_paths:
        # detect if the image has already been processed
        if not always_process and not process_only and is_image_processed(image_path):
            write_to_file("main.log", f"image already processed, loading data: {image_path}\n")
            if is_red_square_image(image_path):
                # There may be multiple indexed red-square data files for one source image.
                # Check for indexed variants first; fall back to the base name.
                p = Path(image_path)
                data_dir = p.parent / "data"
                indexed_files = sorted(data_dir.glob(f"{p.stem}_redsquare*_processed_data.json"))
                candidate_stems = (
                    [f.stem.replace("_processed_data", "") for f in indexed_files] if indexed_files else [p.stem]
                )
                cached_red_results = []
                red_cache_valid = True
                for stem in candidate_stems:
                    candidate_path = str(p.parent / f"{stem}{p.suffix}")
                    ro, img = load_red_square_image_data(candidate_path)
                    if ro is None or img is None:
                        red_cache_valid = False
                        break
                    cached_red_results.append((candidate_path, ro, img))
                if red_cache_valid and cached_red_results:
                    for candidate_path, ro, img in cached_red_results:
                        image_to_image_path[hash_image(img)] = candidate_path
                        processed_image_data.append((ro, img))
                else:
                    write_to_file("main.log", f"red-square cache invalid, reprocessing image: {image_path}\n")
                    unprocessed_images.append(image_path)
            else:
                # data is in the form (green_square_particle_details, green_square_image)
                gspd, img = load_green_square_image_data(image_path)
                if gspd is not None and img is not None:
                    image_to_image_path[hash_image(img)] = image_path
                    processed_image_data.append((gspd, img))
                else:
                    write_to_file("main.log", f"green-square cache invalid, reprocessing image: {image_path}\n")
                    unprocessed_images.append(image_path)
        else:
            # image has not been processed, so add it to the unprocessed list
            write_to_file("main.log", f"image has not been processed: {image_path}\n")
            unprocessed_images.append(image_path)

    # local worker that processes a single image_path + io_image
    def _process_single(img_idx, image_path, io_image):
        """Process one source image and return cached data plus image-path keys.

        Returns either a single ``((data, image), (hash, path))`` tuple for a
        green-square image, a list of those tuples for a red-square image that
        yields multiple extracted red squares, or ``None`` if processing fails.
        The function also updates progress state and writes detailed errors to
        ``main.log``.
        """
        try:
            progress.set_image(img_idx + 1)
            write_to_file("main.log", f"processing image: {image_path}\n")
            if is_red_square_image(image_path):  # get the red square image and process it
                progress.set_phase("RS: finding squares")
                write_to_file("main.log", "processing red square image\n")
                # Find red squares in the ORIGINAL unprocessed image
                red_square_images_raw = get_red_square_image(io_image, save_dir)

                # Check if any red squares were found
                if not red_square_images_raw:
                    raise ValueError(f"No red squares found in image: {image_path}")

                # Process each red square image separately
                # Total steps: 1 (finding squares) + 6 per red square found
                n_rs = len(red_square_images_raw)
                _rs_total = 1 + n_rs * 6
                results = []
                for rs_idx, red_square_image_raw in enumerate(red_square_images_raw):
                    rs_tag = f"RS {rs_idx + 1}/{n_rs}"
                    _rs_base = 1 + rs_idx * 6
                    write_to_file("main.log", f"processing red square content {rs_idx + 1}/{n_rs}\n")
                    progress.set_phase(f"{rs_tag}: processing image", _rs_base + 1, _rs_total)
                    # Process the extracted red square content
                    red_square_image_processed = process_image(
                        red_square_image_raw,
                        save_dir,
                        save_step_images,
                        progress=progress,
                        progress_prefix=f"{rs_tag}: ",
                    )
                    # Convert back to numpy array (process_image returns a list)
                    red_square_image = np.array(red_square_image_processed, dtype=np.uint8)

                    write_to_file(
                        "main.log",
                        f"detecting red square particles (image {rs_idx + 1}/{n_rs})\n",
                    )
                    progress.set_phase(f"{rs_tag}: detecting particles", _rs_base + 2, _rs_total)
                    red_square_particle_positions = detect_particles(red_square_image)
                    write_to_file(
                        "main.log",
                        f"getting red square particle details (image {rs_idx + 1}/{n_rs})\n",
                    )
                    progress.set_phase(f"{rs_tag}: analyzing particles", _rs_base + 3, _rs_total)
                    red_square_particle_details = get_particle_details(red_square_particle_positions, red_square_image)
                    write_to_file(
                        "main.log",
                        f"getting red square relations object (image {rs_idx + 1}/{n_rs})\n",
                    )
                    progress.set_phase(f"{rs_tag}: building relations", _rs_base + 4, _rs_total)
                    red_square_relations_object = get_relations_object(red_square_particle_details)
                    write_to_file("main.log", f"saving red square image data (image {rs_idx + 1}/{n_rs})\n")

                    # Append index to image_path for multiple red squares.
                    # Must preserve the original parent directory so that
                    # save_red_square_particle_data writes to the correct data/ folder.
                    indexed_image_path = (
                        image_path
                        if len(red_square_images_raw) == 1
                        else str(
                            Path(image_path).parent
                            / f"{Path(image_path).stem}_redsquare{rs_idx}{Path(image_path).suffix}"
                        )
                    )
                    progress.set_phase(f"{rs_tag}: saving data", _rs_base + 5, _rs_total)
                    save_red_square_particle_data(red_square_relations_object, red_square_image, indexed_image_path)

                    write_to_file("main.log", f"creating images (image {rs_idx + 1}/{n_rs})\n")
                    progress.set_phase(f"{rs_tag}: saving images", _rs_base + 6, _rs_total)
                    show_detected_particles(red_square_particle_positions, red_square_image, save_dir)
                    save_particles(f"red_square_particle_details_{rs_idx}", red_square_particle_details, save_dir)
                    show_particle_details(red_square_particle_details, red_square_image.copy(), save_dir)

                    results.append(
                        (
                            (red_square_relations_object, red_square_image),
                            (hash_image(red_square_image), indexed_image_path),
                        )
                    )

                return results
            else:  # get the green square image and process it
                progress.set_phase("GS: extracting square", 1, 7)
                write_to_file("main.log", "getting green square image\n")
                green_square_image = get_green_square_image(io_image, save_dir)
                progress.set_phase("GS: processing image", 2, 7)
                write_to_file("main.log", "processing green square image\n")
                green_processed_image = process_image(
                    green_square_image,
                    save_dir,
                    save_step_images,
                    progress=progress,
                    progress_prefix="GS: ",
                )
                progress.set_phase("GS: cleaning aggregations", 3, 7)
                write_to_file("main.log", "removing large aggregations\n")
                green_square_image = remove_large_aggregations(green_processed_image, save_dir)
                progress.set_phase("GS: detecting particles", 4, 7)
                write_to_file("main.log", "detecting green square particles\n")
                green_square_particle_positions = detect_particles(green_square_image)
                progress.set_phase("GS: analyzing particles", 5, 7)
                write_to_file("main.log", "getting green square particle details\n")
                green_square_particle_details = get_particle_details(
                    green_square_particle_positions, green_square_image
                )
                progress.set_phase("GS: saving data", 6, 7)
                write_to_file("main.log", "saving green square image data\n")
                save_green_square_particle_data(green_square_particle_details, green_square_image, image_path)
                progress.set_phase("GS: saving images", 7, 7)
                write_to_file("main.log", "creating images\n")
                show_detected_particles(green_square_particle_positions, green_square_image, save_dir)
                save_particles("green_square_image_particle_details", green_square_particle_details, save_dir)
                show_particle_details(green_square_particle_details, green_square_image.copy(), save_dir)
                return (green_square_particle_details, green_square_image), (hash_image(green_square_image), image_path)
        except Exception as e:
            # write full traceback to the log for easier debugging
            write_to_file("main.log", f"Error processing {image_path}: {e}\n{traceback.format_exc()}\n")
            progress.milestone(f"ERROR: {Path(image_path).name}: {e}")
            return None

    # process images sequentially (directory-level concurrency is handled in main())
    if unprocessed_images:
        for img_idx, img_path in enumerate(unprocessed_images):
            progress.set_phase("loading")
            write_to_file("main.log", f"loading image from disk: {img_path}\n")
            io_img = load_io_image(img_path)
            res = _process_single(img_idx, img_path, io_img)
            if res is None:
                continue
            # Handle both single results (green square) and multiple results (red squares)
            if isinstance(res, list):
                # Multiple red squares from a single image (or empty if none found)
                for processed_tuple, mapping in res:
                    processed_image_data.append(processed_tuple)
                    image_to_image_path[mapping[0]] = mapping[1]
            else:
                # Single green square result
                processed_tuple, mapping = res
                processed_image_data.append(processed_tuple)
                image_to_image_path[mapping[0]] = mapping[1]

    # if only processing images, return now (caller checks process_only)
    if process_only:
        write_to_file("main.log", f"processing only flag set, done with {samples_dir.name}.\n")
        return

    # separate processed data into red square relations objects and green square particle details lists
    red_square_tuples = []
    green_square_tuples = []
    for data in processed_image_data:
        if isinstance(data[0], Relations):
            red_square_tuples.append(data)
        else:
            green_square_tuples.append(data)

    # for each red square relations object, find the percentage matches in each green square particle details list
    # determine the shared samples_dir for the consolidated CSVs
    # (all green square images share the same parent directory)
    if not green_square_tuples:
        write_to_file("main.log", "no green square images found, skipping percentage matching.\n")
        progress.milestone(f"WARNING: no green square images found in {samples_dir.name}")
    elif not red_square_tuples:
        write_to_file("main.log", "no red square images found, skipping percentage matching.\n")
        progress.milestone(f"WARNING: no red square images found in {samples_dir.name}")
    else:
        progress.set_phase("matching")
        first_green_samples_dir = Path(image_to_image_path[hash_image(green_square_tuples[0][1])]).parent
        horiz_csv_path = first_green_samples_dir / "data" / "horizontal_distance.csv"
        pct_csv_path = first_green_samples_dir / "data" / "percentage_data.csv"
        horiz_rows: list[tuple[str, str, float]] = []
        pct_rows: list[tuple[str, str, float, float, str, float, float, float]] = []
        for green_square_particle_details, green_square_image in green_square_tuples:
            img_samples_dir = Path(image_to_image_path[hash_image(green_square_image)]).parent
            gsi_name = Path(image_to_image_path[hash_image(green_square_image)]).stem  # get green square image name
            save_path = img_samples_dir / "best_percentage_images"

            # One conglomerate image per green square — accumulates all red-square annotations
            conglomerate_image = green_square_image.copy()

            for red_square_relations, red_square_image in red_square_tuples:
                rsi_name = Path(image_to_image_path[hash_image(red_square_image)]).stem
                rs_short = re.search(r"redsquare(\d+)", rsi_name)
                rs_label = f"RS{rs_short.group(1)}" if rs_short else rsi_name.split(".")[0][-12:]
                match_label = f"{rs_label} → {gsi_name.split('.')[0][-12:]}"
                write_to_file(
                    "main.log",
                    "getting relations match percentages between red square and green square\n",
                )
                progress.set_phase(f"match {match_label}: scoring")
                full_image_top_percentages = get_percentages(
                    red_square_relations, green_square_particle_details, green_square_image, img_samples_dir, progress
                )
                write_to_file("main.log", "percentages:\n")
                rs_num = re.search(r"redsquare(\d+)", rsi_name)
                gs_mm = re.search(r"(\d+)mm", gsi_name)
                rs_col = rs_num.group(1) if rs_num else "0"
                gs_col = gs_mm.group(1) if gs_mm else gsi_name
                pair_name = f"{rsi_name}_{gsi_name}"

                # accumulate percentage rows for this pair
                for percent_tuple in full_image_top_percentages:
                    (
                        percentage,
                        particle_details,
                        _matched,
                        base_score,
                        completeness_ratio,
                        geometry_consistency,
                        individual_scores,
                    ) = percent_tuple
                    distance = calc_avg_horizontal_distance(
                        red_square_relations, particle_details, green_square_particle_details
                    )
                    write_to_file("main.log", f"% match: {percentage}\n")
                    pct_rows.append(
                        (
                            rs_col,
                            gs_col,
                            percentage,
                            distance,
                            base_score,
                            completeness_ratio,
                            geometry_consistency,
                            individual_scores,
                            str(particle_details),
                        )
                    )

                # calculate the average horizontal distance for the best percentage match
                # (index 0, already sorted descending)
                if full_image_top_percentages:
                    _best_pct, _best_particle, _best_matched_particles, *_ = full_image_top_percentages[0]
                    horizontal_distance = calc_avg_horizontal_distance(
                        red_square_relations, _best_particle, green_square_particle_details
                    )
                    write_to_file("main.log", f"horizontal distance (best match): {horizontal_distance}\n")
                    horiz_rows.append((rs_col, gs_col, horizontal_distance))

                # display the best objects — individual images saved inside; also drawn onto conglomerate
                write_to_file("main.log", "displaying best percentages\n")
                progress.set_phase(f"match {match_label}: drawing results")
                display_best_percentages(
                    pair_name,
                    full_image_top_percentages,
                    green_square_particle_details,
                    red_square_relations,
                    green_square_image,
                    save_path,
                    conglomerate_image,
                    save_step_images,
                )

            # Save the conglomerate image for this green square (all red-square annotations combined)
            write_to_file("main.log", f"saving conglomerate best percentages image for {gsi_name}\n")
            save_io_image(f"{gsi_name}_all_best_percentages", conglomerate_image, str(save_path), True)

        # Write both CSVs sorted by green_square then red_square
        write_sorted_horizontal_distance_csv(horiz_csv_path, horiz_rows)
        write_sorted_percentage_data_csv(pct_csv_path, pct_rows)

    write_to_file("main.log", f"directory processing complete: {samples_dir.name}\n")
    progress.set_phase("")


def main() -> None:
    """Parse command-line flags and process every discovered sample directory.

    The entry point resolves the root directory, discovers eligible sample
    directories, chooses directory-level concurrency, drives
    ``process_sample_dir`` for each directory, and records run timing to the
    log.
    """
    # read the args--if you see --process or -p then process images
    args = os.sys.argv[1:]
    help_flags = {"--help", "-h", "/?", "--h", "-help"}
    if any(arg in help_flags for arg in args):
        print(
            "Usage: python main.py [options]\n\n"
            "Options:\n"
            "  --process-only, -po       Only process images and save data; skip matching and CSV generation.\n"
            "  --single-threaded, -st    Process images using a single thread (no parallelism).\n"
            "  --parallel-mode, -pm      Choose 'thread' or 'process' directory workers.\n"
            "  --always-process, -ap     Reprocess all images even if cached data exists.\n"
            "  --save-step-images, -ssi Save intermediate processing step images for debugging.\n"
            "  --max-dirs, -md <n>       Cap directory-level worker count at <n>.\n"
            "  --resume-last-run         Resume from the most recent run manifest.\n"
            "  --resume-run <run_id>     Resume from a specific prior run manifest.\n"
            "  --retry-failed            Requeue failed directories during resume.\n"
            "  --root-dir <path>        Specify the root directory containing sample subdirectories (overrides default).\n"
            "  --help, -h, /?, --h      Show this help message and exit.\n",
        )
        return
    process_only = ("--process-only" in args) or ("-po" in args)
    single_threaded = ("--single-threaded" in args) or ("-st" in args)
    always_process = ("--always-process" in args) or ("-ap" in args)
    save_step_images = ("--save-step-images" in args) or ("-ssi" in args)
    retry_failed = "--retry-failed" in args
    max_dirs_arg = get_flag_value(args, ("--max-dirs", "-md"))
    parallel_mode = resolve_parallel_mode(args, single_threaded)

    # get the time for logging purposes
    start_time = datetime.now()

    current_dir = Path(__file__).parent

    # Resolve root_dir; override with --root-dir <path> if provided on the command line
    root_dir_arg = get_flag_value(args, ("--root-dir",))
    if root_dir_arg:
        root_dir = Path(root_dir_arg)
    elif DEBUG:
        root_dir = current_dir / "Image_Samples" / "root_dir"
    else:
        root_dir = Path(r"D:\Codes!!\Supercomputer_calc\Samples")

    resume_run_id = resolve_resume_run_id(args, current_dir)

    # log that the run is starting
    write_to_file("main.log", "=" * 62)
    write_to_file(
        "main.log",
        f" starting run at {datetime.today().strftime('%Y-%m-%d--%H-%M-%S')} ",
    )
    write_to_file("main.log", "=" * 62 + "\n")

    sample_dirs = get_sample_dirs(root_dir)

    if max_dirs_arg is not None:
        max_concurrent = max(1, int(max_dirs_arg))
    elif single_threaded:
        max_concurrent = 1
    else:
        max_concurrent = min(get_available_worker_count(), max(1, len(sample_dirs)))

    previous_run_state = load_run_state(current_dir, resume_run_id) if resume_run_id is not None else None
    previous_attempts = previous_run_state["attempts"] if previous_run_state is not None else {}
    manifest = RunManifest(
        current_dir,
        root_dir,
        parallel_mode,
        process_only,
        always_process,
        max_concurrent,
        resume_from_run_id=resume_run_id,
        retry_failed=retry_failed,
        previous_attempts=previous_attempts,
    )

    write_to_file(
        "main.log",
        f"parallel mode: {parallel_mode}; worker count: {max_concurrent}; debug logs enabled: {os.environ.get('SIP_DEBUG_LOGS', '0')}\n",
    )

    write_to_file("main.log", f"found {len(sample_dirs)} sample directories to process\n")
    write_to_file("main.log", f"run manifest directory: {manifest.run_dir}\n")
    if resume_run_id is not None:
        write_to_file("main.log", f"resuming from run manifest: {resume_run_id}; retry_failed={retry_failed}\n")

    try:
        dirs_to_process = []
        skipped_completed = 0
        skipped_failed = 0
        previous_states = previous_run_state["directory_states"] if previous_run_state is not None else {}

        for sample_dir in sample_dirs:
            sample_dir_key = manifest.sample_dir_key(sample_dir)
            previous_attempt = previous_attempts.get(sample_dir_key, 0)
            manifest.emit("directory_discovered", sample_dir=sample_dir_key, attempt=previous_attempt)

            if previous_run_state is None:
                dirs_to_process.append(sample_dir)
                continue

            previous_state = previous_states.get(sample_dir_key)
            if previous_state == "completed":
                manifest.emit(
                    "directory_skipped",
                    sample_dir=sample_dir_key,
                    attempt=previous_attempt,
                    skip_reason="completed_in_previous_run",
                    resume_source_run_id=resume_run_id,
                )
                skipped_completed += 1
                continue

            if previous_state == "failed" and not retry_failed:
                manifest.emit(
                    "directory_skipped",
                    sample_dir=sample_dir_key,
                    attempt=previous_attempt,
                    skip_reason="failed_in_previous_run",
                    resume_source_run_id=resume_run_id,
                )
                skipped_failed += 1
                continue

            dirs_to_process.append(sample_dir)

        if resume_run_id is not None:
            write_to_file(
                "main.log",
                f"resume plan: processing {len(dirs_to_process)} directories, skipped completed={skipped_completed}, skipped failed={skipped_failed}\n",
            )

        if parallel_mode == "process":
            print(
                "Process mode active: live multi-directory progress is disabled; completed directories will be reported as they finish."
            )
            with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {}
                future_meta = {}
                for sample_dir in dirs_to_process:
                    sample_dir_key = manifest.sample_dir_key(sample_dir)
                    attempt = manifest.start_attempt(sample_dir_key)
                    manifest.emit("directory_queued", sample_dir=sample_dir_key, attempt=attempt)
                    future = executor.submit(
                        process_sample_dir_in_subprocess,
                        sample_dir,
                        process_only,
                        always_process,
                        save_step_images,
                    )
                    started_at = datetime.now()
                    manifest.emit(
                        "directory_started",
                        sample_dir=sample_dir_key,
                        attempt=attempt,
                        parallel_mode=parallel_mode,
                    )
                    futures[future] = sample_dir
                    future_meta[future] = (sample_dir_key, attempt, started_at)

                completed_dirs = 0
                for fut in as_completed(futures):
                    sample_dir = futures[fut]
                    sample_dir_key, attempt, started_at = future_meta[fut]
                    duration_seconds = round((datetime.now() - started_at).total_seconds(), 3)
                    try:
                        dir_name = fut.result()
                        completed_dirs += 1
                        manifest.emit(
                            "directory_completed",
                            sample_dir=sample_dir_key,
                            attempt=attempt,
                            duration_seconds=duration_seconds,
                            parallel_mode=parallel_mode,
                        )
                        print(f"Completed {completed_dirs}/{len(dirs_to_process)}: {dir_name}")
                    except Exception as e:
                        manifest.emit(
                            "directory_failed",
                            sample_dir=sample_dir_key,
                            attempt=attempt,
                            duration_seconds=duration_seconds,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            parallel_mode=parallel_mode,
                        )
                        write_to_file(
                            "main.log",
                            f"Unhandled error in directory {sample_dir.name}: {e}\n{traceback.format_exc()}\n",
                        )
        else:
            tracker = MultiProgressTracker(total_dirs=len(dirs_to_process), max_concurrent=max_concurrent)

            def _run_dir(dir_idx: int, samples_dir: Path) -> None:
                """Acquire a progress slot, process one directory, and always release it."""
                state = tracker.acquire_slot(dir_idx, samples_dir.name)
                try:
                    write_to_file("main.log", f"--- processing directory: {samples_dir} ---\n")
                    process_sample_dir(samples_dir, process_only, always_process, save_step_images, state)
                finally:
                    tracker.release_slot(state)

            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {}
                future_meta = {}
                for dir_idx, sample_dir in enumerate(dirs_to_process):
                    sample_dir_key = manifest.sample_dir_key(sample_dir)
                    attempt = manifest.start_attempt(sample_dir_key)
                    manifest.emit("directory_queued", sample_dir=sample_dir_key, attempt=attempt)
                    future = executor.submit(_run_dir, dir_idx, sample_dir)
                    started_at = datetime.now()
                    manifest.emit(
                        "directory_started",
                        sample_dir=sample_dir_key,
                        attempt=attempt,
                        parallel_mode=parallel_mode,
                    )
                    futures[future] = sample_dir
                    future_meta[future] = (sample_dir_key, attempt, started_at)

                for fut in as_completed(futures):
                    sample_dir = futures[fut]
                    sample_dir_key, attempt, started_at = future_meta[fut]
                    duration_seconds = round((datetime.now() - started_at).total_seconds(), 3)
                    try:
                        fut.result()
                        manifest.emit(
                            "directory_completed",
                            sample_dir=sample_dir_key,
                            attempt=attempt,
                            duration_seconds=duration_seconds,
                            parallel_mode=parallel_mode,
                        )
                    except Exception as e:
                        manifest.emit(
                            "directory_failed",
                            sample_dir=sample_dir_key,
                            attempt=attempt,
                            duration_seconds=duration_seconds,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            parallel_mode=parallel_mode,
                        )
                        write_to_file(
                            "main.log",
                            f"Unhandled error in directory {sample_dir.name}: {e}\n{traceback.format_exc()}\n",
                        )

            tracker.done()

        end_time = datetime.now()
        manifest.emit(
            "run_completed",
            parallel_mode=parallel_mode,
            total_dirs_discovered=len(sample_dirs),
            total_dirs_processed=len(dirs_to_process),
            total_dirs_skipped=skipped_completed + skipped_failed,
            duration_seconds=round((end_time - start_time).total_seconds(), 3),
        )

        write_to_file("main.log", "\n" + "=" * 62)
        write_to_file("main.log", f" run completed at {end_time.strftime('%Y-%m-%d--%H-%M-%S')} ")
        write_to_file("main.log", f" total duration: {end_time - start_time} ")
        write_to_file("main.log", "=" * 62 + "\n")
    except KeyboardInterrupt:
        end_time = datetime.now()
        manifest.emit(
            "run_aborted",
            parallel_mode=parallel_mode,
            error_type="KeyboardInterrupt",
            error_message="Run interrupted by user.",
            duration_seconds=round((end_time - start_time).total_seconds(), 3),
        )
        raise
    except Exception as e:
        end_time = datetime.now()
        manifest.emit(
            "run_aborted",
            parallel_mode=parallel_mode,
            error_type=type(e).__name__,
            error_message=str(e),
            duration_seconds=round((end_time - start_time).total_seconds(), 3),
        )
        raise


if __name__ == "__main__":
    main()
