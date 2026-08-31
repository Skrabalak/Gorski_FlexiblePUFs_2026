#!/bin/python3/


# pylint: disable=E0611

import json
import math
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from statistics import mean
from timeit import default_timer as timer
from typing import Optional

import numpy as np
from skimage.draw import circle_perimeter, line

import helper_functions

##########
# GLOBAL VARIABLES
##########

date = datetime.today().strftime("%Y-%m-%d--%H-%M-%S")

# pixel colors for testing
green_pixel = [0, 255, 0]
red_pixel = [255, 0, 0]
blue_pixel = [0, 0, 255]
black_pixel = [0, 0, 0]
white_pixel = [255, 255, 255]


# particle class
class Particle_Details:
    """Represent one detected particle and its derived metadata."""

    # static variables — written by set_full_particle() which can be called
    # from multiple threads simultaneously; protect all reads and writes with
    # this class-level lock.
    _largest_lock = threading.Lock()
    largest_particle_length = 0
    largest_particle_width = 0

    def __init__(self, corner_coordinates: list[list[int]], center_position: np.ndarray) -> None:
        """Initialize a particle with bounding-box geometry and centre position."""
        self.corner_coordinates = corner_coordinates
        self.center_position = center_position
        self.full_particle = []
        self.color = []
        self.nearby_particles = -1
        # unique identifier for serialization; set via assign_particle_ids()
        self.id = None

    def __str__(self):
        """Return the same compact representation produced by ``to_str``."""
        return self.to_str()

    def to_str(self):
        """Return a one-line JSON string representation of this particle."""
        # create a string representation of the particle for debugging purposes
        # ensure that it is a one-liner
        json_string = json.dumps(self.to_json())
        return json_string

    def to_json(self):
        """Return a JSON-serializable dictionary for this particle.

        Nearby-particle object references are converted to ID lists and NumPy
        values are converted to plain Python lists or integers.
        """
        obj = {
            "id": int(self.id) if self.id is not None else None,
            "corner_coordinates": _convert_to_serializable(self.corner_coordinates),
            "center_position": _convert_to_serializable(self.center_position),
            "color": _convert_to_serializable(self.color) if self.color is not None else None,
            "nearby_particle_ids": None,
            "full_particle": None,
            "largest_particle_length": Particle_Details.largest_particle_length,
            "largest_particle_width": Particle_Details.largest_particle_width,
        }

        if isinstance(self.nearby_particles, list):
            obj["nearby_particle_ids"] = [p.id if hasattr(p, "id") else None for p in self.nearby_particles]

        if isinstance(self.full_particle, list) and len(self.full_particle) > 0:
            obj["full_particle"] = _convert_to_serializable(self.full_particle)

        return obj

    @staticmethod
    def from_json(data: dict):
        """Reconstruct a particle from ``to_json`` output.

        Nearby-particle IDs are preserved but not resolved into object
        references; callers perform that second pass after loading the full
        collection.
        """
        # corner_coordinates may be nested lists; keep as-is
        corner_coordinates = data.get("corner_coordinates")

        # center_position should be converted back to numpy array for compatibility
        center = data.get("center_position")
        try:
            center_np = np.array(center).astype(int)
        except Exception:
            center_np = center

        p = Particle_Details(corner_coordinates, center_np)

        # id
        pid = data.get("id")
        if pid is not None:
            try:
                p.id = int(pid)
            except Exception:
                p.id = pid

        # color
        color = data.get("color")
        if color is not None:
            p.color = _convert_to_serializable(color)

        # full_particle - already plain nested lists from JSON
        if data.get("full_particle") is not None:
            p.full_particle = data.get("full_particle")

        # preserve the raw nearby_particle_ids for lossless round-trip; the loader
        # will resolve these into object references. Keep a placeholder in
        # `nearby_particles` so calling code can detect unresolved state.
        p._nearby_particle_ids = data.get("nearby_particle_ids")
        if p._nearby_particle_ids is None:
            p.nearby_particles = -1
        else:
            # placeholder list of same length; will be replaced with actual objects
            p.nearby_particles = [None for _ in p._nearby_particle_ids]

        # restore class-level metadata if present
        lpl = data.get("largest_particle_length")
        lpw = data.get("largest_particle_width")
        if isinstance(lpl, int):
            Particle_Details.largest_particle_length = lpl
        if isinstance(lpw, int):
            Particle_Details.largest_particle_width = lpw

        return p

    def distance_to(self, other: "Particle_Details") -> float:
        """Return the Euclidean distance to another particle centre."""
        return np.sqrt(np.sum(np.square(np.asarray(self.center_position) - np.asarray(other.center_position))))

    def sum_distances(self) -> float:
        """Return the sum of distances to particles stored in ``nearby_particles``."""
        total_distance = 0
        for particle in self.nearby_particles:
            total_distance += self.distance_to(particle)
        return total_distance

    def size(self) -> int:
        """Return a size metric derived from the particle bounding box."""
        total_size = 0
        upper_left = self.corner_coordinates[0]
        bottom_right = self.corner_coordinates[1]

        x_difference = abs(int(upper_left[0]) - int(bottom_right[0]))
        y_difference = abs(int(upper_left[1]) - int(bottom_right[1]))

        if x_difference == 0 or y_difference == 0:
            total_size = x_difference + y_difference
        else:
            total_size = x_difference * y_difference

        return total_size

    def set_full_particle(self, particle_position: list[list[int]], image: np.ndarray) -> None:
        """Copy the particle's bounding-box pixels out of ``image``.

        The copied pixels are stored in ``self.full_particle`` and the class
        maxima used by ``save_particles`` are updated as needed.
        """
        # variables
        row_delta = particle_position[0][0]
        current_particle_length = 0
        current_particle_width = 0

        for image_row_index in range(particle_position[0][0], particle_position[1][0] + 1):
            # make sure that full_particle doesn't have an aneurism
            self.full_particle.append([])
            current_particle_length += 1
            current_particle_width = 0
            for image_col_index in range(particle_position[0][1], particle_position[1][1] + 1):
                # add one to the current particle width
                current_particle_width += 1
                # do a deep copy of the pixel, not a shallow copy
                current_pixel = []
                for value in image[image_row_index][image_col_index]:
                    current_pixel.append(value)

                self.full_particle[image_row_index - row_delta].append(current_pixel)

        # Guard the class-level max update with the lock so concurrent threads
        # cannot interleave their read-compare-write and produce a stale value.
        with Particle_Details._largest_lock:
            if current_particle_length > Particle_Details.largest_particle_length:
                Particle_Details.largest_particle_length = current_particle_length
            if current_particle_width > Particle_Details.largest_particle_width:
                Particle_Details.largest_particle_width = current_particle_width

    ##
    # sets the average color of the particle
    #  currently uses self.full_particle for access to all possible pixels
    ##
    def set_color(self) -> None:
        """Set ``self.color`` to the running average of non-black particle pixels."""
        # start the function with current_color set to a default value
        current_color = [-1]
        # the default value must be the first 'color' encountered that isn't black--we are ignoring all black pixels found in full_particle
        for row in self.full_particle:
            for pixel in row:
                if not helper_functions.is_black(pixel):
                    current_color = pixel
                    # once we find the first we can break our way out of the loop--we have our default, why keep going?
                    break

        # now get the actual average
        for row in self.full_particle:
            for pixel in row:
                if not helper_functions.is_black(pixel):
                    # set the current color to the rounded integer value of the average [current_color, pixel] values
                    current_color = np.rint(np.average([current_color, pixel], axis=0)).astype(int)

        # current_color[3] = 255
        self.color = current_color

    def set_nearby_particles(self, particle_details_array: list["Particle_Details"], search_radius: int) -> None:
        """Populate ``nearby_particles`` with particles inside ``search_radius``."""
        # print("handling particle", self)
        self.nearby_particles = []
        for particle in particle_details_array:
            if particle == self:
                continue
            if self.distance_to(particle) < search_radius:
                self.nearby_particles.append(particle)

    def clean_particle_in_image(self, image: np.ndarray) -> None:
        """Restore this particle's stored pixels back into ``image`` in place."""
        coords = self.corner_coordinates
        # modifier for the rows and cols
        #  used to change from index->image to index->shape
        row_delta = coords[0][0]
        col_delta = coords[0][1]  # min col is now correctly in coords[0][1]

        for row in range(coords[0][0], (coords[1][0] + 1)):
            for col in range(coords[0][1], (coords[1][1] + 1)):
                image[row][col] = self.full_particle[row - row_delta][col - col_delta]

    def get_dummy_particle(
        self, related_particle: "Particle_Details", full_image_particle: "Particle_Details"
    ) -> "Particle_Details":
        """
        Return a synthetic particle at the expected position of a related
        particle relative to a candidate full-image anchor.

        Args:
            related_particle (Particle_Details): Reference relation particle
                whose offset from ``self`` defines the expected displacement.
            full_image_particle (Particle_Details): Candidate anchor particle in
                the full image.

        Returns:
            Particle_Details: A placeholder particle with a calculated centre
            position, used only for distance checks.
        """
        # find the relationship between primary_particle and related_particle
        delta_x = related_particle.center_position[0] - self.center_position[0]
        delta_y = related_particle.center_position[1] - self.center_position[1]

        # create and return a nonexistant 'dummy' particle for the purpose of checking distances
        #   NOTE: DO NOT USE THIS PARTICLE FOR ANYTHING OTHER THAN DISTANCE CHECKS
        #   it will break and you will cry
        new_center_position = [
            full_image_particle.center_position[0] + delta_x,
            full_image_particle.center_position[1] + delta_y,
        ]
        return Particle_Details([-1, -1], new_center_position)


class Relations:
    """Represent a red-square particle template and its candidate matches."""

    # STATIC CLASS VARIABLES
    offset_angle = 30
    SEARCH_DISTANCE_MODIFIER = 0.75
    MAX_SEARCH_DISTANCE = 50

    def __init__(self, primary_particle: "Particle_Details") -> None:
        """Initialize a relation template around one primary particle."""
        self.primary_particle = primary_particle
        self.relation_array = []
        self.equivalent_particles_dict = {}

    def to_json(self):
        """Return a JSON-serializable dictionary for this relations object."""
        primary_id = None
        if hasattr(self.primary_particle, "id") and self.primary_particle.id is not None:
            try:
                primary_id = int(self.primary_particle.id)
            except Exception:
                primary_id = self.primary_particle.id

        relation_ids = []
        for p in self.relation_array:
            if p is None:
                relation_ids.append(None)
                continue
            if hasattr(p, "id") and p.id is not None:
                try:
                    relation_ids.append(int(p.id))
                except Exception:
                    relation_ids.append(p.id)
            else:
                relation_ids.append(None)

        # convert equivalent_particles_dict (object->object) to list of [src_id, dest_id]
        eq_list = []
        for src, dest in getattr(self, "equivalent_particles_dict", {}).items():
            src_id = getattr(src, "id", None)
            dest_id = getattr(dest, "id", None) if dest is not None else None
            eq_list.append([src_id, dest_id])

        return {
            "primary_particle_id": primary_id,
            "relation_ids": relation_ids,
            "equivalent_particles": eq_list,
            "offset_angle": self.offset_angle,
            "search_distance_modifier": self.SEARCH_DISTANCE_MODIFIER,
            "max_search_distance": self.MAX_SEARCH_DISTANCE,
        }

    @staticmethod
    def from_json(data: dict):
        """Reconstruct a relations object from serialized JSON data.

        Particle references are left as IDs until the caller resolves them
        against a particle-ID map.
        """
        primary_id = data.get("primary_particle_id")

        # create a lightweight placeholder primary particle; real object should be
        # swapped in during resolution using the id mapping.
        placeholder_primary = Particle_Details([-1, -1], [0, 0])
        placeholder_primary.id = primary_id

        rel = Relations(placeholder_primary)

        # store ids for later resolution
        rel._primary_particle_id = primary_id
        rel._relation_ids = data.get("relation_ids", [])

        # restore search parameters (set on instance to avoid changing class defaults)
        rel.offset_angle = data.get("offset_angle", rel.offset_angle)
        rel.SEARCH_DISTANCE_MODIFIER = data.get("search_distance_modifier", rel.SEARCH_DISTANCE_MODIFIER)
        rel.MAX_SEARCH_DISTANCE = data.get("max_search_distance", rel.MAX_SEARCH_DISTANCE)

        # relation_array will be resolved by the loader; keep empty for now
        rel.relation_array = []

        return rel

    def add_relation(self, new_particle: "Particle_Details") -> None:
        """Add ``new_particle`` unless it is the primary particle or a duplicate."""
        # if these are true, don't add the particle
        if new_particle == self.primary_particle:
            return
        if new_particle in self.relation_array:
            return

        # everything is okay, time to add it
        self.relation_array.append(new_particle)

    def get_search_distance(self, target_particle: "Particle_Details") -> int:
        """Return the capped search radius for one related particle."""
        search_distance = round(self.primary_particle.distance_to(target_particle) * self.SEARCH_DISTANCE_MODIFIER)
        return int(min([search_distance, self.MAX_SEARCH_DISTANCE]))

    def draw_on_image(
        self, primary_particle: "Particle_Details", particle_details: list["Particle_Details"], image: np.ndarray
    ) -> None:
        """Draw template geometry, search radii, and resolved matches onto ``image``."""
        # find all equivalent particles
        equivalent_particles_dict = self.get_equivalent_particles_dict(primary_particle, particle_details)

        # draw the lines "connecting" each relation particle to the primary_particle
        this_center_pos = primary_particle.center_position
        color_dict = {}

        for particle in self.relation_array:
            # get dummy particle
            dummy_particle = self.primary_particle.get_dummy_particle(particle, primary_particle)
            other_center_pos = dummy_particle.center_position

            # draw line to connected particle
            current_color = helper_functions.get_random_pixel_color()
            color_dict[particle] = current_color
            rr, cc = line(
                this_center_pos[0],
                this_center_pos[1],
                other_center_pos[0],
                other_center_pos[1],
            )
            rr, cc = inbounds_line(rr, cc, image)
            image[rr, cc] = current_color

            # draw line from dummy to best particle
            if equivalent_particles_dict != None:
                best_particle = equivalent_particles_dict[particle]
                if best_particle != None:
                    rr, cc = line(
                        best_particle.center_position[0],
                        best_particle.center_position[1],
                        other_center_pos[0],
                        other_center_pos[1],
                    )
                    rr, cc = inbounds_line(rr, cc, image)
                    image[rr, cc] = current_color

        # clean up the particles in the full_image
        for particle in particle_details:
            particle.clean_particle_in_image(image)

        for particle in self.relation_array:
            # get dummy particle
            dummy_particle = self.primary_particle.get_dummy_particle(particle, primary_particle)
            other_center_pos = dummy_particle.center_position

            # S P H E R E around connected particle
            search_distance = self.get_search_distance(particle)

            (
                rr,
                cc,
            ) = circle_perimeter(other_center_pos[0], other_center_pos[1], search_distance)
            rr, cc = inbounds_line(rr, cc, image)
            image[rr, cc] = color_dict[particle]

        # clean up the primary particle
        # NOTE: must use the `primary_particle` parameter (the full-image particle), NOT
        # `self.primary_particle` (the red-square particle). self.primary_particle has
        # coordinates relative to the small red-square sub-image, so calling
        # clean_particle_in_image with it on the full green-square image stamps the
        # red-square particle's pixels into the upper-left corner, producing ghost particles.
        primary_particle.clean_particle_in_image(image)

    def find_equivalent_particles(
        self,
        related_particle: "Particle_Details",
        full_image_particle: "Particle_Details",
        particle_details: list["Particle_Details"],
    ) -> list["Particle_Details"] | None:
        """Return full-image particles that fall within the related-particle search window."""
        # create a nonexistant 'dummy' particle for checking distances
        dummy_particle = self.primary_particle.get_dummy_particle(related_particle, full_image_particle)

        # given the previous relationship pare down the list of particle_details
        good_matches = []
        search_distance = self.get_search_distance(related_particle)

        for fi_particle in particle_details:
            if dummy_particle.distance_to(fi_particle) < search_distance:
                good_matches.append(fi_particle)

        if len(good_matches) == 0:
            return None

        return good_matches

    def get_equivalent_particles_dict(
        self, image_particle: "Particle_Details", particle_details: list["Particle_Details"]
    ) -> dict["Particle_Details", "Particle_Details | None"] | None:
        """Build the best one-to-one mapping from relation particles to image particles."""
        # get all particles within search distance
        related_particle_good_matches = {}
        for related_particle in self.relation_array:
            # get all potential EPs
            arr = self.find_equivalent_particles(related_particle, image_particle, particle_details)
            if arr != None:
                related_particle_good_matches[related_particle] = arr

        if not related_particle_good_matches:
            helper_functions.write_to_file("timing.log", "No good matches found for any related particles.\n")
            return None

        # src:dest stuff takes up too much memory, so I have to try and compress it somehow
        #   first attempt: don't change anything other than using ints as opposed to objects
        current_num_items_processed = 0
        num_key_dict = {}
        particle_to_id = {}  # Map each unique particle to a single ID
        converted_dict = {}
        for key, matches in related_particle_good_matches.items():
            # Assign ID to source particle if not already assigned
            if key not in particle_to_id:
                particle_to_id[key] = current_num_items_processed
                num_key_dict[current_num_items_processed] = key
                current_num_items_processed += 1
            key_num = particle_to_id[key]

            converted_array = []
            for item in matches:
                # Assign ID to destination particle if not already assigned
                if item not in particle_to_id:
                    particle_to_id[item] = current_num_items_processed
                    num_key_dict[current_num_items_processed] = item
                    current_num_items_processed += 1
                item_num = particle_to_id[item]
                converted_array.append(item_num)
            converted_dict[key_num] = converted_array

        # print(f"converted_dict: {converted_dict}")

        # create combinations of all possible distances
        max_calcs = 1
        for _, key in enumerate(converted_dict):
            max_calcs *= len(converted_dict[key])

        outmessage = f"There is a maximum of {max_calcs:,} possible combinations.\n"
        helper_functions.write_to_file("timing.log", outmessage)

        start = timer()
        best_distance_collection = self.get_best_combination(converted_dict, num_key_dict, image_particle)
        end = timer()

        if best_distance_collection is None:
            helper_functions.write_to_file("timing.log", "No valid combination found.\n")
            return None

        best_distance_collection = parse_string_to_2d_int_array(best_distance_collection)

        # print("==============================================================================================")
        # print(f"best_distance_collection: {best_distance_collection}")
        # print("==============================================================================================")

        helper_functions.write_to_file("timing.log", f"unique_combinations time: {end - start}\n")

        if best_distance_collection is None:
            return

        # build dict with best SRC:DEST relations (map numeric ids back to objects)
        final_dict = {}
        # initialize all related particles to None
        for related_particle in self.relation_array:
            final_dict[related_particle] = None

        # Track which destination particles have been assigned (safety check)
        assigned_destinations = set()

        # best_distance_collection contains pairs of integer ids (src_id, dest_id)
        # use num_key_dict to map them back to Particle_Details objects
        for src_id, dest_id in best_distance_collection:
            src_obj = num_key_dict.get(src_id)
            dest_obj = num_key_dict.get(dest_id)
            if src_obj is None:
                # unexpected id — skip
                continue

            # Safety check: verify dest_obj hasn't already been assigned
            if dest_obj in assigned_destinations:
                helper_functions.write_to_file(
                    "timing.log",
                    f"WARNING: Duplicate destination detected for particle {dest_obj}. This should not happen.\n",
                )
                continue

            final_dict[src_obj] = dest_obj
            assigned_destinations.add(dest_obj)

        return final_dict

    def get_best_combination(
        self,
        converted_dict: dict[int, list[int]],
        num_key_dict: dict[int, "Particle_Details"],
        image_particle: "Particle_Details",
    ) -> str | None:
        """
        Return the best assignment as the legacy comma-separated string format.

        Args:
            converted_dict: Mapping of source particle IDs to candidate
                destination IDs.
            num_key_dict: Lookup table from integer IDs back to particle
                objects.
            image_particle: Candidate anchor particle in the full image.

        Returns:
            A string like ``"0 12, 1 9, "`` or ``None`` when no valid
            assignment exists.
        """
        keys = list(converted_dict.keys())
        n = len(keys)

        # Build choices as lists of tuples (candidate_id, precomputed_distance)
        choices_with_dist = []
        for k in keys:
            related_particle = num_key_dict[k]
            dummy = self.primary_particle.get_dummy_particle(related_particle, image_particle)
            cand_list = []
            for cand in converted_dict[k]:
                cand_particle = num_key_dict[cand]
                dist = float(dummy.distance_to(cand_particle))
                cand_list.append((cand, dist))
            choices_with_dist.append(cand_list)

        # Run branch-and-bound search. No explicit max_evals here (exhaustive),
        # but the method supports a cap if desired in the future.
        best_distance, best_assignment = self.evaluate_with_pruning(
            keys, choices_with_dist, num_key_dict, image_particle, max_evals=None
        )

        if best_assignment is None:
            helper_functions.write_to_file("main.log", "No valid best_combination found.\n")
            return None

        # Build the string format expected by callers: "key cand, key cand, ... , "
        # Exclude skipped sources (sentinel -1) so they remain unmatched (None) in
        # the final equivalent_particles_dict.
        SKIP_SENTINEL = -1
        combination_str = (
            ", ".join(f"{keys[i]} {best_assignment[i]}" for i in range(n) if best_assignment[i] != SKIP_SENTINEL) + ", "
        )
        # print(f"Best combination found: {combination_str} with distance {best_distance}")
        return combination_str

    def evaluate_with_pruning(
        self,
        keys: list[int],
        choices: list[list[tuple[int, float] | int]],
        num_key_dict: dict[int, "Particle_Details"],
        image_particle: "Particle_Details",
        max_evals: Optional[int] = None,
    ) -> tuple[float, Optional[list[int]]]:
        """
        Evaluate the minimum-distance assignment with branch-and-bound search.

        Args:
            keys: Source particle IDs in evaluation order.
            choices: Candidate lists for each source, where entries are either
                candidate IDs or ``(candidate_id, precomputed_distance)`` pairs.
            num_key_dict: mapping from integers to Particle_Details objects.
            image_particle: Candidate anchor particle in the full image.
            max_evals: Optional cap on how many complete assignments to test.

        Returns:
            ``(best_distance, best_assignment)`` where ``best_assignment`` is a
            list aligned with ``keys``. If no valid assignment exists, returns
            ``(float('inf'), None)``.
        """
        # Validate inputs
        n = len(keys)
        if n == 0:
            return float("inf"), None

        # Penalty for leaving a source particle unmatched.  Equal to the
        # maximum search radius so that any real match within that radius
        # is preferred over skipping.
        SKIP_PENALTY = float(self.MAX_SEARCH_DISTANCE)
        # Sentinel value used in assignment[] to mean "no match for this source".
        SKIP_SENTINEL = -1

        best_distance = float("inf")
        best_assignment = None
        used = set()
        assignment = [None] * n
        eval_counter = 0

        # We'll capture pruning events for lightweight logging
        pruned_branches = 0

        def backtrack(index: int, running_total: float):
            nonlocal best_distance, best_assignment, eval_counter, pruned_branches, current_index
            if max_evals is not None and eval_counter >= max_evals:
                return
            if index == n:
                eval_counter += 1
                # found a complete assignment
                if running_total < best_distance:
                    best_distance = running_total
                    best_assignment = list(assignment)
                return

            current_index = index
            for item in choices[index]:
                # resolve candidate and distance
                if isinstance(item, tuple) and len(item) == 2:
                    cand_id, cand_dist = item
                else:
                    cand_id = item
                    cand_particle = num_key_dict[cand_id]
                    dummy = self.primary_particle.get_dummy_particle(num_key_dict[keys[index]], image_particle)
                    cand_dist = dummy.distance_to(cand_particle)

                if cand_id in used:
                    continue

                new_total = running_total + cand_dist
                # branch-and-bound pruning
                if new_total >= best_distance:
                    pruned_branches += 1
                    continue

                used.add(cand_id)
                assignment[index] = cand_id
                backtrack(index + 1, new_total)
                used.remove(cand_id)
                assignment[index] = None

            # Also try skipping this source (no match).  The skip penalty ensures
            # that a real close match is always preferred over skipping, but when
            # all real candidates are taken by other sources we still find a valid
            # (partial) assignment instead of returning None entirely.
            skip_total = running_total + SKIP_PENALTY
            if skip_total < best_distance:
                assignment[index] = SKIP_SENTINEL
                backtrack(index + 1, skip_total)
                assignment[index] = None

        # Start search
        current_index = 0
        backtrack(0, 0.0)

        # # Log pruning info if available
        # try:
        #     helper_functions.write_to_file(
        #         "timing.log",
        #         f"evaluate_with_pruning: evals={eval_counter}, pruned_branches={pruned_branches}, best_distance={best_distance}\n",
        #     )
        # except Exception:
        #     pass

        if best_assignment is None:
            return float("inf"), None
        return best_distance, best_assignment

    def _selectivity_penalty(
        self,
        related_particle: "Particle_Details",
        full_image_particle: "Particle_Details",
        best_particle: "Particle_Details",
        candidates: list["Particle_Details"],
    ) -> float:
        """Return a multiplier that penalizes ambiguous candidate matches.

        Computes the ratio of the best candidate's distance to the second-best
        candidate's distance from the expected (dummy) position:

            selectivity = best_dist / second_best_dist

        - selectivity near 0  → best candidate is dramatically closer than all
          others → high confidence → penalty near 0 → multiplier near 1.0
        - selectivity near 1  → best and second-best are almost equidistant →
          ambiguous match → heavy penalty → multiplier near 0

        If there is only one candidate there is no ambiguity, so the multiplier
        is 1.0 (no penalty).

        Returns:
            A float in ``(0.0, 1.0]`` where values near 1 indicate an
            unambiguous match.
        """
        if len(candidates) < 2:
            return 1.0

        dummy_particle = self.primary_particle.get_dummy_particle(related_particle, full_image_particle)

        best_dist = dummy_particle.distance_to(best_particle)
        # Second-best is the closest candidate that is NOT best_particle
        other_distances = [dummy_particle.distance_to(c) for c in candidates if c is not best_particle]
        if not other_distances:
            return 1.0
        second_best_dist = min(other_distances)

        # Avoid division by zero if best_dist is 0 (exact hit)
        if second_best_dist < 1e-6:
            return 1.0

        selectivity = best_dist / second_best_dist  # 0 = unambiguous, 1 = totally ambiguous

        # SELECTIVITY_SCALE controls how harshly ambiguity is penalised.
        # With SCALE=0.5: selectivity=0.9 → penalty≈0.16, selectivity=0.5 → penalty≈0.37
        SELECTIVITY_SCALE = 0.5

        # penalty rises from 0 (unambiguous) toward 1 (fully ambiguous)
        # multiplier = 1 - penalty, so unambiguous → multiplier 1.0
        penalty = 1.0 - math.exp(-selectivity / SELECTIVITY_SCALE)
        return 1.0 - penalty

    def _geometry_consistency_score(
        self,
        equivalent_particles_dict: dict["Particle_Details", "Particle_Details | None"],
    ) -> float:
        """Score how well matched particles preserve the template geometry.

        For every pair of relation particles that both have a matched destination,
        compare the pairwise distance in the source (relation) space to the pairwise
        distance in the destination (full-image) space.  Each pair contributes an
        exponential-decay score based on how much the ratio deviates from 1.0.

        A perfect geometric match (all inter-particle distances preserved exactly)
        returns 1.0.  A completely random arrangement returns a value near 0.

        Only pairs where both particles were matched are included; unmatched
        particles are handled by the completeness_ratio in percentage_match().

        Returns:
            A value in ``[0.0, 1.0]`` where 1 means pairwise source distances are
            preserved exactly.
        """
        # Collect only the matched (src, dst) pairs
        matched_pairs: list[tuple["Particle_Details", "Particle_Details"]] = [
            (src, dst) for src, dst in equivalent_particles_dict.items() if dst is not None
        ]

        if len(matched_pairs) < 2:
            # Cannot compute pairwise geometry with fewer than 2 matched particles
            return 1.0

        # SCALE controls how harshly ratio deviations are penalised.
        # A ratio of 1.5 (50% stretch) gives exp(-0.5/1.0) ≈ 0.61 with SCALE=1.0.
        GEOMETRY_SCALE = 1.0

        scores = []
        n = len(matched_pairs)
        for i in range(n):
            for j in range(i + 1, n):
                src_i, dst_i = matched_pairs[i]
                src_j, dst_j = matched_pairs[j]

                src_dist = src_i.distance_to(src_j)
                dst_dist = dst_i.distance_to(dst_j)

                # Avoid division by zero for coincident source particles
                if src_dist < 1e-6:
                    continue

                # ratio = how much the distance changed; 1.0 is perfect preservation
                ratio = dst_dist / src_dist
                # deviation from perfect: |ratio - 1|
                deviation = abs(ratio - 1.0)
                pair_score = math.exp(-deviation / GEOMETRY_SCALE)
                scores.append(pair_score)

        if not scores:
            return 1.0

        return mean(scores)

    def get_percent_score(
        self,
        related_particle: "Particle_Details",
        full_image_particle: "Particle_Details",
        best_particle: "Particle_Details | None",
    ) -> float:
        """Return the exponential-decay score for one matched related particle."""
        # if best_particle does not exist
        if best_particle is None:
            return 0

        dummy_particle = self.primary_particle.get_dummy_particle(related_particle, full_image_particle)

        # Measure the deviation: how far is best_particle from its expected position (dummy)
        deviation = dummy_particle.distance_to(best_particle)

        # Use exponential decay to score - penalizes distance much more severely
        # score = 100 * e^(-deviation/scale)
        # Smaller SCALE = harsher penalty for deviations
        SCALE = 20.0  # Tune this: smaller values = stricter scoring

        score = 100 * math.exp(-deviation / SCALE)

        return score

    def percentage_match(
        self, full_image_particle: "Particle_Details", particle_details: list["Particle_Details"]
    ) -> tuple[float, list["Particle_Details"], float, float, float, list[float]]:
        """Score how well this relation template matches one anchor particle.

        Returns the total score, matched destination particles, and the scoring
        components used to build that total.
        """
        total_percentage_score = 0

        # find the equivalent particles for each related particle
        equivalent_particles_dict = self.get_equivalent_particles_dict(full_image_particle, particle_details)

        if equivalent_particles_dict is None:
            return total_percentage_score, [], 0.0, 0.0, 0.0, []

        # find the percentage score for each matched particle
        scores = []
        matched_scores = []
        num_matched = 0
        for related_particle in self.relation_array:
            best_particle = equivalent_particles_dict[related_particle]
            if best_particle is not None:
                individual_percent_score = self.get_percent_score(related_particle, full_image_particle, best_particle)
                # penalise ambiguous matches: if many candidates were equally close,
                # the chosen one may not be meaningful
                candidates = (
                    self.find_equivalent_particles(related_particle, full_image_particle, particle_details) or []
                )
                selectivity_multiplier = self._selectivity_penalty(
                    related_particle, full_image_particle, best_particle, candidates
                )
                individual_percent_score *= selectivity_multiplier
                scores.append(individual_percent_score)
                matched_scores.append(individual_percent_score)
                num_matched += 1
            else:
                # Unmatched particles contribute 0 to the full scores list (for reporting)
                scores.append(0)

        # base_score is the mean of matched-only scores; completeness_ratio is the
        # sole penalty for missing particles, avoiding the double-count that occurred
        # when zeros were included in the mean AND multiplied by completeness_ratio.
        if matched_scores:
            base_score = mean(matched_scores)
        else:
            base_score = 0

        num_expected = len(self.relation_array)
        if num_expected > 0:
            completeness_ratio = num_matched / num_expected
        else:
            completeness_ratio = 0

        geometry_consistency = self._geometry_consistency_score(equivalent_particles_dict)

        total_percentage_score = base_score * completeness_ratio * geometry_consistency

        # collect the matched green particles (non-None values from the equivalent_particles_dict)
        matched_particles = [p for p in equivalent_particles_dict.values() if p is not None]

        return total_percentage_score, matched_particles, base_score, completeness_ratio, geometry_consistency, scores


##########
# FUNCTIONS
##########


def parse_string_to_2d_int_array(s: str) -> list[list[int]]:
    """
    Convert a comma-separated ``"a b"`` string into a nested integer list.

    Args:
        s: String produced by ``get_best_combination``.

    Returns:
        A list of ``[source_id, dest_id]`` integer pairs.
    """
    result = []
    for pair in s.strip().split(","):
        pair = pair.strip()
        if not pair:
            continue
        nums = [int(x) for x in pair.split()]
        result.append(nums)
    return result


def detect_particles(image: np.ndarray) -> list[list[list[int]]]:
    """Find connected non-black particle regions in an image.

    Returns a sanitized list of particle descriptors in
    ``[top_left, bottom_right, pixel_count]`` form.
    """
    # checking_pixels_array is intentionally local (not the module-level global) so
    # that concurrent calls from multiple threads each maintain their own independent
    # visited state.  Using the global caused a race condition: one thread's BFS would
    # overwrite another thread's checking_pixels_array mid-scan, causing valid seed
    # pixels to appear already-visited and skipping particles entirely.
    particle_array = []
    checking_pixels_array = []

    for row in image:
        row_array = []
        for col in row:
            row_array.append(0)
        checking_pixels_array.append(row_array)

    for row in range(0, len(image) - 1):
        for col in range(0, len(image[0]) - 1):
            # if a particle wasn't found in this position
            if checking_pixels_array[row][col] == 0:
                if not helper_functions.is_black(image[row][col]):
                    particle_array.append(get_particle_corners(row, col, image, checking_pixels_array))
                    # print(image[row][col])

    # get the median size of the particles to display for a sanity check
    particle_sizes = [get_particle_size(particle) for particle in particle_array]
    if len(particle_sizes) > helper_functions.MIN_PARTICLE_SIZE:
        median_size = np.median(particle_sizes)
    else:
        raise ValueError("No particles detected in the image.")

    # display the particles for a sanity check
    small_particles = []
    for particle in particle_array:
        if get_particle_size(particle) < median_size:
            small_particles.append(get_particle_size(particle))

    small_particles.sort()
    final_particle_array = sanitize_particle_array(particle_array)

    return final_particle_array


def sanitize_particle_array(particle_array: list[list[list[int]]]) -> list[list[list[int]]]:
    """Discard particle descriptors whose pixel count does not exceed the threshold."""
    # this function is meant to remove any particles that are too small to be real particles
    #   this is done by checking the size of the particle and removing it if it's below a certain threshold
    sanitized_particle_array = []

    for particle in particle_array:
        if get_particle_size(particle) > helper_functions.MIN_PARTICLE_SIZE:
            sanitized_particle_array.append(particle)
    return sanitized_particle_array


def get_particle_size(particle: list) -> int:
    """Return the stored non-black pixel count for a particle descriptor."""
    # particle[2] is the exact non-black pixel count recorded by the BFS in get_particle_corners
    return particle[2]


def is_in_bounds(row: int, col: int, image: np.ndarray) -> bool:
    """Return whether the coordinate lies inside ``image``."""
    return row >= 0 and col >= 0 and row < len(image) and col < len(image[0])


def inbounds_line(rr: np.ndarray, cc: np.ndarray, image: np.ndarray) -> tuple[list[int], list[int]]:
    """Filter line or circle coordinates down to points within the image bounds."""
    new_rr = []
    new_cc = []
    for index, _ in enumerate(rr):
        if is_in_bounds(rr[index], cc[index], image):
            new_rr.append(rr[index])
            new_cc.append(cc[index])
    return new_rr, new_cc


def draw_box_given_coords(corner_coordinates: list[list[int]], color_pixel: list[int], image: np.ndarray) -> np.ndarray:
    """Draw a one-pixel rectangle just outside a particle bounding box."""
    # corner_coordinates layout (post Bug-1 fix):
    #   corner_coordinates[0] = [min_row, min_col]  (top-left)
    #   corner_coordinates[1] = [max_row, max_col]  (bottom-right)
    #
    # The box is drawn one pixel outside the particle boundary on each side.

    # Ensure color_pixel matches the image's channel count
    color_pixel = color_pixel[: image.shape[2]]

    min_row = corner_coordinates[0][0]
    min_col = corner_coordinates[0][1]
    max_row = corner_coordinates[1][0]
    max_col = corner_coordinates[1][1]

    # top line — one row above the particle
    if min_row - 1 >= 0:
        for col_index in range(min_col, max_col + 1):
            image[min_row - 1][col_index] = color_pixel
    # bottom line — one row below the particle
    if max_row + 1 < len(image):
        for col_index in range(min_col, max_col + 1):
            image[max_row + 1][col_index] = color_pixel
    # left line — one column to the left of the particle
    if min_col - 1 >= 0:
        for row_index in range(min_row, max_row + 1):
            image[row_index][min_col - 1] = color_pixel
    # right line — one column to the right of the particle
    if max_col + 1 < len(image[0]):
        for row_index in range(min_row, max_row + 1):
            image[row_index][max_col + 1] = color_pixel

    return image


def get_particle_corners(row: int, col: int, image: np.ndarray, checking_pixels_array: list) -> list[list[int]]:
    """Flood-fill one connected non-black region and return its bounds and size."""
    # checking_pixels_array is passed in (not read from the global) so that each
    # concurrent detect_particles call operates on its own independent visited state.

    # print("get particle corners")
    # print("\tposition:", row, col)
    q = queue.Queue()
    checking_dict = {}
    top_left_pos = [row, col]
    bottom_right_pos = [row, col]

    q.put((row, col))
    checking_dict[(row, col)] = 1
    checking_pixels_array[row][col] = 1

    while not q.empty():
        pos = q.get()
        # print("\t\tposition:", pos[0], pos[1])
        if pos[0] < top_left_pos[0]:
            top_left_pos[0] = pos[0]
        if pos[1] < top_left_pos[1]:  # min col → top_left (was wrongly stored in bottom_right)
            top_left_pos[1] = pos[1]
        if pos[1] > bottom_right_pos[1]:  # max col → bottom_right (was wrongly stored in top_left)
            bottom_right_pos[1] = pos[1]
        if pos[0] > bottom_right_pos[0]:
            bottom_right_pos[0] = pos[0]

        for modifier in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
            delta_x = pos[0] + modifier[0]
            delta_y = pos[1] + modifier[1]
            if is_in_bounds(delta_x, delta_y, image) and not helper_functions.is_black(image[delta_x][delta_y]):
                temp_tuple = (delta_x, delta_y)
                if is_in_bounds(delta_x, delta_y, image) and temp_tuple not in checking_dict:
                    q.put(temp_tuple)
                    checking_dict[temp_tuple] = 1
                    checking_pixels_array[delta_x][delta_y] = 1

    return [top_left_pos, bottom_right_pos, len(checking_dict)]


def show_detected_particles(particle_array: list[list[list[int]]], image: np.ndarray, save_dir: str) -> None:
    """Draw bounding boxes for detected particles and save the annotated image."""
    for corner_coordinates in particle_array:
        image = draw_box_given_coords(corner_coordinates, red_pixel, image)

    helper_functions.save_io_image("detected_particles", image, save_dir)


def get_particle_details(particle_positions, image) -> list[Particle_Details]:
    """Convert raw particle descriptors into populated ``Particle_Details`` objects."""
    particle_details = []

    for particle_position in particle_positions:
        # get center position of particle — use only [top_left, bottom_right], not the pixel count at [2]
        center_position = np.array(np.floor(np.mean(particle_position[:2], axis=0))).astype(int)

        # use center_position and color to create a particle object and add it to the return array
        particle_details.append(Particle_Details(particle_position[:2], center_position))

    # derive search radius from nearest-neighbour distances now that all centres are known;
    # falls back to the image-size heuristic when fewer than 2 particles are present
    search_radius = get_nearest_neighbor_search_radius(particle_details, image)

    # once all particles are created, don't forget to call the
    #  set_full_particle and set_nearby_particles functions
    for particle_detail_index, particle in enumerate(particle_positions):
        particle_details[particle_detail_index].set_full_particle(particle_positions[particle_detail_index][:2], image)
        particle_details[particle_detail_index].set_color()
        particle_details[particle_detail_index].set_nearby_particles(particle_details, search_radius)
    return particle_details


def show_particle_details(particle_details: list[Particle_Details], image: np.ndarray, save_dir: str) -> None:
    """Draw neighbour connections between particles and save the result."""
    # draw the lines connecting each particle to the other
    for particle in particle_details:
        this_center_pos = particle.center_position
        for other in particle.nearby_particles:
            # draw lines from particle to all connected particles
            other_center_pos = other.center_position
            rr, cc = line(
                this_center_pos[0],
                this_center_pos[1],
                other_center_pos[0],
                other_center_pos[1],
            )
            image[rr, cc] = red_pixel
            # profit

    # clean up the image--redraw the particles on top of where they used to be
    for particle in particle_details:
        coords = particle.corner_coordinates
        # modifier for the rows and cols
        #  used to change from index->image to index->shape
        row_delta = coords[0][0]
        col_delta = coords[0][1]  # min col is now correctly in coords[0][1]

        for row in range(coords[0][0], (coords[1][0] + 1)):
            for col in range(coords[0][1], (coords[1][1] + 1)):
                image[row][col] = particle.full_particle[row - row_delta][col - col_delta]

    helper_functions.save_io_image("particle_connections", image, save_dir)
    return


def save_particles(name: str, particle_details_array: list[Particle_Details], save_dir: str) -> None:
    """Save each particle crop as an image and record the saved paths in a CSV."""
    # stupid variables that I shouldn't need but WHATEVER
    troublesome_black_pixel = [np.uint8(0), np.uint8(0), np.uint8(0)]

    # Create subdirectory for saving particles.
    # Previously the directory was cleared before writing, which meant two threads
    # using the same name would delete each other's output.  Instead we no longer
    # clear: helper_functions.save_io_image already uses an atomic per-directory
    # counter so filenames are always unique, and the CSV is opened in "w" mode
    # once per call which safely replaces any prior CSV from a previous run.
    particles_dir = os.path.join(save_dir, f"{name}_images")
    csv_path = os.path.join(particles_dir, "data.csv")

    # Ensure directory exists
    os.makedirs(particles_dir, exist_ok=True)
    # Create (or truncate) the CSV for this batch — safe because each call owns
    # a uniquely-named particles_dir via the `name` parameter.
    with open(csv_path, "w") as f:
        pass

    for particle in particle_details_array:
        # Deep-copy full_particle before padding so the live particle object is
        # never mutated.  The original assignment was a bare reference:
        #   cur_particle_array = particle.full_particle
        # which meant every append() below permanently corrupted the particle's
        # data, breaking any later use of full_particle (e.g. clean_particle_in_image).
        cur_particle_array = [list(row) for row in particle.full_particle]

        # adjust size to fit
        with Particle_Details._largest_lock:
            target_width = Particle_Details.largest_particle_width
            target_length = Particle_Details.largest_particle_length

        if len(cur_particle_array[0]) < target_width:
            for row_index in range(len(cur_particle_array)):
                while len(cur_particle_array[row_index]) < target_width:
                    cur_particle_array[row_index].append(troublesome_black_pixel)
        if len(cur_particle_array) < target_length:
            while len(cur_particle_array) < target_length:
                temp_row = []
                for _ in cur_particle_array[0]:
                    temp_row.append(troublesome_black_pixel)
                cur_particle_array.append(temp_row)

        # save particle — save_io_image uses an atomic counter so the filename
        # is unique even if another thread is writing to the same directory.
        plot_name = helper_functions.save_io_image(name, np.asarray(cur_particle_array), particles_dir)
        full_path = os.path.join(particles_dir, plot_name) if plot_name else ""
        with open(csv_path, "a") as f:
            f.write(f"{full_path}\n")


def get_nearest_neighbor_search_radius(particle_details: list[Particle_Details], image: np.ndarray) -> int:
    """
    Derive a connectivity radius from the K-th nearest-neighbour distance.

    For each particle, all other particles are sorted by distance and the distance
    to the K-th nearest neighbour (the furthest of the K closest) is recorded.
    The median of those per-particle K-th-nearest distances is then scaled by
    ``helper_functions.NN_RADIUS_MULTIPLIER`` to yield a radius that covers
    roughly K neighbours per particle without connecting distant, unrelated particles.

    Falls back to ``helper_functions.NN_FALLBACK_RADIUS`` when there are not enough
    particles to compute K neighbours (fewer than K + 1 particles present).

    Args:
        particle_details: Detected particles with centre positions already set.
        image: Unused and retained for API compatibility.

    Returns:
        Search radius in pixels, with a minimum of 1 and a fixed fallback when
        too few particles exist.
    """
    k = helper_functions.NN_K
    if len(particle_details) <= k:
        return helper_functions.NN_FALLBACK_RADIUS

    kth_nn_distances = []
    for p in particle_details:
        distances = sorted(p.distance_to(q) for q in particle_details if q is not p)
        kth_nn_distances.append(distances[k - 1])  # 0-indexed: index k-1 is the K-th nearest

    median_kth_nn = float(np.median(kth_nn_distances))
    return max(1, round(median_kth_nn * helper_functions.NN_RADIUS_MULTIPLIER))


def get_relations_object(red_square_particle_details: list[Particle_Details]) -> Relations:
    """Build a ``Relations`` template from one processed red-square image."""
    # get largest particle
    largest_particle = red_square_particle_details[0]
    for particle in red_square_particle_details:
        if particle.size() > largest_particle.size():
            largest_particle = particle

    # create Relations object
    relations_object = Relations(largest_particle)

    # Only add particles that are within the primary particle's proximity cluster.
    # Each particle's nearby_particles list was populated by set_nearby_particles()
    # using SEARCH_RADIUS, so it already encodes which particles are spatially
    # related.  Adding only those particles keeps the relation_array high-signal:
    # it represents the local cluster around the primary rather than every particle
    # in the entire image.
    #
    # Fall back to all particles if nearby_particles is uninitialised (-1 sentinel)
    # so the function never silently produces an empty relation_array.
    nearby = largest_particle.nearby_particles
    if nearby == -1 or not isinstance(nearby, list):
        # uninitialised — add everything (safe fallback)
        candidates = red_square_particle_details
    else:
        candidates = nearby

    for particle in candidates:
        relations_object.add_relation(particle)

    # return
    return relations_object


def assign_particle_ids(particle_list, start_id: int = 1):
    """Assign stable integer IDs to each Particle_Details in-place.

    Args:
        particle_list: Iterable of particle objects to modify in place.
        start_id: First integer ID to assign.

    Returns:
        A dictionary mapping assigned IDs to particle objects.
    """
    mapping = {}
    cur = start_id
    for p in particle_list:
        p.id = cur
        mapping[cur] = p
        cur += 1
    return mapping


def serialize_particle_collection(particles: list[Particle_Details], metadata: dict | None = None):
    """Return a single JSON-serializable dict representing the whole collection.

    IDs are assigned first if needed.
    """
    # Ensure IDs
    needs_ids = any(getattr(p, "id", None) is None for p in particles)
    if needs_ids:
        assign_particle_ids(particles)

    coll = {
        "metadata": metadata or {},
        "particles": [p.to_json() for p in particles],
    }
    return coll


def save_particle_collection(
    path: str, particles: list[Particle_Details], metadata: dict | None = None, encoding: str = "utf-8"
):
    """Save the particle collection to a single JSON file (lossless, with IDs).

    This writes pretty-printed JSON for human readability.
    """
    coll = serialize_particle_collection(particles, metadata)
    with open(path, "w", encoding=encoding) as f:
        json.dump(coll, f, indent=2, ensure_ascii=False)


def load_particle_collection(path: str, encoding: str = "utf-8"):
    """Load a particle collection saved by `save_particle_collection` and rehydrate objects.

    Returns (particles_list, metadata_dict).
    """
    with open(path, "r", encoding=encoding) as f:
        coll = json.load(f)

    particles_data = coll.get("particles", [])
    metadata = coll.get("metadata", {})

    # First pass: create Particle_Details objects without resolving nearby references
    id_map = {}
    particles = []
    for pdata in particles_data:
        p = Particle_Details.from_json(pdata)
        particles.append(p)
        if getattr(p, "id", None) is not None:
            id_map[int(p.id)] = p

    # Second pass: resolve nearby_particle_ids -> nearby_particles list of objects
    for pdata, p in zip(particles_data, particles):
        nid_list = pdata.get("nearby_particle_ids")
        if nid_list is None:
            p.nearby_particles = -1
            continue
        # Resolve ids to objects while preserving None placeholders and order
        resolved = []
        for nid in nid_list:
            if nid is None:
                resolved.append(None)
            else:
                resolved.append(id_map.get(int(nid)))
        # preserve exact structure (including None entries) to be lossless
        p.nearby_particles = resolved

    return particles, metadata


def _convert_to_serializable(value):
    """Recursively convert NumPy-backed values into JSON-safe Python types.

    Arrays become lists and NumPy scalars become plain Python scalars.
    """
    # handle None
    if value is None:
        return None

    # numpy ndarray or numpy scalar
    try:
        import numpy as _np

        if isinstance(value, _np.ndarray):
            return _convert_to_serializable(value.tolist())
        if isinstance(value, _np.generic):
            try:
                return int(value.item())
            except Exception:
                return value.item()
    except Exception:
        pass

    # list or tuple
    if isinstance(value, (list, tuple)):
        return [_convert_to_serializable(v) for v in value]

    # numeric types that can be cast to int
    try:
        if isinstance(value, (int,)):
            return value
        if hasattr(value, "item"):
            return _convert_to_serializable(value.item())
        # fallback for numpy scalars already handled above
    except Exception:
        pass

    return value


def get_percentages(
    relations_object: Relations,
    full_image_particle_details: list[Particle_Details],
    image: np.ndarray,
    samples_dir: Path,
    progress=None,
) -> list[tuple[float, Particle_Details, list[Particle_Details]]]:
    """Evaluate all particles in the full image and return the top matches.

    For each particle in full_image_particle_details, calculate its percentage match
    against the relations_object. Return the top `max_groups` (default 16) results
    sorted by percentage in descending order.

    Args:
        relations_object: The Relations object containing the reference pattern.
        full_image_particle_details: List of all particles to evaluate.
        image: The full image, retained for API consistency and currently unused.
        samples_dir: Directory where ``data/particles.csv`` is written.
        progress: Optional progress tracker updated during scoring.

    Returns:
        List of tuples (percentage_score, particle, matched_particles, base_score,
        completeness_ratio, geometry_consistency, individual_scores) for the top matches.
        The three breakdown floats are the individual scoring components that compose
        percentage_score: percentage_score = base_score * completeness_ratio * geometry_consistency.
        individual_scores is the per-particle list of scores (0 for unmatched) whose mean is base_score.

    Side Effects:
        Overwrites ``data/particles.csv`` for the current sample directory.
    """
    max_groups = 16
    percentages = []
    total = len(full_image_particle_details)
    # Update the progress phase every ~5% of particles (at least every 10, at most every 1).
    update_step = max(1, total // 20)

    # Evaluate each particle in the full image
    for idx, particle in enumerate(full_image_particle_details):
        if progress is not None and idx % update_step == 0:
            progress.set_phase("scoring particle", idx + 1, total)
        percent_match, matched_particles, base_score, completeness_ratio, geometry_consistency, individual_scores = (
            relations_object.percentage_match(particle, full_image_particle_details)
        )

        # Skip invalid results
        if percent_match is None or percent_match == -1:
            helper_functions.write_to_file(
                "timing.log", f"Skipping particle {particle.id} due to invalid percentage match: {percent_match}\n"
            )
            continue

        percentages.append(
            (
                percent_match,
                particle,
                matched_particles,
                base_score,
                completeness_ratio,
                geometry_consistency,
                individual_scores,
            )
        )

    particles_csv_path = Path(samples_dir) / "data" / "particles.csv"

    # save to particles.csv in the data directory
    with particles_csv_path.open("w") as f:
        for _, particle, *_ in percentages:
            f.write(f"{particle.id},{str(particle)}\n")

    # trim to the top 16 overall
    if percentages:
        percentages.sort(key=lambda x: x[0], reverse=True)
        percentages = percentages[:max_groups]

    return percentages


def display_best_percentages(
    red_green_name: str,
    percentages_array: list,
    particle_details: list,
    relations_object: Relations,
    image: np.ndarray,
    save_dir: Path,
    conglomerate_image: np.ndarray,
    save_step_images: bool = False,
):
    """
    Draw and optionally save the top-scoring match visualizations for one image pair.

    The top ``BEST_PERCENTAGES_TO_DISPLAY`` results are drawn onto per-result
    copies of ``image`` and also onto ``conglomerate_image`` in place so the
    caller can accumulate annotations across many red/green comparisons.
    """
    num_best_percentages = helper_functions.BEST_PERCENTAGES_TO_DISPLAY

    # Use save_dir as a string
    save_dir = str(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Sort percentages_array by the first element (percentage) in descending order
    percentages_array = sorted(percentages_array, key=lambda x: x[0], reverse=True)
    best_percentages = percentages_array[:num_best_percentages]

    # Draw a box around every detected particle so that all particles are visible
    # on both the individual best-percentage images and the conglomerate image.
    # draw_on_image only repaints particle pixels and draws connection lines/circles;
    # it never draws bounding boxes, so without this step particles that were not
    # part of the best match have no visual indicator on the saved images.
    for particle in particle_details:
        draw_box_given_coords(particle.corner_coordinates, red_pixel, conglomerate_image)

    # Create a separate copy of the image for each of the top best percentages (individual saves)
    # Draw boxes onto each copy after the conglomerate has been stamped so the
    # per-image copies are independent and also show all particle boxes.
    image_array = [image.copy() for _ in range(num_best_percentages)]
    for idx in range(num_best_percentages):
        for particle in particle_details:
            draw_box_given_coords(particle.corner_coordinates, red_pixel, image_array[idx])

    # For each of the top best percentages, draw the relations on its corresponding image copy
    # and also onto the conglomerate image
    for idx, (percent, particle, _matched_particles, *_) in enumerate(best_percentages):
        relations_object.draw_on_image(particle, particle_details, image_array[idx])
        relations_object.draw_on_image(particle, particle_details, conglomerate_image)

    # Save individual images
    if save_step_images:
        for image_index in range(num_best_percentages):
            helper_functions.save_io_image(
                f"{red_green_name}s_best_percentages_image_{image_index}",
                image_array[image_index],
                save_dir,
                True,
            )
