import sys
import threading
from datetime import datetime


class DirState:
    """
    Holds the mutable display state for a single active directory slot.

    All public methods (set_images, set_image, set_phase, set_sub_phase,
    clear_sub_phase, milestone) mirror the old ProgressTracker interface so
    that process_sample_dir requires no body changes.  Every mutating method
    acquires the parent MultiProgressTracker's lock before touching fields, so
    concurrent directory threads never produce interleaved terminal writes.
    """

    def __init__(
        self,
        slot_idx: int,
        dir_idx: int,
        dir_name: str,
        total_dirs: int,
        parent: "MultiProgressTracker",
    ) -> None:
        """Initialize the mutable progress state for one directory slot."""
        self.slot_idx = slot_idx
        self.dir_idx = dir_idx
        self.dir_name = dir_name
        self.total_dirs = total_dirs
        self.total_images = 0
        self.image_idx = 0
        self.phase = ""
        self.phase_step = 0
        self.phase_total = 0
        self.sub_phase = ""
        self.sub_phase_step = 0
        self.sub_phase_total = 0
        self._parent = parent

    def set_images(self, total: int) -> None:
        """Set the total number of images for this directory and rerender."""
        with self._parent._lock:
            self.total_images = total
            self.image_idx = 0
            self._parent._rerender_locked()

    def set_image(self, idx: int) -> None:
        """Update the currently active image index and rerender."""
        with self._parent._lock:
            self.image_idx = idx
            self._parent._rerender_locked()

    def set_phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        """Set the primary phase label and clear any sub-phase state."""
        with self._parent._lock:
            self.phase = phase
            self.phase_step = step
            self.phase_total = total
            self.sub_phase = ""
            self.sub_phase_step = 0
            self.sub_phase_total = 0
            self._parent._rerender_locked()

    def set_sub_phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        """Set the secondary phase label displayed after the main phase."""
        with self._parent._lock:
            self.sub_phase = phase
            self.sub_phase_step = step
            self.sub_phase_total = total
            self._parent._rerender_locked()

    def clear_sub_phase(self) -> None:
        """Clear the secondary phase label and rerender the tracker."""
        with self._parent._lock:
            self.sub_phase = ""
            self.sub_phase_step = 0
            self.sub_phase_total = 0
            self._parent._rerender_locked()

    def milestone(self, msg: str) -> None:
        """Emit a scrolling milestone message through the parent tracker."""
        self._parent.milestone(msg)


class NullProgress:
    """No-op progress sink used when work runs outside the live TTY tracker."""

    def set_images(self, _total: int) -> None:
        return

    def set_image(self, _idx: int) -> None:
        return

    def set_phase(self, _phase: str, _step: int = 0, _total: int = 0) -> None:
        return

    def set_sub_phase(self, _phase: str, _step: int = 0, _total: int = 0) -> None:
        return

    def clear_sub_phase(self) -> None:
        return

    def milestone(self, _msg: str) -> None:
        return


class MultiProgressTracker:
    """
    Manages an N-row ANSI progress block — one row per active directory slot —
    that overwrites itself in-place, plus scrolling milestone messages above it.

    In TTY mode, ANSI cursor-movement codes (\033[NA to move cursor up N rows,
    \r\033[2K to clear a line) keep the block pinned while milestones accumulate
    above.  In non-TTY mode (pipe / redirect), no ANSI codes are ever emitted:
    milestone() calls print() directly and progress updates produce no output,
    keeping redirected logs clean.

    IMPORTANT: nothing outside this class should call print() or write to
    sys.stdout directly.  All user-visible output must go through milestone().

    Usage:
        tracker = MultiProgressTracker(total_dirs=n, max_concurrent=k)
        state = tracker.acquire_slot(dir_idx, dir_name)  # once per directory thread
        state.set_images(10)
        state.set_image(1)
        state.set_phase("green square")
        state.milestone("WARNING: ...")
        tracker.release_slot(state)                       # always in a finally block
        tracker.done()
    """

    _LINE_WIDTH = 160
    # Fixed inner width of the phase bracket so [ and ] never shift position.
    # Widest phase label is "RS 1/1: saving images" (21 chars) + step prefix
    # "99/99: " (7 chars) = 28. Round up to 40 for safety.
    _PHASE_WIDTH = 40
    # Fixed width of the image counter field ("Image NNN/NNN").
    # Assumes image counts never exceed 3 digits.
    _IMG_WIDTH = 13
    # Fixed inner width of the sub-phase bracket (process_image step detail).
    # Widest label is "remove color aggregations" (24 chars) + "11/11: " (7) = 31.
    # Set to 40 to match _PHASE_WIDTH and avoid overflow.
    _SUB_PHASE_WIDTH = 45

    def __init__(self, total_dirs: int, max_concurrent: int) -> None:
        """Create a tracker that can display up to ``max_concurrent`` active directories."""
        self.total_dirs = total_dirs
        self.max_concurrent = max(1, max_concurrent)
        self._slots: list[DirState | None] = [None] * self.max_concurrent
        self._lock = threading.Lock()
        self._start_time = datetime.now()
        self._tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        # True once the block has been drawn at least once (so we know to move
        # the cursor up before the next render).
        self._block_rendered = False
        # Width of the dir-name column; recalculated whenever slots change.
        self._dir_name_width = 0

    def acquire_slot(self, dir_idx: int, dir_name: str) -> DirState:
        """Claim the first free display row and return its ``DirState``."""
        with self._lock:
            for i, slot in enumerate(self._slots):
                if slot is None:
                    state = DirState(
                        slot_idx=i,
                        dir_idx=dir_idx,
                        dir_name=dir_name,
                        total_dirs=self.total_dirs,
                        parent=self,
                    )
                    self._slots[i] = state
                    self._recalc_dir_name_width_locked()
                    self._rerender_locked()
                    return state
        raise RuntimeError("No available progress slots — max_concurrent exceeded")

    def release_slot(self, state: DirState) -> None:
        """Free a slot, rerender the display, and announce completion."""
        with self._lock:
            self._slots[state.slot_idx] = None
            self._recalc_dir_name_width_locked()
            self._rerender_locked()
        # milestone() acquires the lock itself; call outside our lock to avoid nesting
        self.milestone(f"Completed: {state.dir_name}")

    def milestone(self, msg: str) -> None:
        """Print a scrolling message above the progress block and redraw it."""
        with self._lock:
            if self._tty and self._block_rendered:
                # Move cursor to block row 0 (N dir rows + 1 timer row above cursor).
                sys.stdout.write(f"\033[{self.max_concurrent + 1}A")
                sys.stdout.write("\r\033[2K" + msg + "\n")
                self._render_block_from_here_locked()
            else:
                print(msg)
                if self._tty:
                    self._rerender_locked()

    def done(self) -> None:
        """Finalize terminal output and print the overall completion message."""
        with self._lock:
            elapsed = self._elapsed_str()
            msg = f"All directories processed.  Total time: {elapsed}"
            if self._tty and self._block_rendered:
                # Block is N dir rows + 1 timer row.  Replace row 0 with the
                # completion message, clear remaining rows, then park the cursor
                # on the line right below the message.
                block_height = self.max_concurrent + 1
                sys.stdout.write(f"\033[{block_height}A")
                sys.stdout.write("\r\033[2K" + msg + "\n")
                for _ in range(block_height - 1):
                    sys.stdout.write("\r\033[2K\n")
                if block_height > 1:
                    sys.stdout.write(f"\033[{block_height - 1}A")
                sys.stdout.flush()
            else:
                print(msg)

    def _elapsed_str(self) -> str:
        """Return elapsed wall-clock time since construction as ``HH:MM:SS``."""
        delta = datetime.now() - self._start_time
        total_seconds = int(delta.total_seconds())
        h, remainder = divmod(total_seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _rerender_locked(self) -> None:
        """Rerender the entire progress block while holding ``_lock``."""
        if not self._tty:
            return
        if self._block_rendered:
            # Move up past N dir rows + 1 timer row.
            sys.stdout.write(f"\033[{self.max_concurrent + 1}A")
        self._render_block_from_here_locked()

    def _recalc_dir_name_width_locked(self) -> None:
        """Recompute the padded directory-name column width from active slots."""
        active_names = [s.dir_name for s in self._slots if s is not None]
        self._dir_name_width = max((len(n) for n in active_names), default=0)

    def _render_block_from_here_locked(self) -> None:
        """Write every directory row plus the elapsed-time row while locked."""
        for state in self._slots:
            line = self._format_row(state)
            sys.stdout.write("\r\033[2K" + line.ljust(self._LINE_WIDTH) + "\n")
        # Dedicated timer row below all directory rows.
        timer_line = f"  +{self._elapsed_str()}"
        sys.stdout.write("\r\033[2K" + timer_line + "\n")
        self._block_rendered = True
        sys.stdout.flush()

    def _estimate_dir_progress_pct(self, state: DirState) -> int:
        """Return an integer per-directory progress percentage for one active row."""
        if state.total_images <= 0:
            return 0

        completed_images = max(0, min(state.image_idx - 1, state.total_images))
        current_image_fraction = 0.0

        if state.image_idx > 0:
            if state.phase_total > 0:
                current_image_fraction = min(max(state.phase_step / state.phase_total, 0.0), 1.0)
            elif state.phase:
                # A named phase with no explicit step count means the current
                # image has started but cannot be measured more precisely.
                current_image_fraction = 0.0

        total_fraction = (completed_images + current_image_fraction) / state.total_images
        return min(100, max(0, int(total_fraction * 100)))

    def _format_row(self, state: "DirState | None") -> str:
        """Format one display row, or return an empty string for an idle slot."""
        if state is None:
            return ""
        dir_w = len(str(self.total_dirs)) if self.total_dirs else 1
        dir_pct = self._estimate_dir_progress_pct(state)
        dir_part = f"Dir {state.dir_idx + 1:0{dir_w}}/{self.total_dirs:0{dir_w}} ({dir_pct:3d}%)  {state.dir_name:<{self._dir_name_width}}"
        if state.total_images:
            img_w = len(str(state.total_images))
            img_inner = f"Image {state.image_idx:0{img_w}}/{state.total_images:0{img_w}}"
        else:
            img_inner = ""
        img_part = f"  {img_inner:<{self._IMG_WIDTH}}"
        if state.phase:
            if state.phase_total:
                width = len(str(state.phase_total))
                step_part = f"{state.phase_step:0{width}}/{state.phase_total:0{width}}: "
            else:
                step_part = ""
            inner = f"{step_part}{state.phase}"
            # Truncate so the bracket width is always exactly _PHASE_WIDTH.
            inner = inner[: self._PHASE_WIDTH]
            phase_part = f"  [{inner:<{self._PHASE_WIDTH}}]"
        else:
            phase_part = f"  [{' ' * self._PHASE_WIDTH}]"
        if state.sub_phase:
            if state.sub_phase_total:
                width = len(str(state.sub_phase_total))
                sub_step_part = f"{state.sub_phase_step:0{width}}/{state.sub_phase_total:0{width}}: "
            else:
                sub_step_part = ""
            sub_inner = f"{sub_step_part}{state.sub_phase}"
            sub_phase_part = f"  [{sub_inner:<{self._SUB_PHASE_WIDTH}}]"
        else:
            sub_phase_part = ""
        return f"{dir_part}{img_part}{phase_part}{sub_phase_part}"
