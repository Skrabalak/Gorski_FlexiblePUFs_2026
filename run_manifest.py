import json
from datetime import datetime
from pathlib import Path

RUN_STATE_DIRNAME = "run_state"


def get_run_state_root(base_dir: Path) -> Path:
    """Return the repository-local root directory for persisted run manifests."""
    return base_dir / RUN_STATE_DIRNAME


def _make_run_id(run_state_root: Path) -> str:
    base_run_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    run_id = base_run_id
    suffix = 1
    while (run_state_root / run_id).exists():
        run_id = f"{base_run_id}-{suffix}"
        suffix += 1
    return run_id


def get_latest_run_id(base_dir: Path) -> str | None:
    """Return the newest run ID under ``run_state/`` or ``None`` when absent."""
    run_state_root = get_run_state_root(base_dir)
    if not run_state_root.exists():
        return None

    run_ids = sorted(path.name for path in run_state_root.iterdir() if path.is_dir())
    return run_ids[-1] if run_ids else None


def get_run_dir(base_dir: Path, run_id: str) -> Path:
    """Return the manifest directory for one run ID."""
    return get_run_state_root(base_dir) / run_id


def _directory_state_for_event(event_type: str) -> str | None:
    return {
        "directory_discovered": "discovered",
        "directory_queued": "queued",
        "directory_started": "running",
        "directory_completed": "completed",
        "directory_failed": "failed",
        "directory_skipped": "skipped",
    }.get(event_type)


def load_run_state(base_dir: Path, run_id: str) -> dict:
    """Load and derive the latest effective state for a prior run."""
    run_dir = get_run_dir(base_dir, run_id)
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {events_path}")

    derived_state = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "root_dir": None,
        "parallel_mode": None,
        "resume_from_run_id": None,
        "run_status": "unknown",
        "directory_states": {},
        "attempts": {},
    }

    with events_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("event_type", "")
            sample_dir = event.get("sample_dir")
            attempt = event.get("attempt")

            if event_type == "run_started":
                derived_state["root_dir"] = event.get("root_dir")
                derived_state["parallel_mode"] = event.get("parallel_mode")
                derived_state["resume_from_run_id"] = event.get("resume_from_run_id")
                derived_state["run_status"] = "running"
            elif event_type == "run_completed":
                derived_state["run_status"] = "completed"
            elif event_type == "run_aborted":
                derived_state["run_status"] = "aborted"

            directory_state = _directory_state_for_event(event_type)
            if directory_state is not None and sample_dir:
                derived_state["directory_states"][sample_dir] = directory_state
                if isinstance(attempt, int):
                    derived_state["attempts"][sample_dir] = max(
                        attempt,
                        derived_state["attempts"].get(sample_dir, 0),
                    )

    return derived_state


class RunManifest:
    """Append-only run event log plus derived summary for scheduler state."""

    def __init__(
        self,
        base_dir: Path,
        root_dir: Path,
        parallel_mode: str,
        process_only: bool,
        always_process: bool,
        max_concurrent: int,
        resume_from_run_id: str | None = None,
        retry_failed: bool = False,
        previous_attempts: dict[str, int] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.root_dir = root_dir.resolve()
        self.parallel_mode = parallel_mode
        self.process_only = process_only
        self.always_process = always_process
        self.max_concurrent = max_concurrent
        self.resume_from_run_id = resume_from_run_id
        self.retry_failed = retry_failed
        self.run_state_root = get_run_state_root(base_dir)
        self.run_state_root.mkdir(parents=True, exist_ok=True)
        self.run_id = _make_run_id(self.run_state_root)
        self.run_dir = self.run_state_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.latest_directory_states: dict[str, str] = {}
        self.latest_attempts: dict[str, int] = dict(previous_attempts or {})
        self.run_status = "running"

        self.emit(
            "run_started",
            parallel_mode=self.parallel_mode,
            root_dir=str(self.root_dir),
            process_only=self.process_only,
            always_process=self.always_process,
            max_concurrent=self.max_concurrent,
            retry_failed=self.retry_failed,
            resume_from_run_id=self.resume_from_run_id,
        )

    def sample_dir_key(self, sample_dir: Path) -> str:
        """Return the manifest key used for one sample directory."""
        try:
            return sample_dir.resolve().relative_to(self.root_dir).as_posix()
        except ValueError:
            return sample_dir.resolve().as_posix()

    def start_attempt(self, sample_dir_key: str) -> int:
        """Increment and return the attempt number for one directory."""
        attempt = self.latest_attempts.get(sample_dir_key, 0) + 1
        self.latest_attempts[sample_dir_key] = attempt
        return attempt

    def emit(
        self,
        event_type: str,
        sample_dir: str | None = None,
        attempt: int | None = None,
        **extra_fields,
    ) -> None:
        """Append one JSONL event and refresh the derived summary."""
        event = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
        }
        if sample_dir is not None:
            event["sample_dir"] = sample_dir
        if attempt is not None:
            event["attempt"] = attempt
        for key, value in extra_fields.items():
            if value is not None:
                event[key] = value

        with self.events_path.open("a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(event, sort_keys=True) + "\n")

        self._apply_event(event)
        self._write_summary()

    def _apply_event(self, event: dict) -> None:
        event_type = event.get("event_type", "")
        sample_dir = event.get("sample_dir")
        attempt = event.get("attempt")

        if event_type == "run_completed":
            self.run_status = "completed"
        elif event_type == "run_aborted":
            self.run_status = "aborted"

        directory_state = _directory_state_for_event(event_type)
        if directory_state is not None and sample_dir:
            self.latest_directory_states[sample_dir] = directory_state
            if isinstance(attempt, int):
                self.latest_attempts[sample_dir] = max(attempt, self.latest_attempts.get(sample_dir, 0))

    def _write_summary(self) -> None:
        counts: dict[str, int] = {}
        for state in self.latest_directory_states.values():
            counts[state] = counts.get(state, 0) + 1

        summary = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "root_dir": str(self.root_dir),
            "parallel_mode": self.parallel_mode,
            "process_only": self.process_only,
            "always_process": self.always_process,
            "max_concurrent": self.max_concurrent,
            "retry_failed": self.retry_failed,
            "resume_from_run_id": self.resume_from_run_id,
            "run_status": self.run_status,
            "directory_states": self.latest_directory_states,
            "attempts": self.latest_attempts,
            "counts": counts,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with self.summary_path.open("w", encoding="utf-8") as outfile:
            json.dump(summary, outfile, indent=2, sort_keys=True)
