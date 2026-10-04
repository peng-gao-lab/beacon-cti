from __future__ import annotations

from pathlib import Path


# Path validation for Alignmentor inputs and outputs. Any user-specified
# output location is accepted; only existence and write errors are reported.


class PathValidationError(ValueError):
    """Raised when a path cannot be resolved or created."""


def _resolve_path(path_like: str | Path, *, must_exist: bool) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    else:
        path = path.resolve(strict=False)
    if must_exist:
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PathValidationError(f"Path does not exist: {path}") from exc

    return path


def resolve_existing_path(path_like: str | Path) -> Path:
    """Resolve an existing input path."""
    return _resolve_path(path_like, must_exist=True)


def ensure_within_project(path_like: str | Path, *, must_exist: bool) -> Path:
    """Resolve a path. The project-root restriction of the original tool is removed."""
    return _resolve_path(path_like, must_exist=must_exist)


def ensure_output_dir(path_like: str | Path) -> Path:
    path = ensure_within_project(path_like, must_exist=False)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PathValidationError(f"Failed to create output dir: {path}") from exc
    return path
