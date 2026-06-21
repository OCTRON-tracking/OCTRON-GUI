"""
OCTRON user configuration.

A single, GUI- and CLI-friendly home for *user-tunable runtime settings*, kept
deliberately separate from ``octron/yolo_octron/constants.py``:

- ``constants.py``  : developer-owned, code-coupled catalogs and theme that are
                      NOT meant to be hand-edited (e.g. ``ALL_REGION_PROPERTIES``,
                      the skimage capability catalog, and ``TASK_COLORS``, the UI
                      theme).  Their job is to *define and validate* choices.
- ``config.yaml``   : user-tunable runtime settings (this module), e.g. the
                      prediction cache directory.  Their job is to *remember a
                      user's choice* across runs.

The two never compete: a catalog in ``constants.py`` (e.g. the list of all
region properties) is used to validate the *value* a user stores in
``config.yaml`` (e.g. which region properties they selected) — the catalog is
the menu, the config records the order.

Settings are described by a schema (``SETTINGS`` below): one ``SettingSpec`` per
key with its default, a human-readable description, and a GUI/CLI ``kind`` hint.
The schema drives loading, validation, an ``octron config`` CLI, and a future
settings dialog, which can introspect ``specs()`` and render itself generically.

Precedence: in-code default  <  ``config.yaml``  <  per-invocation override
(the per-invocation override — e.g. a CLI flag — is applied by the caller).

Location: the platform's per-user config directory (via
``platformdirs.user_config_dir``) — e.g. ``~/Library/Application Support/octron``
(macOS), ``~/.config/octron`` (Linux, honoring ``$XDG_CONFIG_HOME``), or
``%LOCALAPPDATA%\\octron`` (Windows).  Set the ``OCTRON_CONFIG_PATH`` environment
variable to override the location (used by tests and power users).

Example ``config.yaml``::

    # OCTRON user configuration
    prediction_cache_dir: /scratch/nvme/octron
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import platformdirs
import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SettingSpec:
    """One user setting.

    Attributes
    ----------
    key : str
        Name used in ``config.yaml`` and by ``get_value`` / ``set_value``.
    default : Any
        Value used when the key is absent from the file or fails validation.
    description : str
        Human-readable text for the settings dialog and ``octron config``.
    kind : str
        Hint for how to edit the value in a GUI/CLI: ``"text"``, ``"dir"``,
        ``"path"``, ``"bool"``, ``"int"``, ``"float"``, or ``"choice"``
        (with ``choices`` populated).
    choices : tuple
        Allowed values when ``kind == "choice"``.
    coerce : callable, optional
        ``(raw_value) -> stored_value``; raises ``ValueError`` on invalid input.
        Applied on both load and set so the stored representation is normalized.
    """
    key: str
    default: Any
    description: str
    kind: str = "text"
    choices: tuple = ()
    coerce: Optional[Callable[[Any], Any]] = None


def _coerce_optional_dir(value):
    """Normalize a directory setting: ``None``/``""`` -> ``None``, else a string."""
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError("expected a path string or null")
    text = str(value).strip()
    return text or None


# The settings schema.  Add new user settings here; everything else (loading,
# validation, the CLI, and a settings dialog) picks them up automatically.
SETTINGS: "tuple[SettingSpec, ...]" = (
    SettingSpec(
        key="prediction_cache_dir",
        default=None,
        kind="dir",
        description=(
            "Local directory used to stage prediction output before moving it to "
            "the final destination (e.g. fast NVMe scratch). Leave empty to write "
            "directly to the destination (caching off)."
        ),
        coerce=_coerce_optional_dir,
    ),
    SettingSpec(
        key="model_cache_dir",
        default=None,
        kind="dir",
        description=(
            "Directory where downloaded model weights and SAM checkpoints are "
            "stored. Leave empty to use the per-user cache directory "
            "(platformdirs.user_cache_dir('octron')). Point this at shared/NAS "
            "storage to reuse downloads across machines or installs."
        ),
        coerce=_coerce_optional_dir,
    ),
)

_SPECS = {spec.key: spec for spec in SETTINGS}


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

def config_path() -> Path:
    """Return the path to ``config.yaml``.

    Honors the ``OCTRON_CONFIG_PATH`` environment variable; otherwise uses the
    platform's per-user config directory via ``platformdirs.user_config_dir``
    (e.g. ``~/Library/Application Support/octron`` on macOS, ``~/.config/octron``
    on Linux, ``%LOCALAPPDATA%/octron`` on Windows).
    """
    override = os.environ.get("OCTRON_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_config_dir("octron")) / "config.yaml"


# ---------------------------------------------------------------------------
# Load / read
# ---------------------------------------------------------------------------

def _defaults() -> dict:
    return {spec.key: spec.default for spec in SETTINGS}


def _read_raw() -> dict:
    """Read the raw YAML mapping from disk (``{}`` when missing/empty/invalid)."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not parse OCTRON config at {path}: {e}. Ignoring it.")
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        logger.warning(f"OCTRON config at {path} is not a mapping; ignoring it.")
        return {}
    return data


def load() -> dict:
    """Return the effective config: in-code defaults overlaid by ``config.yaml``.

    Unknown keys are ignored with a warning; values that fail validation fall
    back to their default with a warning, so a malformed file never breaks OCTRON.
    """
    cfg = _defaults()
    for key, value in _read_raw().items():
        spec = _SPECS.get(key)
        if spec is None:
            logger.warning(f"Ignoring unknown OCTRON config key {key!r} in {config_path()}.")
            continue
        try:
            cfg[key] = spec.coerce(value) if spec.coerce else value
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid value for {key!r} in {config_path()} ({value!r}): {e}. "
                f"Using default {spec.default!r}."
            )
    return cfg


def get_value(key: str):
    """Return the effective value for ``key`` (default overlaid by ``config.yaml``)."""
    if key not in _SPECS:
        raise KeyError(f"Unknown OCTRON config key: {key!r}")
    return load()[key]


def specs() -> "tuple[SettingSpec, ...]":
    """Return the settings schema (for ``octron config`` and a settings dialog)."""
    return SETTINGS


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def set_value(key: str, value) -> Path:
    """Validate ``value`` and persist it to ``config.yaml``.

    Keys already present in the file are preserved.  Returns the written path.
    """
    spec = _SPECS.get(key)
    if spec is None:
        raise KeyError(f"Unknown OCTRON config key: {key!r}")
    coerced = spec.coerce(value) if spec.coerce else value

    data = _read_raw()
    data[key] = coerced

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# OCTRON user configuration\n")
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# Typed accessors (convenience wrappers used by the rest of OCTRON)
# ---------------------------------------------------------------------------

def get_prediction_cache_dir() -> Optional[Path]:
    """Return the configured prediction cache directory, or ``None`` if unset.

    ``~`` is expanded in the returned path.  ``None`` means caching is off.
    """
    raw = get_value("prediction_cache_dir")
    if not raw:
        return None
    return Path(raw).expanduser()


def get_model_cache_dir() -> Path:
    """Return the base directory for downloaded model weights / checkpoints.

    Resolution: the ``model_cache_dir`` setting (``~`` expanded) when set,
    otherwise the per-user cache directory ``platformdirs.user_cache_dir("octron")``
    (e.g. ``~/Library/Caches/octron`` on macOS). The directory is NOT created
    here; use :func:`get_yolo_models_dir` / :func:`get_sam_checkpoints_dir` for
    the concrete (created) subdirectories.
    """
    raw = get_value("model_cache_dir")
    if raw:
        return Path(raw).expanduser()
    return Path(platformdirs.user_cache_dir("octron"))


def get_yolo_models_dir() -> Path:
    """Return ``<model_cache_dir>/models`` (created if missing).

    Home for downloaded YOLO weight files. See :func:`get_model_cache_dir`.
    """
    d = get_model_cache_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_sam_checkpoints_dir() -> Path:
    """Return ``<model_cache_dir>/checkpoints`` (created if missing).

    Home for downloaded SAM2/SAM3 checkpoints (and the SAM3 ``config.json``).
    See :func:`get_model_cache_dir`.
    """
    d = get_model_cache_dir() / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d
