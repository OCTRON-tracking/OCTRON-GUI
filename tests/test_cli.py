"""CLI smoke tests — verifies every subcommand and flag via --help.

No heavy dependencies (torch, cv2, ultralytics) are imported; all tests run
quickly in CI with no GPU and no model files required.

Covered subcommands and flags
------------------------------
gui         --help
gpu-test    --help; gpu-test runs (skipped if torch DLLs unavailable)
split       --help: --mode, --train, --val, --seed, --dry-run
train       --help: --model, --mode, --device, --epochs, --imagesz,
                    --save-period, --overwrite, --resume, --no-split,
                    --train, --val, --seed
predict     --help: --model, --tracker, --tracker-config, --device,
                    --conf-thresh, --iou-thresh, --skip-frames,
                    --one-object-per-label, --opening-radius, --overwrite,
                    --detailed, --buffer-size, --output-dir, --local-cache-dir
dump-tracker-config  TRACKER [-o PATH]: print/write a tracker's default config
render      --help: --video, --output, --preset, --start, --end, --alpha,
                    --masks/--no-masks, --boxes/--no-boxes,
                    --labels/--no-labels, --tracklets,
                    --tracklet-size, --tracklet-smooth-sigma,
                    --tracklet-interpolate, --track-ids, --min-observations,
                    --min-confidence, --bbox-sizes
transcode   --help: --output, --crf, --fps, --no-audio, --overwrite
gif         --help (GUI launcher; body not invoked by --help)
download-yolo / download-sam2 / download-sam3   --help: --force
            (download bodies are not invoked by --help)

auto_device returns 'cuda', 'mps', or 'cpu' (skipped if torch unavailable)
"""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from octron.cli import app

# Render help at a wide, plain (un-styled) width so option-name substring checks
# are stable across environments:
#  - COLUMNS=200 keeps Rich from wrapping long flags (e.g. --tracklet-smooth-sigma).
#    NOTE: do NOT set TERM=dumb — Rich then forces an 80-col "dumb terminal" and
#    ignores COLUMNS, which wraps long flags onto two lines.
#  - NO_COLOR + clearing FORCE_COLOR stop CI from forcing ANSI styling (which
#    otherwise splits substrings like '--mode'). _plain() below strips any that remain.
runner = CliRunner(
    env={"COLUMNS": "200", "NO_COLOR": "1", "FORCE_COLOR": None}
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text):
    """Strip ANSI SGR escape codes so help substring checks are robust.

    Rich/typer may style help text (e.g. bold option names) depending on the
    environment; that splits substrings like '--mode' across escape sequences.
    Stripping makes the assertions environment-independent.
    """
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Root help
# ---------------------------------------------------------------------------


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "OCTRON" in result.output


def test_root_no_args_shows_help():
    # Bare `octron` must behave like `octron --help` (no GUI launch).
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output
    assert "gui" in result.output  # the GUI is now an explicit subcommand


# ---------------------------------------------------------------------------
# gui
# ---------------------------------------------------------------------------


def test_gui_help():
    result = runner.invoke(app, ["gui", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# gpu-test
# ---------------------------------------------------------------------------


def test_gpu_test_help():
    result = runner.invoke(app, ["gpu-test", "--help"])
    assert result.exit_code == 0


def test_gpu_test_runs():
    try:
        import torch  # noqa: F401
    except OSError:
        pytest.skip("torch DLLs could not be loaded in this environment")
    result = runner.invoke(app, ["gpu-test"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_help():
    result = runner.invoke(app, ["split", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--mode" in out
    assert "--train" in out
    assert "--val" in out
    assert "--seed" in out
    assert "--dry-run" in out


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--model" in out
    assert "--mode" in out
    assert "--device" in out
    assert "--epochs" in out
    assert "--imagesz" in out
    assert "--save-period" in out
    assert "--overwrite" in out
    assert "--resume" in out
    assert "--no-split" in out
    assert "--train" in out
    assert "--val" in out
    assert "--seed" in out


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_help():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--model" in out
    assert "--tracker" in out
    assert "--tracker-config" in out
    assert "--device" in out
    assert "--conf-thresh" in out
    assert "--iou-thresh" in out
    assert "--skip-frames" in out
    assert "--one-object-per-label" in out
    assert "--opening-radius" in out
    assert "--overwrite" in out
    assert "--detailed" in out
    assert "--buffer-size" in out
    assert "--output-dir" in out
    assert "--local-cache-dir" in out


def test_detailed_help_lists_region_property_names():
    from octron.cli import _DETAILED_HELP, _REGION_PROPERTY_NAMES

    assert _REGION_PROPERTY_NAMES  # non-empty catalog
    for name in _REGION_PROPERTY_NAMES:
        assert name in _DETAILED_HELP


def test_parse_region_properties_valid():
    from octron.cli import _REGION_PROPERTY_NAMES, _parse_region_properties

    assert _parse_region_properties(None) is None
    assert _parse_region_properties("") is None
    assert _parse_region_properties("all") == _REGION_PROPERTY_NAMES
    assert _parse_region_properties("area, eccentricity") == (
        "area",
        "eccentricity",
    )
    assert _parse_region_properties("area,area") == ("area",)  # de-duplicated


def test_parse_region_properties_rejects_unknown():
    import typer

    from octron.cli import _parse_region_properties

    with pytest.raises(typer.BadParameter):
        _parse_region_properties("not_a_real_prop")


def test_predict_detailed_unknown_property_errors():
    # --detailed is validated before any heavy import or prediction.
    result = runner.invoke(
        app, ["predict", "v.mp4", "--model", "m.pt", "--detailed", "bogus"]
    )
    assert result.exit_code != 0
    assert "Unknown region property" in result.output


# ---------------------------------------------------------------------------
# dump-tracker-config
# ---------------------------------------------------------------------------


def test_dump_tracker_config_help():
    result = runner.invoke(app, ["dump-tracker-config", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)


def test_dump_tracker_config_to_file(tmp_path):
    out = tmp_path / "tracker.yaml"
    result = runner.invoke(
        app, ["dump-tracker-config", "botsort", "-o", str(out)]
    )
    assert result.exit_code == 0
    assert (
        "current_value" in out.read_text()
    )  # bundled default config, copied verbatim


def test_dump_tracker_config_to_stdout():
    result = runner.invoke(app, ["dump-tracker-config", "bytetrack"])
    assert result.exit_code == 0
    assert "current_value" in result.output


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_help():
    result = runner.invoke(app, ["render", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--video" in out
    assert "--output" in out
    assert "--preset" in out
    assert "--start" in out
    assert "--end" in out
    assert "--alpha" in out
    assert "--masks" in out
    assert "--no-masks" in out
    assert "--boxes" in out
    assert "--no-boxes" in out
    assert "--labels" in out
    assert "--no-labels" in out
    assert "--tracklets" in out
    assert "--tracklet-size" in out
    assert "--tracklet-smooth-sigma" in out
    assert "--tracklet-interpolate" in out
    assert "--track-ids" in out
    assert "--min-observations" in out
    assert "--min-confidence" in out
    assert "--skip-empty" in out
    assert "--bbox-sizes" in out
    assert "--debug" in out


# ---------------------------------------------------------------------------
# transcode
# ---------------------------------------------------------------------------


def test_transcode_help():
    result = runner.invoke(app, ["transcode", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--output" in out
    assert "--crf" in out
    assert "--fps" in out
    assert "--no-audio" in out
    assert "--overwrite" in out


# ---------------------------------------------------------------------------
# gif
# ---------------------------------------------------------------------------


def test_gif_help():
    # --help must not import/launch the Qt GUI (body lazy-imports mp4_to_gif).
    result = runner.invoke(app, ["gif", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# download-yolo / download-sam2 / download-sam3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd", ["download-yolo", "download-sam2", "download-sam3"]
)
def test_download_help(cmd):
    # --help must not trigger any download (bodies lazy-import the check_* fns).
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    assert "--force" in _plain(result.output)


# ---------------------------------------------------------------------------
# auto_device
# ---------------------------------------------------------------------------


def test_auto_device_returns_valid():
    try:
        import torch  # noqa: F401
    except OSError:
        pytest.skip("torch DLLs could not be loaded in this environment")
    from octron.test_gpu import auto_device

    device = auto_device()
    assert device in ("cuda", "mps", "cpu")


# ---------------------------------------------------------------------------
# Dynamic enum building (cli._sanitize_identifier / _enum_from_yaml)
# ---------------------------------------------------------------------------


def test_sanitize_identifier_handles_invalid_names():
    from octron.cli import _sanitize_identifier

    assert _sanitize_identifier("bot-sort") == "bot_sort"
    assert _sanitize_identifier("bot.sort") == "bot_sort"
    # Leading digit / spaces are not valid identifiers on their own.
    assert _sanitize_identifier("3d sort").isidentifier()
    assert _sanitize_identifier("3d sort").startswith("_")
    assert _sanitize_identifier("") == "_"


def test_enum_from_yaml_missing_file_uses_fallback():
    from octron.cli import _enum_from_yaml

    enum = _enum_from_yaml(
        "Tmp", Path("/no/such/catalog.yaml"), fallback="bytetrack"
    )
    assert enum("bytetrack").value == "bytetrack"


def test_enum_from_yaml_sanitizes_member_names(tmp_path):
    from octron.cli import _enum_from_yaml

    cat = tmp_path / "cat.yaml"
    cat.write_text(
        "3d-sort:\n  available: true\nbyte.track:\n  available: true\n"
    )
    enum = _enum_from_yaml(
        "Tmp2", cat, available_only=True, fallback="bytetrack"
    )
    # Values preserve the lower-cased catalog keys; members are valid identifiers.
    values = {m.value for m in enum}
    assert "3d-sort" in values
    assert "byte.track" in values
    assert all(m.name.isidentifier() for m in enum)
