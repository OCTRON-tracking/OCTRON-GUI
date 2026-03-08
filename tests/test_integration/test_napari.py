from types import SimpleNamespace

import pytest

from octron.main import octron_widget as OctronWidget

# Mock dictionaries returned by check_*_models functions,
# to avoid actual HF downloads during testing.
MOCK_SAM2 = {
    "sam2_base_plus": {"name": "SAM2 Base Plus"},
    "sam2_large": {"name": "SAM2 Large"},
    "sam2_large_hq": {"name": "SAM2 Large HQ"},
}
MOCK_SAM3 = {
    "sam3_mode_a": {
        "name": "SAM3",
        "checkpoint_path": "checkpoints/sam3.pt",
        "semantic": False,
    },
    "sam3_mode_b": {
        "name": "SAM3 multi",
        "checkpoint_path": "checkpoints/sam3.pt",
        "semantic": True,
    },
}
MOCK_COTRACKER = {
    "cotracker": {
        "name": "Cotracker3",
        "checkpoint_path": "checkpoints/scaled_online.pth",
    },
}


@pytest.fixture
def octron_widget(make_napari_viewer, monkeypatch):
    """Instantiate octron_widget with all model-check functions mocked."""
    # Mock functions that check models exist locallty
    monkeypatch.setattr(
        "octron.main.check_sam2_models",
        lambda *a, **kw: MOCK_SAM2,
    )
    monkeypatch.setattr(
        "octron.main.check_sam3_models",
        lambda *a, **kw: MOCK_SAM3,
    )
    monkeypatch.setattr(
        "octron.main.check_cotracker_models",
        lambda *a, **kw: MOCK_COTRACKER,
    )

    # Mock function that loads boxmot trackers
    monkeypatch.setattr(
        "octron.main.load_boxmot_trackers",
        lambda *a, **kw: {},
    )

    # Mock YOLO_octron object with .models_dict attribute
    monkeypatch.setattr(
        "octron.main.YOLO_octron",
        lambda *a, **kw: SimpleNamespace(models_dict={}),
    )

    # Return napari viewer with octron_widget loaded
    viewer = make_napari_viewer()
    return OctronWidget(viewer)


def test_widget_loads(octron_widget):
    """Test that the octron_widget fixture loads without error"""
    assert octron_widget is not None


def test_prediction_model_list(octron_widget):
    """Test prediction model dropdown contains expected items."""

    list_models = octron_widget.prediction_model_list
    list_models_names = [list_models.itemText(i) for i in range(list_models.count())]

    expected_models = [
        "SAM2 Base Plus",
        "SAM2 Large",
        "SAM2 Large HQ",
        "SAM3",
        "SAM3 multi",
        "Cotracker3",
    ]
    assert all(name in list_models_names for name in expected_models), (
        f"Not all expected models found in dropdown. "
        f"Expected: {expected_models}, got: {list_models_names}"
    )       