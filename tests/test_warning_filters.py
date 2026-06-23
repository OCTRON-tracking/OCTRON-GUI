"""Warning-filter tests for dependency deprecations.

These are intentionally lightweight: they exercise the warnings machinery
directly instead of importing/launching napari/Qt, which can be fragile in CI.
"""

import warnings

import octron


def test_pydantic_json_encoders_warning_is_suppressed():
    """The known dependency warning should not be shown to OCTRON users."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Re-install after simplefilter("always") so the OCTRON-specific ignore
        # filter has precedence inside this isolated warnings context.
        octron._install_warning_filters()
        warnings.warn_explicit(
            "`json_encoders` is deprecated. See https://docs.pydantic.dev/2.12/concepts/serialization/#custom-serializers for alternatives.",
            DeprecationWarning,
            filename="pydantic/_internal/_generate_schema.py",
            lineno=319,
            module="pydantic._internal._generate_schema",
        )
    assert caught == []


def test_warning_filter_does_not_hide_other_deprecations():
    """Keep unrelated deprecations visible."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        octron._install_warning_filters()
        warnings.warn_explicit(
            "some other dependency deprecation",
            DeprecationWarning,
            filename="pydantic/_internal/_generate_schema.py",
            lineno=320,
            module="pydantic._internal._generate_schema",
        )
    assert len(caught) == 1
    assert "some other dependency deprecation" in str(caught[0].message)
