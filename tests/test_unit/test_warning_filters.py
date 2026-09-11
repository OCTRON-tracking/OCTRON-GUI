"""Warning-filter tests for dependency deprecations.

These are intentionally lightweight: they exercise the warnings machinery
directly instead of importing/launching napari/Qt, which can be fragile in CI.
"""

import os
import subprocess
import sys
import warnings

from pydantic import PydanticDeprecatedSince20

import octron


def test_pydantic_deprecation_warnings_are_suppressed():
    """Known PydanticDeprecationWarning variants should not reach users.

    napari/npe2/psygnal trigger several distinct deprecated-pydantic-API
    warnings (``json_encoders``, extra ``Field`` kwargs, ``allow_mutation``,
    etc.), each with different message text but all raised as
    ``PydanticDeprecatedSince20`` (a ``PydanticDeprecationWarning``
    subclass). The filter suppresses the whole category rather than
    matching individual message strings, so exercise more than one
    message here to guard against regressing to a message-specific filter.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Re-install after simplefilter("always") so the OCTRON-specific ignore
        # filter has precedence inside this isolated warnings context.
        octron._install_warning_filters()
        warnings.warn_explicit(
            "`json_encoders` is deprecated. See "
            "https://docs.pydantic.dev/2.12/concepts/serialization/"
            "#custom-serializers for alternatives.",
            PydanticDeprecatedSince20,
            filename="pydantic/_internal/_generate_schema.py",
            lineno=319,
            module="pydantic._internal._generate_schema",
        )
        warnings.warn_explicit(
            "Using extra keyword arguments on `Field` is deprecated and "
            "will be removed. Use `json_schema_extra` instead. "
            "(Extra keys: 'min_ver').",
            PydanticDeprecatedSince20,
            filename="npe2/manifest/_package_metadata.py",
            lineno=45,
            module="npe2.manifest._package_metadata",
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


def test_import_analysis_results_is_quiet_in_fresh_process():
    """`from octron import AnalysisResults` must not emit dep warnings."""
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "default"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from octron import AnalysisResults; "
            "print(AnalysisResults.__name__)",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip() == "AnalysisResults"
    assert "json_encoders" not in result.stderr
    assert result.stderr == ""
