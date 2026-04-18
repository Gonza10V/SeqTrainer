"""Centralized deprecation utilities and migration timeline."""

from __future__ import annotations

import warnings

REMOVAL_DATE = "2026-12-31"
REMOVAL_VERSION = "0.4.0"


def warn_deprecated(*, old: str, new: str, kind: str = "module", stacklevel: int = 2) -> None:
    """Emit a standardized deprecation warning with removal timeline."""
    warnings.warn(
        (
            f"{kind.title()} '{old}' is deprecated and will be removed no earlier than "
            f"{REMOVAL_DATE} (target release {REMOVAL_VERSION}). "
            f"Use '{new}' instead."
        ),
        DeprecationWarning,
        stacklevel=stacklevel,
    )
