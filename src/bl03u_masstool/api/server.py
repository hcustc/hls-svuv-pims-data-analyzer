"""Backward-compatible API entrypoint.

New code should import the FastAPI application from
``bl03u_masstool.api.app``. This module is kept so existing deployment
commands using ``bl03u_masstool.api.server:app`` continue to work.
"""

from __future__ import annotations

from bl03u_masstool.api.app import *  # noqa: F401,F403
