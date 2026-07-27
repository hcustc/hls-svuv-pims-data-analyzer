from __future__ import annotations

from pathlib import Path

from bl03u_masstool.core.project_lifecycle import project_root
from bl03u_masstool.core.project_settings import ProjectSettings, ProjectSettingsManager


def record_project_artifact(
    widget,
    field_name: str,
    file_path: str | Path,
    *,
    message: str | None = None,
    project_settings: ProjectSettings | None = None,
) -> bool:
    """Persist an analysis-result path in project settings and refresh the main window if present."""
    if not file_path:
        return False

    manager = ProjectSettingsManager()
    settings = manager.get()
    if (
        project_settings is not None
        and project_root(settings).resolve() != project_root(project_settings).resolve()
    ):
        return False
    if not hasattr(settings, field_name):
        return False

    normalized_path = str(Path(file_path))
    already_registered = str(getattr(settings, field_name, "") or "") == normalized_path
    setattr(settings, field_name, normalized_path)
    manager.set(settings)
    manager.save()
    if already_registered:
        return True

    window = widget.window() if hasattr(widget, "window") else None
    if window is not None and window is not widget:
        if hasattr(window, "_read_project_settings_to_ui"):
            window._read_project_settings_to_ui(settings)
        if hasattr(window, "_sync_project_settings_to_tool_pages"):
            window._sync_project_settings_to_tool_pages(settings)
        if hasattr(window, "refresh_project_lifecycle"):
            window.refresh_project_lifecycle(settings)
        if hasattr(window, "refresh_project_parameter_summary"):
            window.refresh_project_parameter_summary()
        statusbar = getattr(window, "statusbar", None)
        if message and statusbar is not None:
            statusbar.showMessage(message, 4000)

    return True
