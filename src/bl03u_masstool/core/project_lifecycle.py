from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import os
from pathlib import Path
import re
import shutil
import zipfile

import yaml

from .config import writable_project_path
from .project_settings import ProjectSettings, portable_project_settings_dict

DATA_SOURCE_FILE_COUNT_LIMIT = 5000


class WorkflowProfile(Enum):
    """定义可执行的分析工作流"""
    SPECTRUM_ONLY = "spectrum_only"
    TEMPERATURE_SCAN = "temperature_scan"
    PIE_ANALYSIS = "pie_analysis"
    FULL_ANALYSIS = "full_analysis"


class DependencyOperator(Enum):
    """依赖规则操作符"""
    ALL_OF = "all_of"  # 所有源都必需
    ANY_OF = "any_of"  # 至少一个源必需


@dataclass(frozen=True)
class DependencyRule:
    """单个依赖规则"""
    operator: DependencyOperator
    sources: tuple[str, ...]  # 数据源键名


@dataclass(frozen=True)
class WorkflowRequirement:
    """工作流依赖定义"""
    profile: WorkflowProfile
    required_sources: tuple[DependencyRule, ...]  # 多个规则用AND结合


# 工作流依赖定义表
WORKFLOW_REQUIREMENTS = {
    WorkflowProfile.SPECTRUM_ONLY: WorkflowRequirement(
        profile=WorkflowProfile.SPECTRUM_ONLY,
        required_sources=(
            DependencyRule(DependencyOperator.ANY_OF, ("single_spectrum", "sum_spectrum")),
        ),
    ),
    WorkflowProfile.TEMPERATURE_SCAN: WorkflowRequirement(
        profile=WorkflowProfile.TEMPERATURE_SCAN,
        required_sources=(
            DependencyRule(DependencyOperator.ALL_OF, ("temperature_scan",)),
        ),
    ),
    WorkflowProfile.PIE_ANALYSIS: WorkflowRequirement(
        profile=WorkflowProfile.PIE_ANALYSIS,
        required_sources=(
            DependencyRule(DependencyOperator.ALL_OF, ("pie_scan",)),
        ),
    ),
    WorkflowProfile.FULL_ANALYSIS: WorkflowRequirement(
        profile=WorkflowProfile.FULL_ANALYSIS,
        required_sources=(
            DependencyRule(DependencyOperator.ANY_OF, ("temperature_scan", "pie_scan")),
        ),
    ),
}


@dataclass(frozen=True)
class WorkflowCapability:
    """工作流可执行性评估结果"""
    profile: WorkflowProfile
    can_execute: bool
    missing_sources: tuple[str, ...]  # 缺少的数据源
    detail: str = ""


@dataclass
class WorkflowAnalysisResult:
    """工作流分析的完整结果"""
    available_workflows: list[WorkflowProfile] = field(default_factory=list)  # 可执行的工作流
    unavailable_workflows: dict[WorkflowProfile, str] = field(default_factory=dict)  # 不可执行及原因
    data_source_status: dict[str, bool] = field(default_factory=dict)  # 每个源的有效性
    recommended_next_step: str = ""  # 推荐的下一步操作


class ProjectUIState(Enum):
    """项目管理UI主操作按钮的状态枚举"""
    UNINITIALIZED = ("初始化项目", "项目信息未填或目录未初始化")
    AWAITING_DATA_IMPORT = ("配置项目数据源", "已初始化，等待导入原始数据")
    READY_TO_ANALYZE = ("前往质谱工作台", "数据已导入，至少一项分析可执行")
    ANALYSIS_IN_PROGRESS = ("查看分析进度", "分析进行中")
    ANALYSIS_COMPLETE = ("查看分析结果", "分析已完成")

    def __init__(self, button_text: str, description: str):
        self.button_text = button_text
        self.description = description


class DataSourceValidationStatus(Enum):
    """数据源导入状态枚举"""
    UNCONFIGURED = ("未配置", "还未指定数据源路径")
    PARTIAL = ("部分完成", "已导入部分数据源")
    COMPLETE = ("数据已就绪", "所有必需数据源均已导入")
    INVALID = ("路径失效", "已配置的路径不存在或不可读")

    def __init__(self, label: str, description: str):
        self.label = label
        self.description = description


@dataclass(frozen=True)
class DataSourceValidationRecord:
    """单个数据源的验证结果"""
    source_key: str
    source_label: str
    path: str | None
    exists: bool
    is_readable: bool
    file_count: int
    detail: str
    last_modified: str = ""

    @property
    def is_valid(self) -> bool:
        """路径存在且可读"""
        return bool(self.path) and self.exists and self.is_readable


@dataclass(frozen=True)
class ProjectDirectorySpec:
    key: str
    label: str
    relative_path: str
    purpose: str


@dataclass(frozen=True)
class ProjectSourceSpec:
    key: str
    label: str
    field_name: str
    directory_key: str
    subdirectory: str


@dataclass(frozen=True)
class ProjectStageSpec:
    key: str
    label: str
    directory_key: str
    required_fields: tuple[str, ...]
    nav_page: str
    next_action: str


@dataclass(frozen=True)
class ProjectStageStatus:
    key: str
    label: str
    completed: bool
    warning: bool
    detail: str
    nav_page: str
    next_action: str


@dataclass(frozen=True)
class ProjectImportResult:
    source: Path
    destination: Path
    field_name: str
    label: str
    mode: str = "link"


@dataclass(frozen=True)
class ProjectFileRecord:
    path: Path
    section_key: str
    section_label: str
    relative_path: str
    size_bytes: int
    modified_at: str
    registered: bool = False
    missing: bool = False


def _count_files_limited(path: Path, limit: int = DATA_SOURCE_FILE_COUNT_LIMIT) -> tuple[int, bool]:
    """Count files without walking an arbitrarily large data tree.

    Project data sources can point at raw beamline folders with thousands of
    files. Validation is used from the UI, so it should confirm readability
    quickly instead of blocking while recursively counting every file.
    """
    count = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            if count >= limit:
                return count, True
    return count, False


class ArtifactStatus(Enum):
    """产物有效性状态枚举"""
    VALID = "有效"
    MISSING = "缺失"
    EXPIRED = "已过期"
    INCOMPLETE = "不完整"


class ArtifactCategory(Enum):
    """产物分类枚举"""
    INTERMEDIATE = ("intermediate", "中间结果", "spectrum_analysis")  # 寻峰、卡峰范围、高斯拟合
    TEMPERATURE = ("temperature_scan", "温度扫描", "temperature_scan")
    PIE = ("pie", "PIE拟合", "pie_analysis")
    MOLE_FRACTION = ("mole_fraction", "摩尔分数", "mole_fraction")
    PICS = ("pics", "PICS截面数据库", "pics")
    SNAPSHOTS = ("snapshots", "版本快照", "versions")

    def __init__(self, key: str, label: str, directory_key: str):
        self.key = key
        self.label = label
        self.directory_key = directory_key


@dataclass(frozen=True)
class ArtifactRecord:
    """单个产物的元数据记录"""
    artifact_type: str  # 产物类型（如 temperature_scan_result, pie_identification_result）
    category: str  # 分类 key (对应 ArtifactCategory)
    path: str  # 产物绝对路径
    generation_time: str  # ISO 8601 格式生成时间
    size_bytes: int  # 文件大小（bytes）
    source_module: str  # 生成模块（如 TemperatureModule, PIEModule）
    status: str  # 有效性状态（valid/missing/expired/incomplete）
    detail: str = ""  # 状态详情说明


PROJECT_DIRECTORIES: tuple[ProjectDirectorySpec, ...] = (
    ProjectDirectorySpec("config", "项目配置", "config", "项目级设置（YAML）"),
    ProjectDirectorySpec("raw_data", "原始输入", "raw_data", "原始谱图、样品信息和导入记录"),
    ProjectDirectorySpec("calibration", "标定", "analysis/calibration", "定标点、定标参数和标定结果"),
    ProjectDirectorySpec("spectrum_analysis", "谱图分析", "analysis/spectrum", "寻峰、卡峰范围和高斯拟合产物"),
    ProjectDirectorySpec("temperature_scan", "温度扫描", "analysis/temperature_scan", "温度扫描曲线、表格和图像"),
    ProjectDirectorySpec("pie_analysis", "PIE拟合", "analysis/pie", "PIE 曲线、物种拟合和鉴定结果"),
    ProjectDirectorySpec("mole_fraction", "摩尔分数", "analysis/mole_fraction", "摩尔分数计算输入、结果和图像"),
    ProjectDirectorySpec("pics", "PICS计算", "analysis/pics", "PICS 计算结果和导入记录"),
    ProjectDirectorySpec("versions", "版本快照", "versions", "项目级 zip 快照和备份"),
)

PROJECT_SOURCE_SPECS: dict[str, ProjectSourceSpec] = {
    "single_spectrum": ProjectSourceSpec(
        "single_spectrum", "单谱文件", "single_spectrum_file", "raw_data", "single_spectrum"
    ),
    "sum_spectrum": ProjectSourceSpec(
        "sum_spectrum", "累计谱目录", "sum_spectrum_folder", "raw_data", "sum_spectrum"
    ),
    "temperature_scan": ProjectSourceSpec(
        "temperature_scan", "温度扫描目录", "temperature_scan_folder", "raw_data", "temperature_scan"
    ),
    "pie_scan": ProjectSourceSpec("pie_scan", "PIE扫描目录", "pie_scan_folder", "raw_data", "pie_scan"),
    "manual_peak": ProjectSourceSpec(
        "manual_peak", "手动卡峰文件", "manual_peak_file", "spectrum_analysis", "manual_peaks"
    ),
    "kr_calibration": ProjectSourceSpec(
        "kr_calibration", "Kr定标扫描目录", "kr_calibration_folder", "raw_data", "kr_calibration"
    ),
    "kr_calibration_peak": ProjectSourceSpec(
        "kr_calibration_peak", "Kr定标卡峰文件", "kr_calibration_peak_file", "spectrum_analysis", "kr_manual_peaks"
    ),
}

PROJECT_STAGES: tuple[ProjectStageSpec, ...] = (
    ProjectStageSpec(
        "project_setup",
        "项目初始化",
        "raw_data",
        ("project_name", "system"),
        "project",
        "填写项目名/实验体系并初始化目录",
    ),
    ProjectStageSpec(
        "raw_data",
        "数据导入",
        "raw_data",
        ("temperature_scan_folder", "pie_scan_folder"),
        "project",
        "导入温度扫描或 PIE 原始目录",
    ),
    ProjectStageSpec(
        "calibration",
        "标定",
        "calibration",
        ("calibration_points",),
        "spectrum",
        "在质谱工作台完成 TOF-m/z 标定",
    ),
    ProjectStageSpec(
        "spectrum_analysis",
        "寻峰/高斯拟合",
        "spectrum_analysis",
        ("manual_peak_file",),
        "spectrum",
        "在质谱工作台寻峰、修正卡峰并导出峰范围",
    ),
    ProjectStageSpec(
        "temperature_scan",
        "温度扫描",
        "temperature_scan",
        ("temperature_scan_result_file",),
        "temperature",
        "运行温度扫描并导出结果",
    ),
    ProjectStageSpec(
        "pie_analysis",
        "PIE拟合",
        "pie_analysis",
        ("pie_identification_result_file",),
        "pie",
        "运行 PIE 拟合并导出鉴定结果",
    ),
    ProjectStageSpec(
        "mole_fraction",
        "摩尔分数",
        "mole_fraction",
        ("mole_fraction_result_file",),
        "mole_fraction",
        "载入温度/PIE结果并计算摩尔分数",
    ),
)


def directory_spec(key: str) -> ProjectDirectorySpec:
    for spec in PROJECT_DIRECTORIES:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown project directory key: {key}")


def sanitize_project_slug(value: str, fallback: str = "project") -> str:
    cleaned = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", value.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def project_root(settings: ProjectSettings) -> Path:
    raw = settings.output_dir.strip() if settings.output_dir else "output"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = writable_project_path(path)
    return path


def project_directory(settings: ProjectSettings, key: str) -> Path:
    spec = directory_spec(key)
    return project_root(settings) / spec.relative_path


def ensure_project_structure(settings: ProjectSettings) -> dict[str, Path]:
    root = project_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    directories: dict[str, Path] = {}
    for spec in PROJECT_DIRECTORIES:
        path = root / spec.relative_path
        path.mkdir(parents=True, exist_ok=True)
        directories[spec.key] = path
    _write_project_readme(settings)
    _write_project_marker(settings)
    return directories


def _write_project_marker(settings: ProjectSettings) -> Path:
    """Write project initialization marker file to project root."""
    root = project_root(settings)
    marker_path = root / ".bl03u_project"
    if not marker_path.exists():
        marker_content = {
            "initialized_at": utc_now_iso(),
            "version": "0.2.0",
        }
        marker_path.write_text(yaml.dump(marker_content, allow_unicode=True, default_flow_style=False))
    return marker_path


def _write_project_readme(settings: ProjectSettings) -> Path:
    root = project_root(settings)
    path = root / "README.md"
    if path.exists():
        return path
    project_title = settings.project_name or settings.system or root.name
    lines = [
        f"# {project_title}",
        "",
        "This folder is managed by BL03U MassSpectrumTool.",
        "",
        "## Directory Map",
    ]
    for spec in PROJECT_DIRECTORIES:
        lines.append(f"- `{spec.relative_path}/`: {spec.purpose}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(2, 10_000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not choose a unique destination for {path}")


def import_project_source(
    settings: ProjectSettings,
    source_path: str | Path,
    source_key: str,
    *,
    mode: str = "link",
) -> ProjectImportResult:
    spec = PROJECT_SOURCE_SPECS[source_key]
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if mode not in {"link", "copy"}:
        raise ValueError("mode must be 'link' or 'copy'")

    ensure_project_structure(settings)
    root = project_root(settings).resolve()

    # Safety check: prevent copying/linking directory into itself or its subdirectories.
    try:
        # Check if destination would be inside source
        target_dir = project_directory(settings, spec.directory_key) / spec.subdirectory
        target_dir = target_dir.resolve()

        # If source is a directory, ensure target is not inside it
        if source.is_dir():
            try:
                target_dir.relative_to(source)
                # If we can compute relative_to without error, target IS inside source - FAIL
                raise ValueError(
                    f"Cannot import '{source.name}' to '{target_dir}': destination is inside source directory. "
                    f"This would create a recursive copy. Please select a different source or destination."
                )
            except ValueError as e:
                if "not in the same drive" not in str(e) and "does not start with" not in str(e):
                    # Re-raise our custom error
                    if "Cannot import" in str(e):
                        raise
                # If relative_to fails with a path error, it's safe (target NOT inside source)
                pass

        # Check if source is already inside the project directory
        if source != root:
            try:
                source.relative_to(root)
                # If source is inside project root, allow it (internal reorganization)
            except ValueError:
                # Source is outside project, which is fine
                pass

    except ValueError as e:
        if "Cannot import" in str(e):
            raise
        # Other relative_to errors are acceptable
        pass

    target_dir = project_directory(settings, spec.directory_key) / spec.subdirectory
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(target_dir / source.name)

    try:
        if mode == "link":
            destination.symlink_to(source, target_is_directory=source.is_dir())
        elif source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    except Exception as e:
        raise RuntimeError(f"Failed to import '{source.name}': {str(e)}")

    setattr(settings, spec.field_name, str(destination))
    if source_key == "pie_scan":
        settings.pie_scan_folders = [str(destination)]
    _write_raw_data_link_manifest(settings, source_key, source, destination, mode)
    return ProjectImportResult(
        source=source,
        destination=destination,
        field_name=spec.field_name,
        label=spec.label,
        mode=mode,
    )


def _write_raw_data_link_manifest(
    settings: ProjectSettings,
    source_key: str,
    source: Path,
    destination: Path,
    mode: str,
) -> Path:
    manifest_path = project_directory(settings, "raw_data") / "_sources.yaml"
    data: dict = {}
    if manifest_path.exists():
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    sources = data.setdefault("sources", {})
    entry = {
        "mode": mode,
        "source": str(source),
        "project_path": destination.relative_to(project_root(settings)).as_posix(),
        "updated_at": utc_now_iso(),
    }
    if source_key == "pie_scan" and settings.pie_multi_folder_mode:
        sources.setdefault(source_key, entry)
        segments = sources.setdefault("pie_scan_segments", [])
        if not isinstance(segments, list):
            segments = []
            sources["pie_scan_segments"] = segments
        segments[:] = [item for item in segments if item.get("source") != str(source)]
        segments.append(entry)
    else:
        sources[source_key] = entry
    manifest_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return manifest_path


def import_initial_project_data(
    settings: ProjectSettings,
    sources: dict[str, str | Path | None],
    *,
    mode: str = "link",
) -> list[ProjectImportResult]:
    results: list[ProjectImportResult] = []
    for source_key, source_path in sources.items():
        if not source_path:
            continue
        results.append(import_project_source(settings, source_path, source_key, mode=mode))
    return results


CORE_WORKFLOW_SOURCE_KEYS: tuple[str, ...] = (
    "temperature_scan",
    "pie_scan",
)

MANAGED_PROJECT_SOURCE_KEYS: tuple[str, ...] = (
    *CORE_WORKFLOW_SOURCE_KEYS,
    "kr_calibration",
    "kr_calibration_peak",
)


def materialize_project_data_sources(
    settings: ProjectSettings,
    *,
    source_keys: tuple[str, ...] = MANAGED_PROJECT_SOURCE_KEYS,
    mode: str = "copy",
) -> list[ProjectImportResult]:
    """Copy/link registered raw data sources into the managed project folder.

    This keeps experiment-specific directory structures intact. For example,
    a temperature-scan folder is copied as a whole under
    ``raw_data/temperature_scan/<folder name>`` instead of being normalized into
    a fixed child layout.
    """
    ensure_project_structure(settings)
    root = project_root(settings).resolve()
    results: list[ProjectImportResult] = []

    for source_key in source_keys:
        spec = PROJECT_SOURCE_SPECS[source_key]
        if source_key == "pie_scan":
            raw_values = settings.effective_pie_scan_folders()
            settings.pie_multi_folder_mode = len(raw_values) > 1
        else:
            raw_value = str(getattr(settings, spec.field_name, "") or "").strip()
            raw_values = [raw_value] if raw_value else []
        if not raw_values:
            continue

        materialized_values: list[str] = []
        for raw_value in raw_values:
            source = Path(raw_value).expanduser()
            if not source.exists():
                raise FileNotFoundError(f"{spec.label}不存在：{source}")

            try:
                source.resolve().relative_to(root)
            except ValueError:
                result = import_project_source(settings, source, source_key, mode=mode)
                results.append(result)
                materialized_values.append(str(result.destination))
            else:
                materialized_values.append(str(source))

        if source_key == "pie_scan":
            settings.pie_scan_folders = materialized_values
            settings.pie_scan_folder = materialized_values[0] if materialized_values else ""
            settings.pie_multi_folder_mode = len(materialized_values) > 1

    return results


def build_project_stage_statuses(settings: ProjectSettings) -> list[ProjectStageStatus]:
    statuses: list[ProjectStageStatus] = []
    for stage in PROJECT_STAGES:
        completed, warning, detail = _stage_completion(settings, stage)
        statuses.append(
            ProjectStageStatus(
                key=stage.key,
                label=stage.label,
                completed=completed,
                warning=warning,
                detail=detail,
                nav_page=stage.nav_page,
                next_action=stage.next_action,
            )
        )
    return statuses


def next_project_stage(settings: ProjectSettings) -> ProjectStageStatus | None:
    for status in build_project_stage_statuses(settings):
        if not status.completed:
            return status
    return None


def _stage_completion(settings: ProjectSettings, stage: ProjectStageSpec) -> tuple[bool, bool, str]:
    if stage.key == "project_setup":
        has_identity = bool(settings.project_name.strip() or settings.system.strip())
        root = project_root(settings)
        marker_exists = (root / ".bl03u_project").exists()
        standard_dirs_exist = all((root / spec.relative_path).exists() for spec in PROJECT_DIRECTORIES)

        if has_identity and marker_exists and standard_dirs_exist:
            return True, False, f"项目目录：{root}"
        if has_identity:
            return False, False, "项目信息已填写，尚未初始化目录结构"
        return False, False, "缺少项目名或实验体系"

    registered_fields = [field for field in stage.required_fields if _field_has_value(settings, field)]
    missing_paths = [field for field in registered_fields if _field_points_to_missing_path(settings, field)]
    directory_has_files = _directory_has_files(project_directory(settings, stage.directory_key))

    if stage.key == "calibration" and settings.calibration_points:
        return True, False, f"已登记 {len(settings.calibration_points)} 个定标点"

    if registered_fields and not missing_paths:
        return True, False, f"已登记：{', '.join(_field_label(field) for field in registered_fields)}"
    if registered_fields and missing_paths:
        return True, True, f"已登记但文件缺失：{', '.join(_field_label(field) for field in missing_paths)}"
    if directory_has_files:
        return True, False, f"{directory_spec(stage.directory_key).label}目录已有文件"
    return False, False, stage.next_action


def _field_has_value(settings: ProjectSettings, field_name: str) -> bool:
    value = getattr(settings, field_name, None)
    if isinstance(value, list | tuple | dict):
        return bool(value)
    if value is None:
        return False
    return bool(str(value).strip())


def _field_points_to_missing_path(settings: ProjectSettings, field_name: str) -> bool:
    value = getattr(settings, field_name, "")
    if not value or isinstance(value, list | tuple | dict):
        return False
    return not Path(str(value)).expanduser().exists()


def _field_label(field_name: str) -> str:
    labels = {
        "project_name": "项目名",
        "system": "实验体系",
        "single_spectrum_file": "单谱文件",
        "sum_spectrum_folder": "累计谱目录",
        "temperature_scan_folder": "温度扫描目录",
        "pie_scan_folder": "PIE扫描目录",
        "manual_peak_file": "手动卡峰文件",
        "kr_calibration_folder": "Kr定标扫描目录",
        "kr_calibration_peak_file": "Kr定标卡峰文件",
        "temperature_scan_result_file": "温度扫描结果",
        "pie_identification_result_file": "PIE鉴定结果",
        "mole_fraction_result_file": "摩尔分数结果",
        "calibration_points": "定标点",
    }
    return labels.get(field_name, field_name)


def _directory_has_files(path: Path) -> bool:
    if not path.exists():
        return False
    return any(child.is_file() for child in path.rglob("*"))


def collect_project_files(settings: ProjectSettings) -> list[ProjectFileRecord]:
    root = project_root(settings)
    records: list[ProjectFileRecord] = []
    registered_paths = _registered_artifact_paths(settings)

    for spec in PROJECT_DIRECTORIES:
        section_root = root / spec.relative_path
        if not section_root.exists():
            continue
        for path in sorted(section_root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            records.append(
                ProjectFileRecord(
                    path=path,
                    section_key=spec.key,
                    section_label=spec.label,
                    relative_path=str(path.relative_to(root)),
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    registered=path.resolve() in registered_paths,
                )
            )

    scanned = {record.path.resolve() for record in records if record.path.exists()}
    for field_name, path in _registered_artifact_fields(settings).items():
        if not path:
            continue
        resolved = Path(path).expanduser()
        if resolved.exists() and resolved.resolve() in scanned:
            continue
        spec = _section_for_registered_field(field_name)
        size = resolved.stat().st_size if resolved.exists() else 0
        modified = datetime.fromtimestamp(resolved.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if resolved.exists() else ""
        records.append(
            ProjectFileRecord(
                path=resolved,
                section_key=spec.key,
                section_label=spec.label,
                relative_path=f"registered/{field_name}/{resolved.name}",
                size_bytes=size,
                modified_at=modified,
                registered=True,
                missing=not resolved.exists(),
            )
        )
    return records


def _registered_artifact_fields(settings: ProjectSettings) -> dict[str, str]:
    fields = {
        "single_spectrum_file": settings.single_spectrum_file,
        "sum_spectrum_folder": settings.sum_spectrum_folder,
        "temperature_scan_folder": settings.temperature_scan_folder,
        "pie_scan_folder": settings.pie_scan_folder,
        "manual_peak_file": settings.manual_peak_file,
        "kr_calibration_folder": settings.kr_calibration_folder,
        "kr_calibration_peak_file": settings.kr_calibration_peak_file,
        "temperature_scan_result_file": settings.temperature_scan_result_file,
        "pie_identification_result_file": settings.pie_identification_result_file,
        "mole_fraction_result_file": settings.mole_fraction_result_file,
    }
    for index, folder in enumerate(settings.effective_pie_scan_folders()[1:], start=2):
        fields[f"pie_scan_folder_{index}"] = folder
    return {key: value for key, value in fields.items() if value}


def _registered_artifact_paths(settings: ProjectSettings) -> set[Path]:
    paths: set[Path] = set()
    for value in _registered_artifact_fields(settings).values():
        path = Path(value).expanduser()
        if path.exists() and path.is_file():
            paths.add(path.resolve())
    return paths


def _section_for_registered_field(field_name: str) -> ProjectDirectorySpec:
    if field_name.startswith("pie_scan_folder_"):
        return directory_spec("raw_data")
    mapping = {
        "single_spectrum_file": "raw_data",
        "sum_spectrum_folder": "raw_data",
        "temperature_scan_folder": "raw_data",
        "pie_scan_folder": "raw_data",
        "manual_peak_file": "spectrum_analysis",
        "kr_calibration_folder": "raw_data",
        "kr_calibration_peak_file": "spectrum_analysis",
        "temperature_scan_result_file": "temperature_scan",
        "pie_identification_result_file": "pie_analysis",
        "mole_fraction_result_file": "mole_fraction",
    }
    return directory_spec(mapping.get(field_name, "spectrum_analysis"))


def export_project_archive(
    settings: ProjectSettings,
    destination: str | Path | None = None,
    *,
    include_metadata: bool = True,
) -> Path:
    ensure_project_structure(settings)
    root = project_root(settings)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = sanitize_project_slug(settings.project_name or settings.system or root.name)
    if destination is None:
        destination_path = root / "versions" / f"{slug}_backup_{timestamp}.zip"
    else:
        destination_path = Path(destination).expanduser()
        if destination_path.suffix.lower() != ".zip":
            destination_path = destination_path / f"{slug}_backup_{timestamp}.zip"
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(destination_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if include_metadata:
            archive.writestr(
                f"{root.name}/project_state.yaml",
                yaml.safe_dump(
                    portable_project_settings_dict(settings, root),
                    allow_unicode=True,
                    sort_keys=False,
                ),
            )
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                continue
            if not path.is_file() or path.resolve() == destination_path.resolve():
                continue
            # Exclude existing snapshots to prevent recursive packaging
            if path.parent.name == "versions" and path.suffix == ".zip":
                continue
            archive.write(path, f"{root.name}/{path.relative_to(root)}")
        for field_name, value in _registered_artifact_fields(settings).items():
            path = Path(value).expanduser()
            if not path.exists():
                continue
            try:
                path.relative_to(root)
                continue
            except ValueError:
                if path.is_file():
                    archive.write(path, f"{root.name}/_registered_external/{field_name}/{path.name}")
                elif path.is_dir():
                    for child in sorted(path.rglob("*")):
                        if child.is_file():
                            archive.write(
                                child,
                                f"{root.name}/_registered_external/{field_name}/{path.name}/{child.relative_to(path)}",
                            )
    return destination_path


def create_project_snapshot(settings: ProjectSettings, note: str = "") -> Path:
    ensure_project_structure(settings)
    root = project_root(settings)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_slug = sanitize_project_slug(note, fallback="snapshot")
    destination = root / "versions" / f"{timestamp}_{note_slug}.zip"
    return export_project_archive(settings, destination)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_data_source(settings: ProjectSettings, source_key: str) -> DataSourceValidationRecord:
    """验证单个数据源的有效性"""
    spec = PROJECT_SOURCE_SPECS.get(source_key)
    if not spec:
        return DataSourceValidationRecord(
            source_key=source_key,
            source_label="未知",
            path=None,
            exists=False,
            is_readable=False,
            file_count=0,
            detail="未知的数据源类型",
        )

    if source_key == "pie_scan":
        folders = settings.effective_pie_scan_folders()
        if len(folders) > 1:
            segment_records = [
                validate_data_source(
                    replace(settings, pie_scan_folder=folder, pie_scan_folders=[]),
                    source_key,
                )
                for folder in folders
            ]
            invalid = [record for record in segment_records if not record.is_valid]
            detail = (
                f"{len(folders)} 段，合计 {sum(record.file_count for record in segment_records)} 个文件"
                if not invalid
                else f"{len(invalid)}/{len(folders)} 个能段目录不可用"
            )
            return DataSourceValidationRecord(
                source_key=source_key,
                source_label=spec.label,
                path=";".join(folders),
                exists=all(record.exists for record in segment_records),
                is_readable=not invalid,
                file_count=sum(record.file_count for record in segment_records),
                detail=detail,
                last_modified=max(
                    (record.last_modified for record in segment_records),
                    default="",
                ),
            )

    field_value = getattr(settings, spec.field_name, None)
    path_str = str(field_value).strip() if field_value else None

    if not path_str:
        return DataSourceValidationRecord(
            source_key=source_key,
            source_label=spec.label,
            path=None,
            exists=False,
            is_readable=False,
            file_count=0,
            detail="未配置",
        )

    path = Path(path_str).expanduser()

    # 检查路径存在性
    if not path.exists():
        return DataSourceValidationRecord(
            source_key=source_key,
            source_label=spec.label,
            path=path_str,
            exists=False,
            is_readable=False,
            file_count=0,
            detail=f"路径不存在",
        )

    # 检查可读性和文件数量
    try:
        if path.is_file():
            file_count = 1
            file_count_limited = False
            is_readable = os.access(path, os.R_OK)
        elif path.is_dir():
            try:
                file_count, file_count_limited = _count_files_limited(path)
                is_readable = os.access(path, os.R_OK | os.X_OK)
            except (PermissionError, OSError):
                file_count = 0
                file_count_limited = False
                is_readable = False
        else:
            return DataSourceValidationRecord(
                source_key=source_key,
                source_label=spec.label,
                path=path_str,
                exists=False,
                is_readable=False,
                file_count=0,
                detail="既不是文件也不是目录",
            )
    except Exception as e:
        return DataSourceValidationRecord(
            source_key=source_key,
            source_label=spec.label,
            path=path_str,
            exists=True,
            is_readable=False,
            file_count=0,
            detail=f"读取失败: {str(e)[:50]}",
        )

    # 获取修改时间
    try:
        mtime = path.stat().st_mtime
        last_modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except:
        last_modified = ""

    if is_readable and file_count > 0:
        detail = f"至少 {file_count} 个文件" if file_count_limited else f"{file_count} 个文件"
    else:
        detail = "文件数为0"

    return DataSourceValidationRecord(
        source_key=source_key,
        source_label=spec.label,
        path=path_str,
        exists=True,
        is_readable=is_readable,
        file_count=file_count,
        detail=detail,
        last_modified=last_modified,
    )


def validate_all_data_sources(settings: ProjectSettings) -> list[DataSourceValidationRecord]:
    """验证所有数据源"""
    records = []
    for source_key in PROJECT_SOURCE_SPECS.keys():
        records.append(validate_data_source(settings, source_key))
    return records


def get_data_source_validation_status(
    settings: ProjectSettings,
    records: list[DataSourceValidationRecord] | None = None,
) -> DataSourceValidationStatus:
    """获取总体数据导入状态"""
    if records is None:
        records = validate_all_data_sources(settings)

    # 关键数据源（必需）。单谱/累计谱是质谱工作台来源，可由温度/PIE目录派生，
    # 不作为项目级原始数据源完整性的判据。
    essential_sources = set(CORE_WORKFLOW_SOURCE_KEYS)
    essential_records = [r for r in records if r.source_key in essential_sources]

    valid_count = sum(1 for r in essential_records if r.is_valid)
    essential_count = len(essential_records)

    # 检查是否有失效的已配置项（路径存在但不可读）
    invalid_records = [r for r in records if r.path and not r.is_valid]

    if invalid_records:
        return DataSourceValidationStatus.INVALID

    if valid_count == 0:
        return DataSourceValidationStatus.UNCONFIGURED

    if valid_count < essential_count:
        return DataSourceValidationStatus.PARTIAL

    return DataSourceValidationStatus.COMPLETE


def get_project_ui_state(settings: ProjectSettings) -> ProjectUIState:
    """判断项目当前应展示的UI主操作按钮状态"""
    # 检查项目信息和目录初始化
    has_identity = bool(settings.project_name.strip() or settings.system.strip())
    root = project_root(settings)
    marker_exists = (root / ".bl03u_project").exists()
    standard_dirs_exist = all((root / spec.relative_path).exists() for spec in PROJECT_DIRECTORIES)

    if not has_identity or not marker_exists or not standard_dirs_exist:
        return ProjectUIState.UNINITIALIZED

    # 检查是否至少有一个分析工作流可执行
    analysis_result = analyze_workflow_capabilities(settings)

    if not analysis_result.available_workflows:
        return ProjectUIState.AWAITING_DATA_IMPORT

    # 数据已完成导入且至少一个工作流可执行，检查是否有分析进行中
    statuses = build_project_stage_statuses(settings)
    # 跳过前两个阶段（项目初始化、数据导入），检查后面的分析阶段
    analysis_started = any(s.completed for s in statuses[2:])

    if not analysis_started:
        return ProjectUIState.READY_TO_ANALYZE

    # 检查是否已完成当前配置中的最后一个分析阶段
    final_stage = statuses[-1]
    if final_stage.completed:
        return ProjectUIState.ANALYSIS_COMPLETE

    return ProjectUIState.ANALYSIS_IN_PROGRESS


def analyze_workflow_capabilities(
    settings: ProjectSettings,
    validation_records: list[DataSourceValidationRecord] | None = None,
) -> WorkflowAnalysisResult:
    """分析当前项目可执行的工作流和缺少的依赖"""
    result = WorkflowAnalysisResult()

    # 收集所有数据源的有效性
    data_source_status = {}
    if validation_records is None:
        validation_records = validate_all_data_sources(settings)
    for validation in validation_records:
        data_source_status[validation.source_key] = validation.is_valid

    result.data_source_status = data_source_status

    # 评估每个工作流
    for workflow_profile, requirement in WORKFLOW_REQUIREMENTS.items():
        can_execute = _check_workflow_requirements(requirement, data_source_status)

        if can_execute:
            result.available_workflows.append(workflow_profile)
        else:
            missing = _get_missing_sources_for_workflow(requirement, data_source_status)
            result.unavailable_workflows[workflow_profile] = ", ".join(missing)

    # 生成推荐的下一步操作
    if result.available_workflows:
        result.recommended_next_step = "可执行以下工作流"
    else:
        # 找出最关键缺少的数据源
        missing_by_frequency = {}
        for workflow, reasons in result.unavailable_workflows.items():
            for source_key in CORE_WORKFLOW_SOURCE_KEYS:
                if not data_source_status.get(source_key):
                    missing_by_frequency[source_key] = missing_by_frequency.get(source_key, 0) + 1

        if missing_by_frequency:
            most_critical = max(missing_by_frequency, key=missing_by_frequency.get)
            source_label = PROJECT_SOURCE_SPECS[most_critical].label
            result.recommended_next_step = f"请先导入：{source_label}"

    return result


def _check_workflow_requirements(
    requirement: WorkflowRequirement,
    data_source_status: dict[str, bool],
) -> bool:
    """检查工作流的所有依赖是否满足"""
    for rule in requirement.required_sources:
        if rule.operator == DependencyOperator.ANY_OF:
            # 至少一个源有效
            if not any(data_source_status.get(src, False) for src in rule.sources):
                return False
        elif rule.operator == DependencyOperator.ALL_OF:
            # 所有源都有效
            if not all(data_source_status.get(src, False) for src in rule.sources):
                return False
    return True


def _get_missing_sources_for_workflow(
    requirement: WorkflowRequirement,
    data_source_status: dict[str, bool],
) -> list[str]:
    """获取工作流缺少的所有数据源"""
    missing = set()
    for rule in requirement.required_sources:
        if rule.operator == DependencyOperator.ANY_OF:
            if not any(data_source_status.get(src, False) for src in rule.sources):
                for src in rule.sources:
                    if src in PROJECT_SOURCE_SPECS and not data_source_status.get(src, False):
                        missing.add(PROJECT_SOURCE_SPECS[src].label)
        elif rule.operator == DependencyOperator.ALL_OF:
            for src in rule.sources:
                if src in PROJECT_SOURCE_SPECS and not data_source_status.get(src, False):
                    missing.add(PROJECT_SOURCE_SPECS[src].label)
    return sorted(list(missing))



def scan_project_artifacts(settings: ProjectSettings) -> list[ArtifactRecord]:
    """扫描项目的所有产物并返回记录，包括已登记和目录内未登记文件"""
    records: list[ArtifactRecord] = []
    root = project_root(settings)

    # Track which files we've already recorded (by resolved path) to avoid duplicates
    recorded_paths: set[Path] = set()

    # 定义每个产物类型与其相关字段的映射
    artifact_mappings = [
        ("temperature_scan_result", "temperature_scan", "温度扫描", "TemperatureModule", settings.temperature_scan_result_file),
        ("pie_identification_result", "pie", "PIE鉴定", "PIEModule", settings.pie_identification_result_file),
        ("mole_fraction_result", "mole_fraction", "摩尔分数", "MoleFractionModule", settings.mole_fraction_result_file),
        ("manual_peak_file", "intermediate", "手动卡峰", "SpectrumModule", settings.manual_peak_file),
    ]

    for artifact_type, category_key, label, source_module, path_str in artifact_mappings:
        if not path_str:
            continue

        path = Path(path_str).expanduser()
        try:
            resolved_path = path.resolve()
            recorded_paths.add(resolved_path)

            if path.exists():
                stat = path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                generation_time = mtime.isoformat()
                size = stat.st_size
                status = ArtifactStatus.VALID.value
                detail = ""
            else:
                generation_time = ""
                size = 0
                status = ArtifactStatus.MISSING.value
                detail = "文件不存在或已删除"

            records.append(
                ArtifactRecord(
                    artifact_type=artifact_type,
                    category=category_key,
                    path=str(path),
                    generation_time=generation_time,
                    size_bytes=size,
                    source_module=source_module,
                    status=status,
                    detail=detail,
                )
            )
        except Exception as e:
            records.append(
                ArtifactRecord(
                    artifact_type=artifact_type,
                    category=category_key,
                    path=str(path),
                    generation_time="",
                    size_bytes=0,
                    source_module=source_module,
                    status=ArtifactStatus.INCOMPLETE.value,
                    detail=f"读取失败: {str(e)[:50]}",
                )
            )

    # 扫描项目目录内的所有文件，补充未登记的产物
    category_mapping = {
        "spectrum_analysis": "intermediate",
        "temperature_scan": "temperature_scan",
        "pie_analysis": "pie",
        "mole_fraction": "mole_fraction",
        "pics": "pics",
    }

    for spec in PROJECT_DIRECTORIES:
        if spec.key == "versions":
            # Already handled below
            continue

        category_key = category_mapping.get(spec.key, spec.key)
        section_path = root / spec.relative_path

        if not section_path.exists():
            continue

        try:
            for file_path in sorted(section_path.rglob("*")):
                if not file_path.is_file():
                    continue

                # Skip if already recorded as registered artifact
                try:
                    if file_path.resolve() in recorded_paths:
                        continue
                except Exception:
                    pass

                try:
                    stat = file_path.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    generation_time = mtime.isoformat()

                    records.append(
                        ArtifactRecord(
                            artifact_type=f"{spec.key}_unregistered",
                            category=category_key,
                            path=str(file_path),
                            generation_time=generation_time,
                            size_bytes=stat.st_size,
                            source_module=spec.label,
                            status=ArtifactStatus.VALID.value,
                            detail="项目目录内文件（未登记）",
                        )
                    )
                except Exception as e:
                    pass
        except Exception:
            pass

    # 扫描版本快照目录
    snapshots_dir = root / "versions"
    if snapshots_dir.exists():
        for snapshot_file in sorted(snapshots_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                stat = snapshot_file.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                records.append(
                    ArtifactRecord(
                        artifact_type="snapshot",
                        category="snapshots",
                        path=str(snapshot_file),
                        generation_time=mtime.isoformat(),
                        size_bytes=stat.st_size,
                        source_module="ProjectManager",
                        status=ArtifactStatus.VALID.value,
                        detail="项目快照",
                    )
                )
            except Exception as e:
                pass

    return records
