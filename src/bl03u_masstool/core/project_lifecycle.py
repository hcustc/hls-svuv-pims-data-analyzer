from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import os
from pathlib import Path
import re
import shutil
import zipfile

import yaml

from .config import writable_project_path
from .project_settings import ProjectSettings


class ProjectUIState(Enum):
    """项目管理UI主操作按钮的状态枚举"""
    UNINITIALIZED = ("初始化项目", "项目信息未填或目录未初始化")
    AWAITING_DATA_IMPORT = ("前往数据导入", "已初始化，等待导入原始数据")
    READY_TO_ANALYZE = ("前往质谱工作台", "数据已导入，可开始分析")
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
    PICS = ("pics", "PICS数据库", "final_report")
    REPORTS = ("reports", "综合报告", "final_report")
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
    ProjectDirectorySpec("raw_data", "原始输入", "raw_data", "原始谱图、样品信息和导入记录"),
    ProjectDirectorySpec("calibration", "标定", "calibration", "定标点、定标参数和标定结果"),
    ProjectDirectorySpec("spectrum_analysis", "谱图分析", "spectrum_analysis", "寻峰、卡峰范围和高斯拟合产物"),
    ProjectDirectorySpec("temperature_scan", "温度扫描", "temperature_scan", "温度扫描曲线、表格和图像"),
    ProjectDirectorySpec("pie_analysis", "PIE拟合", "pie_analysis", "PIE 曲线、物种拟合和鉴定结果"),
    ProjectDirectorySpec("mole_fraction", "摩尔分数", "mole_fraction", "摩尔分数计算输入、结果和图像"),
    ProjectDirectorySpec("final_report", "综合报告", "final_report", "汇总报告、说明文档和发表用图表"),
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
    "sample_info": ProjectSourceSpec("sample_info", "样品信息", "sample_info_file", "raw_data", "sample_info"),
    "manual_peak": ProjectSourceSpec(
        "manual_peak", "手动卡峰文件", "manual_peak_file", "spectrum_analysis", "manual_peaks"
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
        ("single_spectrum_file", "sum_spectrum_folder", "temperature_scan_folder", "pie_scan_folder", "sample_info_file"),
        "project",
        "导入原始谱图与样品信息",
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
    ProjectStageSpec(
        "final_report",
        "综合报告",
        "final_report",
        (),
        "project",
        "汇总关键表格、图像和版本快照",
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
    return directories


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


def import_project_source(settings: ProjectSettings, source_path: str | Path, source_key: str) -> ProjectImportResult:
    spec = PROJECT_SOURCE_SPECS[source_key]
    source = Path(source_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)

    ensure_project_structure(settings)
    target_dir = project_directory(settings, spec.directory_key) / spec.subdirectory
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(target_dir / source.name)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    setattr(settings, spec.field_name, str(destination))
    return ProjectImportResult(source=source, destination=destination, field_name=spec.field_name, label=spec.label)


def import_initial_project_data(
    settings: ProjectSettings,
    sources: dict[str, str | Path | None],
) -> list[ProjectImportResult]:
    results: list[ProjectImportResult] = []
    for source_key, source_path in sources.items():
        if not source_path:
            continue
        results.append(import_project_source(settings, source_path, source_key))
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
        root_exists = project_root(settings).exists()
        if has_identity and root_exists:
            return True, False, f"项目目录：{project_root(settings)}"
        if has_identity:
            return False, False, "项目信息已填写，尚未初始化目录"
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
        "sample_info_file": "样品信息",
        "manual_peak_file": "手动卡峰文件",
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
        "sample_info_file": getattr(settings, "sample_info_file", ""),
        "manual_peak_file": settings.manual_peak_file,
        "temperature_scan_result_file": settings.temperature_scan_result_file,
        "pie_identification_result_file": settings.pie_identification_result_file,
        "mole_fraction_result_file": settings.mole_fraction_result_file,
    }
    return {key: value for key, value in fields.items() if value}


def _registered_artifact_paths(settings: ProjectSettings) -> set[Path]:
    paths: set[Path] = set()
    for value in _registered_artifact_fields(settings).values():
        path = Path(value).expanduser()
        if path.exists() and path.is_file():
            paths.add(path.resolve())
    return paths


def _section_for_registered_field(field_name: str) -> ProjectDirectorySpec:
    mapping = {
        "single_spectrum_file": "raw_data",
        "sum_spectrum_folder": "raw_data",
        "temperature_scan_folder": "raw_data",
        "pie_scan_folder": "raw_data",
        "sample_info_file": "raw_data",
        "manual_peak_file": "spectrum_analysis",
        "temperature_scan_result_file": "temperature_scan",
        "pie_identification_result_file": "pie_analysis",
        "mole_fraction_result_file": "mole_fraction",
    }
    return directory_spec(mapping.get(field_name, "final_report"))


def export_project_archive(settings: ProjectSettings, destination: str | Path | None = None) -> Path:
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
        archive.writestr(
            f"{root.name}/project_state.yaml",
            yaml.safe_dump(asdict(settings), allow_unicode=True, sort_keys=False),
        )
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.resolve() == destination_path.resolve():
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
            is_readable = os.access(path, os.R_OK)
        elif path.is_dir():
            try:
                files = list(path.glob("**/*"))
                file_count = sum(1 for f in files if f.is_file())
                is_readable = os.access(path, os.R_OK | os.X_OK)
            except (PermissionError, OSError):
                file_count = 0
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

    detail = f"{file_count} 个文件" if is_readable and file_count > 0 else "文件数为0"

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


def get_data_source_validation_status(settings: ProjectSettings) -> DataSourceValidationStatus:
    """获取总体数据导入状态"""
    records = validate_all_data_sources(settings)

    # 关键数据源（必需）
    essential_sources = {"single_spectrum", "sum_spectrum", "temperature_scan", "pie_scan"}
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
    root_exists = project_root(settings).exists()

    if not has_identity or not root_exists:
        return ProjectUIState.UNINITIALIZED

    # 检查原始数据导入（使用严格验证）
    data_status = get_data_source_validation_status(settings)

    if data_status in (DataSourceValidationStatus.UNCONFIGURED, DataSourceValidationStatus.INVALID):
        return ProjectUIState.AWAITING_DATA_IMPORT

    if data_status == DataSourceValidationStatus.PARTIAL:
        return ProjectUIState.AWAITING_DATA_IMPORT

    # 数据已完成导入（COMPLETE），检查是否有任何分析产物
    statuses = build_project_stage_statuses(settings)
    # 跳过前两个阶段（项目初始化、数据导入），检查后面的分析阶段
    analysis_started = any(s.completed for s in statuses[2:])

    if not analysis_started:
        return ProjectUIState.READY_TO_ANALYZE

    # 检查是否已完成所有阶段
    final_stage = statuses[-1]  # "综合报告"阶段
    if final_stage.completed:
        return ProjectUIState.ANALYSIS_COMPLETE

    return ProjectUIState.ANALYSIS_IN_PROGRESS


def scan_project_artifacts(settings: ProjectSettings) -> list[ArtifactRecord]:
    """扫描项目的所有产物并返回记录"""
    records: list[ArtifactRecord] = []
    root = project_root(settings)

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
