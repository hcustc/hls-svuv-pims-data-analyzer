import pytest

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.temperature_scan import (
    analyze_temperature_folder,
    build_temperature_curves,
)


pytestmark = pytest.mark.real_data


def test_real_temperature_fixture_builds_grouped_curves(bl03u_real_data_dir):
    folder = bl03u_real_data_dir / "顺-13-二甲基环己烷 30 torr" / "温度扫描" / "11.5eV"
    if not folder.is_dir():
        pytest.skip(f"real temperature fixture not found: {folder}")

    df = analyze_temperature_folder(
        folder,
        calibration=Calibration(),
        detection_min_idx=0,
        threshold_end=2,
        min_intensity=3,
        prefer_gaussian=False,
        reference_mode="sum",
    )
    curves = build_temperature_curves(df)

    assert len(df) > 0
    assert len(curves) > 0
