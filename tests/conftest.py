import os
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REAL_DATA_ENV_VAR = "BL03U_REAL_DATA_DIR"


def bl03u_real_data_path() -> Path:
    configured = os.environ.get(REAL_DATA_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / "tests" / "fixtures" / "bl03u_sample"


@pytest.fixture
def bl03u_real_data_dir() -> Path:
    data_dir = bl03u_real_data_path()
    if not data_dir.is_dir():
        pytest.skip(
            f"optional BL03U real-data fixture directory not found: {data_dir}. "
            f"Set {REAL_DATA_ENV_VAR} or keep a local copy at tests/fixtures/bl03u_sample."
        )
    return data_dir
