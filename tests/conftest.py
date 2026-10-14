import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    from make_synthetic_data import make

    return make(tmp_path_factory.mktemp("data"), sources_per_class=6, copies=4, external_per_class=3)
