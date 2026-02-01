import pytest
import sys
from pathlib import Path
from unittest.mock import patch

@pytest.fixture
def sample_context():
    return {"key":"value", "count": 42}

@pytest.fixture
def sample_config():
    return {
        "slice_size": 512,
        "overlap_ratio": 0.2,
        "threshold": 0.5,
    }

@pytest.fixture
def tmp_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir

@pytest.fixture(autouse=False)
def mock_config_dir(tmp_config_dir):
    with patch("kelp_coverage.config.get_config_dir", return_value=tmp_config_dir):
        yield tmp_config_dir

