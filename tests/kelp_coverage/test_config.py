import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from kelp_coverage.config import (
    get_config_dir,
    config_from_model_path,
    change_config,
    load_config_file,
    load_model_config,
    get_config,
)
from kelp_coverage.core.errors import KelpFileNotFoundError, KelpDirNotFoundError

class TestGetConfigDir:
    def test_raises_when_configs_dir_missing(self, tmp_path):
        fake_module_path = tmp_path / "kelp_coverage" / "config.py"
        fake_module_path.parent.mkdir(parents=True)
        fake_module_path.write_text("")

        with patch("kelp_coverage.config.Path") as mock_path_class:
            mock_path_instance = MagicMock()
            mock_path_class.return_value = mock_path_instance
            
            mock_path_instance.parent.parent.parent = tmp_path
            with pytest.raises(KelpDirNotFoundError):
                get_config_dir()

class TestConfigFromModelPath:
    @pytest.mark.parametrize("model_path,expected", [
        ("/path/to/mobile_sam.pt", "mobile_sam"),
        ("/path/to/MOBILE_SAM.pth", "mobile_sam"),
        ("/path/to/sam_vit_b.pt", "default"),
        ("mobile_sam_encoder.onnx", "mobile_sam"),
    ])
    def test_various_model_paths(self, model_path, expected):
        assert config_from_model_path(model_path) == expected

class TestChangeConfig:
    def test_applies_changes(self, sample_config):
        changes = {"threshold": 0.7}
        result = change_config(sample_config, changes)
        assert result["threshold"] == 0.7

    def test_preserves_unchanged_values(self, sample_config):
        changes = {"threshold": 0.7}
        result = change_config(sample_config, changes)
        assert result["slice_size"] == 512

    def test_ignores_none_values(self, sample_config):
        changes = {"threshold": None, "slice_size": 256}
        result = change_config(sample_config, changes)
        assert result["threshold"] == 0.5
        assert result["slice_size"] == 256

    def test_does_not_mutate_original(self, sample_config):
        original_threshold = sample_config["threshold"]
        changes = {"threshold": 0.9}
        change_config(sample_config, changes)
        assert sample_config["threshold"] == original_threshold

    def test_empty_changes_returns_copy(self, sample_config):
        result = change_config(sample_config, {})
        assert result == sample_config

@pytest.mark.usefixtures("mock_config_dir")
class TestLoadConfigFile:
    def test_loads_json_file(self, tmp_config_dir, sample_config):
        config_file = tmp_config_dir / "test.json"
        config_file.write_text(json.dumps(sample_config))
        result = load_config_file("test")
        assert result == sample_config

    def test_raises_for_missing_file(self):
        with pytest.raises(KelpFileNotFoundError):
            load_config_file("nonexistent")

@pytest.mark.usefixtures("mock_config_dir")
class TestLoadModelConfig:
    def test_loads_matching_config(self, tmp_config_dir, sample_config):
        mobile_config = sample_config.copy()
        mobile_config["threshold"] = 0.99
        
        config_file = tmp_config_dir / "mobile_sam.json"
        config_file.write_text(json.dumps(mobile_config))

        result = load_model_config("/models/mobile_sam.pt")
        assert result["threshold"] == 0.99

    def test_falls_back_to_default(self, tmp_config_dir, sample_config):
        default_file = tmp_config_dir / "default.json"
        default_file.write_text(json.dumps(sample_config))

        result = load_model_config("/models/unknown_model.pt")
        assert result == sample_config

@pytest.mark.usefixtures("mock_config_dir")
class TestGetConfig:
    def test_returns_base_config_without_changes(self, tmp_config_dir, sample_config):
        default_file = tmp_config_dir / "default.json"
        default_file.write_text(json.dumps(sample_config))

        result = get_config("/models/sam.pt")
        assert result == sample_config

    def test_applies_changes_to_config(self, tmp_config_dir, sample_config):
        default_file = tmp_config_dir / "default.json"
        default_file.write_text(json.dumps(sample_config))
        changes = {"threshold": 0.8}

        result = get_config("/models/sam.pt", changes=changes)
        
        assert result["threshold"] == 0.8
        assert result["slice_size"] == 512