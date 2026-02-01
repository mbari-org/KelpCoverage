import json
from pathlib import Path
from kelp_coverage.core.errors import KelpFileNotFoundError, KelpDirNotFoundError

def get_config_dir():
    root = Path(__file__).parent.parent.parent
    config_dir = root / "configs"
    if not config_dir.exists():
        raise KelpDirNotFoundError(str(config_dir))
    return config_dir

def load_config_file(config_name):
    config_dir = get_config_dir()
    config_path = config_dir / f"{config_name}.json"
    if not config_path.exists():
        raise KelpFileNotFoundError(str(config_path))
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def config_from_model_path(model_path):
    model_name = Path(model_path).name.lower()
    if "mobile_sam" in model_name:
        return "mobile_sam"
    return "default"

def load_model_config(model_path):
    config_name = config_from_model_path(model_path)
    try:
        config = load_config_file(config_name)
    except KelpFileNotFoundError:
        config = load_config_file("default")
    return config

def change_config(base_config, changes):
    changed = base_config.copy()
    for key, value in changes.items():
        if value is not None:
            changed[key] = value
    return changed

def get_config(model_path, changes=None):
    base_config = load_model_config(model_path)
    if changes:
        return change_config(base_config, changes)
    return base_config
