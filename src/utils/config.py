from typing import Any, Dict, List

import yaml


def load_configs(config_paths: List[str]) -> Dict[str, Any]:
    """
    Load and merge multiple configuration files.

    Args:
        config_paths: List of paths to configuration files

    Returns:
        Merged configuration dictionary
    """
    merged_config = {}

    for path in config_paths:
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
                if config:
                    merged_config.update(config)
        except FileNotFoundError:
            print(f"Warning: Configuration file {path} not found")
        except yaml.YAMLError:
            print(f"Warning: Error parsing YAML in {path}")

    return merged_config
