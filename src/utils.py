import yaml, re
from src import PROJECT_ROOT_PATH

def replace_project_root_path(config, project_root):
        # replace 
        if isinstance(config, dict):
            return {k: replace_project_root_path(v, project_root) for k, v in config.items()}
        elif isinstance(config, list):
            return [replace_project_root_path(i, project_root) for i in config]
        elif isinstance(config, str):
            return re.sub(r'{PROJECT_ROOT_PATH}', project_root, config)
        return config

def read_config(config_path):
    # read configs and format
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    # Recursively replace {PROJECT_ROOT_PATH} in all values
    cfg = replace_project_root_path(cfg, PROJECT_ROOT_PATH)
    return cfg