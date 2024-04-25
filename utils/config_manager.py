import os
import json
import pathlib
DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "conf/config.json"
CONFIG_PATH = os.environ.get("CONFIG_PATH",  str(DEFAULT_CONFIG_PATH))
from pathlib import Path
class ConfigManager():
    def __init__(self, config_path: Path) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)
    def get(self, key:str):
        return self.config[key]

    def __getitem__(self, key):
        return self.get(key)
