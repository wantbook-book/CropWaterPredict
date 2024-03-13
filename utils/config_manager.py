import os
import json
import pathlib
DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "conf/config.json"
CONFIG_PATH = os.environ.get("CONFIG_PATH",  str(DEFAULT_CONFIG_PATH))
class ConfigManager():
    def __init__(self) -> None:
        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)
    def get(self, key:str):
        return self.config[key]
