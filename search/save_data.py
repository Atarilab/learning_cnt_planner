import yaml
import os


FILE_NAME = "mcts_data.yaml"

class DataSaver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.datafile_path = os.path.join(self.save_dir, FILE_NAME)
        self._data = None  # Lazy-loaded cache

    def _ensure_data_loaded(self):
        if self._data is None:
            if os.path.exists(self.datafile_path):
                with open(self.datafile_path, 'r') as file:
                    self._data = yaml.safe_load(file) or {}
            else:
                self._data = {}

    def append(self, **kwargs):
        self._ensure_data_loaded()

        for k, v in kwargs.items():
            if k in self._data:
                self._data[k].append(v)
            else:
                self._data[k] = [v]

        with open(self.datafile_path, 'w') as file:
            yaml.safe_dump(self._data, file, default_flow_style=False)

    def load(self):
        self._ensure_data_loaded()
        return self._data