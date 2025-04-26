import yaml
import os


FILE_NAME = "mcts_data.yaml"

class DataSaver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.datafile_path = os.path.join(self.save_dir, FILE_NAME)
            
    def append(self, **kwargs):
        
        data = self.load()
        if data is None:
            data = {}
            
        for k, v in kwargs.items():
            if k in data:
                data[k].append(v)
            else:
                data[k] = [v]

        with open(self.datafile_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)
            
    def load(self):
        if not os.path.exists(self.datafile_path):
            return None
        
        with open(self.datafile_path, 'r') as file:
            data = yaml.safe_load(file)
        return data