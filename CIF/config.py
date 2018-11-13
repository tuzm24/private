import os
import yaml

class ConfigMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

class Config(dict):
    def __init__(self, file_path):
        assert os.path.exists(file_path), "ERROR: Config File doesn't exist."
        with open(file_path, 'r') as f:
            self.member = yaml.load(f)
            f.close()
#        self.PRETRAINED_MODEL_PATH = self.MODEL_PATH + self.PRETRAINED_MODEL_PATH
        self.TENSORBOARD_LOG_PATH = self.MODEL_PATH + self.TENSORBOARD_LOG_PATH

        os.makedirs(self.MODEL_PATH,exist_ok=True)
        os.makedirs(self.TENSORBOARD_LOG_PATH,exist_ok=True)

    def __getattr__(self, name):
        value = self.member[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

def write_yml(config):
    path = config.NET_INFO_PATH
    with open(path, 'w+') as fp:
        for key, value in config.member.items():
            if type(value)==str:
                fp.write("{}: '{}'\n".format(key, value))
            else:
                fp.write("{}: {}\n".format(key, value))
        fp.close()
