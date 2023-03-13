import yaml


class Config:
    def __init__(self, config_file_path=None):
        # if config path is none search for config.yaml in the current directory
        config_file_path = 'config.yaml' if config_file_path is None else config_file_path
        self.update(config_file_path)

    def update(self, config_file_path):
        with open(config_file_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


if __name__ == '__main__':
    config = Config()
    print(config.dict)