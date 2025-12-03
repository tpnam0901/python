from configs.base import Config as BaseConfig


class Config(BaseConfig):
    """Example configuration class extending the base Config."""

    def __init__(self):
        super(Config, self).__init__()

        # Modify default parameters
        self.name = "example_config"

        self.unlock()
        self.learning_rate = 1e-6
        # Add more configuration parameters as needed
        self.input_size = 1
        self.hidden_size = 2
        self.output_size = 1

        # Lock the config to prevent further modifications
        self.lock()
