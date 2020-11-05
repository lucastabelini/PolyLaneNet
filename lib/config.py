import yaml
import torch

import lib.models as models
import lib.datasets as datasets


class Config(object):
    def __init__(self, config_path):
        """
        Initialize the config file.

        Args:
            self: (todo): write your description
            config_path: (str): write your description
        """
        self.config = {}
        self.load(config_path)

    def load(self, path):
        """
        Load config file.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        with open(path, 'r') as file:
            self.config_str = file.read()
        self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)

    def __repr__(self):
        """
        Return a repr string__ method for the __repr__ method.

        Args:
            self: (todo): write your description
        """
        return self.config_str

    def get_dataset(self, split):
        """
        Returns the dataset.

        Args:
            self: (str): write your description
            split: (str): write your description
        """
        return getattr(datasets,
                       self.config['datasets'][split]['type'])(**self.config['datasets'][split]['parameters'])

    def get_model(self):
        """
        Get the model object.

        Args:
            self: (todo): write your description
        """
        name = self.config['model']['name']
        parameters = self.config['model']['parameters']
        return getattr(models, name)(**parameters)

    def get_optimizer(self, model_parameters):
        """
        Get optimizer.

        Args:
            self: (todo): write your description
            model_parameters: (todo): write your description
        """
        return getattr(torch.optim, self.config['optimizer']['name'])(model_parameters,
                                                                      **self.config['optimizer']['parameters'])

    def get_lr_scheduler(self, optimizer):
        """
        Gets the optimizer.

        Args:
            self: (todo): write your description
            optimizer: (todo): write your description
        """
        return getattr(torch.optim.lr_scheduler,
                       self.config['lr_scheduler']['name'])(optimizer, **self.config['lr_scheduler']['parameters'])

    def get_loss_parameters(self):
        """
        Returns the loss parameters

        Args:
            self: (todo): write your description
        """
        return self.config['loss_parameters']

    def get_test_parameters(self):
        """
        Returns a list of test parameters.

        Args:
            self: (todo): write your description
        """
        return self.config['test_parameters']

    def __getitem__(self, item):
        """
        Return the value of item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.config[item]
