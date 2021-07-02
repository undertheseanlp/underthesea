import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Union
import torch.nn

from underthesea import file_utils
from underthesea.utils import device


class Model(torch.nn.Module):
    r"""
    Abstract base class for all downstream task models
    Every new type of model must implement these methods.

    Source: FlairNLP
    """

    @abstractmethod
    def evaluate(self):
        r"""Evaluates the model. Returns a Result object containing evaluation
        """
        pass

    def save(self, model_file: Union[str, Path]):
        """
        Saves the current model to the provided file.

        Args:
            model_file (Union[str, Path]): the model file

        """
        model_state = self._get_state_dict()

        torch.save(model_state, str(model_file), pickle_protocol=4)

    @staticmethod
    @abstractmethod
    def _fetch_model(model_name) -> str:
        return model_name

    @staticmethod
    @abstractmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary. Implementing this enables the load() and load_checkpoint()
        functionality."""
        pass

    @classmethod
    def load(cls, model: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = file_utils.load_big_file(str(model_file))
        state = torch.load(f, map_location='cpu')

        model = cls._init_model_with_state_dict(state)

        model.eval()
        model.to(device)

        return model
