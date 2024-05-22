from copy import deepcopy

from typing import Optional, Sequence, Union, List

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins.evaluation import default_logger

import os
import math


class EvalStoredWeightsPlugin(StrategyPlugin):
    def __init__(self, path_to_weights: str):
        super().__init__()
        print("\nUsing EvalStoredWeightsPlugin!\n")
    
        self.path_to_weights = path_to_weights
        self.model_weights = self._discover_weight_files(path_to_weights)
        print(self.model_weights)
        return
    
    def _discover_weight_files(self, path):
        """
        Return list of all .pth files in the path directory
        """
        # Discover weight files
        weight_files = [f for f in os.listdir(path) if f.endswith('.pth')]
        # Stort them by task
        weight_files = sorted(weight_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return weight_files

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        if strategy.training_exp_counter == 0:
            self.orig_train_epochs = strategy.train_epochs

        # Load weights to model according to current experience
        weight_file = self.path_to_weights + self.model_weights[strategy.training_exp_counter]
        try:
            strategy.model.load_state_dict(torch.load(weight_file), strict=False)
            print("Loaded weights from: ", weight_file)
        except Exception:
            print("COULD NOT LOAD WEIGHTS from: ", weight_file)
        strategy.train_epochs = 0
        return 
    
    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        strategy.clock.train_iterations += self.orig_train_epochs * \
                math.ceil(len(strategy.adapted_dataset)/strategy.train_mb_size)
        return 


class EvalStoredWeights(BaseStrategy):
    def __init__(self, model: Module, 
                 path_to_weights: str,
                 optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, 
                 train_epochs: int = 1,
                 eval_mb_size: int = 1, 
                 device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, 
                 eval_every=-1):
        
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

        self.path_to_weights = path_to_weights
        self.model_weights = self._discover_weight_files(path_to_weights)
        print(self.model_weights)
        ##import sys;sys.exit()
        self.orig_train_epochs = train_epochs
        return
    
    def _discover_weight_files(self, path):
        """
        Return list of all .pth files in the path directory
        """
        # Discover weight files
        weight_files = [f for f in os.listdir(path) if f.endswith('.pth')]
        # Stort them by task
        weight_files = sorted(weight_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return weight_files
    
    def _before_training_exp(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        # Load weights to model according to current experience
        weight_file = self.path_to_weights + self.model_weights[self.training_exp_counter]
        self.model.load_state_dict(torch.load(weight_file), strict=True)
        print("Loaded weights from: ", weight_file)
        self.train_epochs = 0

        super()._before_training_exp(**kwargs)
        return
    
    def _after_training_exp(self, **kwargs):
        self.clock.train_iterations += self.orig_train_epochs * \
                math.ceil(len(self.adapted_dataset)/self.train_mb_size)
        return super()._after_training_exp(**kwargs)

    