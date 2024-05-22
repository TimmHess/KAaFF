#  Copyright (c) 2023. Timm Hess (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452


from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy

import numpy as np


class EpochLengthAdapterPlugin(StrategyPlugin):
    def __init__(self, epochs=None, increase_epochs=False):
        super().__init__()

        self.epochs = epochs
        self.increase_epochs = increase_epochs
        self.default_train_epochs = None
        return

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        # Grab the default training epochs by the first time this function is called
        if self.default_train_epochs is None:
            self.default_train_epochs = strategy.train_epochs
        return super().before_training(strategy, **kwargs)

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        if self.epochs: # if epochs is defined
            if strategy.clock.train_exp_counter < len(self.epochs):
                if self.increase_epochs:
                    strategy.train_epochs = np.sum(self.epochs[:(strategy.clock.train_exp_counter+1)])
                else:
                    strategy.train_epochs = self.epochs[strategy.clock.train_exp_counter]
                    print("\nUsing", self.epochs[strategy.clock.train_exp_counter], " epochs during experience", strategy.clock.train_exp_counter, "\n")
            # Check if the number of epochs is 0
            if strategy.train_epochs == 0:
                # Increase the counter of train_iterations by an arbitrary number to preven results from overwriting one-another
                print("Adjusting the number of training iterations to prevent overwriting of results")
                strategy.clock.train_iterations += 100
        else:
            print("Increasing training epochs by factor:", strategy.clock.train_exp_counter+1)
            strategy.train_epochs = self.default_train_epochs * (strategy.clock.train_exp_counter+1)
        return
