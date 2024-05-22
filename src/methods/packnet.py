#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.

#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452


import copy

from avalanche.models import MultiTaskModule
from avalanche.training import BaseStrategy
from avalanche.training.plugins.strategy_plugin import StrategyPlugin

import torch
from typing import Optional, Sequence, Union, List

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import default_logger
from typing import TYPE_CHECKING

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy
from src.utils import get_grad_normL2
from src.eval.continual_eval import ContinualEvaluationPhasePlugin

if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin

from libs.packnet.packnet_modules import *


class PackNetPlugin(StrategyPlugin):
    def __init__(self, post_prune_epochs: int,
        prune_proportion: t.Union[float, t.Callable[[int], float], t.List[float]] = 0.5,):
        """The PackNetPlugin calls PackNet's pruning and freezing procedures at
        the appropriate times.

        :param post_prune_epochs: The number of epochs to finetune the model
            after pruning the parameters. Must be less than the number of
            training epochs.
        :param prune_proportion: The proportion of parameters to prune
            during each task. Can be a float, a list of floats, or a function
            that takes the task id and returns a float. Each value must be
            between 0 and 1.
        """
        super().__init__()

        self.post_prune_epochs = post_prune_epochs
        self.total_epochs: Union[int, None] = None
        self.prune_proportion: t.Callable[[int], float] = prune_proportion

        if isinstance(prune_proportion, float):
            assert 0 <= self.prune_proportion <= 1, (
                f"`prune_proportion` must be between 0 and 1, got "
                f"{self.prune_proportion}"
            )
            self.prune_proportion = lambda _: prune_proportion
        elif isinstance(prune_proportion, list):
            assert all(0 <= x <= 1 for x in prune_proportion), (
                "all values in `prune_proportion` must be between 0 and 1,"
                f" got {prune_proportion}"
            )
            self.prune_proportion = lambda i: prune_proportion[i]
        else:
            self.prune_proportion = prune_proportion

        return
    

    def before_training(self, strategy: BaseStrategy, **kwargs):
        # Wrap feature_extractor of the model if needed
        if not isinstance(strategy.model.feature_extractor, PackNetModel):
            strategy.model.feature_extractor = PackNetModel(strategy.model.feature_extractor)

            # Only one time adjust the training epochs by adding the post-pruning phase epochs
            strategy.train_epochs += self.post_prune_epochs

        # Check the scenario has enough epochs for the post-pruning phase
        self.total_epochs = strategy.train_epochs
        if self.post_prune_epochs >= self.total_epochs:
            raise ValueError(
                f"`PackNetPlugin` can only be used with a `BaseStrategy`"
                "that has a `train_epochs` attribute greater than "
                f"{self.post_prune_epochs}. "
                f"Strategy has only {self.total_epochs} training epochs."
            )

        return super().before_training(strategy, **kwargs)
    

    def before_training_epoch(self, strategy, *args, **kwargs):
        """
        When the initial training phase is over, prune the model and
        transition to the post-pruning phase.
        """
        epoch = strategy.clock.train_exp_epochs
        model = strategy.model.feature_extractor

        if epoch == (self.total_epochs - self.post_prune_epochs):
            print("\n\n Prune function was called... for exp,", strategy.clock.train_exp_counter, "\n\n")
            model.prune(self.prune_proportion(strategy.clock.train_exp_counter))
        return
    

    def after_training_exp(self, strategy, *args, **kwargs):
        """
        After each experience, commit the model so that the next experience
        does not interfere with the previous one.
        """
        model = strategy.model.feature_extractor
        model.freeze_pruned()
        print("\n\n Freeze pruned function was called... \n\n")
        return
