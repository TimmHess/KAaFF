#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from enum import unique
from typing import Union, TYPE_CHECKING
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
from avalanche.training import BaseStrategy

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.models.dynamic_modules import MultiHeadClassifier

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy


class IterationsInsteadOfEpochs(StrategyPlugin):
    """Stop training based on number of iterations instead of epochs."""

    def __init__(self, max_iterations: int):
        super().__init__()
        self.max_iterations = max_iterations -1 # -1 because we start at 0

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        if self.max_iterations == 0:
            strategy.stop_training()
        return super().before_training_exp(strategy, **kwargs)

    def after_training_iteration(self, strategy, **kwargs):
        if strategy.clock.train_exp_iterations == self.max_iterations:
            print(f"Stopping training, reached max iterations: {self.max_iterations}")
            strategy.stop_training()


class MetricOverSeed:
    logging_token = "[SEED-AVGED-RESULTS]="
    logging_result_format = "{:.5f}\pm{:.5f}"
    loggin_result_separator = "\t"

    def __init__(self, name, extract_name, extract_idx=-1, mul_factor=100):
        """
        :param name: Name to give in logging
        :param extract_name: Dict name in all_metrics dict after end of training seed.
        :param extract_idx: Which idx to extract, -1 for final one.
        :param mul_factor: Multiplication factor before return result.
        """
        self.name = name
        self.extract_name = extract_name
        self.extract_idx = extract_idx
        self.mul_factor = mul_factor

        # Results appended sequentially
        self.seeds = []
        self.seed_results = []

    def extract_metric_fn(self, all_metrics: dict):
        """ Extract dict of all eval results on end of seed.
        Name-idx returns tuple of (list(<STEPS>),list(<METRIC-VALS>))
        Select latter with [1], and apply the extraction idx.
        """
        return all_metrics[self.extract_name][1][self.extract_idx] * self.mul_factor

    def add_result(self, all_metrics: dict, seed: int):
        try:
            result = self.extract_metric_fn(all_metrics)
        except Exception as e:
            print(f"[WARNING] No SEED result for metric {self.name}, because of error: {e}")
            return

        self.seeds.append(seed)
        self.seed_results.append(result)

    def get_mean_std_results(self):
        result_t = torch.tensor(self.seed_results)  # list to tensor
        mean, std = result_t.mean().item(), result_t.std().item()
        return mean, std


def get_grad_normL2(model, norm_type: float = 2):
    """Returns the gradient norm of the model.
    Calculated the same way as torch.clip_grad_norm_"""

    # Params with grad
    parameters = model.parameters()
    if isinstance(model.parameters(), torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return None
    device = parameters[0].grad.device

    # calc norm
    total_norm = torch.norm(torch.stack(
        [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.item()


@torch.no_grad()
def get_prototypes_from_classifier(classifier: Union[torch.nn.Linear, MultiHeadClassifier], get_clone: bool):
    """
    Returns individual prototypes for given classifier.
    :param classifier:
    :param get_clone: Get a clone of the original parameter references instead of the original ones.
    :return:
    """
    protos_weight = {}
    protos_bias = {}

    if isinstance(classifier, MultiHeadClassifier):  # Multi-head
        y_offset = 0
        for taskid, taskhead in classifier.classifiers.items():  # Iterate heads
            nb_task_protos = 0
            for param_name, param in taskhead.named_parameters():  # Weight/bias of Linear layer heads
                nb_task_protos = param.shape[0]
                if 'weight' in param_name:
                    for y in range(nb_task_protos):  # Per head params restart from 0, but total protos has offset
                        protos_weight[y + y_offset] = param[y]
                elif 'bias' in param_name:
                    for y in range(nb_task_protos):
                        protos_bias[y + y_offset] = param[y]
            y_offset += nb_task_protos

    elif isinstance(classifier, torch.nn.Linear):  # Single head
        for param_name, param in classifier.named_parameters():
            nb_protos = param.shape[0]
            if 'weight' in param_name:
                for y in range(nb_protos):
                    protos_weight[y] = param[y]
            elif 'bias' in param_name:
                for y in range(nb_protos):
                    protos_bias[y] = param[y]
    else:
        raise Exception()

    if get_clone:  # Make clones of original references
        protos_weight = {y: param.detach().clone() for y, param in protos_weight.items()}
        protos_bias = {y: param.detach().clone() for y, param in protos_bias.items()}
    return protos_weight, protos_bias


class ExpLRSchedulerPlugin(StrategyPlugin):
    """
    Learning Rate Scheduler Plugin
    This plugin manages learning rate scheduling inside of a strategy using the
    PyTorch scheduler passed to the constructor. The step() method of the
    scheduler is called after each training epoch.
    """

    def __init__(self, scheduler, min_lr=0.00001):
        """
        Creates a ``LRSchedulerPlugin`` instance, step per experience.

        :param scheduler: a learning rate scheduler that can be updated through
            a step() method and can be reset by setting last_epoch=0
        """
        super().__init__()
        self.initial_scheduler = deepcopy(scheduler)
        self.scheduler = deepcopy(self.initial_scheduler)
        self.min_lr = min_lr

    def print_lrs(self):
        print(f"[LR SCHEDULER] Current lrs: "
              f"{['{:.1e}'.format(group['lr']) for group in self.scheduler.optimizer.param_groups]}")

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self.print_lrs()
        
    def after_training_exp(self, strategy, **kwargs):
        # Reset the current optimizer in the scheduler (Should not be neccessary but is...)
        self.scheduler.optimizer = strategy.optimizer 
        if self.scheduler.optimizer.param_groups[0]['lr'] > self.min_lr:
            self.scheduler.step()
        self.print_lrs()


class EpochLRSchedulerPlugin(StrategyPlugin):
    def __init__(self, scheduler):
        super().__init__()
        self.initial_scheduler = deepcopy(scheduler)
        self.scheduler = scheduler

    def print_lrs(self):
        print(f"[LR SCHEDULER] Current lrs: "
              f"{['{:.1e}'.format(group['lr']) for group in self.scheduler.optimizer.param_groups]}")

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        #self.print_lrs()
        return
        
    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self.scheduler.step()
        self.print_lrs()
        return super().after_training_epoch(strategy, **kwargs)

    def after_training_exp(self, strategy, **kwargs):
        # Resetting the scheulder
        self.scheduler = deepcopy(self.initial_scheduler)
        self.print_lrs()


class ItrLRSchedulerPlugin(StrategyPlugin):
    def __init__(self, scheduler):
        super().__init__()
        self.initial_scheduler = deepcopy(scheduler)
        self.scheduler = scheduler

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self.scheduler.step()
        #self.print_lrs()
        return super().after_training_iteration(strategy, **kwargs)

    def after_training_exp(self, strategy, **kwargs):
        # Resetting the scheulder
        self.scheduler = deepcopy(self.initial_scheduler)
        #self.print_lrs()

class OneCycleSchedulerPlugin(StrategyPlugin):
    """
    Basically a wrapper for the OneCycleLR scheduler that is able to handle changes 
    to the optimizer's parameter_groups that happen during begin_training
    """
    def __init__(self, 
                 optimizer,
                 max_lr,
                 start_lr,
                 final_lr, 
                 epochs,
                 warmup_epochs,  
                 three_phase):
        super().__init__()
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.three_phase = three_phase
    
    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        print("exp counter", strategy.training_exp_counter)
        try:
            num_epochs = self.epochs[strategy.training_exp_counter]
        except Exception:
            num_epochs = self.epochs[-1]
        #total_steps = int((num_epochs+1)*((len(strategy.experience.dataset)//strategy.train_mb_size))+1)
        total_steps = int((num_epochs)*(len(strategy.experience.dataset)//strategy.train_mb_size))
        self.scheduler = OneCycleLR(
                    self.optimizer, 
                    max_lr=self.max_lr, 
                    div_factor=(self.max_lr/self.start_lr), # NOTE: initial_lr = max_lr/div_factor
                    final_div_factor=(self.max_lr/self.final_lr), # NOTE: final_lr = max_lr/final_div_factor
                    pct_start=self.warmup_epochs/num_epochs, 
                    anneal_strategy='cos', # NOTE: uses cosin annaling
                    total_steps=total_steps,
                    three_phase=False,
                    verbose=False
                )
        #self.scheduler.step()
        return super().before_training_exp(strategy, **kwargs)

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self.scheduler.step()
        #self.print_lrs()
        return super().after_training_iteration(strategy, **kwargs)