#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, Dict, TypeVar
from collections import deque, defaultdict
import torch

from avalanche.models.dynamic_modules import MultiTaskModule

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.loss import Loss
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task
from avalanche.models.utils import avalanche_forward
from avalanche.models.dynamic_modules import MultiHeadClassifier, IncrementalClassifier

from src.model import _get_classifier
from src.model import ConcatFeatClassifierModel

from src.eval.continual_eval import ContinualEvaluationPhasePlugin # NOTE: used to call 'get_strategy_state' and 'restore_strategy_'

import copy

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')

# NOTE: PluginMetric->GenericPluginMetric->AccuracyPluginMetric
    # ->'SpecificMetric' (in our case this will be the LinearProbingAccuracyMetric)
    # in avalnache this could be, e.g. MinibatchAccuracy...

class ConcatDownStreamProbingAccuracyMetric(GenericPluginMetric[float]):
    """
    Evaluation plugin for down-stream tasks.

    Params:
        down_stream_task: The task to evaluate on
        scenario_loader: The scenario loader to use for the down-stream task, i.e. 'get_scenario' from helper.py # NOTE: this can't be importet due to cyclic dependece here..
        num_finetune_epochs: Number of epochs to finetune the model on the down-stream task
        batch_size: Batch size to use for the down-stream task
        num_workers: Number of workers to use for the down-stream task
        skip_initial_eval: If True, the initial evaluation on the down-stream task is skipped   
    """
    def __init__(self, args, downstream_task, train_set, eval_set, n_classes): #args, downstream_task, dset_rootpath, scenario_loader):
        self._accuracy = Accuracy() # metric calculation container
        super(ConcatDownStreamProbingAccuracyMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval')

        self.args = args
        self.downstream_task = downstream_task
        self.train_set = train_set
        self.eval_set = eval_set
        self.ds_n_classes = n_classes


        self.batch_size = args.bs
        self.num_workers = args.num_workers

        self.num_fintune_epochs = args.lp_finetune_epochs

        self.reduce_dim_in_head = args.reduce_dim_in_head
        self.reduce_layer = None
        
        self.combined_head = None # NOTE: local copy of the model's head used for linear probing
        self.local_optim = None

        self.eval_complete = False
        self.initial_out_features = None

        self.skip_initial_eval = args.skip_initial_eval
        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "Top1_Concatenated_"+self.downstream_task+"_Acc"

    def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
        metric_value = self.result(strategy)
        
        add_exp = self._emit_at == 'experience'
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
            return metrics
        
        metric_name = get_metric_name(self, strategy,
                                        add_experience=add_exp,
                                        add_task=True)
        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]


    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            print(self._metric.result(task_label=0))
            return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
        return self._metric.result(0)
        
    def before_training(self, strategy: 'BaseStrategy'):
        # Sanity check this plugin can be used - calling strategy has to be a SeparatedNetwork strategy
        assert isinstance(strategy.model, ConcatFeatClassifierModel), "This plugin only works with ConcatFeatClassifierModel!"

        super().before_training(strategy)
        return

    def eval(self, strategy):
        """
        Runs a custom evaluation loop on the down-stream scenario
        """
        # Init datalaoder for the down-stream task
        ds_dataloader = torch.utils.data.DataLoader(self.eval_set, 
                                                    batch_size=self.batch_size, 
                                                    shuffle=False, 
                                                    num_workers=self.num_workers, 
                                                    drop_last=False) 
        
        print("\nEval concatenated representation on downstream task - ", self.downstream_task)
        for _, mbatch in tqdm(enumerate(ds_dataloader), total=len(ds_dataloader)):
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

            x = x.to(strategy.device)
            y = y.to(strategy.device)
            
            # Get representation from backbone
            x_rep_concat = strategy.model.forward_all_feats(x).detach()
            if self.reduce_dim_in_head:
                x_rep_concat = self.reduce_layer(x_rep_concat)
            out = self.combined_head(x_rep_concat)
            
            # Update the accuracy measure
            self._accuracy.update(out, y, 0) # NOTE: removed tid
        return  
    
    def before_eval(self, strategy: 'BaseStrategy'):
        super().before_eval(strategy)

        # Initialize and prepare the linear probing head
        with torch.enable_grad(): # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            lp_dataloader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, 
                        shuffle=True, num_workers=self.num_workers, drop_last=True) 
        
            # Prepare the linear-probing head
            num_input_features = strategy.model.feature_extractor.feature_size * len(strategy.model.feature_extractors) # One full representation for each 
            print("num_input_features for combined representation:", num_input_features)

            if self.reduce_dim_in_head:
                reduced_dim = strategy.model.feature_extractor.feature_size
                self.reduce_layer = torch.nn.Linear(num_input_features, reduced_dim)
                self.reduce_layer = self.reduce_layer.to(strategy.device)
                self.reduce_layer.train()

                self.combined_head = _get_classifier(
                    classifier_type="linear", 
                    n_classes=self.ds_n_classes, 
                    feat_size=reduced_dim, 
                    initial_out_features=None, 
                    task_incr=False) # NOTE: no needed because we will only have 1 task (the downstream task)
            else: 
                self.combined_head = _get_classifier(
                    classifier_type="linear", 
                    n_classes=self.ds_n_classes, 
                    feat_size=num_input_features, 
                    initial_out_features=None, 
                    task_incr=False)

            print("combined_head:", self.combined_head)

            # Move novel probe head to common device and (re-)initialize
            self.combined_head = self.combined_head.to(strategy.device)
            self.combined_head.train() # set to train mode (for safety)
            
            # Initialize local optimizer for the new head
            self.local_optim = None
            if self.reduce_dim_in_head:
                params = list(self.reduce_layer.parameters()) + list(self.combined_head.parameters())
                self.local_optim = torch.optim.AdamW(params, lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999))
            else:
                self.local_optim = torch.optim.AdamW(self.combined_head.parameters(), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999))

            for _ in tqdm(range(self.num_fintune_epochs)):
                for _, mbatch in enumerate(lp_dataloader):
                    self.local_optim.zero_grad()

                    x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                    x = x.to(strategy.device)
                    y = y.to(strategy.device)
                    
                    x_rep_concat = strategy.model.forward_all_feats(x).detach()
                    if self.reduce_dim_in_head:
                        x_rep_concat = self.reduce_layer(x_rep_concat)
                    out = self.combined_head(x_rep_concat)

                    loss = strategy._criterion(out, y)
                    loss.backward()
                    self.local_optim.step()                
            print("\nLinear Probe for downstream training complete...\n")
        return

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Evaluate 
        self.eval(strategy)

        # In case of initial evaluation, do not increase the exp_seen counter
        if self.is_initial_eval_run:
            # Reset the state of the continual learner
            assert(not self._prev_state is None)
            assert(not self._prev_training_modes is None)
            ContinualEvaluationPhasePlugin.restore_strategy_(strategy, self._prev_state, self._prev_training_modes)
            self.is_initial_eval_run = False # Reset flag
        return super().after_eval(strategy) # NOTE: This return is maximally necessary because otherwise it will not log porperly


    