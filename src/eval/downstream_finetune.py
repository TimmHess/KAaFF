#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, Dict, TypeVar
import torch
import torch.nn.functional as F

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

from src.model import freeze_from_to

import copy
from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')


class DownstreamFinetuneAccuracyMetric(GenericPluginMetric[float]):
    """
    Downstream finetune accuracy metric.
    NOTE: Finetuned model needs to have head with same number of classes as the downstream task
    """
    def __init__(self, 
                 args, 
                 downstream_task, 
                 train_set, 
                 eval_set, 
                 n_classes, 
                 criterion, 
                 train_mb_size, 
                 eval_mb_size,
                 buffer_lp_dataset=True,
                 nomralize_features=False,
                 freeze_from=None,
                 freeze_to=None):
        self._accuracy = Accuracy() # metric calculation container
        super(DownstreamFinetuneAccuracyMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval')

        self.args = args
        # Init the scenario for the downstream task
        self.downstream_task = downstream_task
        self.train_set = train_set
        self.eval_set = eval_set
        self.ds_n_classes = n_classes
        self.criterion = criterion
        self.buffer_lp_dataset = buffer_lp_dataset
        self.normalize_features = nomralize_features
    
        self.train_mb_size = train_mb_size
        self.eval_mb_size = eval_mb_size
        
        self.num_workers = args.num_workers
        self.num_fintune_epochs = args.lp_finetune_epochs

        self.model_copy = None
        self.freeze_from_layer_name = freeze_from
        self.freeze_to_layer_name = freeze_to
        self.local_optim = None
        return

    def __str__(self):
        return "Top1_DS_"+self.downstream_task+"_Acc"

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

        try: # NOTE: the try-except is a bit hacky, but necessary to avoid crash for initial eval
            self._metric.reset(phase_and_task(strategy)[1])
        except Exception:
            pass
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            #print(self._metric.result(task_label=0))
            return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
        return self._metric.result(0)
    

    def eval(self, strategy):
        """
        Runs a custom evaluation loop on the down-stream scenario
        """
        # Init datalaoder for the down-stream task
        ds_dataloader = torch.utils.data.DataLoader(
                                self.eval_set, 
                                batch_size=self.eval_mb_size, 
                                shuffle=False, 
                                num_workers=self.num_workers, 
                                drop_last=False
        ) 
        
        for _, mbatch in tqdm(enumerate(ds_dataloader), total=len(ds_dataloader)):
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

            x = x.to(strategy.device)
            y = y.to(strategy.device)
            
            # Get representation from backbone
            out = self.model_copy(x)
            
            # Update the accuracy measure
            self._accuracy.update(out, y, 0) # NOTE: removed tid
        return  
    
    def _prepare_model(self, model):
        model_copy = copy.deepcopy(model)
        # freeze specified layers
        freeze_from_to(model=self.model_copy, 
                       freeze_from_layer=self.freeze_from_layer_name, 
                       freeze_until_layer=self.freeze_to_layer_name) 
        return model_copy
    
    def before_eval(self, strategy: 'BaseStrategy'):
        super().before_eval(strategy)

        # Prepare copy of current model (backbone + head)
        self.model_copy = self._prepare_model(strategy.model)

        # Train non-frozen layers (entire train-set)
        with torch.enable_grad(): # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            self.local_optim = torch.optim.AdamW(self.model_copy.parameters(), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999)) # TODO: implement get optim function
        
            lp_dataloader = torch.utils.data.DataLoader(
                self.train_set, 
                batch_size=self.train_mb_size, 
                shuffle=True, 
                num_workers=self.num_workers, 
                drop_last=True
            ) 

            # Run local optimization
            for _ in tqdm(range(self.num_fintune_epochs)):
                for _, mbatch in enumerate(lp_dataloader):
                    self.local_optim.zero_grad()

                    x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                    y = y.to(strategy.device)
                    
                    out = self.model_copy(x)

                    loss = self.criterion(out, y)
                    loss.backward()
                    self.local_optim.step()                
            print("\nDownstream training complete...\n")
        return

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Evaluate 
        self.eval(strategy)
        return super().after_eval(strategy)


# Class DownstreamCorruptionPlugin

class CorruptTransformsPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()
        return
    
    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        from torchvision import transforms
        print("experience dataset type:", type(strategy.experience.dataset._original_dataset.transform_groups)) # .replace_transforms()
        print("")
        print(strategy.experience.dataset._original_dataset.transform_groups["train"][0].transforms[1])
        # Access transform groups
        strategy.experience.dataset._original_dataset.transform_groups["eval"][0].transforms[0] = \
        transforms.Compose([transforms.Resize(size=(32, 32))])

        strategy.train_epochs = 0
        # This works...
        # TODO: How to know where to add the corruption?
        return