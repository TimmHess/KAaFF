#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, Dict, TypeVar
import torch
import torch.nn.functional as F

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

from src.model import _get_classifier
from src.methods.replay import ERPlugin

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')

# NOTE: PluginMetric->GenericPluginMetric->AccuracyPluginMetric
    # ->'SpecificMetric' (in our case this will be the LinearProbingAccuracyMetric)
    # in avalnache this could be, e.g. MinibatchAccuracy...

class DownstreamLinearProbeAccuracyMetric(GenericPluginMetric[float]):
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
    def __init__(self, 
                 args, 
                 downstream_task, 
                 train_set, 
                 eval_set, 
                 n_classes, 
                 criterion, 
                 train_mb_size=32, 
                 eval_mb_size=32,
                 buffer_lp_dataset=True,
                 nomralize_features=False,
                 train_stream_from_ER_buffer=False):
        self._accuracy = Accuracy() # metric calculation container
        super(DownstreamLinearProbeAccuracyMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval')

        self.args = args
        # Init the scenario for the downstream task
        self.downstream_task = downstream_task
        self.train_set = train_set
        self.train_stream_from_ER_buffer = train_stream_from_ER_buffer
        self.eval_set = eval_set
        self.ds_n_classes = n_classes
        self.criterion = criterion
        self.buffer_lp_dataset = buffer_lp_dataset
        self.normalize_features = nomralize_features
    
        self.train_mb_size = train_mb_size
        self.eval_mb_size = eval_mb_size
        self.num_workers = args.num_workers

        self.num_fintune_epochs = args.lp_finetune_epochs

        self.ds_head = None # NOTE: local copy of the model's head used for linear probing
        self.local_optim = None

        self.eval_complete = False
        self.initial_out_features = None

        self.skip_initial_eval = args.skip_initial_eval
        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "Top1_"+self.downstream_task+"_Acc"

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
            return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
        return self._metric.result(0)
        
    @torch.no_grad()
    def prepare_tensordataset(self, model, dataset, device):
        x_reprs = []
        ys = []
        ts = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.train_mb_size, 
                         shuffle=False, num_workers=self.num_workers, drop_last=False) 
        print("Preparing dataset for downstream eval...")
        for _, mbatch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
            
            x = x.to(device)
            y = y.to(device)
            
            # Get representation from backbone
            x_rep = model(x).detach() # detach() is most likely not necessary here but I want to be sure
            if self.normalize_features:
                x_rep = F.normalize(x_rep, dim=1)
            x_reprs.append(x_rep.cpu())
            ys.append(y.cpu())
            ts.append(tid)

        x_reprs = torch.concat(x_reprs)
        ys = torch.concat(ys)
        ts = torch.concat(ts)
        return x_reprs, ys, ts
    
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
            x_rep = strategy.model.feature_extractor(x).detach() # detach() is most likely not necessary here but I want to be sure
            x_rep = x_rep.view(x_rep.shape[0], -1)
            
            out = self.ds_head(x_rep)
            
            # Update the accuracy measure
            self._accuracy.update(out, y, 0) # NOTE: removed tid
        return  
    
    def before_eval(self, strategy: 'BaseStrategy'):
        super().before_eval(strategy)

        print("\nPreparing Down-Stream Linear-Probe for", self.downstream_task)
        # Initialize and prepare the linear probing head
        with torch.enable_grad(): # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            self.ds_head = _get_classifier(classifier_type=self.args.classifier, 
                                           n_classes=self.ds_n_classes, 
                                           feat_size=strategy.model.feature_extractor.feature_size, 
                                           initial_out_features=None, 
                                           task_incr=False, # NOTE: no needed because we will only have 1 task (the downstream task))
                                           lin_bias=self.args.lin_bias)

            # Move novel probe head to common device and (re-)initialize
            self.ds_head = self.ds_head.to(strategy.device)
            self.ds_head.train() # set to train mode (for safety)
            
            # Initialize local optimizer for the new head
            self.local_optim = torch.optim.AdamW(self.ds_head.parameters(), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999))
            
            # Prepare dataet and dataloader
            train_set = self.train_set
            
            if self.train_stream_from_ER_buffer: # Grab train_set from ERPlugin buffer
                for plugin in strategy.plugins:
                    if isinstance(plugin, ERPlugin):
                        lp_dataset = plugin.storage_policy.buffer
                        print(lp_dataset.targets)
                        print("len of buffer:", len(lp_dataset))
                        break
                if lp_dataset is None:
                    raise ValueError("No ERPlugin found in strategy.plugins, or lp_dataset None!")
                train_set = lp_dataset

            if self.buffer_lp_dataset:
                xs, ys, _ = self.prepare_tensordataset(strategy.model.feature_extractor, train_set, strategy.device)
                tensor_train_set = torch.utils.data.TensorDataset(xs, ys)
                lp_dataloader = torch.utils.data.DataLoader(
                                        tensor_train_set, 
                                        batch_size=self.train_mb_size, 
                                        shuffle=True, 
                                        num_workers=self.num_workers, 
                                        drop_last=True
                ) 
            else:
                lp_dataloader = torch.utils.data.DataLoader(
                                        train_set, 
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

                    x_rep = x.to(strategy.device)
                    if not self.buffer_lp_dataset:
                        # Get representation from backbone
                        x_rep = strategy.model.feature_extractor(x).detach() # detach() is most likely not necessary here but I want to be sure
                    y = y.to(strategy.device)
                    
                    out = self.ds_head(x_rep)

                    loss = self.criterion(out, y)
                    loss.backward()
                    self.local_optim.step()                
            print("\nLinear Probe for downstream training complete...\n")
        return

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Evaluate 
        self.eval(strategy)
        return super().after_eval(strategy)


    