#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, Dict, TypeVar
import torch
import torch.nn.functional as F

from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.training.utils import unfreeze_everything

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task
from avalanche.models.dynamic_modules import MultiHeadClassifier, IncrementalClassifier
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence

from src.model import reinit_weights
from src.methods.replay import ERPlugin

import copy
from collections import OrderedDict

from tqdm import tqdm

import numpy as np
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')

# NOTE: PluginMetric->GenericPluginMetric->AccuracyPluginMetric
    # ->'SpecificMetric' (in our case this will be the LinearProbingAccuracyMetric)
    # in avalnache this could be, e.g. MinibatchAccuracy...

class LinearProbingAccuracyMetric(GenericPluginMetric[float]):
    def __init__(self, 
                 train_stream, 
                 test_stream, 
                 criterion,
                 eval_all=False, 
                 force_task_eval=False,
                 num_finetune_epochs=1,
                 num_head_copies=1, 
                 train_mb_size=32, 
                 num_workers=0, 
                 skip_initial_eval=False, 
                 buffer_lp_dataset=True,
                 normalize_features=False,
                 train_stream_from_ER_buffer=False,
                 reduce_dim_in_head=None,
                 pca_on_subset=False):
        self._accuracy = Accuracy() # metric calculation container
        super(LinearProbingAccuracyMetric, self).__init__(
            self._accuracy, reset_at='experience', emit_at='experience',
            mode='eval')

        self.criterion = criterion
        self.train_mb_size = train_mb_size
        self.num_workers = num_workers
        self.buffer_lp_dataset = buffer_lp_dataset
        self.normalize_features = normalize_features

        self.pca = None
        self.reduce_dim_in_head = reduce_dim_in_head
        self.pca_on_subset = pca_on_subset

        self.train_stream = train_stream
        self.train_stream_from_ER_buffer = train_stream_from_ER_buffer
        self.test_stream = test_stream
        self.num_exps_seen = 0
        self.num_fintune_epochs = num_finetune_epochs

        self.head_copy = None # NOTE: local copy of the model's head used for linear probing
        self.num_head_copies = num_head_copies
        self.head_copies = []
        self.local_optim = None

        self.eval_all = eval_all # flag to indicate forced evaluation on all experiences for each tasks (including yet unseed ones)
        self.force_task_eval = force_task_eval # flag to indicate forced evaluation on all experiences for the current task (including yet unseed ones)
        self.initial_out_features = None

        self.skip_initial_eval = skip_initial_eval
        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "Top1_LP_Acc_Exp"

    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            if self.force_task_eval:
                print(self._metric.result())
                import sys;sys.exit()
                result = self._metric.result(torch.div(strategy.mb_y, self.initial_out_features, rounding_mode='trunc')) 
                return result
            
            return self._metric.result(phase_and_task(strategy)[1])

    @torch.no_grad()
    def prepare_tensordataset(self, model, dataset, device):
        x_reprs = []
        ys = []
        ts = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.train_mb_size, 
                         shuffle=False, num_workers=self.num_workers, drop_last=False) 
        
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

    def update(self, strategy=None):
        task_labels = strategy.mb_task_id
        y = strategy.mb_y

        # Get representation of current mbatch from backbone
        x_rep = strategy.model.last_features.detach()
    
        if self.reduce_dim_in_head:
            x_rep = self.pca.transform(x_rep.cpu())
            x_rep = torch.tensor(x_rep, dtype=torch.float32).to(strategy.device)

        for head_copy in self.head_copies:
            if isinstance(head_copy, MultiTaskModule): # NOTE: this is copied from the 'avalanche_forward'-function 
                out = head_copy(x_rep, task_labels)
            else:  # no task labels
                out = head_copy(x_rep)
            # Update the accuracy measure
            self._accuracy.update(out, y, task_labels)
        return

    def before_eval(self, strategy: 'BaseStrategy'):
        # NOTE: (it is enought to train once because this will include all sub-tasks already)
        print("\nLocked LinearProbe Training")
        # Initialize and prepare the linear probing head
        print("\nPreparing Linear Probe(s)")
        with torch.enable_grad(): # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            print("Initializing new head(s)...")
            # Check number of current heads against max numbre of heads possible
            self.head_copy = copy.deepcopy(strategy.model.classifier)
            self.head_copy.apply(reinit_weights) # NOTE: maybe this can be removed because it is redundant
            # For safety reasons unfreeze everything in the head_copy
            unfreeze_everything(self.head_copy)
            print("\nis multi-headed:", isinstance(self.head_copy, MultiTaskModule))
               
            if self.force_task_eval: # NOTE: Re-initialize self.head_copy as MultiTaskModule for task-wise evaluation
                print("Forcing multi-headed eval!")
                feat_size = self.head_copy.in_features
                self.initial_out_features = len(torch.unique(torch.tensor(self.train_stream[0].dataset.targets))) # NOTE: not the best way because it assumes that the first experience is representative
                lin_bias = self.head_copy.bias is not None
                print("\nReplaced head_copy with MultiHeadModule using feat_size: {} and initial_out_features: {}".format(feat_size, self.initial_out_features))
                self.head_copy = MultiHeadClassifier(in_features=feat_size,
                                            initial_out_features=self.initial_out_features,
                                            use_bias=lin_bias)
                # Force adapation of MultiHeadClassifier by hand.. NOTE: this is needed because original adaptation method does not work with CI datasets
                for exp_id, _ in enumerate(self.train_stream):
                    tid = str(exp_id)  # need str keys
                    if tid not in self.head_copy.classifiers:
                        new_head = IncrementalClassifier(feat_size, self.initial_out_features)
                        self.head_copy.classifiers[tid] = new_head
                        
            if isinstance(self.head_copy, MultiTaskModule): # NOTE: this adds classifiers for every task possible
                if len(self.head_copy.classifiers) < len(self.train_stream):
                    for exp in self.train_stream:
                        self.head_copy.adaptation(exp.dataset)

            self.head_copies = []
            for copy_id in range(self.num_head_copies):
                self.head_copies.append(copy.deepcopy(self.head_copy))
                # Move novel probe head to common device and (re-)initialize
                self.head_copies[copy_id] = self.head_copies[copy_id].to(strategy.device)
                self.head_copies[copy_id].apply(reinit_weights) #initialize_weights(self.head_copies[copy_id])
                self.head_copies[copy_id].train() # set to train mode (for safety)
            
            # Prepare dataet and dataloader
            if self.eval_all: # NOTE: Override the number of experiences to use in each step with max value
                self.num_exps_seen = len(self.train_stream) -1 # -1 to make up for +1 in next step
                print("\nNum seen experiences is maxed out!")
            
            lp_datasets = []
            lp_dataset = None
            if self.train_stream_from_ER_buffer:
                for plugin in strategy.plugins:
                    if isinstance(plugin, ERPlugin):
                        lp_dataset = plugin.storage_policy.buffer
                        print("len of buffer:", len(lp_dataset))
                        break
                if lp_dataset is None:
                    raise ValueError("No ERPlugin found in strategy.plugins, or lp_dataset None!")
            else:
                curr_exp_data_stream = self.train_stream[:(self.num_exps_seen+1)] # Grab the curent subset of experiences from train_stream
                #curr_exp_data = []

                # Prepare dimensionality reduction if requested
                if self.reduce_dim_in_head:
                    curr_exp_data = []
                    for i, exp in enumerate(curr_exp_data_stream):
                        if self.pca_on_subset and i == strategy.training_exp_counter:
                            break
                        curr_exp_data.append(exp.dataset.eval())
                    pca_dataset = torch.utils.data.ConcatDataset(curr_exp_data)
                    xs, _, _ = self.prepare_tensordataset(
                                        model=strategy.model, # NOTE: because I need all backbones 
                                        dataset=pca_dataset, 
                                        device=strategy.device,
                                        normalize_features=(self.normalize_features)
                    )
                    print(xs.shape)
                    print("\nsklearn pca")
                    pca = PCA(n_components=self.reduce_dim_in_head, whiten=self.normalize_features)
                    if self.pca_on_subset:
                        print("pca fitted on subset...", len(pca_dataset))
                        pca.fit(xs)
                    else:
                        pca.fit(xs)
                    # Store the principle component fit (needed in eval)
                    self.pca = pca
                    print("pca fitted...")

                # Create a ConcatDataset and respective Dataloader
                for i, exp in enumerate(curr_exp_data_stream):
                    if self.force_task_eval:
                        print("Forcing task labels for eval!")
                        task_labels = ConstantSequence(i, len(exp.dataset))
                        exp.dataset.targets_task_labels = task_labels
                    
                    lp_dataset = exp.dataset.eval()
                    if self.buffer_lp_dataset:
                        # Prepare tensor dataset to prevent massive compute overhead
                        xs, ys, ts = self.prepare_tensordataset(
                                            model=strategy.model.feature_extractor, 
                                            dataset=lp_dataset, 
                                            device=strategy.device
                        )
                        # Reduce dim by pca if requested
                        if self.reduce_dim_in_head:
                            xs = self.pca.transform(xs.cpu())
                            xs = torch.tensor(xs, dtype=torch.float32)#.to(strategy.device)
                        lp_dataset = torch.utils.data.TensorDataset(xs, ys, ts)
                    lp_datasets.append(lp_dataset)
            print("lp datasets", len(lp_datasets),"\n")

            # Initialize local optimizer for the new head
            from torch.optim.lr_scheduler import MultiStepLR
            for _, head_copy in tqdm(enumerate(self.head_copies)):

                for lp_dataset in lp_datasets:
                    lp_dataloader = torch.utils.data.DataLoader(
                                        lp_dataset, 
                                        batch_size=self.train_mb_size, #self.train_mb_size*len(self.train_stream), 
                                        shuffle=True, 
                                        num_workers=self.num_workers, 
                                        drop_last=True, 
                                        pin_memory=True)

                    #self.local_optim = torch.optim.AdamW(head_copy.parameters(), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999))
                    self.local_optim = torch.optim.SGD(head_copy.parameters(), lr=0.1, weight_decay=0.0, momentum=0.9)
                    local_scheduler = MultiStepLR(self.local_optim, milestones=[60,75,90], gamma=0.2)
                    #local_scheduler = MultiStepLR(self.local_optim, milestones=[30,37,45], gamma=0.2)
                
                    for _ in range(self.num_fintune_epochs):
                        for _, mbatch in enumerate(lp_dataloader):
                            self.local_optim.zero_grad()

                            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
                            
                            # On-the-fly update labels and targets for task-incremental learning
                            if self.force_task_eval: 
                                y, tid = y % self.initial_out_features, torch.div(y, self.initial_out_features, rounding_mode='trunc') # NOTE: This assumes that the number of classes per head is constant!

                            x_rep = x.to(strategy.device)
                            if not self.buffer_lp_dataset:
                                # Get representation from backbone
                                x_rep = strategy.model.feature_extractor(x).detach() # detach() is most likely not necessary here but I want to be sure
                            
                            y = y.to(strategy.device)
                            
                            # Forward representation through new head 
                            if isinstance(head_copy, MultiTaskModule): # NOTE: this is the avalanche_forward function copied
                                out = head_copy(x_rep, tid)
                            else:  # no task labels
                                out = head_copy(x_rep)

                            loss = self.criterion(out, y)
                            loss.backward()
                            self.local_optim.step()
                        local_scheduler.step()
            print("\nLinear Probe training complete...")
        
            # Reset the task_lables in the dataset to avoid interference with training
            if self.force_task_eval:
                for i, exp in enumerate(curr_exp_data_stream):
                    task_labels = ConstantSequence(0, len(exp.dataset))
                    exp.dataset.targets_task_labels = task_labels

        super().before_eval(strategy) # NOTE: this will do the reset of results etc.
        return

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Increase the counter on seen experiences
        self.num_exps_seen += 1 
        return


    