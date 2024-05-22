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

from src.utils import get_grad_normL2
from src.model import initialize_weights, _get_classifier
from src.methods.separate_networks import SeparateNetworks

from src.eval.continual_eval import ContinualEvaluationPhasePlugin # NOTE: used to call 'get_strategy_state' and 'restore_strategy_'

import numpy as np
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')

# NOTE: PluginMetric->GenericPluginMetric->AccuracyPluginMetric
    # ->'SpecificMetric' (in our case this will be the LinearProbingAccuracyMetric)
    # in avalnache this could be, e.g. MinibatchAccuracy...

def get_concatenated_rep(strategy, x):
    x_rep_concat = []
    for bb_idx in strategy.backbones:
        #x_rep = strategy.model.feature_extractor(x).detach() # detach() is most likely not necessary here but I want to be sure
        x_rep = strategy.backbones[bb_idx](x).detach() # NOTE: backbones is a member of SeparateNetworks class
        x_rep_concat.append(x_rep)  
    x_rep_concat = torch.concat(x_rep_concat, dim=1)
    return x_rep_concat



class ConcatenatedPCAProbingMetric(GenericPluginMetric[float]):
    def __init__(self,  args, downstream_task, train_set, eval_set, n_classes, pca_threshold=0.99):
        self._accuracy = Accuracy() # metric calculation container
        super(ConcatenatedPCAProbingMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval')

        self.args = args
        self.downstream_task = downstream_task
        self.train_set = train_set
        self.eval_set = eval_set
        self.ds_n_classes = n_classes
        self.pca_threshold = pca_threshold
        self.last_result_value = None


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
        return "PCA_"+self.downstream_task

    def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
        #metric_value = self.result(strategy)
        metric_value = self.last_result_value

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
        
        self._metric.reset(phase_and_task(strategy)[1])
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            print(self._metric.result(task_label=0))
            return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
        return self._metric.result(0)
    
    def get_barplot(self, data):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        sns.barplot(x=list(range(len(data))), y=data, ax=ax)
        plt.locator_params(axis='x', nbins=len(data)//10)
        plt.xlabel('Singular Values')
        plt.ylabel('Magnitude')
        fig.tight_layout()
        return fig

    def compute_embeddings(self, loader, strategy, scaler, do_normalize=False):
        # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
        feat_size = strategy.model.feature_extractor.feature_size * len(strategy.backbones)
        if self.reduce_dim_in_head:
            feat_size = strategy.model.feature_extractor.feature_size
        total_embeddings = np.zeros((len(loader)*loader.batch_size, feat_size))
        total_labels = np.zeros(len(loader)*loader.batch_size)

        for idx, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.cuda()
            bsz = labels.shape[0]
            if scaler:
                with torch.cuda.amp.autocast():
                    #embed = model(images)
                    embed = get_concatenated_rep(strategy, images)
                    if self.reduce_dim_in_head:
                        embed = self.reduce_layer(embed)
                    if do_normalize:
                        embed = torch.nn.functional.normalize(embed, dim=1)
                    total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
                    total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()
            else:
                #embed = model(images)
                embed = get_concatenated_rep(strategy, images)
                if self.reduce_dim_in_head:
                    embed = self.reduce_layer(embed)
                if do_normalize:
                    embed = torch.nn.functional.normalize(embed, dim=1)
                total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
                total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()

            del images, labels, embed
            torch.cuda.empty_cache()

        return np.float32(total_embeddings), total_labels.astype(int)

        
    def before_training(self, strategy: 'BaseStrategy'):
        # Sanity check this plugin can be used - calling strategy has to be a SeparatedNetwork strategy
        assert isinstance(strategy, SeparateNetworks), "This plugin can only be used with the SeparateNetworks strategy"

        super().before_training(strategy)
        return


    def before_eval(self, strategy: 'BaseStrategy'):
        super().before_eval(strategy)

        # Initialize and prepare the linear probing head
        with torch.enable_grad(): # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            lp_dataloader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, 
                        shuffle=True, num_workers=self.num_workers, drop_last=True) 
        
            # Prepare the linear-probing head
            num_input_features = strategy.model.feature_extractor.feature_size * strategy.clock.train_exp_counter # One full representation for each 

            if self.reduce_dim_in_head:
                reduced_dim = strategy.model.feature_extractor.feature_size
                self.reduce_layer = torch.nn.Linear(num_input_features, reduced_dim)
                self.reduce_layer = self.reduce_layer.to(strategy.device)
                self.reduce_layer.train()

                self.combined_head = _get_classifier(
                        classifier_type=self.args.classifier, 
                        n_classes=self.ds_n_classes, 
                        feat_size=reduced_dim, 
                        initial_out_features=None, 
                        task_incr=False, # NOTE: no needed because we will only have 1 task (the downstream task)
                        lin_bias=self.args.lin_bias) 
            else:
                self.combined_head = _get_classifier(
                        classifier_type=self.args.classifier, 
                        n_classes=self.ds_n_classes, 
                        feat_size=num_input_features, initial_out_features=None, 
                        task_incr=False,
                        lin_bias=self.args.lin_bias)

            # Move novel probe head to common device and (re-)initialize
            self.combined_head = self.combined_head.to(strategy.device)
            self.combined_head.train() # set to train mode (for safety)
            
            # Initialize local optimizer for the new head
            self.local_optim = None
            if self.reduce_dim_in_head:
                params = list(self.reduce_layer.parameters()) + list(self.combined_head.parameters())
                self.local_optim = torch.optim.AdamW(params, lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999))
            else:
                self.local_optim = torch.optim.AdamW(self.combined_head.parameters(), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999)) # #self.local_optim = torch.optim.Adam(self.head_copy.parameters(), lr=0.01, weight_decay=0.0, betas=(0.9, 0.999))        

            for _ in tqdm(range(self.num_fintune_epochs)):
                for _, mbatch in enumerate(lp_dataloader):
                    self.local_optim.zero_grad()

                    x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                    x = x.to(strategy.device)
                    y = y.to(strategy.device)
                    
                    x_rep_concat = []
                    # Get representations from backbone
                    for bb_idx in strategy.backbones:
                        x_rep = strategy.backbones[bb_idx](x).detach()
                        x_rep_concat.append(x_rep)
                        
                    x_rep_concat = torch.concat(x_rep_concat, dim=1)
                    
                    if self.reduce_dim_in_head:
                        x_rep_concat = self.reduce_layer(x_rep_concat)
                    out = self.combined_head(x_rep_concat)

                    loss = strategy._criterion(out, y)
                    loss.backward()
                    self.local_optim.step()                
            print("\nLinear Probe for downstream training complete...\n")
        return


    def eval(self, strategy):
        """
        Runs a custom evaluation loop on the down-stream scenario
        """
        # Init datalaoder for the down-stream task
        ds_dataloader = torch.utils.data.DataLoader(self.eval_set, batch_size=self.batch_size, 
                                                    shuffle=False, num_workers=self.num_workers, drop_last=False) 
        
        embeddings, _ = self.compute_embeddings(loader=ds_dataloader, strategy=strategy, 
                scaler=False, do_normalize=True)
        
        print("\nsklearn pca")
        pca = PCA(n_components=None)
        pca.fit(embeddings)
        #self.last_result_value = len(pca.singular_values_)
        #print(self.last_result_value)
        self.last_result_value = self.get_barplot(data=pca.singular_values_)
        print("")
        return  
 

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Evaluate 
        self.eval(strategy)

        # In case of initial evaluation, do not increase the exp_seen counter
        # if self.is_initial_eval_run:
        #     # Reset the state of the continual learner
        #     assert(not self._prev_state is None)
        #     assert(not self._prev_training_modes is None)
        #     ContinualEvaluationPhasePlugin.restore_strategy_(strategy, self._prev_state, self._prev_training_modes)
        #     self.is_initial_eval_run = False # Reset flag
        return super().after_eval(strategy) # NOTE: This return is maximally necessary because otherwise it will not log porperly


    