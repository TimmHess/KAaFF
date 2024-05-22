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


class PCAProbingMetric(GenericPluginMetric[float]):
    def __init__(self,  args, downstream_task, eval_set, pca_threshold=0.99):
        self._accuracy = Accuracy() # metric calculation container
        super(PCAProbingMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval')

        self.args = args
        self.downstream_task = downstream_task
        self.eval_set = eval_set
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

    # def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
    #     #metric_value = self.result(strategy)
    #     metric_value = self.last_result_value

    #     add_exp = self._emit_at == 'experience'
    #     plot_x_position = strategy.clock.train_iterations

    #     if isinstance(metric_value, dict):
    #         metrics = []
    #         for k, v in metric_value.items():
    #             metric_name = get_metric_name(
    #                 self, strategy, add_experience=add_exp, add_task=k)
    #             metrics.append(MetricValue(self, metric_name, v,
    #                                        plot_x_position))
    #         return metrics
        
    #     metric_name = get_metric_name(self, strategy,
    #                                     add_experience=add_exp,
    #                                     add_task=True)
    #     print("metric_name: ", metric_name)
    #     return [MetricValue(self, metric_name, metric_value,
    #                         plot_x_position)]

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
        print("metric_name: ", metric_name)
        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]

    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        try:
            self._metric.reset(phase_and_task(strategy)[1])
        except Exception:
            pass
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            print(self._metric.result(task_label=0))
            return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
        return self._metric.result(0)

    def get_barplot(self, data, ylim=None, put_lines = None):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        sns.barplot(x=list(range(len(data))), y=data, ax=ax)
        #plt.locator_params(axis='x', nbins=len(data)//10)
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::len(xticks) // 10])
        if ylim:
            ax.set_ylim(ylim)
        if put_lines:
            for k,v in put_lines.items():
                plt.axvline(x=v, color='red', linestyle='--')
                plt.text(v+0.1,0,k,rotation=90)
        plt.xlabel('Singular Values')
        plt.ylabel('Magnitude')
        fig.tight_layout()
        return fig

    def compute_embeddings(self, loader, strategy, scaler, do_normalize=False):
        # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
        feat_size = strategy.model.feature_extractor.feature_size
        total_embeddings = np.zeros((len(loader)*loader.batch_size, feat_size))
        total_labels = np.zeros(len(loader)*loader.batch_size)

        for idx, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.cuda()
            bsz = labels.shape[0]
            if scaler:
                with torch.cuda.amp.autocast():
                    embed = strategy.model.feature_extractor(images)
                    if do_normalize:
                        embed = torch.nn.functional.normalize(embed, dim=1)
                    total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
                    total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()
            else:
                embed = strategy.model.feature_extractor(images)
                if do_normalize:
                    embed = torch.nn.functional.normalize(embed, dim=1)
                total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
                total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()

            del images, labels, embed
            torch.cuda.empty_cache()

        return np.float32(total_embeddings), total_labels.astype(int)


    def eval(self, strategy):
        """
        Runs a custom evaluation loop on the down-stream scenario
        """
        # Init datalaoder for the down-stream task
        ds_dataloader = torch.utils.data.DataLoader(self.eval_set, batch_size=self.batch_size, 
                                                    shuffle=False, num_workers=self.num_workers, drop_last=True) 
        
        embeddings, _ = self.compute_embeddings(loader=ds_dataloader, strategy=strategy, 
                scaler=False, do_normalize=True)
        
        print("\nsklearn pca...")
        pca = PCA(n_components=None, whiten=True)
        pca.fit(embeddings)
        #self.last_result_value = len(pca.singular_values_)
        #print(self.last_result_value)
        self.last_result_value = {}
        self.last_result_value[0] = self.get_barplot(data=pca.singular_values_)
        print("PCA explained variance ratio:")
        print(pca.explained_variance_ratio_)
        # sum up pca.expalined_variance_ratio_ until 90% variance is explained
        sum = 0
        put_lines = {}
        for i in range(len(pca.explained_variance_ratio_)):
            sum += pca.explained_variance_ratio_[i]
            if sum >= 0.9:
                if not "90%" in put_lines:
                    put_lines["90%"] = i
            if sum >= 0.95:
                if not "95%" in put_lines:
                    put_lines["95%"] = i
            if sum >= 0.99:
                if not "99%" in put_lines:
                    put_lines["99%"] = i
                break
        self.last_result_value[1] = self.get_barplot(data=pca.explained_variance_ratio_, 
                                                    ylim=(0, 0.11),
                                                    put_lines=put_lines)
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


    