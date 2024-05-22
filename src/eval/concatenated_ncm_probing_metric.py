from typing import TYPE_CHECKING, Dict, TypeVar
import torch

from avalanche.models.dynamic_modules import MultiTaskModule

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset, \
    AvalancheConcatDataset

from src.model import ConcatFeatClassifierModel, ExRepMultiHeadClassifier

from src.eval.downstream_knn import KNNClassifier
from src.methods.replay import ERPlugin

from src.eval.continual_eval import ContinualEvaluationPhasePlugin

import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')

# NOTE: PluginMetric->GenericPluginMetric->AccuracyPluginMetric
    # ->'SpecificMetric' (in our case this will be the LinearProbingAccuracyMetric)
    # in avalnache this could be, e.g. MinibatchAccuracy...


def compute_embeddings(loader, strategy, scaler, do_normalize=False, mean=False):
    """
    Code adapted from: https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/losses.py
    """
    # NOTE: it's okay to do len(loader) * bs, since drop_last=True is enabled
    print("Computing embeddings, with dim: ", 
            strategy.model.feature_extractor.feature_size*len(strategy.model.feature_extractors))
    total_embeddings = []
    total_labels = []

    for idx, (images, labels, _) in tqdm(enumerate(loader), total=len(loader)):
        images = images.cuda()
        bsz = labels.shape[0]
        
        embed = strategy.model.forward_all_feats(images).detach()
        if do_normalize:
            embed = torch.nn.functional.normalize(embed, dim=1)
        total_embeddings.append(embed.detach().cpu().numpy())
        total_labels.append(labels.detach().numpy())

        del images, labels, embed
        torch.cuda.empty_cache()
    total_embeddings = np.concatenate(total_embeddings, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)

    # Compute mean embeddings per class
    if mean:
        mean_embeddings = []
        unique_labels = []
        print("unique labels:", np.unique(total_labels))
        for cl in np.unique(total_labels):
            #print(total_embeddings[total_labels==cl].shape)
            mean_embeddings.append(np.mean(total_embeddings[total_labels==cl], axis=0))
            unique_labels.append(cl)
        mean_embeddings = np.stack(mean_embeddings, axis=0)
        return np.float32(mean_embeddings), np.asarray(unique_labels, dtype=int)
    
    return np.float32(total_embeddings), total_labels.astype(int)


class ConcatenatedNCMProbingAccuracyMetric(GenericPluginMetric[float]):
    def __init__(self, 
                 train_stream, 
                 test_stream, 
                 k, 
                 eval_all=False, 
                 force_task_eval=False,
                 num_finetune_epochs=1, 
                 train_mb_size=32, 
                 num_workers=0, 
                 skip_initial_eval=False,
                 train_stream_from_ER_buffer=False,
                 reduce_dim_in_head=None,
                 pca_on_subset=False):
        self._accuracy = Accuracy() # metric calculation container
        super(ConcatenatedNCMProbingAccuracyMetric, self).__init__(
            self._accuracy, reset_at='experience', emit_at='experience',
            mode='eval')

        self.train_mb_size = train_mb_size
        self.num_workers = num_workers

        self.reduce_dim_in_head = reduce_dim_in_head
        self.pca_on_subset = pca_on_subset
        self.pca = None

        self.train_stream = train_stream
        self.train_stream_from_ER_buffer = train_stream_from_ER_buffer
        self.test_stream = test_stream
        self.num_exps_seen = 0
        self.k = k

        self.knn_classifier = None
        self.knn_acc_k = dict()
        for k_i in self.k:
            self.knn_acc_k[k_i] = Accuracy()

        self.eval_all = eval_all # flag to indicate forced evaluation on all experiences for each tasks (including yet unseed ones)
        self.force_task_eval = force_task_eval # flag to indicate forced evaluation on all experiences for the current task (including yet unseed ones)
        self.initial_out_features = None

        self.skip_initial_eval = skip_initial_eval
        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "Top1_NCM_Acc_Exp"

    def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
        #metric_value = self.result(strategy)        
        add_exp = self._emit_at == 'experience'
        plot_x_position = strategy.clock.train_iterations

        metrics = []
        #for exp_i in range(self.num_exps_seen):
        for k_i, acc in self.knn_acc_k.items(): # NOTE: v is an Accuracy() instance
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            #metric_name += "/k_{}".format(k_i)
            metric_value = acc.result(task_label=strategy.experience.current_experience)
            metric_value = metric_value[list(metric_value.keys())[0]]

            metrics.append(MetricValue(self, metric_name, metric_value,
                                        plot_x_position)
            )
        return metrics


    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
            for k_i in self.k:
                self.knn_acc_k[k_i].reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            if self.force_task_eval:
                result = self._metric.result(torch.div(strategy.mb_y, self.initial_out_features, rounding_mode='trunc')) 
                return result
            
            return self._metric.result(phase_and_task(strategy)[1])


    def before_training(self, strategy: 'BaseStrategy'):
        assert isinstance(strategy.model, ConcatFeatClassifierModel), "This plugin only works with ConcatFeatClassifierModel!"
        print("Found ConcatFeatClassifierModel!")
        return super().before_training(strategy)


    def before_eval(self, strategy: 'BaseStrategy'):
        print("\nLocked KNN Preparation Phase")
        # Initialize and prepare the linear probing head
        with torch.enable_grad(): # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            # Reset all KNN classifiers
            self.knn_classifier = dict()
            if isinstance(strategy.model.classifier, MultiTaskModule): # NOTE: this adds classifiers for every task possible
                for exp_idx,_ in enumerate(self.train_stream):
                    self.knn_classifier[exp_idx] = dict()#KNNClassifier(k=self.k)
                    for k_i in self.k:
                        self.knn_classifier[exp_idx][k_i] = KNNClassifier(k=k_i)
            else:
                for k_i in self.k:
                    self.knn_classifier[k_i] = KNNClassifier(k=k_i)
            
            # Basically the reset for the accuracy metric
            for k_i in self.k:
                self.knn_acc_k[k_i] = Accuracy()

            # Prepare dataet and dataloader
            if self.eval_all: # NOTE: Override the number of experiences to use in each step with max value
                self.num_exps_seen = len(self.train_stream) -1 # -1 to make up for +1 in next step
                print("\nNum seen experiences is maxed out!")

            # Prepare stream subset from ER buffer
            if self.train_stream_from_ER_buffer:
                lp_dataset = None
                for plugin in strategy.plugins:
                    if isinstance(plugin, ERPlugin):
                        lp_dataset = plugin.storage_policy.buffer  #NOTE: buffer -> AvalancheDataset
                        break
                if lp_dataset is None:
                    raise ValueError("No ERPlugin found in strategy.plugins!")
                
                if(isinstance(strategy.model.classifier, MultiTaskModule)):
                    # Get sample idxs per class
                    cl_idxs = {}
                    for idx, _ in enumerate(lp_dataset.targets):
                        if lp_dataset.targets_task_labels[idx] not in cl_idxs:
                            cl_idxs[lp_dataset.targets_task_labels[idx]] = []
                        cl_idxs[lp_dataset.targets_task_labels[idx]].append(idx)
                    print("cl_idxs:")
                    print(cl_idxs)
                    
                    for exp_idx in cl_idxs.keys():
                        dataloader = torch.utils.data.DataLoader(
                                        AvalancheSubset(lp_dataset, indices=cl_idxs[exp_idx]).eval(),
                                        batch_size=self.train_mb_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        drop_last=False
                        ) 
                        # Get embeddings and labels
                        embeddings, labels = compute_embeddings(loader=dataloader, 
                                                strategy=strategy, 
                                                scaler=False, 
                                                do_normalize=True,
                                                mean=True
                                            )
                        print("Training KNN for task: ", exp_idx)
                        for k_i in self.k:
                            self.knn_classifier[exp_idx][k_i].fit(embeddings, labels)
                        print("KNNs task {} prepared..".format(exp_idx))
                else:
                    dataloader = torch.utils.data.DataLoader(
                                        lp_dataset.eval(),
                                        batch_size=self.train_mb_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        drop_last=False
                        ) 
                    # Get embeddings and labels
                    embeddings, labels = compute_embeddings(loader=dataloader, 
                                            strategy=strategy, 
                                            scaler=False, 
                                            do_normalize=True,
                                            mean=True
                                        )
                    for k_i in self.k:
                        self.knn_classifier[k_i].fit(embeddings, labels)
                    print("KNNs prepared..")
            
            # Prepare stream subset from train stream
            else:
                curr_exp_data_stream = self.train_stream[:(self.num_exps_seen+1)] # Grab the curent subset of experiences from train_stream
                
                if self.reduce_dim_in_head:
                    print("prepareing PCA...")
                    pca_embeddings = []
                    if self.pca_on_subset:
                        exp_subset_stream = self.train_stream[:strategy.training_exp_counter]
                        #print("num_subsets", len(exp_subset_stream))
                    for exp_idx, exp in enumerate(exp_subset_stream):
                        #print("exp_idx:", exp_idx)
                        dataloader = torch.utils.data.DataLoader(
                                        exp.dataset.eval(), 
                                        batch_size=self.train_mb_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        drop_last=False)
                        
                        # Get embeddings and labels
                        exp_embeds, _ = compute_embeddings(loader=dataloader, 
                                                strategy=strategy, 
                                                scaler=False, 
                                                do_normalize=True,
                                                mean=False)
                        pca_embeddings.append(exp_embeds)
                    pca_embeddings = np.concatenate(pca_embeddings, axis=0)
                    
                    # Prepare 
                    self.pca = PCA(n_components=self.reduce_dim_in_head, whiten=True)
                    self.pca.fit(pca_embeddings)

                # Do KNN preparation
                for exp_idx, exp in enumerate(curr_exp_data_stream):
                    dataloader = torch.utils.data.DataLoader(
                                    exp.dataset.eval(), 
                                    batch_size=self.train_mb_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    drop_last=False)
                    
                    # Get embeddings and labels
                    embeddings, labels = compute_embeddings(loader=dataloader, 
                                            strategy=strategy, 
                                            scaler=False, 
                                            do_normalize=True,
                                            mean=True)
                    
                    # Reduce dim in head if requested
                    if self.reduce_dim_in_head:
                        embeddings = self.pca.transform(embeddings)
                        print("embeddings.shape after reduce layer:", embeddings.shape)

                    # Train the KNN classifier  
                    if(isinstance(strategy.model.classifier, MultiTaskModule)):
                        print("Training KNN for task: ", exp_idx)
                        for k_i in self.k:
                            self.knn_classifier[exp_idx][k_i].fit(embeddings, labels)
                        print("KNNs task {} prepared..".format(exp_idx))
                    else:
                        for k_i in self.k:
                            self.knn_classifier[k_i].fit(embeddings, labels)
            print("\nKNN preparation complete...")

        super().before_eval(strategy) # NOTE: this will do the reset of results etc.
        return


    def before_eval_exp(self, strategy: 'BaseStrategy'):
        #super().before_eval_exp(strategy) # TODO: need to fix the resetting of the metric
        lp_dataloader = strategy.dataloader

        for _, mbatch in tqdm(enumerate(lp_dataloader), total=len(lp_dataloader)):
                x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                task_label = torch.unique(tid)
                assert len(task_label) == 1, "Only one task per batch is supported"
                task_label = task_label[0].item()

                x = x.to(strategy.device)
                #y = y.to(strategy.device)

                # Get representation from backbone
                x_rep_concat = []
                # Get representations from backbone
                x_rep_concat = strategy.model.forward_all_feats(x).detach() # detach() is most likely not necessary here but I want to be sure

                #print("x_rep_concat.shape:", x_rep_concat.shape)
                #x_rep_concat = x_rep_concat.view(x_rep_concat.shape[0], -1) # TODO: is this even necessary?
                x_rep_concat = torch.nn.functional.normalize(x_rep_concat, dim=1)
                x_rep_concat = x_rep_concat.cpu().numpy()

                # Reduce dim in head if requested
                if self.reduce_dim_in_head:
                    x_rep_concat = self.pca.transform(x_rep_concat)
                print("x_rep_concat.shape:", x_rep_concat.shape)
                
                # Forward representation through  all knns
                for k_i in self.k:
                    if isinstance(strategy.model.classifier, MultiTaskModule): # NOTE: this is copied from the 'avalanche_forward'-function 
                        out = self.knn_classifier[task_label][k_i].predict(x_rep_concat)
                    else:  # no task labels
                        out = self.knn_classifier[k_i].predict(x_rep_concat)
                    
                    self.knn_acc_k[k_i].update(out, y, task_label)
                # NOTE: original self._accuracy does nothing!
                return
                
                # for head_copy in self.head_copies:
                #     if isinstance(head_copy, MultiTaskModule): # NOTE: this is copied from the 'avalanche_forward'-function 
                #         out = head_copy(x_rep_concat, tid)
                #     else:  # no task labels
                #         out = head_copy(x_rep_concat)
                
                #     self._accuracy.update(out, y, tid)
        return 
        
    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Do reset of metric here that is normaly done in "before_eval_exp"
        # if self._reset_at == 'experience' and self._mode == 'eval':
        #     self.reset(strategy)
        self._accuracy.reset()

        # In case of initial evaluation, do not increase the exp_seen counter
        if self.is_initial_eval_run:
            # Reset the state of the continual learner
            assert(not self._prev_state is None)
            assert(not self._prev_training_modes is None)
            ContinualEvaluationPhasePlugin.restore_strategy_(strategy, self._prev_state, self._prev_training_modes)
            self.is_initial_eval_run = False # Reset flag
        # Increase the counter on seen experiences
        self.num_exps_seen += 1 
        return



    # def update(self, strategy=None):
    #     task_label = torch.unique(strategy.mb_task_id)
    #     assert len(task_label) == 1, "Only one task per batch is supported"
    #     task_label = task_label[0].item()
    #     y = strategy.mb_y

    #     # Get representation of current mbatch from backbone
    #     x_rep = strategy.model.last_features.detach()
    #     x_rep = x_rep.view(x_rep.shape[0], -1)
    #     x_rep = torch.nn.functional.normalize(x_rep, dim=1)
    #     x_rep = x_rep.cpu().numpy()
        
    #     # Forward representation through  all knns
    #     for k_i in self.k:
    #         if isinstance(strategy.model.classifier, MultiTaskModule): # NOTE: this is copied from the 'avalanche_forward'-function 
    #             out = self.knn_classifier[task_label][k_i].predict(x_rep)
    #         else:  # no task labels
    #             out = self.knn_classifier[k_i].predict(x_rep)
            
    #         self.knn_acc_k[k_i].update(out, y, strategy.mb_task_id)
    #     # NOTE: original self._accuracy does nothing!
    #     return

    # def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
    #     # Increase the counter on seen experiences
    #     self.num_exps_seen += 1 
    #     return


    