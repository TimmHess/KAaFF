from typing import TYPE_CHECKING, Dict, TypeVar
import torch

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

from src.methods.replay import ERPlugin

import numpy as np
import faiss

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')


def compute_embeddings(loader, model, scaler, do_normalize=False):
    """
    Code adapted from: https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/losses.py
    """
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    print("Computing embeddings, with dim: ", model.feature_size)
    total_embeddings = []
    total_labels = []

    #for idx, (images, labels, _) in tqdm(enumerate(loader), total=len(loader)):
    for idx, mbatch in tqdm(enumerate(loader), total=len(loader)):
        images = mbatch[0].cuda()
        labels = mbatch[1]
        #bsz = labels.shape[0]
        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images)
                if do_normalize:
                    embed = torch.nn.functional.normalize(embed, dim=1)
                total_embeddings.append(embed.detach().cpu().numpy())
                total_labels.append(labels.detach().numpy())
        else:
            embed = model(images)
            if do_normalize:
                embed = torch.nn.functional.normalize(embed, dim=1)
            total_embeddings.append(embed.detach().cpu().numpy())
            total_labels.append(labels.detach().numpy())

        del images, labels, embed
        torch.cuda.empty_cache()
    total_embeddings = np.concatenate(total_embeddings, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    return np.float32(total_embeddings), total_labels.astype(int)


class KNNClassifier():
    """
    k := number of neighbors to consider
    use_inner_prduct := flag to indicate whether to use inner-product or L2 distance for nearest-neighbor search


    To use the cosine-metric for neares-neighbor search we need to use inner-product with normalized vectors! 
    Basically, in this case 'X' is expected to be comprised of normalized vectors!
    """

    def __init__(self, k=5, use_inner_prduct=True):
        self.index = None
        self.y = None
        self.k = k

        self.use_inner_product = use_inner_prduct

    def fit(self, X, y):
        if self.use_inner_product:
            self.index = faiss.IndexFlatIP(X.shape[1])
        else:
            self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        if self.index is None:
            print("X shape:", X.shape)
            return np.array([0]*X.shape[0])
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


class DownstreamKKNNAccuracyMetric(GenericPluginMetric[float]):
    def __init__(self, 
                 args, 
                 downstream_task, 
                 train_set, 
                 eval_set, 
                 k,
                 train_stream_from_ER_buffer=False):
        self._accuracy = Accuracy() # metric calculation container
        super(DownstreamKKNNAccuracyMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval')

        self.args = args
        # Init the scenario for the downstream task
        self.downstream_task = downstream_task
        self.train_set = train_set
        self.train_stream_from_ER_buffer = train_stream_from_ER_buffer
        self.eval_set = eval_set


        self.knn_classifier = None
        self.knn_classifiers_k = dict()
        self.knn_acc_k = dict()
        self.k = k # NOTE: list of k's

        self.batch_size = args.bs
        self.num_workers = args.num_workers

        self.num_fintune_epochs = args.lp_finetune_epochs

        self.eval_complete = False
        self.initial_out_features = None

        self.skip_initial_eval = args.skip_initial_eval
        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "Top1_"+self.downstream_task+"_KNN_Acc"

    def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
        #metric_value = self.result(strategy)
        metric_value = self.knn_acc_k

        add_exp = self._emit_at == 'experience'
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = self.__str__() + "/eval_phase/test_stream/k_" + str(k)
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
            self.knn_acc_k = dict()
            
        try: # NOTE: the try-except is a bit hacky, but necessary to avoid crash for initial eval
            self._metric.reset(phase_and_task(strategy)[1])
        except Exception:
            pass
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            print(self._metric.result(task_label=0))
            return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
        return self._metric.result(0)
        

    def before_eval(self, strategy: 'BaseStrategy'):
        super().before_eval(strategy)

        print("\nPreparing KNN classifier for", self.downstream_task)
        # Prepare dataet and dataloader
        train_set = self.train_set

        if self.train_stream_from_ER_buffer: # Grab train_set from ERPlugin buffer
            for plugin in strategy.plugins:
                if isinstance(plugin, ERPlugin):
                    lp_dataset = plugin.storage_policy.buffer
                    print("len of buffer:", len(lp_dataset))
                    break
            if lp_dataset is None:
                raise ValueError("No ERPlugin found in strategy.plugins, or lp_dataset None!")
            train_set = lp_dataset

        dataloader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, 
                    shuffle=True, num_workers=self.num_workers, drop_last=False) 
        # Initialize and prepare KNN classifier
        for k_i in self.k:
            self.knn_classifiers_k[k_i] = KNNClassifier(k=k_i, use_inner_prduct=True)

        # Get the embeddings for the classifier
        embeddings, labels = compute_embeddings(loader=dataloader, model=strategy.model.feature_extractor, scaler=False, do_normalize=True)
        print("Embeddings shape:", embeddings.shape)
        # Train the KNN classifier  
        for k_i in self.k:
            print("fitting with k=", k_i, "...")
            self.knn_classifiers_k[k_i].fit(embeddings, labels)
        print("KNNs prepared...")
        return
    
    
    def eval(self, strategy):
        """
        Runs a custom evaluation loop on the down-stream scenario
        """
        ds_dataloader = torch.utils.data.DataLoader(self.eval_set, batch_size=self.batch_size, 
                                                    shuffle=False, num_workers=self.num_workers, drop_last=False) 

        embeddings, labels = compute_embeddings(loader=ds_dataloader, model=strategy.model.feature_extractor, scaler=False, do_normalize=True)
        for k_i in self.k:
            out = self.knn_classifiers_k[k_i].predict(embeddings)
            # Compare out with labels for accuracy caluclation
            self.knn_acc_k[k_i] = np.sum(out == labels)/len(out)
        return  
    

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Evaluate 
        self.eval(strategy)
        return super().after_eval(strategy)



    