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
from src.model import initialize_weights, reinit_after, freeze_up_to

from src.eval.continual_eval import ContinualEvaluationPhasePlugin # NOTE: used to call 'get_strategy_state' and 'restore_strategy_'

import copy

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')

class LinearProbingLastNAccuracyMetricLastN(GenericPluginMetric[float]):
    def __init__(self, layer_N, train_stream, test_stream, eval_all=False,
                force_task_eval=False, num_finetune_epochs=1, batch_size=32, num_workers=0, skip_initial_eval=False):
        self._accuracy = Accuracy() # metric calculation container
        super(LinearProbingLastNAccuracyMetricLastN, self).__init__(
            self._accuracy, reset_at='experience', emit_at='experience',
            mode='eval')

        self.layer_N = layer_N # NOTE: N is the number of layer to re-train

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_stream = train_stream
        self.test_stream = test_stream
        self.num_exps_seen = 0
        self.num_fintune_epochs = num_finetune_epochs
        
        self.model_copy = None
        self.model_original = None
        self.local_optim = None

        self.training_complete = False

        self.eval_all = eval_all # flag to indicate forced evaluation on all experiences for each tasks (including yet unseed ones)
        self.force_task_eval = force_task_eval # flag to indicate forced evaluation on all experiences for the current task (including yet unseed ones)
        self.initial_out_features = None

        self.skip_initial_eval = skip_initial_eval
        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "Top1_LP_last_" + str(self.layer_N) + "_Acc_Exp"

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
                result = self._metric.result(torch.div(strategy.mb_y, self.initial_out_features, rounding_mode='trunc')) 
                return result
            
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy=None):
        task_labels = strategy.mb_task_id
        y = strategy.mb_y

        # Adjust task_labels if 'forced_task_eval' flag active
        if self.force_task_eval:
            task_labels = torch.div(strategy.mb_y, self.initial_out_features, rounding_mode='trunc') # equivalent to '//' operation
            y = y % self.initial_out_features

        # Get representation of current mbatch from backbone
        x_rep = strategy.model.last_features.detach() # NOTE: At time of this call 'strategy.model' is the version trained by this module
    
        # Forward representation through new head (linear probe)
        if isinstance(self.model_copy.classifier, MultiTaskModule): # NOTE: this is copied from the 'avalanche_forward'-function 
            out = self.model_copy.classifier(x_rep, task_labels) # shouldn_t this be (x_rep, strategy.mb_task_id) ?
        else:  # no task labels
            out = self.model_copy.classifier(x_rep)
        
        # Update the accuracy measure
        self._accuracy.update(out, y, task_labels) # TODO: replace task_labels with strategy.mb_task_id?s
        return

    def before_training_exp(self, strategy: 'BaseStrategy'):
        # Check exp_clock if this is the '0th' exp and do a linear probing step 
        if strategy.clock.train_exp_counter == 0 and not self.skip_initial_eval:
            print("\nDoing linear probing on random-weighted model")
            self.is_initial_eval_run = True # Set flag to indicate that this is a special evaluation run
           
            # Need to store the state of the trainer and model befor running eval inside train-loop
            self._prev_state, self._prev_training_modes = ContinualEvaluationPhasePlugin.get_strategy_state(strategy)
            
            # Trigger the evaluation by calling strategy.eval()? -> This will run entire evaluation... 
            print("Triggering initial evaluation before training starts...")
            strategy.eval(self.test_stream)
        return

    def before_eval_exp(self, strategy: 'BaseStrategy'):
        # Check if LinearProbe already trained
        if not self.training_complete:
            # Set flag that will prevent retraining of LinearProbe for each sub-task 
            # NOTE: (it is enought to train once because this will include all sub-tasks already)
            self.training_complete = True
            print("\nLocked LinearProbe Training")
            # Initialize and prepare the last N layers
            with torch.enable_grad(): # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
                print("\nPreparing layers after", self.layer_N, "for probing")
                # Check number of current heads against max numbre of heads possible
                self.model_original = copy.deepcopy(strategy.model) # NOTE: This will be needed to revert to the original model after probing
                self.model_copy = copy.deepcopy(strategy.model) # NOTE: This is probed for evaluation
                print("\nis multi-headed:", isinstance(self.model_copy.classifier, MultiTaskModule))

                #######
                if self.force_task_eval: 
                    print("Forcing multi-headed eval!")
                    feat_size = self.model_copy.classifier.in_features
                    self.initial_out_features = len(torch.unique(torch.tensor(self.train_stream[0].dataset.targets))) # NOTE: not the best way because it assumes that the first experience is representative
                    lin_bias = self.model_copy.classifier.bias is not None
                    print("\nReplaced head_copy with MultiHeadModule using feat_size: {} and initial_out_features: {}".format(feat_size, self.initial_out_features))
                    self.model_copy.classifier = MultiHeadClassifier(in_features=feat_size,
                                             initial_out_features=self.initial_out_features,
                                             use_bias=lin_bias)
                    # Force adapation of MultiHeadClassifier by hand.. NOTE: this is needed because original adaptation method does not work with CI datasets
                    for exp_id, _ in enumerate(self.train_stream):
                        tid = str(exp_id)  # need str keys
                        if tid not in self.model_copy.classifier.classifiers:
                            new_head = IncrementalClassifier(feat_size, self.initial_out_features)
                            self.model_copy.classifier.classifiers[tid] = new_head
                            #print("\nAdding new heads to Linear Probe")
                #######
                
                if isinstance(self.model_copy.classifier, MultiTaskModule): # NOTE: this adds classifiers for every task possible
                    if len(self.model_copy.classifier.classifiers) < len(self.train_stream):
                        for exp in self.train_stream:
                            self.model_copy.classifier.adaptation(exp.dataset)
   
                # Move novel probe head to common device and (re-)initialize
                self.model_copy.classifier = self.model_copy.classifier.to(strategy.device)
                print("Reinitializing weights of the head...")
                initialize_weights(self.model_copy.classifier)
                self.model_copy.train() # Set entire model to train for safety (will be partially reset to eval in 'freeze_up_to' function)
                print("Freeze backbone up tp layer", self.layer_N)
                freeze_up_to(model=self.model_copy, freeze_until_layer=self.layer_N)
                print("Reinitialize weights after layer", self.layer_N)
                reinit_after(model=self.model_copy.feature_extractor, reinit_after=self.layer_N)
                
                # Initialize local optimizer for the new head
                #self.local_optim = torch.optim.AdamW(self.model_copy.parameters(), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999)) # #self.local_optim = torch.optim.Adam(self.head_copy.parameters(), lr=0.01, weight_decay=0.0, betas=(0.9, 0.999))
                self.local_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model_copy.parameters()), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999))
                # Prepare dataet and dataloader
                if self.eval_all: # NOTE: Override the number of experiences to use in each step with max value
                    self.num_exps_seen = len(self.train_stream) -1 # -1 to make up for +1 in next step
                    print("\nNum seen experiences is maxed out!")

                curr_exp_data_stream = self.train_stream[:(self.num_exps_seen+1)] # Grab the curent subset of experiences from train_stream
                curr_exp_data = []
                # Create a ConcatDataset and respective Dataloader
                for exp in curr_exp_data_stream:
                    curr_exp_data.append(exp.dataset)
                
                lp_dataset = torch.utils.data.ConcatDataset(curr_exp_data)
                lp_dataloader = torch.utils.data.DataLoader(lp_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers, drop_last=True) 
                print("Collected dataset and loader...")
                
                # Train the new head(s)
                if isinstance(self.model_copy.classifier, MultiTaskModule):
                    print("Training new head(s)...", "num heads:", len(self.model_copy.classifier.classifiers))
                    
                for _ in tqdm(range(self.num_fintune_epochs)):
                    for _, mbatch in enumerate(lp_dataloader):
                        self.local_optim.zero_grad()

                        x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                        # On-the-fly update labels and targets for task-incremental learning
                        if self.force_task_eval: 
                            y, tid = y % self.initial_out_features, torch.div(y, self.initial_out_features, rounding_mode='trunc') # NOTE: This assumes that the number of classes per head is constant!

                        x = x.to(strategy.device)
                        y = y.to(strategy.device)
                        
                        # Get representation from backbone
                        x_rep = self.model_copy.feature_extractor(x)

                        # Forward representation through new head 
                        if isinstance(self.model_copy.classifier, MultiTaskModule): # NOTE: this is the avalanche_forward function copied
                            out = self.model_copy.classifier(x_rep, tid)
                        else:  # no task labels
                            out = self.model_copy.classifier(x_rep)

                        loss = strategy._criterion(out, y)
                        loss.backward()
                        self.local_optim.step()
                print("\nLinear Probe training complete...")

        super().before_eval_exp(strategy)
        # Replace strategy.model with the probed version
        strategy.model = self.model_copy
        if self._reset_at == 'experience' and self._mode == 'eval':
            self.reset(strategy)


    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Release the lock on LinearProbe training
        self.training_complete = False
        print("\nReleased Flag for Linear Probe Training")
        # Increase the counter on seen experiences
        self.num_exps_seen += 1 
        # In case of initial evaluation, do not increase the exp_seen counter
        if self.is_initial_eval_run:
            #self.num_exps_seen -= 1 # NOTE: this will be done by 'restore_strategy* call
            # Reset the state of the continual learner
            assert(not self._prev_state is None)
            assert(not self._prev_training_modes is None)
            ContinualEvaluationPhasePlugin.restore_strategy_(strategy, self._prev_state, self._prev_training_modes)
            self.is_initial_eval_run = False # Reset flag
        # Replace strategy.model with the original model
        strategy.model = self.model_original
        return


    