from copy import deepcopy

from typing import Optional, Sequence, Union, List

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from avalanche.training import BaseStrategy

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins.evaluation import default_logger

from src.model import ConcatFeatClassifierModel, ExRepMultiHeadClassifier

class SeparateNetworksPlugin(StrategyPlugin):
    """
    Manages training of a separate model for each experience for multi-headed learning.
    """
    def __init__(self):
        super().__init__()

        # Map of task to backbone
        self.backbones = {}
        # Copy of the current backbone to switch back to after eval #NOTE: during eval previous backbones will be loaded 
        self.curr_backbone = None
        return

    def after_training_exp(self, strategy, **kwargs):
        # Store backbone separately
        self.backbones[strategy.training_exp_counter] = deepcopy(strategy.model.feature_extractor)
        # Store current backbone to be able to return to it
        self.curr_backbone = deepcopy(strategy.model.feature_extractor)
        return

    def before_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        # Switch backbone according to self.experience <- this is used to determine the dataloader so it should be fine...
        exp_idx = strategy.experience.current_experience
        print("Switching backbone to: ", exp_idx)
        if exp_idx in self.backbones:
            strategy.model.feature_extractor = self.backbones[exp_idx] # NOTE: the head should switch automatically
        else:
            strategy.model.feature_extractor = self.curr_backbone
        return

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        # Switch back to current backbone to continue next training
        strategy.model.feature_extractor = self.curr_backbone
        return

class ConcatFeatClassifierAdapterPlugin(StrategyPlugin):
    def __init__(self,
                 reinit_all_backbones=False):
        super().__init__()

        self.reinit_all_backbones = reinit_all_backbones
        return
    
    def before_training(self, strategy: BaseStrategy, **kwargs):
        assert isinstance(strategy.model, ConcatFeatClassifierModel), "This plugin only works with ConcatFeatClassifierModel!"
        return super().before_training(strategy, **kwargs)
    

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        # Add the current feature extractor to the dict of feature_extractors
        strategy.model.extent_feature_extractor(exp_idx=strategy.training_exp_counter)
        if isinstance(strategy.model.classifier, ExRepMultiHeadClassifier):
            strategy.model.classifier.extend_in_features(exp_idx=strategy.training_exp_counter)
        # Need to recall to(device) to make sure the new feature extractor is on the right device (because it is update after model adaptation)
        strategy.model.to(strategy.device)
        # Reinit all backbones
        if self.reinit_all_backbones:
            strategy.model.reinit_all_feature_extractors() # HACK: this can hardly be more hacky...
        return super().before_training_exp(strategy, **kwargs)

    def after_training_exp(self, strategy: BaseStrategy, **kwargs):
        strategy.model.store_feature_extractor(exp_idx=strategy.training_exp_counter-1)

        #strategy.model.feature_extractors[strategy.training_exp_counter-1] = strategy.model.feature_extractor
        #print("ConcatFeatClassifierAdapterPlugin: Extending feature extractor!", strategy.training_exp_counter)
        #strategy.model.extent_feature_extractor(exp_idx=strategy.training_exp_counter-1)
        #if isinstance(strategy.model.classifier, ExRepMultiHeadClassifier):
        #    strategy.model.classifier.extend_in_features(exp_idx=strategy.training_exp_counter-1)
        
        return super().after_training_exp(strategy, **kwargs)


class SeparateNetworks(BaseStrategy):
    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1):
        
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

        # Map of task to backbone
        self.backbones = {}
        # Copy of the current backbone to switch back to after eval #NOTE: during eval previous backbones will be loaded 
        self.curr_backbone = None
        return

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        # Store backbone separately
        self.backbones[self.training_exp_counter-1] = deepcopy(self.model.feature_extractor)
        # Store current backbone to be able to return to it
        self.curr_backbone = deepcopy(self.model.feature_extractor)
        
        return 

    def _before_eval_exp(self, **kwargs):
        super()._before_eval_exp(**kwargs)
        # Switch backbone according to self.experience <- this is used to determine the dataloader so it should be fine...
        exp_idx = self.experience.current_experience
        if exp_idx in self.backbones:
            self.model.feature_extractor = self.backbones[exp_idx] # NOTE: the head should switch automatically
        else:
            self.model.backbone = self.curr_backbone
        return

    def _after_eval(self, **kwargs):
        super()._after_eval(**kwargs)
        # Switch back to current backbone to continue next training
        self.model.feature_extractor = self.curr_backbone
        return


            