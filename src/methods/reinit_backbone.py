import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from typing import NamedTuple, List, Optional, Tuple, Callable

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import freeze_everything, get_layers_and_params

import copy

class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor

class ReInitBackbonePlugin(StrategyPlugin):
    def __init__(self, exp_to_reinit_on=0, reinit_until_exp=100, reinit_after_layer_name=None,
                 freeze=False, reinit_deterministically=False):
        super().__init__()

        self.exp_to_reinit_on = exp_to_reinit_on
        self.reinit_until_exp = reinit_until_exp
        self.reinit_after_layer_name = reinit_after_layer_name
        self.freeze = freeze

        self.reinit_deterministically = reinit_deterministically
        self.initial_weights_copy = None

        self.is_frozen = False
        return

    def get_layers_and_params(self, model, prefix=''):
        result: List[LayerAndParameter] = []
        layer_name: str
        layer: Module
        for layer_name, layer in model.named_modules():
            if layer == model:
                continue
            if isinstance(layer, nn.Sequential): # NOTE: cannot include Sequentials because this is basically a repetition of parameter listings
                continue
            layer_complete_name = prefix + layer_name + "."

            layers_and_params = get_layers_and_params(layer, prefix=layer_complete_name) #NOTE: this calls to avalanche function! (not self)
            result += layers_and_params
        return result


    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
    
    def reinit_after(self, model, reinit_after=None, freeze=False, module_prefix=""):
        print("\nEntering Reinit function...")
        do_skip = True # NOTE: flag to skip the first layers in reinitialization
        for param_def in self.get_layers_and_params(model, prefix=module_prefix):
            if reinit_after in param_def.layer_name: # NOTE: if reinit_after is None, nothing is reinitialized!
                do_skip = False
            
            if do_skip: # NOTE: this will skip the first n layers in execution
                print("Skipping layer {}".format(param_def.layer_name))
                continue
            
            # TODO: re-add layer filter option (it was too annoying to implement)
    
            self.initialize_weights(param_def.layer)
            print("Reinitialized {}".format(param_def.layer_name))
        return 

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        if self.reinit_deterministically and self.initial_weights_copy is None:
            self.initial_weights_copy = copy.deepcopy(strategy.model.feature_extractor)
            print("Stored initial weights for deterministic reinitialization!")
        return super().before_training(strategy, **kwargs)

    def before_training_exp(self, strategy, **kwargs):
        """
        Reinitialize after every experience
        """
        if (strategy.clock.train_exp_counter >= self.exp_to_reinit_on) and not (strategy.clock.train_exp_counter > self.reinit_until_exp):
            if self.reinit_after_layer_name is None:
                if self.reinit_deterministically: # Affecting backbone only
                    strategy.model.feature_extractor.load_state_dict(self.initial_weights_copy.state_dict(), strict=True)
                    print("\nRe-Initialized ALL weights to SAME state before training!\n")
                    return
                else:
                    strategy.model.apply(self.initialize_weights)
                    print("\nRe-Initialized ALL weights from distribution!\n")
                    return

            if not self.is_frozen: # NOTE: Trapdoor flag to avoid multiple reinits on frozen model
                # Applying reinitialization partly
                self.reinit_after(strategy.model, self.reinit_after_layer_name)
                print("\nRe-Initialized weights after {}!\n".format(self.reinit_after_layer_name))
            
            if self.freeze and not self.is_frozen:
                # Set the freeze flag
                self.is_frozen = True 
                freeze_everything(strategy.model.feature_extractor)
                print("Backbone frozen after reinitialization!")                
        return