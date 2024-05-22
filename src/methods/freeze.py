import torch
from avalanche.training import BaseStrategy

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import freeze_everything

from src.model import freeze_up_to, freeze_from_to


class FreezeModelPlugin(StrategyPlugin): # TODO: renamce to "FreezeParametersPlugin"
    def __init__(self, 
                 exp_to_freeze_on=0, 
                 freeze_from_layer_name=None,
                 freeze_up_to_layer_name=None,
                 backbone_only=False):
        super().__init__()

        self.exp_to_freeze_on = exp_to_freeze_on
        self.freeze_from_layer_name = freeze_from_layer_name
        self.freeze_up_to_layer_name = freeze_up_to_layer_name
        
        self.backbone_only = backbone_only
        return

    def before_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > (self.exp_to_freeze_on -1): # NOTE: -1 is required to be able to freeze on the 0th experience
            if self.backbone_only:
                print("\n\nFreezing entire backbone...")
                freeze_everything(strategy.model.feature_extractor)
            elif self.freeze_up_to_layer_name is None and self.freeze_from_layer_name is None:
                print("Freezing entire model...")
                freeze_everything(strategy.model)
            else: 
                print("Freezing model layer {}... to layer {}".format(self.freeze_from_layer_name, self.freeze_up_to_layer_name))
                frozen_layers, _ = freeze_from_to(strategy.model, 
                                                  freeze_from_layer=self.freeze_from_layer_name,
                                                  freeze_until_layer=self.freeze_up_to_layer_name)
                for layer_name in frozen_layers:
                    print("Froze layer: {}".format(layer_name))
        return
    
def freeze_all_but_bn(m):
    "Source: https://discuss.pytorch.org/t/retrain-batchnorm-layer-only/61324"
    if not (isinstance(m, torch.nn.modules.batchnorm._BatchNorm) or \
        isinstance(m, torch.nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class FreezeAllButBNPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()
        return

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        strategy.model.feature_extractor.apply(freeze_all_but_bn)
        return super().before_training_exp(strategy, **kwargs)