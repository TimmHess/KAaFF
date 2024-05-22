from typing import Optional, Sequence, Union, List
from collections import defaultdict

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from src.model import ConcatFeatClassifierModel, ExRepMultiHeadClassifier
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins.evaluation import default_logger
from avalanche.benchmarks.utils import AvalancheConcatDataset

from avalanche.models.utils import avalanche_forward

from src.methods.exclude_experience import ExcludeExperiencePlugin


class JointTrainingPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()
        print("\nUsing JointTrainingPlugin!\n")
    
        self.injection_done = False # NOTE: Trapdoor bool to only 1 time allow for adjusting the eval function of strategy
        return

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        #NOTE: Super does nothing - no need to call
        if self.injection_done: # NOTE: Only inject once
            return
        
        # Define all injection functions here  
        def adj_train_dataset_adaptation(obj, **kwargs):
            """
            Concatenates all the datastream.
            """
            # If dataset needs to be excluded, the previous dataset will be returned
            try: # NOTE self.exclude_experiences is defined by injection from ExcludeExperiencePlugin
                if obj.clock.train_exp_counter in obj.exclude_experiences:
                    print("Not adapting dataset! For exp: ", obj.clock.train_exp_counter)
                    obj.adapted_dataset = obj.accumulating_dataset.train()
                    return obj.adapted_dataset
            except Exception:
                pass
            
            if obj.accumulating_dataset is None:    
                obj.accumulating_dataset = obj.experience.dataset.train()
            else:
                cat_data = AvalancheConcatDataset([obj.accumulating_dataset, obj.experience.dataset])
                obj.accumulating_dataset = cat_data
            obj.adapted_dataset = obj.accumulating_dataset
            obj.adapted_dataset = obj.adapted_dataset.train()
            return

        def adj_make_train_dataloader(obj, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
            """ Data loader initialization.
            Called at the start of each learning experience after the dataset 
            adaptation.

            :param num_workers: number of thread workers for the data loading.
            :param shuffle: True if the data should be shuffled, False otherwise.
            :param pin_memory: If True, the data loader will copy Tensors into CUDA
                pinned memory before returning them. Defaults to True.
            """
            obj.dataloader = torch.utils.data.DataLoader(
                                    obj.adapted_dataset,
                                    num_workers=num_workers,
                                    batch_size=obj.train_mb_size,
                                    shuffle=shuffle,
                                    pin_memory=pin_memory,
                                    drop_last=True
            )
            return 

        # Inject missing members
        strategy.__class__.accumulating_dataset = None
        # Inject overwrite to functions
        strategy.__class__.train_dataset_adaptation = adj_train_dataset_adaptation
        strategy.__class__.make_train_dataloader = adj_make_train_dataloader
        
        self.injection_done = True
        return
    
    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        print("len strategy dataloder:", len(strategy.dataloader))
        return

class ConcatJointTrainingPlugin(JointTrainingPlugin):
    def __init__(self):
        super().__init__()
        print("\nUsing ConcatJointTrainingPlugin!\n")
        return
    
    def before_training(self, strategy: BaseStrategy, **kwargs):
        assert isinstance(strategy.model, ConcatFeatClassifierModel)
        assert isinstance(strategy.model.classifier, ExRepMultiHeadClassifier)
        # Do injections custom to concat_joint training
        if self.injection_done: # NOTE: injection_done is set true in super().before_training
            return
        
        # inject forward pass
        def adj_forward(obj):
            x_repr = obj.model.forward_all_feats(obj.mb_x)
            #return avalanche_forward(self.model, self.mb_x, self.mb_task_id) 
            return avalanche_forward(obj.model.classifier, x_repr, obj.mb_task_id) 
        
        # inject optimizer reset / adjustment
        def adj_make_optimizer(obj):
            obj.optimizer.state = defaultdict(dict)
            # Add parameters of all stored backbones to the optimizer 
            #obj.optimizer.param_groups[0]['params'] = list(obj.model.parameters()) # NOTE: param_groups[0] need to be the model's parameters
            parameters = None
            for key in obj.model.feature_extractors:
                if not parameters:
                    parameters = list(obj.model.feature_extractors[key].parameters())
                else:
                    parameters += list(obj.model.feature_extractors[key].parameters())
            # Add classifier to parameters
            parameters += list(obj.model.classifier.parameters())

            obj.optimizer.param_groups[0]['params'] = parameters
            print("Optimizer params:", len(obj.optimizer.param_groups[0]['params']))

        # Inject missing members
        strategy.__class__.forward = adj_forward
        strategy.__class__.make_optimizer = adj_make_optimizer

        print("\nInjected forward function")
        print("\nInjected make_optimizer function")
        return super().before_training(strategy, **kwargs)


class JointTraining(BaseStrategy):
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

        self.accumulating_dataset = None
        return

    def _before_training(self, **kwargs):
        # Iterate all plugins and check if any is ExcludeExperiencePlugin
        for plugin in self.plugins:
            if isinstance(plugin, ExcludeExperiencePlugin):
                # Store the exclude_experiences list
                print("\n ExcludeExperiencePlugin found!\n")
                self.exclude_experiences = plugin.exclude_experiences
        
        super()._before_training(**kwargs) # Call to plugins
        return

    def _before_training_exp(self, **kwargs):
        print(len(self.dataloader))
        super()._before_training_exp(**kwargs)
        return

    def train_dataset_adaptation(self, **kwargs):
        """
        Concatenates all the datastream.
        """
        # If dataset needs to be excluded, the previous dataset will be returned
        try: # NOTE self.exclude_experiences is only defined if ExcludeExperiencePlugin is used
            if self.clock.train_exp_counter in self.exclude_experiences:
                print("Not adapting dataset! For exp: ", self.clock.train_exp_counter)
                self.adapted_dataset = self.accumulating_dataset.train()
                return self.adapted_dataset
        except Exception:
            pass
        
        if self.accumulating_dataset is None:    
            self.accumulating_dataset = self.experience.dataset.train()
        else:
            cat_data = AvalancheConcatDataset([self.accumulating_dataset, self.experience.dataset])
            self.accumulating_dataset = cat_data
        self.adapted_dataset = self.accumulating_dataset
        self.adapted_dataset = self.adapted_dataset.train()
        print("Dataset size:", len(self.adapted_dataset))
        return

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """ Data loader initialization.

        Called at the start of each learning experience after the dataset 
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        self.dataloader = torch.utils.data.DataLoader(
                                self.adapted_dataset,
                                num_workers=num_workers,
                                batch_size=self.train_mb_size,
                                shuffle=shuffle,
                                pin_memory=pin_memory,
                                drop_last=True
        )
        return


            