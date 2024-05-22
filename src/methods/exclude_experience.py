import torch

from typing import Optional, Sequence, Union, List
from copy import deepcopy

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from src.methods.mas import MASPlugin
from src.methods.replay import ERPlugin
import math


class ExcludeExperiencePlugin(StrategyPlugin):
    def __init__(self, experiences):
        super().__init__()

        self.exclude_experiences = experiences

        self.prev_train_epochs = None
    
        self.is_eval_adjusted = False # NOTE: Trapdoor bool to only 1 time allow for adjusting the eval function of strategy
        self.injections_done = False
        return
    
    def inject_mas(self, mas_plugin):
        """
        For MAS plugin the after_training_exp method needs to be overwritten.
        It will be adjusted to simply omit the importance evaluation for this task alltogether.
        """
        # define the new after_training_exp method
        mas_plugin.__class__.exclude_experiences = self.exclude_experiences
        mas_plugin.__class__.prev_after_training_exp = deepcopy(mas_plugin.__class__.after_training_exp)
        
        def adj_after_training_exp(obj, strategy, **kwargs):
            if strategy.clock.train_exp_counter in obj.exclude_experiences:
                print("\nPreventing this experience from being used for MAS!\n")
                return
            obj.prev_after_training_exp(strategy, **kwargs)
            return

        mas_plugin.__class__.after_training_exp = adj_after_training_exp
        return

    def inject_replay(self, replay_plugin):
        """
        For ReplayPlugin the after_training_exp method needs to be overwritten.
        It will be adjusted to omit storing any samples for excluded tasks.
        """
        replay_plugin.__class__.exclude_experiences = self.exclude_experiences
        replay_plugin.__class__.prev_after_training_exp = deepcopy(replay_plugin.__class__.after_training_exp)
        
        def adj_after_training_exp(obj, strategy, **kwargs):
            if strategy.clock.train_exp_counter in obj.exclude_experiences:
                print("\nPreventing this experience from being stored in replay memory!\n")
                return
            obj.prev_after_training_exp(strategy, **kwargs)
            return

        replay_plugin.__class__.after_training_exp = adj_after_training_exp
        return

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        """
        Before training, inject the member exclude_experiences into the strategy object.
        For many BaseStrategies this it not necessary, but also not harmful. Some may use it, .e.g joint.
        """
        # TODO: Adjust this function such that it does all injections? 
        # Check the trapdoor bool
        if self.injections_done:
            return
        # Inject the member exclude_experiences into the strategy object
        strategy.__class__.exclude_experiences = self.exclude_experiences
        print("ExcludeExperience::before_training: Injected exclude_experiences into strategy object")
        
        # Check the plugins of strategy for MAS and Replay # NOTE: currently only joint, MAS and replay are supported for exclusion
        for plugin in strategy.plugins:
            if isinstance(plugin, MASPlugin):
                self.inject_mas(plugin)
            elif isinstance(plugin, ERPlugin):
                self.inject_replay(plugin)

        # Set the trapdoor bool
        self.injections_done = True
        return

    def before_training_exp(self, strategy, **kwargs):
        # Set train_epochs to 0 if current experience is in exclude_experiences
        print("Training experience: ", strategy.clock.train_exp_counter, "ommitting?-", (strategy.clock.train_exp_counter in self.exclude_experiences))
        if strategy.clock.train_exp_counter in self.exclude_experiences:
            print("Omitting experience: ", strategy.clock.train_exp_counter, "in training")
            self.prev_train_epochs = strategy.train_epochs
            print("prev_train_epochs: ", self.prev_train_epochs)
            strategy.train_epochs = 0       
        return

    def after_training_exp(self, strategy, **kwargs):
        if not self.prev_train_epochs: # Savety check that can happen when loading weights from exclusion run
            return
        if strategy.train_epochs == 0: #NOTE: if train_epochs is 0, then the training loop is not executed, need to reset train_epochs
            strategy.train_epochs = self.prev_train_epochs
            print("reset prev_train_epochs: ", strategy.train_epochs)
            # Adjust strategy.clock.train_iterations
            #strategy.clock.train_iterations += strategy.train_epochs * math.ceil(len(strategy.experience.dataset)/strategy.train_mb_size)
            strategy.clock.train_iterations += strategy.train_epochs * math.ceil(len(strategy.adapted_dataset)/strategy.train_mb_size) # NOTE: this works also for joint datasets
        return


    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        """
        Overwrite the eval function of the given strategy to omit all tasks that are NOT excluded from
        training. These dont add to this baseline and only slow down the evaluation.
        """
        if self.is_eval_adjusted:
            return 

        print("\n\nINVADING EVAL FUNCTION\n\n")
        # Define adjusted eval function
        @torch.no_grad()
        def adj_eval(obj, exp_list, **kwargs):
            print("Using invaded eval!")
            
            strategy.is_training = False
            strategy.model.eval()
            
            if not isinstance(exp_list, Sequence):
                exp_list = [exp_list]
            strategy.current_eval_stream = exp_list

            strategy._before_eval(**kwargs)
            for exp_idx, strategy.experience in enumerate(exp_list):
                # Omit experience that are trained on from evaluation
                if not exp_idx in self.exclude_experiences:
                    print("skipping", exp_idx)
                    continue

                # Data Adaptation
                strategy._before_eval_dataset_adaptation(**kwargs)
                strategy.eval_dataset_adaptation(**kwargs)
                strategy._after_eval_dataset_adaptation(**kwargs)
                strategy.make_eval_dataloader(**kwargs)

                # Model Adaptation (e.g. freeze/add new units)
                strategy.model = strategy.model_adaptation()

                strategy._before_eval_exp(**kwargs)
                strategy.eval_epoch(**kwargs)
                strategy._after_eval_exp(**kwargs)
            strategy._after_eval(**kwargs)
            res = strategy.evaluator.get_last_metrics()
            return res

        # Inject overwritten eval function
        strategy.__class__.eval = adj_eval

        self.is_eval_adjusted = True
        return 