import torch

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from src.eval.continual_eval import ContinualEvaluationPhasePlugin



class InitialEvalStage(StrategyPlugin):
    def __init__(self, test_stream, only_initial_eval=False):
        super().__init__()

        self.test_stream = test_stream
        self.only_initial_eval = only_initial_eval

        self._prev_state = None
        self._prev_training_modes = None
        return

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        if strategy.clock.train_exp_counter == 0:
            print("\nDoing linear probing on random-weighted model")
            # Need to store the state of the trainer and model befor running eval inside train-loop
            self._prev_state, self._prev_training_modes = ContinualEvaluationPhasePlugin.get_strategy_state(strategy)
            # Trigger the evaluation by calling strategy.eval()? -> This will run entire evaluation... 
            print("Triggering initial evaluation before training starts...")
            strategy.eval(self.test_stream)
        return

    def after_eval(self, strategy: 'BaseStrategy'):
        if self.only_initial_eval:
            import sys; sys.exit(0) # NOTE: terminate the run
        
        # Reset the state of the continual learner
        assert(not self._prev_state is None)
        assert(not self._prev_training_modes is None)
        ContinualEvaluationPhasePlugin.restore_strategy_(strategy, self._prev_state, self._prev_training_modes)
        self.is_initial_eval_run = False # Reset flag
        return
