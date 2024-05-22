from copy import deepcopy
import math
from typing import Optional, Sequence, Union, List

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins.evaluation import default_logger

from libs.barlow_twins.barlow import BarlowTwinLoss

from src.model import BarlowTwinModel


class BarlowTwins(BaseStrategy):
    def __init__(self, 
                 model: Module, 
                 optimizer: Optimizer,
                 train_mb_size: int = 1, 
                 train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1,
                 train_transforms=None,
                 eval_transforms=None,
                 projection_dim=1024):
        
        # Check that a SupConModel is passed
        assert isinstance(model, BarlowTwinModel), "model must be of type BarlowTwinModel!"
        
        # Get a GradScaler
        #self.scaler = torch.cuda.amp.GradScaler()
        self.scaler = None

        # Define SupCon criterion
        self._criterion = BarlowTwinLoss(
                            batch_size=train_mb_size,
                            projection_dim=projection_dim, 
                            lambda_coeff=5e-4
                            #device=device
        )# NOTE: cannot be 'criterion' because that is a function!

        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms

        self.last_epoch_train_loss = None

        super().__init__(
            model, optimizer, self._criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)
        return

    def _before_training_exp(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        # Need to set 'use_projection_head' to true in standard training phase
        self.model.use_projection_head(True)
        for p in self.plugins:
            p.before_training_exp(self, **kwargs)
        return

    def _before_training_iteration(self, **kwargs):
        # Transpose the input to be able to make use of TwoCropTransform
        self.mbatch[0] = self.mb_x.transpose(0,1)
        for p in self.plugins:
            p.before_training_iteration(self, **kwargs)

    def training_epoch(self, **kwargs):
        """ Training epoch.
        
        :param kwargs:
        :return:
        """
        self.last_epoch_train_loss = []
        num_batches_trained = 0
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
            
            self._unpack_minibatch() # NOTE: should put everything on correct device

            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)

            #Below is inline code for: self.mb_output = self.forward()
            embed = self.mb_output
            if self.scaler:
                with torch.cuda.amp.autocast():
                    embed_x1, embed_x2 = self.model(self.mbatch[0]) # NOTE: embed is self.mb_output
                    self.loss += self._criterion(embed_x1, embed_x2) # NOTE: need to calculate loss here because of autocast
            else:
                embed_x1, embed_x2 = self.model(self.mbatch[0]) # NOTE: embed is self.mb_output
                self.loss += self._criterion(embed_x1, embed_x2) 
        
            # Write embed back to the strategy's field (maybe this is needed somewhere)
            self.mb_output = embed 
            self._after_forward(**kwargs)
            
            # Clean caches # NOTE: I cannot call del on attributes...
            del embed 
            torch.cuda.empty_cache()

            #self.loss += self.criterion()  # NOTE: Loss & Backward # already calculated above...

            self._before_backward(**kwargs)
            
            if self.scaler:
                self.scaler.scale(self.loss).backward()
                self._after_backward(**kwargs)
                self._before_update(**kwargs)
                self.scaler.step(self.optimizer)
                self._after_update(**kwargs)
                self.scaler.update()
            else:
                self.loss.backward()
                self._after_backward(**kwargs)
                self._before_update(**kwargs)
                self.optimizer.step()
                self._after_update(**kwargs)           
           
            self._after_training_iteration(**kwargs)

            self.last_epoch_train_loss.append(self.loss.item())
            
            # Effectively an inefficient drop_last in the dataloader
            num_batches_trained += 1
            if num_batches_trained >= len(self.dataloader)-1:
                break

        # Finally, prepare mean loss value for this epoch
        self.last_epoch_train_loss = torch.mean(torch.Tensor(self.last_epoch_train_loss)).item()
        print("\n\nmean loss:", self.last_epoch_train_loss, "\n")
        assert not math.isnan(self.last_epoch_train_loss), "Loss is NaN!"
        return

    def forward(self):
        print( "This should not be called with SupCon, ever!")
        raise NotImplementedError

    def _after_forward(self, **kwargs):
        for p in self.plugins:
            p.after_forward(self, **kwargs)

    def criterion(self):
        """ Loss function. """
        print("criterion function should not have been called!")
        raise NotImplementedError
        #return self._criterion(self.mb_output, self.mb_y)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            # NOTE: forward pass only generates intermediate features but no mb_output
            #self._before_eval_forward(**kwargs)
            #if self.scaler:
            #    with torch.cuda.amp.autocast():
            #        self.model(self.mbatch[0])
            #else:
            self.model(self.mbatch[0])
            
            #self.mb_output = self.forward()
            #self._after_eval_forward(**kwargs)
            
            #self.loss = self.criterion() # NOTE: deactivated loss calculation for supcon eval, because I don't know what it should provide...
            self.loss = self.last_epoch_train_loss

            self._after_eval_iteration(**kwargs)

    def _before_eval(self, **kwargs):
        for p in self.plugins:
            p.before_eval(self, **kwargs)

            