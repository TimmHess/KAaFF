#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Adapted by Timm Hess (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import random
import copy
from pprint import pprint
from typing import TYPE_CHECKING, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.storage_policy import ClassBalancedBuffer, ClassTaskBalancedBuffer

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class ACE_CE_Loss(nn.Module):
    """
    Masked version of CrossEntropyLoss.
    """
    def __init__(self, device):
        super(ACE_CE_Loss, self).__init__()

        self.seen_so_far = torch.zeros(0, dtype=torch.int).to(device) # basically an empty tensor
        return

    def forward(self, logits, labels):
        present = labels.unique()

        mask = torch.zeros_like(logits).fill_(-1e9)
        mask[:, present] = logits[:, present] # add the logits for the currently observed labels
        
        if len(self.seen_so_far) > 0: # if there are seen classes, add them as well (this is for replay buffer loss)
            mask[:, (self.seen_so_far.max()+1):] = logits[:, (self.seen_so_far.max()+1):] # add the logits for the unseen labels
        
        logits = mask
        return F.cross_entropy(logits, labels)

class ERPlugin(StrategyPlugin):
    """
    Rehearsal Revealed: replay plugin.
    Implements two modes: Classic Experience Replay (ER) and Experience Replay with Ridge Aversion (ERaverse).
    """
    store_criteria = ['rnd']

    def __init__(self, 
            n_total_memories, 
            lmbda, 
            device, 
            replay_batch_handling='separate', # NOTE: alterantive is 'combined'
            task_incremental=False, 
            domain_incremental=False,
            lmbda_warmup_steps=0, 
            total_num_classes=100, 
            num_experiences=1,
            do_decay_lmbda=False, 
            ace_ce_loss=False
        ):
        """
        # TODO: add docstring
        """
        super().__init__()

        # Memory
        self.n_total_memories = n_total_memories  # Used dynamically
        # a Dict<task_id, Dataset>
        if task_incremental:
            self.storage_policy = ClassTaskBalancedBuffer(  # Samples to store in memory
                max_size=self.n_total_memories,
                adaptive_size=False,
                total_num_classes=total_num_classes
            )
        elif domain_incremental:
            self.storage_policy = ClassTaskBalancedBuffer(  # Samples to store in memory
                max_size=self.n_total_memories,
                adaptive_size=False,
                total_num_classes=total_num_classes*num_experiences # NOTE: because classes are repeated...
            )
        else:
            self.storage_policy = ClassBalancedBuffer(
                max_size=self.n_total_memories,
                adaptive_size=False,
                total_num_classes=total_num_classes
            )
        print(f"[METHOD CONFIG] n_total_mems={self.n_total_memories} ")
        print(f"[METHOD CONFIG] SUMMARY: ", end='')
        print(f"[METHOD CONFIG] replay_batch_handling={replay_batch_handling} ")
        pprint(self.__dict__, indent=2)

        # device
        self.device = device

        # weighting of replayed loss and current minibatch loss
        self.lmbda = lmbda  # 0.5 means equal weighting of the two losses
        self.do_decay_lmbda = do_decay_lmbda
        self.lmbda_warmup_steps = lmbda_warmup_steps
        self.do_exp_based_lmbda_weighting = False
        self.last_iteration = 0

        # replay batch handling
        self.replay_batch_handling = replay_batch_handling
        self.nb_new_samples = None

        # Losses
        self.replay_criterion = torch.nn.CrossEntropyLoss()
        self.use_ace_ce_loss = ace_ce_loss
        if self.use_ace_ce_loss:
            self.replay_criterion = ACE_CE_Loss(self.device)
            self.ace_ce_loss = ACE_CE_Loss(self.device)
        self.replay_loss = 0

    def before_training(self, strategy: BaseStrategy, **kwargs):
        """ 
        Omit reduction in criterion to be able to 
        separate losses from buffer and batch
        """
        strategy._criterion.reduction = 'none'  # Overwrite
        return super().before_training(strategy, **kwargs)

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        if self.replay_batch_handling == 'combined':
            return
        if strategy.clock.train_exp_counter > 0 and self.do_decay_lmbda:
            lmbda_decay_factor = (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("\nDecaying lmbda by:", lmbda_decay_factor)
            self.lmbda *= lmbda_decay_factor
            print("New lmbda is:", self.lmbda)
        return


    def before_training_iteration(self, strategy, **kwargs):
        """
        Adjust the lmbda weighting according to lmbda warmup settings
        """

        self.nb_new_samples = strategy.mb_x.shape[0]
    
        if strategy.clock.train_exp_counter > 0:
            self.last_iteration += 1
            if not self.last_iteration > self.lmbda_warmup_steps:
                # Apply linear weighting over the number of warmup steps
                self.lmbda_weighting = self.last_iteration / self.lmbda_warmup_steps
        return


    def before_forward(self, strategy, **kwargs):
        """
        Calculate the loss with respect to the replayed data separately here.
        This enables to weight the losses separately.
        Needs to be done here to prevent gradients being zeroed!
        """

        # Sample memory batch
        x_s, y_s, t_s = None, None, None
        if self.n_total_memories > 0 and len(self.storage_policy.buffer) > 0:  # Only sample if there are stored
            x_s, y_s, t_s = self.load_buffer_batch(storage_policy=self.storage_policy, 
                                        strategy=strategy, nb=strategy.train_mb_size)

        # Append to current new-data batch
        if x_s is not None:  # Add
            assert y_s is not None
            assert t_s is not None
            # Assemble minibatch
            strategy.mbatch[0] = torch.cat([strategy.mbatch[0], x_s])
            strategy.mbatch[1] = torch.cat([strategy.mbatch[1], y_s])
            strategy.mbatch[-1] = torch.cat([strategy.mbatch[-1], t_s])
        return


    def before_backward(self, strategy: BaseStrategy, **kwargs):
        # Disentangle losses
        nb_samples = strategy.loss.shape[0]        

        loss_new = self.lmbda * strategy.loss[:self.nb_new_samples].mean()
        loss = loss_new

        # Mem loss
        if nb_samples > self.nb_new_samples:
            loss_reg = (1 - self.lmbda) * strategy.loss[self.nb_new_samples:].mean()
            loss = loss_new + loss_reg  

        # Writeback loss to strategy   
        strategy.loss = loss      
        return super().before_backward(strategy, **kwargs)

    def after_training_exp(self, strategy, **kwargs):
        """ Update memories."""
        self.storage_policy.update(strategy, **kwargs)  # Storage policy: Store the new exemplars in this experience
        self.reset()
        return

    def reset(self):
        """
        Reset internal variables after each experience
        """
        self.last_iteration = 0
        self.lmbda_weighting = 1
        return

    def load_buffer_batch(self, storage_policy, strategy, nb=None):
        """
        Wrapper to retrieve a batch of exemplars from the rehearsal memory
        :param nb: Number of memories to return
        :return: input-space tensor, label tensor
        """

        ret_x, ret_y, ret_t = None, None, None
        # Equal amount as batch: Last batch can contain fewer!
        n_exemplars = strategy.train_mb_size if nb is None else nb
        new_dset = self.retrieve_random_buffer_batch(storage_policy, n_exemplars)  # Dataset object
        
        # Load the actual data
        for sample in DataLoader(new_dset, batch_size=len(new_dset), pin_memory=True, shuffle=False):
            x_s, y_s = sample[0].to(strategy.device), sample[1].to(strategy.device)
            t_s = sample[-1].to(strategy.device)  # Task label (for multi-head)

            ret_x = x_s if ret_x is None else torch.cat([ret_x, x_s])
            ret_y = y_s if ret_y is None else torch.cat([ret_y, y_s])
            ret_t = t_s if ret_t is None else torch.cat([ret_t, t_s])

        return ret_x, ret_y, ret_t

    def retrieve_random_buffer_batch(self, storage_policy, n_samples):
        """
        Retrieve a batch of exemplars from the rehearsal memory.
        First sample indices for the available tasks at random, then actually extract from rehearsal memory.
        There is no resampling of exemplars.

        :param n_samples: Number of memories to return
        :return: input-space tensor, label tensor
        """
        assert n_samples > 0, "Need positive nb of samples to retrieve!"

        # Determine how many mem-samples available
        q_total_cnt = 0  # Total samples
        free_q = {}  # idxs of which ones are free in mem queue
        tasks = []
        for t, ex_buffer in storage_policy.buffer_groups.items():
            mem_cnt = len(ex_buffer.buffer)  # Mem cnt
            free_q[t] = list(range(0, mem_cnt))  # Free samples
            q_total_cnt += len(free_q[t])  # Total free samples
            tasks.append(t)

        # Randomly sample how many samples to idx per class
        free_tasks = copy.deepcopy(tasks)
        tot_sample_cnt = 0
        sample_cnt = {c: 0 for c in tasks}  # How many sampled already
        max_samples = n_samples if q_total_cnt > n_samples else q_total_cnt  # How many to sample (equally divided)
        while tot_sample_cnt < max_samples:
            t_idx = random.randrange(len(free_tasks))
            t = free_tasks[t_idx]  # Sample a task

            if sample_cnt[t] >= len(storage_policy.buffer_group(t)):  # No more memories to sample
                free_tasks.remove(t)
                continue
            sample_cnt[t] += 1
            tot_sample_cnt += 1

        # Actually sample
        s_cnt = 0
        subsets = []
        for t, t_cnt in sample_cnt.items():
            if t_cnt > 0:
                # Set of idxs
                cnt_idxs = torch.randperm(len(storage_policy.buffer_group(t)))[:t_cnt]
                sample_idxs = cnt_idxs.unsqueeze(1).expand(-1, 1)
                sample_idxs = sample_idxs.view(-1)

                # Actual subset
                s = Subset(storage_policy.buffer_group(t), sample_idxs.tolist())
                subsets.append(s)
                s_cnt += t_cnt
        assert s_cnt == tot_sample_cnt == max_samples
        new_dset = ConcatDataset(subsets)
        return new_dset