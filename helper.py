from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from copy import deepcopy
import shutil
from typing import List

import numpy as np

import torch
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR

from torchvision import transforms
from torchvision.datasets import CIFAR100

# Avalanche
from avalanche.benchmarks.classic import SplitCIFAR100 #, SplitCUB200
#from avalanche.benchmarks.classic.cmnist import _get_mnist_dataset
from src.benchmarks.miniimagenet_benchmark import get_miniimgnet_dataset
from avalanche.evaluation.metrics import ExperienceForgetting, StreamForgetting, accuracy_metrics, loss_metrics, \
    StreamConfusionMatrix, timing_metrics
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.strategies import Naive
from src.methods.eval_stored_weights import EvalStoredWeightsPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy

# CUSTOM
from src.benchmarks.miniimagenet_benchmark import SplitMiniImageNet
from src.eval.continual_eval import ContinualEvaluationPhasePlugin
from src.eval.continual_eval_metrics import TaskTrackingLossPluginMetric, \
    TaskTrackingGradnormPluginMetric, TaskTrackingFeatureDriftPluginMetric, TaskTrackingAccuracyPluginMetric, \
    TaskAveragingPluginMetric, WindowedForgettingPluginMetric, \
    TaskTrackingMINAccuracyPluginMetric, TrackingLCAPluginMetric, WCACCPluginMetric, WindowedPlasticityPluginMetric
from src.eval.initial_eval_stage import InitialEvalStage
from src.eval.downstream_finetune import DownstreamFinetuneAccuracyMetric
from src.eval.linear_mh_probing_metric import LinearProbingAccuracyMetric
from src.eval.linear_probing_metric_last_n import LinearProbingLastNAccuracyMetricLastN
from src.eval.knn_probing_metric import KNNProbingAccuracyMetric
from src.eval.knn_pca_probing_metric import KNNPCAProbingAccuracyMetric
from src.eval.ncm_probing_metric import NCMProbingAccuracyMetric
from src.eval.downstream_linear import DownstreamLinearProbeAccuracyMetric
from src.eval.concatenated_linear_mh_probing_metric import ConcatenatedLinearProbingAccuracyMetric
from src.eval.concatenated_knn_probing_metric import ConcatenatedKNNProbingAccuracyMetric
from src.eval.concatenated_ncm_probing_metric import ConcatenatedNCMProbingAccuracyMetric
from src.eval.downstream_concatenated_probing_metric import ConcatDownStreamProbingAccuracyMetric
from src.eval.downstream_knn import DownstreamKKNNAccuracyMetric
from src.eval.pca_probing import PCAProbingMetric
from src.eval.concatenated_pca_probing import ConcatenatedPCAProbingMetric
from src.utils import ExpLRSchedulerPlugin, IterationsInsteadOfEpochs, EpochLRSchedulerPlugin, OneCycleSchedulerPlugin
from src.methods.lwf_standard import LwFStandardPlugin
from src.methods.packnet import PackNetPlugin
from src.methods.mas import MASPlugin
from src.methods.replay import ERPlugin
from src.methods.der import DERPlugin
from src.methods.freeze import FreezeModelPlugin, FreezeAllButBNPlugin
from src.methods.reinit_backbone import ReInitBackbonePlugin
from src.methods.exclude_experience import ExcludeExperiencePlugin
from src.methods.epoch_adapter import EpochLengthAdapterPlugin
from src.methods.store_models import StoreModelsPlugin
from src.methods.separate_networks import SeparateNetworks
from src.methods.separate_networks import ConcatFeatClassifierAdapterPlugin
from src.methods.joint import JointTrainingPlugin, ConcatJointTrainingPlugin
from src.methods.supcon import SupCon
from src.methods.barlow_twins import BarlowTwins

from libs.supcon.utils import TwoCropTransform

def args_to_tensorboard(writer, args):
    """
    Copyright: OCDVAE
    Takes command line parser arguments and formats them to
    display them in TensorBoard text.
    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        args (dict): dictionary of command line arguments
    """
    txt = ""
    for arg in sorted(vars(args)):
        txt += arg + ": " + str(getattr(args, arg)) + "<br/>"

    writer.add_text('command_line_parameters', txt, 0)
    return

def get_dataset(scenario_name, 
                dset_rootpath, 
                train_transform=None, 
                eval_transform=None,
                subsample_classes=None):

    print("\nGetDataset:eval_transform: {}".format(eval_transform))
    train_set = None
    test_set = None
    n_classes = None
    dset_rootpath = deepcopy(dset_rootpath) 

    eval_transform = deepcopy(eval_transform)
    # Add or remove 'to_pil' transform if needed
    if not scenario_name in ["miniimgnet"] \
            and isinstance(eval_transform.transforms[0], transforms.ToPILImage): #.transforms[0]
        eval_transform.transforms = eval_transform.transforms[1:]
    elif scenario_name in ["miniimgnet"] \
            and not isinstance(eval_transform.transforms[0], transforms.ToPILImage):
        eval_transform.transforms.insert(0, transforms.ToPILImage())

    if scenario_name == 'cifar100':
        train_set = CIFAR100(root=dset_rootpath, train=True, download=True, transform=eval_transform)
        test_set = CIFAR100(root=dset_rootpath, train=False, download=True, transform=eval_transform)
        n_classes = 100
    elif scenario_name == 'miniimgnet':
        train_set, test_set = get_miniimgnet_dataset(rootpath=dset_rootpath, transform=eval_transform)
        n_classes = 100
    else:
        raise ValueError("Unknown dataset name: {}".format(scenario_name))
    
    if subsample_classes:
        print("Subsampling classes: {}".format(subsample_classes))
        # Get masks for all classes
        train_mask = (torch.tensor(train_set.targets) == subsample_classes[0])
        test_mask = (torch.tensor(test_set.targets) == subsample_classes[0])
        for c in subsample_classes[1:]:
            train_mask = train_mask | (torch.tensor(train_set.targets) == c)
            test_mask = test_mask | (torch.tensor(test_set.targets) == c)

        # Make tensor that is True for all indices that are in subsample_classes
        train_set.data = train_set.data[train_mask]
        train_set.targets = torch.tensor(train_set.targets)[train_mask].tolist()
        test_set.data = test_set.data[test_mask]
        test_set.targets = torch.tensor(test_set.targets)[test_mask].tolist()
        # Convert the targets to proper task-labes
        subsample_map = {k: v for v, k in enumerate(subsample_classes)}
        train_set.targets = [subsample_map[y] for y in train_set.targets]
        test_set.targets = [subsample_map[y] for y in test_set.targets]
        
        # Adjust n_classes
        n_classes = len(subsample_classes)
    return train_set, test_set, n_classes

def get_transforms(data_aug, 
                   input_size, 
                   norm_mean=(0.0, 0.0, 0.0), 
                   norm_std=(1.0, 1.0, 1.0),
                   use_to_pil=False):
    """
    Single place in codebase where data transforms are defined.
    

    Return_ List of transforms for train and test
    """
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize(size=(input_size[1], input_size[2])) 

    crop_flip = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2]),
                                    scale=(0.1 if input_size[0]>=64 else 0.2, 1.),
                                    ),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    sim_clr = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2]),
                                    scale=(0.1 if input_size[0]>=64 else 0.2, 1.),
                                    ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, 
                                    contrast=0.4,
                                    saturation=0.2, 
                                    hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
                [transforms.GaussianBlur(
                    kernel_size=input_size[0]//20*2+1, 
                    sigma=(0.1, 2.0)
                )], 
            p=0.5 if input_size[0]>32 else 0.0),
    ])
    rand_crop_aug = transforms.Compose([
        transforms.RandomCrop(size=(input_size[1], input_size[2]), padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
    ])
    autoaug_cifar10 = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2])),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10)
    ])
    autoaug_imgnet = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size[1], input_size[2])),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.IMAGENET)
    ])
    
    rand_grayscale = transforms.Compose([transforms.RandomGrayscale(p=0.2)]) 
    rand_gauss_blur = transforms.Compose([
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=input_size[0]//20*2+1, sigma=(0.1, 2.0))
                    ], p=0.5 if input_size[0]>32 else 0.0
                    )
                ])

    normalize = transforms.Normalize(norm_mean, norm_std)
    to_tensor = transforms.ToTensor()

    train_transforms = []
    test_transforms = []
    # Optional addition of ToPILImage transform
    if use_to_pil:
        train_transforms.append(to_pil)
        test_transforms.append(to_pil)
    # Default addtion of Resize transform
    train_transforms.append(resize)
    test_transforms.append(resize)

    if "simclr" in data_aug:
        train_transforms.append(sim_clr)
    elif "crop_flip" in data_aug:
        train_transforms.append(crop_flip)
    elif "rand_crop" in data_aug:
        train_transforms.append(rand_crop_aug)
    elif "auto_cifar10" in data_aug:
        train_transforms.append(autoaug_cifar10)
    elif "auto_imgnet" in data_aug:
        train_transforms.append(autoaug_imgnet)
    
    # Default addition of Normalize and ToTensor transforms
    train_transforms.extend([to_tensor, normalize])
    test_transforms.extend([to_tensor, normalize])
    # Finally, compose everything into a single transform
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def get_scenario(args, scenario_name, dset_rootpath,
                num_experiences=None, use_data_aug=None, seed=42):
    print(f"\n[SCENARIO] {scenario_name}, Task Incr = {args.task_incr}")

    # Check for 'none' string #NOTE: this will happen when using default roots for downstream sets
    if not dset_rootpath is None:
        if dset_rootpath.lower() == "none":
            dset_rootpath = None

    # Prepare general transforms
    train_transform = None
    test_transform = None
    data_transforms = dict()

    if scenario_name in ["cifar100"]:
        input_size = (3, 32, 32)
    elif scenario_name in ["miniimgnet"]:
        input_size = (3, 84, 84)
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")

    if not args.overwrite_input_size is None:
        input_size = (input_size[0], args.overwrite_input_size[0], args.overwrite_input_size[1])

    norm_mean = None
    norm_stddev = None
    if args.overwrite_mean and args.overwrite_stddev:
        norm_mean = tuple(args.overwrite_mean)
        norm_stddev = tuple(args.overwrite_stddev)
        print("Overwriting mean and stddev with", norm_mean, norm_stddev)

    n_classes = None
    fixed_class_order = args.fixed_class_order

    # Prepare datasets/scenarios
    # CIFAR100
    if scenario_name == 'cifar100':
        n_classes = 100
        n_experiences = 10

        if not fixed_class_order:
            fixed_class_order = [i for i in range(n_classes)]
        if args.use_rand_class_ordering:
            assert not args.fixed_class_order, "Cannot use random class ordering with fixed class ordering!"
            fixed_class_order = np.random.permutation(n_classes).tolist()
            print("the order is ", fixed_class_order)

        fixed_class_order = [int(x) for x in fixed_class_order]

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
                                            use_data_aug, 
                                            input_size=input_size, 
                                            use_to_pil=False,
                                            norm_mean=norm_mean if norm_mean else (0.485, 0.456, 0.406),  # (0.5071, 0.4867, 0.4408)
                                            norm_std=norm_stddev if norm_stddev else (0.229, 0.224, 0.225) # (0.2675, 0.2565, 0.2761)
        )
        print("train transform:")
        print(train_transform)
        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy
        if 'supcon' in args.strategy or 'barlow_twins' in args.strategy:
            altered_train_transform = TwoCropTransform(train_transform)

        print("altered transform")
        print(altered_train_transform)
        
        scenario = SplitCIFAR100(
                    n_experiences=n_experiences, 
                    return_task_id=args.task_incr, 
                    seed=seed,
                    fixed_class_order=fixed_class_order,
                    train_transform=altered_train_transform,
                    eval_transform=test_transform,
                    dataset_root=dset_rootpath
        )
        scenario.n_classes = n_classes
        scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes // n_experiences

    # MiniImageNet
    elif scenario_name == 'miniimgnet':
        n_classes = 100
        n_experiences = 20

        fixed_class_order = [i for i in range(n_classes)]
        if args.use_rand_class_ordering:
            print("Using random class ordering for MiniImageNet...")
            fixed_class_order = np.random.permutation(n_classes).tolist()
            print("the order is ", fixed_class_order)

        if not num_experiences is None:
            n_experiences = num_experiences

        train_transform, test_transform = get_transforms(
                                            use_data_aug, 
                                            input_size=input_size, 
                                            use_to_pil=True,
                                            norm_mean=norm_mean if norm_mean else (0.4914, 0.4822, 0.4465), 
                                            norm_std=norm_stddev if norm_stddev else (0.2023, 0.1994, 0.2010)
        )

        altered_train_transform = train_transform #NOTE: need this potentially redundant line for 'supcon' strategy
        if 'supcon' in args.strategy or 'barlow_twins' in args.strategy:
            altered_train_transform = TwoCropTransform(train_transform)
  
        scenario = SplitMiniImageNet(
                    dset_rootpath,
                    n_experiences=n_experiences, 
                    return_task_id=args.task_incr, # NOTE: args.dset_rootpath as first argument (original code)
                    seed=seed, per_exp_classes=args.per_exp_classes_dict,
                    fixed_class_order=fixed_class_order, 
                    preprocessed=True,
                    train_transform=altered_train_transform, 
                    test_transform=test_transform 
        )
        scenario.n_classes = n_classes
        scenario.fixed_class_order = fixed_class_order
        initial_out_features = n_classes // n_experiences  # For Multi-Head

    else:
        raise ValueError("Unknown scenario name.")

    # Cutoff if applicable
    scenario.train_stream = scenario.train_stream[: args.partial_num_tasks]
    scenario.test_stream = scenario.test_stream[: args.partial_num_tasks]

    # Pack transforms #NOTE: this is necessary because I am unable to retrieve the transforms from the scenario object (or avalanche dataset)
    data_transforms["train"] = train_transform
    data_transforms["eval"] = test_transform
    print(f"Scenario = {scenario_name}")

    return scenario, data_transforms, input_size, initial_out_features


def get_continual_evaluation_plugins(args, scenario):
    """Plugins for per-iteration evaluation in Avalanche."""
    assert args.eval_periodicity >= 1, "Need positive "

    if args.eval_with_test_data:
        args.evalstream_during_training = scenario.test_stream  # Entire test stream
    else:
        args.evalstream_during_training = scenario.train_stream  # Entire train stream
    print(f"Evaluating on stream (eval={args.eval_with_test_data}): {args.evalstream_during_training}")

    # Metrics
    loss_tracking = TaskTrackingLossPluginMetric()
    
    # Expensive metrics
    gradnorm_tracking = None
    if args.track_gradnorm:
        gradnorm_tracking = TaskTrackingGradnormPluginMetric() # if args.track_gradnorm else None  # Memory+compute expensive

    # Acc derived plugins
    acc_tracking = TaskTrackingAccuracyPluginMetric()
#
    acc_min = TaskTrackingMINAccuracyPluginMetric()
    #acc_min_avg = TaskAveragingPluginMetric(acc_min)
    #wc_acc_avg = WCACCPluginMetric(acc_min)

    tracking_plugins = [
        loss_tracking, gradnorm_tracking, acc_tracking, 
        #acc_min, acc_min_avg, #wc_acc_avg
    ]
    tracking_plugins = [p for p in tracking_plugins if p is not None]

    trackerphase_plugin = ContinualEvaluationPhasePlugin(tracking_plugins=tracking_plugins,
                                                         max_task_subset_size=args.eval_task_subset_size,
                                                         eval_stream=args.evalstream_during_training,
                                                         eval_max_iterations=args.eval_max_iterations,
                                                         mb_update_freq=args.eval_periodicity,
                                                         num_workers=args.num_workers,
    )
    return [trackerphase_plugin, *tracking_plugins]


def get_metrics(scenario, args, data_transforms):
    """Metrics are calculated efficiently as running avgs."""

    # Pass plugins, but stat_collector must be called first
    minibatch_tracker_plugins = []

    # Stats on external tracking stream
    if args.enable_continual_eval:
        tracking_plugins = get_continual_evaluation_plugins(args, scenario)
        minibatch_tracker_plugins.extend(tracking_plugins)

    metrics = [
        #accuracy_metrics(experience=True, stream=True), 
        #loss_metrics(minibatch=True, experience=True, stream=True),
        #ExperienceForgetting(),  # Test only
        #StreamForgetting(),  # Test only
        #StreamConfusionMatrix(num_classes=scenario.n_classes, save_image=True),

        # CONTINUAL EVAL
        *minibatch_tracker_plugins,

        # LOG OTHER STATS
        # timing_metrics(epoch=True, experience=False),
        # cpu_usage_metrics(experience=True),
        # DiskUsageMonitor(),
        # MinibatchMaxRAM(),
        # GpuUsageMonitor(0),
    ]
    if not "supcon" in args.strategy\
        and not "barlow" in args.strategy\
        and not "concat" in args.strategy:
        metrics.append(accuracy_metrics(experience=True, stream=True))


    # Linear Probing Evaluation
    if args.use_lp_eval:
        print("\nAdding a probing eval plugin")
        if "linear_ER" in args.use_lp_eval:
            print("Using linear probe on replay buffer!")
            metrics.append(LinearProbingAccuracyMetric(
                            criterion=torch.nn.CrossEntropyLoss(),
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            eval_all=args.lp_eval_all, 
                            train_mb_size=128,
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            num_head_copies=args.lp_probe_repetitions,
                            skip_initial_eval=args.skip_initial_eval,
                            num_workers=args.num_workers,
                            buffer_lp_dataset=args.lp_buffer_dataset,
                            normalize_features=args.lp_normalize_features,
                            train_stream_from_ER_buffer=True
                        )
            )

        elif "concat_linear" in args.use_lp_eval:
            print("Using concat linear probe")
            metrics.append(ConcatenatedLinearProbingAccuracyMetric(
                criterion=torch.nn.CrossEntropyLoss(),
                train_stream=scenario.train_stream,
                train_mb_size=args.bs,
                test_stream=scenario.test_stream,
                eval_all=args.lp_eval_all,
                force_task_eval=args.lp_force_task_eval,
                num_finetune_epochs=args.lp_finetune_epochs,
                num_head_copies=args.lp_probe_repetitions,
                skip_initial_eval=args.skip_initial_eval,
                num_workers=args.num_workers,
                buffer_lp_dataset=args.lp_buffer_dataset,
                normalize_features=args.lp_normalize_features)
            )
        
        elif "concat_linear_reduce" in args.use_lp_eval:
            print("Using concat linear probe with PCA reduction")
            metrics.append(ConcatenatedLinearProbingAccuracyMetric(
                criterion=torch.nn.CrossEntropyLoss(),
                train_stream=scenario.train_stream,
                train_mb_size=args.bs,
                test_stream=scenario.test_stream,
                eval_all=args.lp_eval_all,
                force_task_eval=args.lp_force_task_eval,
                num_finetune_epochs=args.lp_finetune_epochs,
                num_head_copies=args.lp_probe_repetitions,
                skip_initial_eval=args.skip_initial_eval,
                num_workers=args.num_workers,
                buffer_lp_dataset=args.lp_buffer_dataset,
                normalize_features=args.lp_normalize_features,
                reduce_dim_in_head=args.lp_reduce_dim,
                pca_on_subset=args.lp_pca_on_subset)
            )

        elif "linear" in args.use_lp_eval and args.lp_eval_last_n is None:
            print("Using linear probe")
            metrics.append(LinearProbingAccuracyMetric(
                            criterion=torch.nn.CrossEntropyLoss(),
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            eval_all=args.lp_eval_all, 
                            train_mb_size=128,
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            num_head_copies=args.lp_probe_repetitions,
                            skip_initial_eval=args.skip_initial_eval,
                            num_workers=args.num_workers,
                            buffer_lp_dataset=args.lp_buffer_dataset,
                            normalize_features=args.lp_normalize_features,
                        )
            )
        elif "linear" in args.use_lp_eval:
            print("Using linear probe and finetune after layer", args.lp_eval_last_n)
            metrics.append(LinearProbingLastNAccuracyMetricLastN(
                            layer_N=args.lp_eval_last_n,
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            eval_all=args.lp_eval_all, 
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            num_head_copies=args.lp_probe_repetitions,
                            skip_initial_eval=args.skip_initial_eval
                        )
            )

            
        if 'knn_ER' in args.use_lp_eval:
            print("\n\nUsing KNN probe on replay buffer!\n\n")
            metrics.append(KNNProbingAccuracyMetric(
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            k=args.knn_k,
                            train_mb_size=args.bs,
                            eval_all=args.lp_eval_all, 
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            skip_initial_eval=args.skip_initial_eval,
                            train_stream_from_ER_buffer=True
                        )
            )

        elif "concat_knn" in args.use_lp_eval:
            print("Using KNN probe")
            metrics.append(ConcatenatedKNNProbingAccuracyMetric(
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            k=args.knn_k,
                            train_mb_size=args.bs,
                            eval_all=args.lp_eval_all, 
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            skip_initial_eval=args.skip_initial_eval,
                            reduce_dim_in_head=args.lp_reduce_dim,
                            pca_on_subset=args.lp_pca_on_subset
                        )
            )

        elif "knn_pca" in args.use_lp_eval:
            print("Using KNN probe with PCA step")
            metrics.append(KNNPCAProbingAccuracyMetric(
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            k=args.knn_k,
                            train_mb_size=args.bs,
                            eval_all=args.lp_eval_all, 
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            skip_initial_eval=args.skip_initial_eval,
                            reduce_dim_in_head=args.lp_reduce_dim,
                            pca_on_subset=args.lp_pca_on_subset
                        )
            )

        elif "knn" in args.use_lp_eval:
            print("Using KNN probe")
            metrics.append(KNNProbingAccuracyMetric(
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            k=args.knn_k,
                            train_mb_size=args.bs,
                            eval_all=args.lp_eval_all, 
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            skip_initial_eval=args.skip_initial_eval
                        )
            )

        if "ncm" in args.use_lp_eval:
            print("Using NCM probe")
            metrics.append(NCMProbingAccuracyMetric(
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            k=[1],
                            train_mb_size=args.bs,
                            eval_all=args.lp_eval_all, 
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            skip_initial_eval=args.skip_initial_eval
                        )
            )
        elif "concat_ncm" in args.use_lp_eval:
            print("Using NCM probe")
            metrics.append(ConcatenatedNCMProbingAccuracyMetric(
                            train_stream=scenario.train_stream, 
                            test_stream=scenario.test_stream,
                            k=[1],
                            train_mb_size=args.bs,
                            eval_all=args.lp_eval_all, 
                            force_task_eval=args.lp_force_task_eval,
                            num_finetune_epochs=args.lp_finetune_epochs,
                            skip_initial_eval=args.skip_initial_eval,
                            reduce_dim_in_head=args.lp_reduce_dim,
                            pca_on_subset=args.lp_pca_on_subset
                        )
            )

    # Down Stream Evaluation (Full Sets)
    if args.downstream_sets and args.downstream_method:
        assert len(args.downstream_sets) == len(args.downstream_rootpaths), "Must specify a rootpath for each downstream set"

        for task_id in range(len(args.downstream_sets)):
            # Get the respective dataset
            print("\nPrepare dataset for downstream task")
            print("args.downstream_sets[task_id]", args.downstream_sets[task_id])
            print("args.downstream_rootpaths[task_id]", args.downstream_rootpaths[task_id])
            train_set, test_set, n_classes = get_dataset(
                    scenario_name=args.downstream_sets[task_id], 
                    dset_rootpath=args.downstream_rootpaths[task_id], 
                    train_transform=data_transforms["train"], 
                    eval_transform=data_transforms["eval"],
                    #subsample_classes= scenario.fixed_class_order[
                    #        args.downstream_subsample_tasks[0]*(scenario.n_classes // len(scenario.train_stream)):
                    #        (args.downstream_subsample_tasks[1]+1)*(scenario.n_classes // len(scenario.train_stream))]
            )
            # Initialize finetune metric
            if "finetune" in args.downstream_method:
                metrics.append(DownstreamFinetuneAccuracyMetric(
                        args=args, 
                        criterion=torch.nn.CrossEntropyLoss(),
                        downstream_task=args.downstream_sets[task_id],
                        train_set=train_set, 
                        eval_set=test_set,
                        train_mb_size=args.bs,
                        eval_mb_size=args.bs,
                        n_classes = n_classes,
                        freeze_from=args.ft_freeze_from,
                        freeze_to=args.ft_freeze_to,
                ))
                print("Added finetune donwstream evaluation for task", args.downstream_sets[task_id])

            # Initialize linear metric
            if "linear" in args.downstream_method:
                if args.use_concatenated_eval:
                    metrics.append(ConcatDownStreamProbingAccuracyMetric(
                                    args=args, 
                                    downstream_task=args.downstream_sets[task_id], 
                                    train_set=train_set, eval_set=test_set, 
                                    n_classes=n_classes
                        ))
                    print("Added concatenated downstream evaluation for task", args.downstream_sets[task_id])
                    continue # NOTE: continue to not append the downstream metric as well because its redundant
                metrics.append(DownstreamLinearProbeAccuracyMetric(
                                    args=args, 
                                    criterion=torch.nn.CrossEntropyLoss(),
                                    downstream_task=args.downstream_sets[task_id],
                                    train_set=train_set, 
                                    eval_set=test_set,
                                    train_mb_size=args.bs,
                                    eval_mb_size=args.bs,
                                    n_classes = n_classes,
                                    buffer_lp_dataset=args.lp_buffer_dataset,
                        )
                    )
                print("Added linear downstream evaluation for", args.downstream_sets[task_id])

            # Initialize downstream metrics
            if "knn" in args.downstream_method:
                metrics.append(DownstreamKKNNAccuracyMetric(
                                args=args, 
                                downstream_task=args.downstream_sets[task_id], 
                                train_set=train_set, 
                                eval_set=test_set,
                                k=args.knn_k
                        )
                    )
                print("Added KNN downstream evaluation for", args.downstream_sets[task_id])
        
    # Down Stream Evaluation (Subsampled Sets)
    if not args.downstream_subsample_sets is None:
        assert len(args.downstream_subsample_sets) == len(args.downstream_subsample_rootpaths), "Must specify a rootpath for each downstream set"
        assert args.downstream_method, "Must specify a downstream method"
        assert len(args.downstream_subsample_tasks) == 2*len(args.downstream_subsample_sets), "Must specify a start and end task for EACH subsampling set"

        for task_id in range(len(args.downstream_subsample_sets)):
            # Get the respective dataset
            train_set, test_set, n_classes = get_dataset(
                    scenario_name=args.downstream_subsample_sets[task_id], 
                    dset_rootpath=args.downstream_subsample_rootpaths[task_id],
                    train_transform=data_transforms["eval"], #train_transform=data_transforms["train"], 
                    eval_transform=data_transforms["eval"],
                    subsample_classes = scenario.fixed_class_order[
                            args.downstream_subsample_tasks[(2*task_id)]*(scenario.n_classes // len(scenario.train_stream)):
                            (args.downstream_subsample_tasks[(2*task_id)+1]+1)*(scenario.n_classes // len(scenario.train_stream))]
            )
            if "linear_ER" in args.downstream_method:
                suffix = "_" + str(args.downstream_subsample_tasks[(2*task_id)]) + "-" + str(args.downstream_subsample_tasks[(2*task_id)+1])
                metrics.append(DownstreamLinearProbeAccuracyMetric(
                                    args=args, 
                                    criterion=torch.nn.CrossEntropyLoss(),
                                    downstream_task=args.downstream_subsample_sets[task_id] + suffix,
                                    train_set=train_set, 
                                    eval_set=test_set,
                                    train_mb_size=args.bs,
                                    eval_mb_size=args.bs,
                                    n_classes = n_classes,
                                    buffer_lp_dataset=args.lp_buffer_dataset,
                                    train_stream_from_ER_buffer=True
                                )
                )
            elif "linear" in args.downstream_method:
                suffix = "_" + str(args.downstream_subsample_tasks[(2*task_id)]) + "-" + str(args.downstream_subsample_tasks[(2*task_id)+1])
                metrics.append(DownstreamLinearProbeAccuracyMetric(
                                    args=args, 
                                    criterion=torch.nn.CrossEntropyLoss(),
                                    downstream_task=args.downstream_subsample_sets[task_id] + suffix,
                                    train_set=train_set, 
                                    eval_set=test_set,
                                    train_mb_size=args.bs,
                                    eval_mb_size=args.bs,
                                    n_classes = n_classes,
                                    buffer_lp_dataset=args.lp_buffer_dataset,
                                )
                )
            if "knn_ER" in args.downstream_method:
                suffix = "_" + str(args.downstream_subsample_tasks[(2*task_id)]) + "-" + str(args.downstream_subsample_tasks[(2*task_id)+1])
                metrics.append(DownstreamKKNNAccuracyMetric(
                                args=args, 
                                downstream_task=args.downstream_subsample_sets[task_id] + suffix, 
                                train_set=train_set, 
                                eval_set=test_set,
                                k=args.knn_k,
                                train_stream_from_ER_buffer=True
                        )
                    )


    # PCA Evaluation
    if args.pca_eval and args.downstream_sets: # NOTE: works because float 0.0 evaluates to False
        print("Using PCA evaluation")
        for task_id in range(len(args.downstream_sets)):
            train_set, test_set, n_classes = get_dataset(
                    scenario_name=args.downstream_sets[task_id], 
                    dset_rootpath=args.downstream_rootpaths[task_id],
                    train_transform=data_transforms["train"], 
                    eval_transform=data_transforms["eval"]
                )
            if args.use_concatenated_eval:
                for threshold in args.pca_eval:
                    metrics.append(ConcatenatedPCAProbingMetric(args=args, 
                            downstream_task=args.scenario, 
                            train_set=train_set, eval_set=test_set, 
                            n_classes=n_classes,
                            pca_threshold=threshold)
                    )
            else:
                metrics.append(PCAProbingMetric(args=args, 
                            downstream_task=args.downstream_sets[task_id], 
                            eval_set=test_set, 
                            pca_threshold=args.pca_eval)
                )
            print("Added PCA downstream evaluation for", args.scenario)

    print("Plugins added...\n")
    return metrics


def get_optimizer(optim_name, 
                  model, 
                  lr, 
                  weight_decay=0.0, 
                  betas=(0.9,0.999), 
                  momentum=0.9,
                  lr_classifier=None):
    params = [{"params": model.parameters(), "lr": lr}]
    if lr_classifier is not None:
        params = [{"params": model.feature_extractor.parameters(), "lr": lr},
                  {"params": model.classifier.parameters(), "lr": lr_classifier}]
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(params,
                                   lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(params, 
                                   lr=lr, weight_decay=weight_decay, betas=betas)
    elif optim_name == 'adamW':
        optimizer = torch.optim.AdamW(params, 
                                      lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        print("No optimizer found for name", optim_name)
        raise ValueError()
    return optimizer

def get_strategy(args, model, eval_plugin, scenario, device, 
            plugins=None, data_transforms=None):
    plugins = [] if plugins is None else plugins

    # CRIT/OPTIM
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optim, 
                              model, 
                              args.lr, 
                              weight_decay=args.weight_decay, 
                              momentum=args.momentum,
                              lr_classifier=args.lr_classifier)

    initial_epochs = args.epochs[0]

    # lr scheduler per experience
    if args.one_cycle_lr_schedule:
        print("\nUsing OneCycleLR")
        # NOTE: requires num_batches, e.g. iterations per epoch
        plugins.append(
            OneCycleSchedulerPlugin(
                optimizer = optimizer,
                max_lr = args.lr,
                start_lr=args.one_cycle_start_lr,
                final_lr=args.one_cycle_final_lr, 
                epochs=args.epochs,
                warmup_epochs=args.one_cycle_warmup_epochs,
                three_phase=True
            )
        )
    else:
        # lr-schedule over experiences
        if args.lr_decay > 0:
            if len(args.epochs) > 1:
                raise NotImplementedError("lr_decay not implemented for dynamic epoch length training")
            print("\nAdding LRScheduler...")
            milestones = list(range(1, len(scenario.train_stream))) #NOTE: this is applying the lr_decay after each experience!
            print("milestines:", milestones)
            sched = ExpLRSchedulerPlugin(MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay))
            plugins.append(sched)
            print("Added ExperienceLRScheduler, with decay", args.lr_decay, "at milestones", milestones)
        elif args.lr_decay_epoch > 0:
            plugins.append(EpochLRSchedulerPlugin(
                    MultiStepLR(optimizer, milestones=args.lr_decay_steps_epoch, gamma=args.lr_decay_epoch)
                )) 
            print("Added EpochLRSchedulerPlugin, with decay", args.lr_decay_epoch, "at milestones", args.lr_decay_steps_epoch)

    # Use Iterations if defined
    if args.iterations_per_task is not None:
        args.epochs = [int(1e9)] # NOTE: something absurdly high to make sure we don't stop early
        initial_epochs = args.epochs[0]
        it_stopper = IterationsInsteadOfEpochs(max_iterations=args.iterations_per_task)
        plugins.append(it_stopper)
        print("\nUsing iterations instead of epochs, with", args.iterations_per_task, "iterations per task")


    # STRATEGY
    if args.strategy == 'finetune':
        strategy = Naive(model, optimizer, criterion,
                         train_epochs=initial_epochs, device=device,
                         train_mb_size=args.bs, evaluator=eval_plugin,
                         plugins=plugins
        )
    elif args.strategy == 'concat_finetune':
        strategy = Naive(model, optimizer, criterion,
                         train_epochs=initial_epochs, device=device,
                         train_mb_size=args.bs, evaluator=eval_plugin,
                         plugins=plugins
        )
        strategy.plugins.append(
                    ConcatFeatClassifierAdapterPlugin()   
        )  
    elif args.strategy == 'supcon':
        strategy = SupCon(
                    model, 
                    optimizer,
                    train_epochs=initial_epochs, 
                    train_mb_size=args.bs,   
                    eval_mb_size=args.bs, 
                    train_transforms=TwoCropTransform(data_transforms['train']),
                    eval_transforms=data_transforms['eval'],
                    evaluator=eval_plugin,
                    plugins=plugins,
                    supcon_temperature=args.supcon_temperature,
                    device=device
        )
    elif args.strategy == 'supcon_spread':
            strategy = SupCon(
                        model, 
                        optimizer,
                        train_epochs=initial_epochs, 
                        train_mb_size=args.bs,   
                        eval_mb_size=args.bs, 
                        train_transforms=TwoCropTransform(data_transforms['train']),
                        eval_transforms=data_transforms['eval'],
                        evaluator=eval_plugin,
                        plugins=plugins,
                        device=device,
                        use_spread_loss=True,
                        alpha=args.supcon_spread_alpha
            )
    elif args.strategy == 'supcon_joint':
        strategy = SupCon(
                    model, 
                    optimizer,
                    train_epochs=initial_epochs, 
                    train_mb_size=args.bs,   
                    eval_mb_size=args.bs, 
                    train_transforms=TwoCropTransform(data_transforms['train']),
                    eval_transforms=data_transforms['eval'],
                    evaluator=eval_plugin,
                    plugins=plugins,
                    device=device
        )        
        strategy.plugins.append(
                    JointTrainingPlugin()
        )

    elif args.strategy == 'barlow_twins':
        strategy = BarlowTwins(
                    model, 
                    optimizer,
                    train_epochs=initial_epochs, 
                    train_mb_size=args.bs,   
                    eval_mb_size=args.bs, 
                    train_transforms=TwoCropTransform(data_transforms['train']),
                    eval_transforms=data_transforms['eval'],
                    evaluator=eval_plugin,
                    plugins=plugins,
                    device=device,
                    projection_dim=args.projector_dim
        )
    elif args.strategy == 'barlow_twins_joint':
        strategy = BarlowTwins(
                    model, 
                    optimizer,
                    train_epochs=initial_epochs, 
                    train_mb_size=args.bs,   
                    eval_mb_size=args.bs, 
                    train_transforms=TwoCropTransform(data_transforms['train']),
                    eval_transforms=data_transforms['eval'],
                    evaluator=eval_plugin,
                    plugins=plugins,
                    device=device,
                    projection_dim=1024
        )
        strategy.plugins.append(
                    JointTrainingPlugin()
        )
    
    elif args.strategy == 'joint':
        strategy = BaseStrategy(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, device=device,
                    plugins=plugins
        )
        strategy.plugins.append(
                    JointTrainingPlugin()
        )
    elif args.strategy == 'concat_joint':
        strategy = BaseStrategy(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, device=device,
                    plugins=plugins
        )
        strategy.plugins.append(
                    ConcatFeatClassifierAdapterPlugin(reinit_all_backbones=True)   
        )
        strategy.plugins.append(
                    ConcatJointTrainingPlugin()
        )

    elif args.strategy == 'separate_networks':
        strategy = SeparateNetworks(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, 
                    eval_mb_size=256, 
                    eval_every=-1, 
                    evaluator=eval_plugin, 
                    device=device,
                    plugins=plugins
        )

    elif args.strategy == 'ER_avl':
        print("\nUsing ER strategy as implemneted in Avalanche")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
                                train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                                evaluator=eval_plugin, device=device,
                                plugins=[ReplayPlugin(mem_size=args.mem_size)]
        )
        
    elif args.strategy == 'ER':
        print("\nUsing custom ER strategy")
        strategy = BaseStrategy(
                    model, 
                    optimizer, 
                    criterion, 
                    train_mb_size=args.bs,
                    train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                    evaluator=eval_plugin, device=device,
                    plugins=plugins
        ) 
        strategy.plugins.append(
                    ERPlugin(
                        n_total_memories=args.mem_size, 
                        device=device,
                        replay_batch_handling=args.replay_batch_handling,
                        task_incremental=args.task_incr,
                        domain_incremental=args.domain_incr,
                        total_num_classes=scenario.n_classes,
                        num_experiences=scenario.n_experiences,
                        lmbda=args.lmbda, 
                        lmbda_warmup_steps=args.lmbda_warmup, 
                        do_decay_lmbda=args.do_decay_lmbda,
                        ace_ce_loss=args.ace_ce_loss 
                    )
        )

    elif args.strategy == 'DER':
        print("\nUsing DER strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
                                train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                                evaluator=eval_plugin, device=device,
                                plugins=plugins)
        strategy.plugins.append(
                DERPlugin(
                    mem_size = args.mem_size,
                    total_num_classes= scenario.n_classes,
                    batch_size_mem = None,
                    alpha = 0.1,
                    beta = 1.0,
                    do_decay_beta = True,
                    task_incremental = args.task_incr,
                    num_experiences = scenario.n_experiences,
                )
        )

    elif args.strategy == 'LwF':
        print("\nUsing LwF strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
                                train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                                evaluator=eval_plugin, device=device,
                                plugins=plugins)
        strategy.plugins.append(
                LwFStandardPlugin(
                    alpha=args.lmbda, 
                    temperature=2)
        )

    elif args.strategy == 'PackNet':
        print("\nUsing PackNet strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
                                train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                                evaluator=eval_plugin, device=device,
                                plugins=plugins)
        strategy.plugins.append(
                PackNetPlugin(
                    post_prune_epochs=args.post_prune_eps,
                    prune_proportion=args.prune_proportion,)
        )

    elif args.strategy == 'MAS':
        print("\nUsing MAS strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
                                train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
                                evaluator=eval_plugin, device=device,
                                plugins=plugins
        )
        strategy.plugins.append(
                    MASPlugin(
                        lambda_reg=args.lmbda, 
                        alpha=0.5)
        )

    else:
        raise NotImplementedError(f"Non existing strategy arg: {args.strategy}")
    
    #################
    # Additional auxiliary plugins
    #################
    # Evaluatie stored models
    if args.eval_stored_weights:
        print("Appending EvalStoredWeightsPlugin")
        strategy.plugins.append(
            EvalStoredWeightsPlugin(
                path_to_weights=args.eval_stored_weights
            )
        )

    # Initial Eval Stage (to probe random network performance)
    if not args.skip_initial_eval:
        print("Prepending an evaluation stage to training")
        strategy.plugins.append(InitialEvalStage(
                        test_stream=scenario.test_stream,
                        only_initial_eval=args.only_initial_eval
                    )
        )
    
    # Freeze backbone
    if args.freeze == "backbone":
        strategy.plugins.append(FreezeModelPlugin(
                exp_to_freeze_on=args.freeze_after_exp,
                backbone_only=True)
        )
    elif args.freeze == "model":
        for i in range(len(args.freeze_from)):
            freeze_from_layer_name=args.freeze_from[i] if not args.freeze_from[i] == "none" else None            
            freeze_up_to_layer_name=args.freeze_up_to[i] if not args.freeze_up_to[i] == "none" else None
            strategy.plugins.append(FreezeModelPlugin(
                exp_to_freeze_on=args.freeze_after_exp,
                freeze_from_layer_name=freeze_from_layer_name,
                freeze_up_to_layer_name=freeze_up_to_layer_name)
            )
        print("Added Plugin to freeze model")
    elif args.freeze == "all_but_bn":
        strategy.plugins.append(FreezeAllButBNPlugin())
        print("Added Plugin to freeze all but BN layers")

    # Dynamic epoch length adapter
    print("\nNum epochs:", len(args.epochs))
    if len(args.epochs) > 1:
        strategy.plugins.append(
            EpochLengthAdapterPlugin(args.epochs)
        )
        print("Added EpochLengthAdapter!")

    # Model Storing to Disk
    if args.store_models:
        strategy.plugins.append(StoreModelsPlugin(model_name=args.backbone, model_store_path=args.results_path))

    # Re-Initialize Model
    if args.reinit_model:
        reinit_until_exp = args.reinit_up_to_exp if (not args.reinit_up_to_exp is None) else (scenario.n_experiences+1)
        reinit_plugin = ReInitBackbonePlugin(
                exp_to_reinit_on=args.reinit_after_exp, 
                reinit_until_exp=reinit_until_exp,
                reinit_after_layer_name=args.reinit_layers_after,
                freeze=args.reinit_freeze,
                reinit_deterministically=args.reinit_deterministic
            )
        strategy.plugins.append(reinit_plugin)
        print("Added re-init plugin!")
        if args.adapt_epoch_length:
            strategy.plugins.append(EpochLengthAdapterPlugin(epochs=None, increase_epochs=True))


    if args.exclude_experiences:
        args.exclude_experiences = [int(x) for x in args.exclude_experiences]
        strategy.plugins.append(ExcludeExperiencePlugin(experiences=args.exclude_experiences))
        print("Added exclude experiences plugin!")
        print("Omitting experiences:", args.exclude_experiences)


    print(f"Running strategy:{strategy}")
    if hasattr(strategy, 'plugins'):
        print(f"with Plugins: {strategy.plugins}")
    return strategy


def overwrite_args_with_config(args):
    """
    Directly overwrite the input args with values defined in config yaml file.
    Only if args.config_path is defined.
    """
    if args.config_path is None:
        return
    assert os.path.isfile(args.config_path), f"Config file does not exist: {args.config_path}"

    import yaml
    with open(args.config_path, 'r') as stream:
        arg_configs = yaml.safe_load(stream)

    for arg_name, arg_val in arg_configs.items():  # Overwrite
        assert hasattr(args, arg_name), \
            f"'{arg_name}' defined in config is not specified in args, config: {args.config_path}"
        print(arg_name, arg_val, type(arg_val))
        setattr(args, arg_name, arg_val)

    print(f"Overriden args with config: {args.config_path}")