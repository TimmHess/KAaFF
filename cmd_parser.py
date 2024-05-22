import argparse
from distutils.util import strtobool
import json

def get_arg_parser():
    parser = argparse.ArgumentParser()

    # Meta hyperparams
    parser.add_argument('--exp_name', default="", type=str, help='Name for the experiment.')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Yaml file with config for the args.')

    parser.add_argument('--exp_postfix', type=str, default='#now,#uid',
                        help='Extension of the experiment name. A static name enables continuing if checkpointing is define'
                            'Needed for fixes/gridsearches without needing to construct a whole different directory.'
                            'To use argument values: use # before the term, and for multiple separate with commas.'
                            'e.g. #cuda,#featsize,#now,#uid')
    parser.add_argument('--save_path', type=str, default='./results/', help='save eval results.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for the dataloaders.')
    parser.add_argument('--disable_pbar', default=True, type=lambda x: bool(strtobool(x)), help='Disable progress bar')
    parser.add_argument('--n_seeds', default=5, type=int, help='Nb of seeds to run.')
    parser.add_argument('--seed', default=None, type=int, help='Run a specific seed.')
    parser.add_argument('--deterministic', default=False, type=lambda x: bool(strtobool(x)),
                        help='Set deterministic option for CUDNN backend.')
    parser.add_argument('--wandb', default=False, type=lambda x: bool(strtobool(x)), help="Use wandb for exp tracking.")

    # Dataset
    parser.add_argument('--scenario', type=str, default='cifar100',
                        choices=['cifar100', 'miniimgnet']
                        )
    parser.add_argument('--dset_rootpath', default=None, type=str, # NOTE: default='./data' (original code)
                        help='Root path of the downloaded dataset for e.g. Mini-Imagenet')  # Mini Imagenet
    parser.add_argument('--use_rand_class_ordering', action='store_true', default=False, help='Whether to use random task ordering.')
    parser.add_argument('--fixed_class_order', nargs='+', default=None, help='Fixed class order for the scenario.')
    parser.add_argument('--partial_num_tasks', type=int, default=None,
                        help='Up to which task to include, e.g. to consider only first 2 tasks of 5-task Split-MNIST')
    parser.add_argument('--exclude_experiences', nargs='+', default=None, help='Experiences to exclude.')
    parser.add_argument('--num_experiences', type=int, default=None, 
                        help='Number of experiences to use in the scenario.')
    parser.add_argument('--per_exp_classes_dict', nargs='+', default=None, 
                        help='Dict of per-experience classes to control non uniform distribution of classes per task.')

    # Feature extractor
    parser.add_argument('--featsize', type=int, default=400,
                        help='The feature size output of the feature extractor.'
                            'The classifier uses this embedding as input.')
    parser.add_argument('--backbone', type=str, choices=['input', 
                                                         'mlp', 
                                                         'resnet18_big_t', 
                                                         'resnet18_big_pt', 
                                                         'resnet18_nf21_t',
                                                         'resnet18_nf32_t', 
                                                         'resnet18_nf128_t', 
                                                         'simple_cnn', 
                                                         'vgg11', 
                                                         'resnet101_t', 
                                                         'resnext50_t'], 
                                                        default='mlp'
                      )
    parser.add_argument('--use_small_resolution_adj', action='store_true', default=False, 
                        help='Whether to adjust for small resolution (e.g. 32x32) in the backbone. Currently only for Resnet18.')
    parser.add_argument('--backbone_weights', type=str, default=None, help='Path to backbone weights.')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to custom pretrained weights.')
    parser.add_argument('--overwrite_input_size', type=int, nargs='+', default=None, help='Overwrite data input_size the backbone and add respective transform to match data.')
    
    parser.add_argument('--show_backbone_param_names', action='store_true', default=False, help='Show parameter names of the backbone.')
    parser.add_argument('--use_GAP', default=True, type=lambda x: bool(strtobool(x)),
                        help="Use Global Avg Pooling after feature extractor (for Resnet18).")
    parser.add_argument('--use_maxpool', action='store_true', default=False, help="Use maxpool after feature extractor (for WRN).")
    parser.add_argument('--use_pooling', type=str, choices=['GAP', 'MAX'], default='GAP')
    
    # Classifier
    parser.add_argument('--classifier', type=str, choices=['linear', 'norm_embed', 'identity', 'concat_linear'], default='linear',
                        help='linear classifier (prototype=weight vector for a class)'
                            'For feature-space classifiers, we output the embedding (identity) '
                            'or normalized embedding (norm_embed)')
    parser.add_argument('--lin_bias', default=True, type=lambda x: bool(strtobool(x)),
                        help="Use bias in Linear classifier")

    # Optimization
    parser.add_argument('--optim', type=str, choices=['sgd', 'adam', 'adamW'], default='sgd')
    parser.add_argument('--bs', type=int, default=128, help='Minibatch size.')
    parser.add_argument('--epochs', type=int, nargs='+', default=[50], 
                        help='Number of epochs per experience. If len(epochs) != n_experiences, then the last epoch is used for the remaining experiences.')
    parser.add_argument('--iterations_per_task', type=int, default=None,
                        help='When this is defined, it overwrites the epochs per task.'
                            'This enables equal compute per task for imbalanced scenarios.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--lr_classifier', type=float, default=None, help='Learning rate for classifier only.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--lr_decay_epoch', type=float, default=0.0, help='Learning rate decay on epoch level.')
    parser.add_argument('--lr_decay_steps_epoch', nargs='+', type=int, default=None, help='Epoch steps to decay lr.')
    parser.add_argument('--lr_milestones', type=str, default=None, help='Learning rate epoch decay milestones.')
    parser.add_argument('--lr_decay', type=float, default=0.0, help='Multiply factor on milestones.')
    parser.add_argument('--one_cycle_lr_schedule', action='store_true', default=False, help='Use OneCricleLR scheduler per experience.')
    parser.add_argument('--one_cycle_warmup_epochs', type=int, default=10, help='Number of warmup epochs for OneCycleLR.')
    parser.add_argument('--one_cycle_start_lr', type=float, default=0.0005, help='Start learning rate for OneCycleLR.')
    parser.add_argument('--one_cycle_final_lr', type=float, default=0.0005, help='Final learning rate for OneCycleLR.')

    parser.add_argument('--use_data_aug', nargs='+', default=[],
                        choices=['simclr', 'crop_flip', 'auto_cifar10', 'auto_imgnet', 'rand_crop'],
                        help='Define one or more data augmentations. This is especially helpful with corruptions')
    parser.add_argument('--overwrite_mean', nargs='+', type=float, default=None, help='Overwrite mean_norm of dataset.')
    parser.add_argument('--overwrite_stddev', nargs='+', type=float, default=None, help='Overwrite std_norm of dataset.')

    # Continual Evaluation
    parser.add_argument('--eval_with_test_data', default=True, type=lambda x: bool(strtobool(x)),
                        help="Continual eval with the test or train stream, default True for test data of datasets.")
    parser.add_argument('--enable_continual_eval', default=True, type=lambda x: bool(strtobool(x)),
                        help='Enable evaluation each eval_periodicity iterations.')
    parser.add_argument('--eval_periodicity', type=int, default=1,
                        help='Periodicity in number of iterations for continual evaluation. (None for no continual eval)')
    parser.add_argument('--eval_task_subset_size', type=int, default=-1, #original: 1000
                        help='Max nb of samples per evaluation task. (-1 if not applicable)')
    parser.add_argument('--eval_max_iterations', type=int, default=-1, help='Max nb of iterations for continual eval.\
                        After this number of iters is reached, no more continual eval is performed. Default value \
                        of -1 means no limit.')
    parser.add_argument('--skip_initial_eval', action='store_true', default=False, help='Skip initial eval.')
    parser.add_argument('--only_initial_eval', action='store_true', default=False, help='Only perform initial eval.')
    parser.add_argument('--only_prepare_data', action='store_true', default=False, help='Only prepare data.')
    parser.add_argument('--terminate_after_exp', type=int, default=None, help='Terminate training after this experience.')

    # Expensive additional continual logging
    parser.add_argument('--track_class_stats', default=False, type=lambda x: bool(strtobool(x)),
                        help="To track per-class prototype statistics, if too many classes might be better to turn off.")
    parser.add_argument('--track_gradnorm', default=False, type=lambda x: bool(strtobool(x)),
                        help="Track the gradnorm of the evaluation tasks."
                            "This accumulates computational graphs from the entire task and is very expensive memory wise."
                            "Can be made more feasible with reducing 'eval_task_subset_size'.")
    parser.add_argument('--track_features', default=False, type=lambda x: bool(strtobool(x)),
                        help="Track the features before and after a single update. This is very expensive as "
                            "entire evaluation task dataset features are forwarded and stored twice in memory."
                            "Can be made more feasible with reducing 'eval_task_subset_size'.")
    parser.add_argument('--reduced_tracking', action='store_true', default=False, 
                        help='Use reduced tracking metrics.')
    parser.add_argument('--use_lp_eval', nargs="+", type=str, default=None, 
                                choices=['linear', 'linear_ER', 'concat_linear', 'concat_linear_reduce',
                                         'knn','knn_ER', 'concat_knn', 'knn_pca'], 
                        help='Usa a probing evaluation metric')
    parser.add_argument('--use_lp_eval_ER', action='store_true', default=False, help="Use ER buffer for linear probing evaluation.")
    parser.add_argument('--lp_buffer_dataset', default=True, type=lambda x: bool(strtobool(x)), 
                        help='Buffer the linear probing dataset in memory.')
    parser.add_argument('--lp_eval_last_n', type=str, default=None, help='Use last n layers for probing evaluation.')
    parser.add_argument('--lp_optim', type=str, choices=['sgd', 'adamW'], default='adamW', 
                        help='Optimizer for linear probing.')
    parser.add_argument('--lp_lr', type=float, default=1e-3, 
                        help='Learning rate for linear probing.')
    parser.add_argument('--lp_eval_all', action='store_true', default=False, 
                        help='Use all tasks, always, for Linear Probing evaluation.')
    parser.add_argument('--lp_finetune_epochs', type=int, default=100, 
                        help='Number of epochs to finetune Linear Probing.')
    parser.add_argument('--lp_force_task_eval', action='store_true', default=False, 
                        help='Force SEPARATE evaluation of all tasks in Linear Probing.')
    parser.add_argument('--lp_normalize_features', action='store_true', default=False, 
                        help='Normalize features before Linear Probing.')
    parser.add_argument('--lp_probe_repetitions', type=int, default=1, 
                        help='Number of repetitions for Linear Probing.')
    parser.add_argument('--lp_reduce_dim', type=int, default=None, 
                        help='When using concat probe - reduce dimensionality of features for Linear Probing.')
    parser.add_argument('--lp_pca_on_subset', action='store_true', default=False, 
                        help='Use PCA on the subset of already observed data instead of whole training dataset.')

    parser.add_argument('--knn_k', nargs='+', type=int, default=[1, 10, 50, 100], 
                        help='Number of nearest neighbors for KNN probing.')
    parser.add_argument('--downstream_method', nargs='+', type=str, default=None, 
                        choices=["linear", "knn", "linear_ER", "knn_ER", "finetune"], 
                        help='Method(s) to use for down-stream tasks.')
    parser.add_argument('--downstream_sets', nargs='+', type=str, default=None, 
                        help='Name of scenarios to take the down-stream tasks from.')
    parser.add_argument('--downstream_rootpaths', nargs='+', type=str, default=None, 
                        help='Root path of the down-stream tasks.')
    parser.add_argument('--downstream_subsample_sets', nargs='+', type=str, default=None,
                        help="todo")
    parser.add_argument('--downstream_subsample_rootpaths', nargs='+', type=str, default=None,
                        help="todo")
    parser.add_argument('--downstream_subsample_tasks', nargs='+', type=int, default=None, 
                        help='Subsample the down-stream tasks. Provide start and end task indices. E.g. 10 19 will use classes from task 10 to 19')
    parser.add_argument('--use_concatenated_eval', action='store_true', default=False, 
                        help='Concatenate the representations from all backbones.')
    parser.add_argument('--pca_eval', nargs='+', type=float, default=None, 
                        help='Use PCA when threshold(s) for explainable variabtion are given.')
    parser.add_argument('--reduce_dim_in_head', action='store_true', default=False, 
                        help='Reduce the dimensionality of the head by an additional linear layer.')

    # Strategy
    parser.add_argument('--strategy', type=str, default='finetune',
                        choices=['finetune', 'joint', 'concat_joint',
                                 'LwF',
                                 'ER',
                                 'DER',
                                 'MAS',
                                 'PackNet',
                                 'supcon', 'supcon_joint'
                                 'barlow_twins', 'barlow_twins_joint',
                                 'separate_networks', 'concat_finetune'], 
                        help='Strategy to use for training.')
    parser.add_argument('--task_incr', action='store_true', default=False,
                        help="Give task ids during training to single out the head to the current task.")
    parser.add_argument('--domain_incr', action='store_true', default=False,
                        help="Rarely ever useful, but will make certain plugins behave like task-incremental\
                        which is desirable, e.g. for replay.")
    parser.add_argument('--eval_stored_weights', type=str, default=None,
                        help="When provided, will use the stored weights from the provided path to evaluate the model.")

    # ER
    parser.add_argument('--Lw_new', type=float, default=0.5,
                        help='Weight for the CE loss on the new data, in range [0,1]')
    parser.add_argument('--record_stability_gradnorm', default=False, type=lambda x: bool(strtobool(x)),
                        help="Record the gradnorm of the memory samples in current batch?")
    parser.add_argument('--mem_size', default=1000, type=int, help='Total nb of samples in rehearsal memory.')
    parser.add_argument('--replay_batch_handling', choices=['separate', 'combined'], default='separate',
                        help='How to handle the replay batch. Separate means that the replay batch is handled separately from the current batch. Combined means that the replay batch is combined with the current batch.')
    
    # LWF
    parser.add_argument('--lmbda', type=float, default=1.0, help='Regularization strength')
    parser.add_argument('--do_decay_lmbda', action='store_true', default=False, help='Decay lambda.')
    parser.add_argument('--lwf_lmbda', type=float, default=1.0, help='LwF lambda.')

    # PackNet
    parser.add_argument('--post_prune_eps', type=int, default=5, help='Number of epochs to finetune after pruning.')
    parser.add_argument('--prune_proportion', type=float, default=0.5, help='Proportion of parameters to prune.')

    # BackboneFreezing
    parser.add_argument('--freeze', type=str, default=None, choices=["model", "backbone", "all_but_bn"])
    parser.add_argument('--freeze_after_exp', type=int, default=0, help='Freeze backbone after experience n.')
    parser.add_argument('--freeze_from', nargs='+', type=str, default=["none"], help='Freeze backbone from layer name x.')
    parser.add_argument('--freeze_up_to', nargs='+', type=str, default=["none"], help='Freeze backbone up to layer name x.')
    parser.add_argument('--ft_freeze_from', type=str, default=None, help='Freeze backbone from layer name x.')
    parser.add_argument('--ft_freeze_to', type=str, default=None, help='Freeze backbone up to layer name x.')
    # Re-Initialize model after each experience
    parser.add_argument('--reinit_model', action='store_true', default=False, help='Re-initialize model after each experience.')
    parser.add_argument('--reinit_deterministic', action='store_true', default=False, 
                        help='Re-initialize the model to the same weights every time.')
    parser.add_argument('--reinit_after_exp', type=int, default=0, help='Re-initialize model after experience n.')
    parser.add_argument('--reinit_up_to_exp', type=int, default=None, help='Re-initialize model only up to experience n.')
    parser.add_argument('--reinit_layers_after', type=str, default=None, help='Reinit backbone after layer name x.')
    parser.add_argument('--reinit_freeze', action='store_true', default=False, help='Freeze backbone after reinit. This is complementary to freeze flag.')
    parser.add_argument('--adapt_epoch_length', action='store_true', default=False, help='Increase number of epochs after each experience.')

    # Store model every experience
    parser.add_argument('--store_models', action='store_true', default=False, help='Store model after each experience.')

    # Self-Supervised Projector
    parser.add_argument('--projector_dim', type=int, default=2048, 
                        help='Dimension of the projector for self-supervised learning.')
    parser.add_argument('--projector_sizes', nargs='+', type=int, default=[512,2048,2048,2048], 
                        help='Sizes of the projector for self-supervised learning.')

    # SupCon
    parser.add_argument('--supcon_temperature', type=float, default=0.07, help="Temperature for SupCon.")
    parser.add_argument('--supcon_projection_dim', type=int, default=512, help="Projection dimension for SupCon.")
    parser.add_argument('--supcon_proj_hidden_dim', type=int, default=None, help='Hidden dimension for SupCon.')

    return parser