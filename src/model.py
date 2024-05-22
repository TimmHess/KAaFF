#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Adapted by Timm Hess (KU Leuven)
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from __future__ import division

from typing import List
from typing import NamedTuple, List, Callable
from collections import OrderedDict

import math
from copy import deepcopy

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.functional import relu, avg_pool2d, max_pool2d

from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary

from avalanche.models.dynamic_modules import MultiHeadClassifier, MultiTaskModule, IncrementalClassifier
from avalanche.models.utils import avalanche_forward
from avalanche.training import utils as avl_utils
from avalanche.training.utils import freeze_everything
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence

import libs.models.pytorch_resnet as custom_pytorch_resnet



def get_feat_size(block, spatial_size, in_channels=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.
    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    Source: https://github.com/TimmHess/OCDVAEContinualLearning/blob/master/lib/Models/architectures.py
    """

    block_device = next(block.parameters()).device
    x = torch.randn(2, in_channels, spatial_size, spatial_size).to(block_device)
    out = block(x)
    if len(out.size()) == 2: # NOTE: catches the case where the block is a linear layer
        num_feat = out.size(1)
        spatial_dim_x = 1
        spatial_dim_y = 1
    else:
        num_feat = out.size(1)
        spatial_dim_x = out.size(2)
        spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y

def initialize_weights(m) -> None:
    """
    Initilaize weights of model m.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    return 

def reinit_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()

def reinit_after(model, reinit_after=None, module_prefix=""):
    do_skip = True # NOTE: flag to skip the first layers in reinitialization
    for param_def in get_layers_and_params(model, prefix=module_prefix):
        if reinit_after in param_def.layer_name: # NOTE: if reinit_after is None, nothing is reinitialized!
            do_skip = False
        
        if do_skip: # NOTE: this will skip the first n layers in execution
            print("Skipping layer {}".format(param_def.layer_name))
            continue
        # TODO: re-add layer filter option (it was too annoying to implement)
        reinit_weights(param_def.layer)
        print("Reinitialized {}".format(param_def.layer_name))
    return
    

class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor

def get_layers_and_params(model, prefix=''):
    """
    Adapted from AvalancheCL lib
    """
    result: List[LayerAndParameter] = []
    layer_name: str
    layer: Module
    for layer_name, layer in model.named_modules():
        if layer == model:
            continue
        if isinstance(layer, nn.Sequential): # NOTE: cannot include Sequentials because this is basically a repetition of parameter listings
            continue
        layer_complete_name = prefix + layer_name + "."
        layers_and_params = avl_utils.get_layers_and_params(layer, prefix=layer_complete_name) #NOTE: this calls to avalanche function! (not self)
        result += layers_and_params

    unique_layers = OrderedDict()
    for param_def in result:
        if param_def.layer_name not in unique_layers:
            unique_layers[param_def.layer_name] = param_def
    unique_layers = list(unique_layers.values())
    return unique_layers

def freeze_from_to(
        model: Module,
        freeze_from_layer: str = None,
        freeze_until_layer: str = None,
        set_eval_mode: bool = True,
        set_requires_grad_false: bool = True,
        module_prefix: str = ""):
    
    frozen_layers = set()
    frozen_parameters = set()
    
    layer_and_params = get_layers_and_params(model, prefix=module_prefix) 

    is_freezing = False # NOTE: status flag to determine if we are freezing or not
    
    for param_def in layer_and_params:
        print("freeze_from_to:: param_def: ", param_def.layer_name)
        # Check if first layer to freeze is reached
        if not is_freezing and ((freeze_from_layer is None) or (freeze_from_layer in param_def.layer_name)):
            is_freezing = True
            print("Start freezing, including:", param_def.layer_name)

        # Check if last layer to freeze was reached
        if is_freezing and (freeze_until_layer is not None) and (freeze_until_layer in param_def.layer_name): 
            print("Stop freezing layers, not freezing:", param_def.layer_name)
            is_freezing = False
        
        if is_freezing:
            if set_requires_grad_false:
                param_def.parameter.requires_grad = False
                frozen_parameters.add(param_def.parameter_name)
            if set_eval_mode:
                param_def.layer.eval()
                frozen_layers.add(param_def.layer_name)
                
    return frozen_layers, frozen_parameters

def freeze_up_to(
            model: Module,
            freeze_until_layer: str = None,
            set_eval_mode: bool = True,
            set_requires_grad_false: bool = True,
            layer_filter: Callable[[LayerAndParameter], bool] = None,
            module_prefix: str = ""):
    """
    A simple utility that can be used to freeze a model.
    :param model: The model.
    :param freeze_until_layer: If not None, the freezing algorithm will continue
        (proceeding from the input towards the output) until the specified layer
        is encountered. The given layer is excluded from the freezing procedure.
    :param set_eval_mode: If True, the frozen layers will be set in eval mode.
        Defaults to True.
    :param set_requires_grad_false: If True, the autograd engine will be
        disabled for frozen parameters. Defaults to True.
    :param layer_filter: A function that, given a :class:`LayerParameter`,
        returns `True` if the parameter must be frozen. If all parameters of
        a layer are frozen, then the layer will be set in eval mode (according
        to the `set_eval_mode` parameter. Defaults to None, which means that all
        parameters will be frozen.
    :param module_prefix: The model prefix. Do not use if non strictly
        necessary.
    :return:
    """
    print("entering freeze-up-to function...")
    frozen_layers = set()
    frozen_parameters = set()

    to_freeze_layers = dict()
    layer_and_params = get_layers_and_params(model, prefix=module_prefix) 
    for param_def in layer_and_params:
        if(freeze_until_layer is not None
            and freeze_until_layer in param_def.layer_name): # freeze_until_layer == param_def.layer_name
            print("Will not freeze:", param_def.layer_name)
            print("Will stop freezing layers...")
            break

        print("Will freeze:", param_def.layer_name)
        freeze_param = layer_filter is None or layer_filter(param_def)
        if freeze_param:
            if set_requires_grad_false:
                param_def.parameter.requires_grad = False
                frozen_parameters.add(param_def.parameter_name)

            if param_def.layer_name not in to_freeze_layers:
                to_freeze_layers[param_def.layer_name] = (True, param_def.layer)
        else:
            # Don't freeze this parameter -> do not set eval on the layer
            to_freeze_layers[param_def.layer_name] = (False, None)

    if set_eval_mode:
        for layer_name, layer_result in to_freeze_layers.items():
            if layer_result[0]:
                layer_result[1].eval()
                frozen_layers.add(layer_name)
                
    return frozen_layers, frozen_parameters


'''
Layer Definitions
'''
class L2NormalizeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        return torch.nn.functional.normalize(x, p=2, dim=1)


class IdentityLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class FeatAvgPoolLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """ This format to be compatible with OpenSelfSup and the classifiers expecting a list."""
        # Pool
        assert x.dim() == 4, \
            "Tensor must has 4 dims, got: {}".format(x.dim())
        x = self.avg_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out



'''
Backbones
'''
class SimpleCNNFeat(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNNFeat, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            #nn.Dropout(p=0.25)
        )

        self.feature_size = self.calc_feature_size(input_size)
    
    def calc_feature_size(self, input_size):
        self.feature_size = self.features(torch.zeros(1, *input_size)).view(1, -1).size(1)
        return self.feature_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x



class VGG11ConvBlocks(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_size[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Flatten feature maps
            nn.Flatten()
        )
    
    def forward(self, x):
        x = self.features(x)
        return x

class VGG11DenseBlock(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.features = nn.Sequential(
            # # Dense
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace=True),
            # Dense 
            nn.Linear(1024, 128)
        )
    
    def forward(self, x):
        x = self.features(x)
        return x

class VGG11Feat(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.conv_blocks = VGG11ConvBlocks(input_size)
        self.conv_block_out_size = self.conv_blocks(torch.zeros(1, *input_size)).view(1, -1).size(1)

        self.dense_block = VGG11DenseBlock(self.conv_block_out_size)

        self.features = nn.Sequential(
           self.conv_blocks,
           self.dense_block
        )

        self.feature_size = self.calc_feature_size(input_size)
        print("vgg11 feature_size:", self.feature_size)
        return

    def calc_feature_size(self, input_size):
        self.feature_size = self.features(torch.zeros(1, *input_size)).size(1)
        return self.feature_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x


class MLPfeat(nn.Module):
    def_hidden_size = 400

    def __init__(self, nonlinear_embedding: bool, input_size=28 * 28,
                 hidden_sizes: tuple = None, nb_layers=2):
        """
        :param nonlinear_embedding: Include non-linearity on last embedding layer.
        This is typically True for Linear classifiers on top. But is false for embedding based algorithms.
        :param input_size:
        :param hidden_size:
        :param nb_layers:
        """
        super().__init__()
        assert nb_layers >= 2
        if hidden_sizes is None:
            hidden_sizes = [self.def_hidden_size] * nb_layers
        else:
            assert len(hidden_sizes) == nb_layers
        self.feature_size = hidden_sizes[-1]
        self.hidden_sizes = hidden_sizes

        # Need at least one non-linear layer
        layers = nn.Sequential(*(nn.Linear(input_size, hidden_sizes[0]),
                                 nn.ReLU(inplace=True)
                                 ))

        for layer_idx in range(1, nb_layers - 1):  # Not first, not last
            layers.add_module(
                f"fc{layer_idx}", nn.Sequential(
                    *(nn.Linear(hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]),
                      nn.ReLU(inplace=True)
                      )))

        # Final layer
        layers.add_module(
            f"fc{nb_layers}", nn.Sequential(
                *(nn.Linear(hidden_sizes[nb_layers - 2],
                            hidden_sizes[nb_layers - 1]),
                  )))

        # Optionally add final nonlinearity
        if nonlinear_embedding:
            layers.add_module(
                f"final_nonlinear", nn.Sequential(
                    *(nn.ReLU(inplace=True),)))

        self.features = nn.Sequential(*layers)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x

'''
ResNet
'''
class ResNet(nn.Module):
    """ ResNet feature extractor, slimmed down according to GEM paper."""

    def __init__(self, block, num_blocks, nf, global_pooling, input_size):
        """

        :param block:
        :param num_blocks:
        :param nf: Number of feature maps in each conv layer.
        """
        super(ResNet, self).__init__()
        self.global_pooling = global_pooling

        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.feature_size = None

        assert len(input_size) >= 3
        input_size = input_size[-3:]  # Only care about last 3

        if nf==20:
            if input_size == (3, 32, 32):  # Cifar10
                self.feature_size = 160 if global_pooling else 2560
            elif input_size == (3, 84, 84):  # Mini-Imagenet
                self.feature_size = 640 if global_pooling else 19360
            elif input_size == (3, 96, 96):  # TinyDomainNet
                self.feature_size = 1440 if global_pooling else 23040
            else:
                raise ValueError(f"Input size not recognized: {input_size}")
        else:
            pass

        # self.linear = nn.Linear(self.feature_size, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.shape) == 4, "Assuming x.view(bsz, C, W, H)"
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.global_pooling:
            out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # Flatten
        return out


class ResNetPT(nn.Module):
    def __init__(self, 
                 arch="resnet18", 
                 pretrained=False, 
                 nf=64,  
                 dropout_rate=0.0, 
                 use_pooling=None,
                 small_resolution_model=False
                ):
        
        super(ResNetPT, self).__init__()
        self.pretrained = pretrained
        self.feature_size = None
        if arch == "resnet18":
            if self.pretrained:
                print("\nUsing pretrained resnet18 model weights from pytorch!")
                self.features = torch.nn.Sequential(*(list((resnet18(
                                        weights=ResNet18_Weights.DEFAULT
                                    )).children())[:-1]))
            else:
                print("\nUsing resnet18 as by pytorch!")
                self.features = torch.nn.Sequential(*(
                                    list((custom_pytorch_resnet.resnet18(
                                        nf=nf, 
                                        dropout_rate=dropout_rate,
                                        use_pooling=use_pooling,
                                        small_resolution_model=small_resolution_model
                                    )).children())[:-1]))
            # else:
            #     print("\nUsing wide_resnet18 as adatped from pytorch")
            #     self.features = torch.nn.Sequential(
            #         *(list((custom_pytorch_resnet.wide_resnet18(nf=nf, dropout_rate=dropout_rate)).children())[:-1]))  
                 
        elif arch == "resnet101":
            print("\nUsing resnet101 as by pytorch!")
            self.features = torch.nn.Sequential(*(list((custom_pytorch_resnet.resnet101()).children())[:-1]))
        elif arch == "resnext50":
            print("\nUsing resnext50 as by pytorch!")
            self.features = torch.nn.Sequential(*(list((custom_pytorch_resnet.resNeXt50()).children())[:-1]))
        return

    def forward(self, x):
        assert len(x.shape) == 4, "Assuming x.view(bsz, C, W, H)"
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten
        return out

def ResNetfeat(arch, 
               input_size, 
               nf=64, 
               use_pooling=None, 
               use_torch_version=False, 
               pretrained=False, 
               dropout_rate=0.0,
               small_resolution_model=False):
    model_backbone = None
    
    if "resnet18" in arch:
        if use_torch_version and pretrained:
            assert nf==64, "Pretrained weights not available for nf!=64"
            model_backbone = ResNetPT(pretrained=pretrained, use_pooling=use_pooling) # use_pooling="GAP"
        elif use_torch_version:
            model_backbone = ResNetPT(pretrained=False, 
                                      nf=nf, 
                                      dropout_rate=dropout_rate, 
                                      use_pooling=use_pooling,
                                      small_resolution_model=small_resolution_model
                            )
        else: 
            model_backbone = ResNet(BasicBlock, [2, 2, 2, 2], 
                                    nf, 
                                    global_pooling=(True if use_pooling=="GAP" else False), 
                                    input_size=input_size
                            )
    
    elif "resnet101" in arch:
        model_backbone = ResNetPT(arch="resnet101")
    elif "resnext50" in arch:
        model_backbone = ResNetPT(arch="resnext50")

    enc_channels, enc_spatial_dim_x, enc_spatial_dim_y = get_feat_size(model_backbone, spatial_size=input_size[1], in_channels=input_size[0])
    model_backbone.feature_size = enc_channels * enc_spatial_dim_x * enc_spatial_dim_y
    print("\nModel Feature Size:", model_backbone.feature_size)
    return model_backbone


'''
Classiifer
'''
class ExRepMultiHeadClassifier(MultiHeadClassifier):
    def __init__(self, in_features, initial_out_features=2, use_bias=True):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__(in_features, 
                         initial_out_features=initial_out_features, 
                         use_bias=use_bias)
        self.exp_idx = 0

    def adaptation(self, dataset: AvalancheDataset):
        #super().adaptation(dataset)
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            tid = str(tid)  # need str keys
            new_head = IncrementalClassifier(self.in_features*(self.exp_idx+1),
                                             self.starting_out_features, 
                                             bias=self.use_bias)
            if tid == "0":
                print("Expanding classifier for task 0")
                print("new_head:", new_head)
            self.classifiers[tid] = new_head
            
    def extend_in_features(self, exp_idx):
        print("extend_in_features", exp_idx)
        self.exp_idx = exp_idx

        for tid in self.classifiers:
            new_head = IncrementalClassifier(self.in_features*(self.exp_idx+1),
                                             self.starting_out_features, 
                                             bias=self.use_bias)
            if tid == "0":
                print("Expanding classifier for task 0")
                print("new_head:", new_head)
                print("")
            self.classifiers[tid] = new_head
        return

    def forward_single_task(self, x, task_label):
        return self.classifiers[str(task_label)](x)


'''
FeatClassifierModel
'''
class FeatClassifierModel(nn.Module):
    def __init__(self, feature_extractor, classifier, with_adaptive_pool=False):
        super().__init__()
        #self.with_adaptive_pool = with_adaptive_pool

        self.feature_extractor = feature_extractor
        self.classifier = classifier  # Linear or MultiTaskHead

        self.last_features = None

    def forward_feats(self, x, task_labels=None):
        x = avalanche_forward(self.feature_extractor, x, task_labels)
        
        # store last computed features
        self.last_features = x.clone().detach()
        return x

    def forward_classifier(self, x, task_labels=None):
        try:  # Multi-task head # TODO: use avalance forward instead?
            x = self.classifier(x, task_labels)
        except:  # Single head
            x = self.classifier(x)
        return x

    def forward(self, x, task_labels=None):
        x = self.forward_feats(x, task_labels)
        x = self.forward_classifier(x, task_labels)
        return x


'''
ConcatFeatClassifierModel
'''
class ConcatFeatClassifierModel(FeatClassifierModel):
    def __init__(self, 
                feature_extractor, 
                classifier, 
                with_adaptive_pool=False,
                #use_full_features_in_train=False,
                freeze_prev_features=True):
        super().__init__(feature_extractor, classifier, with_adaptive_pool)
        
        self.feature_extractors = {0: self.feature_extractor} # NOTE: this is needed
        #self.use_full_features_in_train = use_full_features_in_train
        self.freeze_prev_features = freeze_prev_features
        return

    def forward_feats(self, x, task_labels=None):
        if self.training:
            #x = self.feature_extractor(x)
            x = avalanche_forward(self.feature_extractor, x, task_labels)
        else:
            #x = self.feature_extractors[len(self.feature_extractors)-1](x)
            x = avalanche_forward(
                self.feature_extractors[len(self.feature_extractors)-1], 
                x, task_labels)
        if self.with_adaptive_pool:
            x = self.avg_pool(x)
        # store last computed features
        self.last_features = x
        return x

    def forward_all_feats(self, x, task_labels=None):
        x_reps = []
        for key in self.feature_extractors:
            #x_reps.append(self.feature_extractors[key](x))
            x_reps.append(
                avalanche_forward(self.feature_extractors[key], x, task_labels)
            )
        x_rep = torch.concat(x_reps, dim=1)
        if self.with_adaptive_pool:
            x_rep = self.avg_pool(x_rep)
        self.last_features = x_rep
        return x_rep

    def extent_feature_extractor(self, exp_idx):
        print("\nExtending feature extractor with new branch:", exp_idx)
        # Store a reference to the active feature extractor
        self.feature_extractors[exp_idx] = self.feature_extractor
        return

    def store_feature_extractor(self, exp_idx):
        # Get a true separate copy of the feature extractor
        self.feature_extractors[exp_idx] = deepcopy(self.feature_extractor)
        if self.freeze_prev_features:
            freeze_everything(self.feature_extractors[exp_idx])
    
    def reinit_all_feature_extractors(self):
        for key in self.feature_extractors:
            self.feature_extractors[key].apply(initialize_weights)
        print("Reinitialized all feature extractors")
        return

'''
SupCon Model
'''
class SupConModel(FeatClassifierModel): # NOTE: needs to be a FeatClassifierModel for seamless integration
    def __init__(self, 
                 feature_extractor, 
                 classifier, 
                 with_adaptive_pool=False, 
                 projector_model="linear",
                 projection_dim=128, 
                 projection_hidden_dim=None):
        super(SupConModel, self).__init__(feature_extractor, classifier, with_adaptive_pool=with_adaptive_pool)
        
        self.is_use_projection_head = True
        self.projector_model = projector_model
        self.projection_dim = projection_dim
        self.embed_dim = projection_dim

        self.features_dim = self.feature_extractor.feature_size
        self.projection_hidden_dim = projection_hidden_dim
        if self.projection_hidden_dim is None:
            self.projection_hidden_dim = self.features_dim
        
        #self.projection_head = nn.Linear(self.features_dim, self.features_dim)
        #if self.projector_model == "mlp":
        self.projection_head = nn.Sequential(
                nn.Linear(self.features_dim, self.projection_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.projection_hidden_dim, self.projection_dim)
            )   

    def use_projection_head(self, mode):
        self.is_use_projection_head = mode
        if mode:
            self.embed_dim = self.projection_dim
        else:
            self.embed_dim = self.features_dim

    def forward(self, x):
        feat = self.feature_extractor(x).squeeze()
        self.last_features = feat
        if self.is_use_projection_head:
            # Use projection head
            return F.normalize(self.projection_head(feat), dim=1)
        else:
            # Use classifier
            return self.forward_classifier(F.normalize(feat, dim=1))
            

"""
BarlowTwin Model
"""
class BarlowTwinModel(FeatClassifierModel):
    """
    Code adapted from https://github.com/facebookresearch/barlowtwins/
    Copyright lies with facebookresearch.
    """
    def __init__(self, 
                 feature_extractor, 
                 classifier,
                 projection_dim,
                 projection_sizes,
                 with_adaptive_pool=False):
        super(BarlowTwinModel, self).__init__(feature_extractor, 
                                              classifier, 
                                              with_adaptive_pool=with_adaptive_pool)

        self.is_use_projection_head = True
        self.projection_dim = projection_dim #NOTE: projection_dim is dim_out projection_head
        self.embed_dim = projection_dim

        self.features_dim = self.feature_extractor.feature_size #NOTE: feautres_dim is the dimension of the backbone representation

        # Projection head
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True)) # originally bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projection_head = nn.Sequential(*layers)
        print("Projection head:")
        print(self.projection_head, "\n")

    def use_projection_head(self, mode):
        self.is_use_projection_head = mode
        if mode:
            self.embed_dim = self.projection_dim
        else:
            self.embed_dim = self.features_dim

    def forward(self, x):
        # TrainMode: If x consist of two inputs
        if x.shape[0] == 2: 
            feat_x1 = self.feature_extractor(x[0]).squeeze()
            feat_x2 = self.feature_extractor(x[1]).squeeze()
            self.last_features = [feat_x1, feat_x2]
            proj_feat_x1 = self.projection_head(feat_x1)
            proj_feat_x2 = self.projection_head(feat_x2)
            return proj_feat_x1, proj_feat_x2

        # Eval Mode: If x consist of a single input
        feat = self.feature_extractor(x).squeeze()
        self.last_features = feat
        if self.is_use_projection_head:
            # Return representation encoding
            return feat
        else:
            # Use classifier
            return self.forward_classifier(F.normalize(feat, dim=1))

########################################################################################################################
def get_model(args, n_classes, input_size, initial_out_features, backbone_weights=None, model_weights=None):
    """ 
    Build model from feature extractor and classifier.
    
    n_classes: total number of classes in the dataset
    initial_out_features: number of classes for the first task
    """
    feat_extr = _get_feat_extr(args, input_size=input_size)  # Feature extractor
    classifier = _get_classifier(args.classifier, 
                                 n_classes, 
                                 feat_extr.feature_size, 
                                 initial_out_features, 
                                 task_incr=args.task_incr,
                                 lin_bias=args.lin_bias)  # Classifier
    
    # Load weights for the backbone
    if backbone_weights:
        print("Loading backbone weights from: ", backbone_weights)
        state_dict = torch.load(backbone_weights)
        for key in list(state_dict.keys()):
            new_key = key.replace("encoder", "features")
            new_key = key.replace("feature_extractor.", "")
            state_dict[new_key] = state_dict.pop(key)
        feat_extr.load_state_dict(state_dict, strict=False)

    model = None
    if 'supcon' in args.strategy:
        model = SupConModel(feat_extr, 
                            classifier, 
                            with_adaptive_pool=False, 
                            projection_dim=args.supcon_projection_dim, 
                            projection_hidden_dim=args.supcon_proj_hidden_dim)
    elif 'barlow' in args.strategy:
        model = BarlowTwinModel(feat_extr, 
                                classifier, 
                                with_adaptive_pool=False, 
                                projection_dim=args.projector_dim,
                                projection_sizes=args.projector_sizes)
    elif 'concat_joint' in args.strategy:
        model = ConcatFeatClassifierModel(feat_extr, classifier, freeze_prev_features=False)
    elif "concat" in args.strategy:
        model = ConcatFeatClassifierModel(feat_extr, classifier, freeze_prev_features=True)
    else:
        model = FeatClassifierModel(feat_extr, classifier) # Combined model

    # Load weights for the entire model (backbone + heads)
    if model_weights:
        state_dict = torch.load(model_weights)
        print("state_dict")
        for key in state_dict:
            print(key)
        print("Loading pretrained model weights from: ", model_weights)
        model.load_state_dict(torch.load(model_weights), strict=True)

    return model

def get_model_summary(model, input_size, show_backbone_param_names=False, device='cpu'):
    summary(model.feature_extractor, input_size=input_size, device=device) 
    
    if show_backbone_param_names:
        print("Modules:")
        for module in model.feature_extractor.named_modules():
            print(module)
        print("\nStopping execution here! Remove the 'show_backbone_param_names' flag to continue!")
        import sys;sys.exit()


def _get_feat_extr(args, input_size):
    """ Get embedding network. """
    nonlin_embedding = args.classifier in ['linear']  # Layer before linear should have nonlinearities

    if args.backbone == "mlp":  # MNIST mlp
        feat_extr = MLPfeat(hidden_sizes=(400, args.featsize), nb_layers=2,
                            nonlinear_embedding=nonlin_embedding, input_size=math.prod(input_size))
    elif args.backbone == 'simple_cnn':
        feat_extr = SimpleCNNFeat(input_size=input_size)
    elif args.backbone == 'vgg11':
        feat_extr = VGG11Feat(input_size=input_size)
    elif args.backbone == "resnet18_big_t": # torch version
        feat_extr = ResNetfeat(
                        arch="resnet18", 
                        nf=64, 
                        use_pooling=args.use_pooling, 
                        input_size=input_size, 
                        use_torch_version=True, 
                        pretrained=False, 
                        dropout_rate=args.dropout_rate,
                        small_resolution_model=args.use_small_resolution_adj
                    )
    elif args.backbone == "resnet18_big_pt": # torch version - pretrained
        feat_extr = ResNetfeat(arch="resnet18", nf=64,  use_pooling=args.use_pooling, input_size=input_size, use_torch_version=True, pretrained=True)
    elif args.backbone == "resnet18_nf21_t":
        feat_extr = ResNetfeat(arch="resnet18", 
                               nf=21,  
                               use_pooling=args.use_pooling, 
                               input_size=input_size, 
                               use_torch_version=True, 
                               pretrained=False, 
                               small_resolution_model=args.use_small_resolution_adj
                    )
    elif args.backbone == "resnet18_nf32_t":
        feat_extr = ResNetfeat(arch="resnet18", 
                               nf=32,  
                               use_pooling=args.use_pooling, 
                               input_size=input_size, 
                               use_torch_version=True, 
                               pretrained=False)
    elif args.backbone == "resnet18_nf128_t":
        feat_extr = ResNetfeat(arch="resnet18", nf=128,  use_pooling=args.use_pooling, input_size=input_size, use_torch_version=True, pretrained=False)
    elif args.backbone == "resnet101_t":
        feat_extr = ResNetfeat(arch="resnet101", input_size=input_size)
    elif args.backbone == "resnext50_t":
        feat_extr = ResNetfeat(arch="resnext50", input_size=input_size)
    else:
        raise ValueError()
    return feat_extr


def _get_classifier(classifier_type: str, #args, 
                    n_classes: int, 
                    feat_size: int, 
                    initial_out_features: int, 
                    task_incr: bool,
                    lin_bias: bool = True): 
    """ 
    Get classifier head. For embedding networks this is normalization or identity layer.
    
    feat_size: Input to the linear layer
    initial_out_features: (Initial) output of the linear layer. Potenitally growing with task.
    """
    # No prototypes, final linear layer for classification
    if classifier_type == 'linear':  # Lin layer
        print("_get_classifier::feat_size: ", feat_size)
        if task_incr:
            classifier = MultiHeadClassifier(in_features=feat_size,
                                             initial_out_features=initial_out_features,
                                             use_bias=lin_bias)
        else:
            classifier = torch.nn.Linear(in_features=feat_size, out_features=n_classes, bias=lin_bias)
    elif classifier_type == 'concat_linear':
        print("_get_classifier::feat_size: ", feat_size)
        if task_incr:
            classifier = ExRepMultiHeadClassifier(in_features=feat_size,
                                                  initial_out_features=initial_out_features,
                                                  use_bias=lin_bias)
        else:
            raise NotImplementedError()
        
    # Prototypes held in strategy
    elif classifier_type == 'norm_embed':  # Get feature normalization
        classifier = L2NormalizeLayer()
    elif classifier_type == 'identity':  # Just extract embedding output
        classifier = torch.nn.Flatten()
    else:
        raise NotImplementedError()
    return classifier

