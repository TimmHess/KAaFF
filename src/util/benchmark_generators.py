################################################################################
# Copyright (c) 2021 Timm Hess.                                                #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: TODO                                                                   #
# Author(s): Timm Hess                                                         #
# E-mail: timmfelix.hess@esat.kuleuven.be                                      #
# Website: avalanche.continualai.org                                           #
################################################################################

from functools import partial
from itertools import tee
from typing import Sequence, Optional, Dict, Union, Any, List, Callable, Set, \
    Tuple, Iterable, Generator

import torch

from avalanche.benchmarks import GenericCLScenario, Experience, \
    GenericScenarioStream
from avalanche.benchmarks.scenarios.generic_benchmark_creation import *
from avalanche.benchmarks.scenarios.generic_cl_scenario import \
    TStreamsUserDict, StreamUserDef
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import \
    NCScenario
from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario
from avalanche.benchmarks.utils import concat_datasets_sequentially
from avalanche.benchmarks.utils.avalanche_dataset import SupportedDataset, \
    AvalancheDataset, AvalancheDatasetType, AvalancheSubset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence


from src.util.corruption_scenario import NICorruptionScenario


def ni_corruption_benchmark(
        train_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        test_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        n_experiences: int,
        *,
        task_labels: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        corruption_set: str = None,
        train_transform=None,
        eval_transform=None,
        #reproducibility_data: Optional[Dict[str, Any]] = None) 
        complete_test_set_only=False
        ) -> NICorruptionScenario:
    
    # transform_groups = dict(
    #     train=(train_transform, None),
    #     eval=(eval_transform, None)
    # )

    seq_train_dataset = AvalancheDataset(
        train_dataset,
        #transform_groups=transform_groups, # TODO: Potentially I need to erase the transform_groups here - it is running 2 times I am guessing because of this shit...
        transform_groups=None,
        initial_transform_group='train',
        dataset_type=AvalancheDatasetType.CLASSIFICATION)

    seq_test_dataset = AvalancheDataset(
        test_dataset,
        #transform_groups=transform_groups,
        transform_groups=None,
        initial_transform_group='eval',
        dataset_type=AvalancheDatasetType.CLASSIFICATION)

    return NICorruptionScenario(
        train_dataset=seq_train_dataset, 
        test_dataset=seq_test_dataset,
        n_experiences=n_experiences,
        corruption_set=corruption_set,
        task_labels=task_labels,
        shuffle=shuffle, 
        seed=seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
        complete_test_set_only=complete_test_set_only
    )



def create_multi_dataset_generic_benchmark(
        train_datasets: Sequence[SupportedDataset],
        test_datasets: Sequence[SupportedDataset],
        task_labels: bool = False,
        *,
        other_streams_datasets: Dict[str, Sequence[SupportedDataset]] = None,
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        eval_transform=None, eval_target_transform=None,
        other_streams_transforms: Dict[str, Tuple[Any, Any]] = None,
        dataset_type: AvalancheDatasetType = None) -> GenericCLScenario:
    """
    Creates a benchmark instance given a list of datasets. Each dataset will be
    considered as a separate experience.

    Contents of the datasets must already be set, including task labels.
    Transformations will be applied if defined.

    This function allows for the creation of custom streams as well.
    While "train" and "test" datasets must always be set, the experience list
    for other streams can be defined by using the `other_streams_datasets`
    parameter.

    If transformations are defined, they will be applied to the datasets
    of the related stream.

    :param train_datasets: A list of training datasets.
    :param test_datasets: A list of test datasets.
    :param task_labels: Whether to assign task labels for each dataset.
    :param other_streams_datasets: A dictionary describing the content of custom
        streams. Keys must be valid stream names (letters and numbers,
        not starting with a number) while the value must be a list of dataset.
        If this dictionary contains the definition for "train" or "test"
        streams then those definition will override the `train_datasets` and
        `test_datasets` parameters.
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the benchmark. This means that the ``test_dataset_list``
        parameter must be list with a single element (the complete test set).
        Defaults to False.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.
    :param other_streams_transforms: Transformations to apply to custom
        streams. If no transformations are defined for a custom stream,
        then "train" transformations will be used. This parameter must be a
        dictionary mapping stream names to transformations. The transformations
        must be a two elements tuple where the first element defines the
        X transformation while the second element is the Y transformation.
        Those elements can be None. If this dictionary contains the
        transformations for "train" or "test" streams then those transformations
        will override the `train_transform`, `train_target_transform`,
        `eval_transform` and `eval_target_transform` parameters.
    :param dataset_type: The type of the dataset. Defaults to None, which
        means that the type will be obtained from the input datasets. If input
        datasets are not instances of :class:`AvalancheDataset`, the type
        UNDEFINED will be used.

    :returns: A :class:`GenericCLScenario` instance.
    """

    transform_groups = dict(
        train=(train_transform, train_target_transform),
        eval=(eval_transform, eval_target_transform))

    if other_streams_transforms is not None:
        for stream_name, stream_transforms in other_streams_transforms.items():
            if isinstance(stream_transforms, Sequence):
                if len(stream_transforms) == 1:
                    # Suppose we got only the transformation for X values
                    stream_transforms = (stream_transforms[0], None)
            else:
                # Suppose it's the transformation for X values
                stream_transforms = (stream_transforms, None)

            transform_groups[stream_name] = stream_transforms

    input_streams = dict(
        train=train_datasets,
        test=test_datasets)

    if other_streams_datasets is not None:
        input_streams = {**input_streams, **other_streams_datasets}

    if complete_test_set_only:
        if len(input_streams['test']) != 1:
            raise ValueError('Test stream must contain one experience when'
                             'complete_test_set_only is True')

    stream_definitions = dict()

    for stream_name, dataset_list in input_streams.items():
        initial_transform_group = 'train'
        if stream_name in transform_groups:
            initial_transform_group = stream_name

        stream_datasets = []
        for dataset_idx in range(len(dataset_list)):
            dataset = dataset_list[dataset_idx]
            task_labels = ConstantSequence(dataset_idx, len(dataset))
            stream_datasets.append(AvalancheDataset(
                dataset,
                task_labels=task_labels if task_labels else None,
                transform_groups=transform_groups,
                initial_transform_group=initial_transform_group,
                dataset_type=dataset_type))
        stream_definitions[stream_name] = (stream_datasets,)

    return GenericCLScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only)