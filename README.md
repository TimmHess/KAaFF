# Knowledge Accumulation in Continually Learned Representations and the Issue of Feature Forgetting (KAaFF)
[//]: # (**What is this repo?**)
This repository contains the codebase for our paper "Knowledge Accumulation in Continually Learned Representations and the Issue of Feature Forgetting". 

## Setup
This code uses
- Python 3.8
- [Avalanche](https://github.com/ContinualAI/avalanche) 0.1.0 (library code included in this repository)__*__
- PyTorch 2.0.1

To setup your python environmet we provide a [conda_environment.yml](conda_environment.yml) that mirrors our Anaconda packages.
Please use the following bash command:

    conda env create -n KAaFF --file conda_environment.yml
    conda activate KAaFF

__*__ extended by DER according to Avalanche 0.4.0a. Time constraints prevented a full migration to the latest version at this point. We hope to deliver an update in the future. 

## Reproducing results
The specific configurations for the experimentation can be found in [reproduce/configs](reproduce/configs). The yaml configuration files contain all hyper-parameter configurations we used for the experiments found in our paper.
However, system specific paramters, e.g. path-to-datasets, experiment nameing, seed, you will need to provide for your run: For example:
    
    python train.py --save_path ./results/ --dset_rootpath ./cifar100/ --num_experiences 10 --eval_max_iterations 0 --config_path ./reproduce/configs/cifar100/{CONFIG}.yml


For convenice, we also added a bash file that runs all seeds and respective "exclusion" experiments. Note that this takes some time.

    bash ./reproduce/reproduce_experiment.sh --yaml_config {yaml-config-file} --exp_name {some-name} --dset_rootpath {path-to-dataset} --save_path {path-to-store-results-to}

    # e.g. bash ./reproduce/reproduce_experiment.sh --yaml_config ./reproduce/configs/cifar100/cifar100_adamW_finetune_augmentations.yml --exp_name ft_adamW_aug --dset_rootpath ./cifar100/ --save_path ./results/

We did not run with deterministic CUDNN backbone for computational efficiency, which might result in small deviations in results.
We average all results over 5 initialization seeds, these can be found in the yaml files.


## Visualize results
At this point, we only support Tensorboard visualizations.
To view results for Tensorboard, run:

    tensorboard --logdir=OUTPUT_DIR


## Licence
Code is available under MIT license: A short and simple permissive license with conditions only requiring preservation of copyright and license notices.


## Credit
As a full disclaimer we acknowledge this code-base originated as a fork of https://github.com/Mattdl/ContinualEvaluation and thank the original authors for their open-source contribution. However, our work is not further connected.