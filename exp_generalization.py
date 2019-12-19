import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch
import torchvision
import torchvision.models
from matplotlib import pyplot as plt

from art_of_vectors import AdversarialAttack, ModelFeatureExtracter
from art_of_vectors.dataset_utils import (fix_seed, get_idx2label_map,
                                          get_images_dataloader,
                                          get_images_transforms)

DATA_PATH = './data'
LABELS_PATH = './labels.json'
idx2label = None


def get_all_perturbations(models, layers, seed=0, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Getting perturbations for all models...')
    results = {}
    for model_name, model in models.items():
        print(f'Starting power method for {model_name}...')
        mfe = ModelFeatureExtracter(model, layers[model_name])
        adv_attack = create_adversarial_attack(mfe, device=device)

        fix_seed(seed)
        train_dataloader = get_images_dataloader(DATA_PATH, batch_size, transforms=get_images_transforms())
        adv_attack.fit(mfe, train_dataloader)
        print('Done power method!')

        results[model_name] = adv_attack.get_perturbation().cpu()

    return results


def create_adversarial_attack(mfe, q=10, device=torch.device('cpu'), verbose=1):
    mfe.to(device)
    train_dataloader = get_images_dataloader(DATA_PATH, 1, transforms=get_images_transforms())

    input_img = next(iter(train_dataloader))['image'].to(device)
    input_shape = input_img.shape[1:]
    output_shape = mfe.extract_layer_output(input_img).shape[1:]

    return AdversarialAttack(input_shape, output_shape, q=q, pm_maxiter=20, device=device, verbose=verbose)


def run_all_experiments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = defaultdict(dict)

    models = {
        'vgg16': torchvision.models.vgg16(pretrained=True),
        'vgg19': torchvision.models.vgg19(pretrained=True),
        'resnet50': torchvision.models.resnet50(pretrained=True)
    }
    layers = {
        'vgg16': models['vgg16'].features[4],
        'vgg19': models['vgg19'].features[4],
        'resnet50': models['resnet50'].maxpool
    }

    all_perturbations = get_all_perturbations(models, layers)

    raw_imgs_dataloader = get_images_dataloader(DATA_PATH, 128, transforms=get_images_transforms())
    for model_name, model in models.items():
        mfe = ModelFeatureExtracter(model, layers[model_name])
        adv_attack = create_adversarial_attack(mfe, device=device)
        print(f'Getting initial predictions for {model_name}...')
        initial_predictions = adv_attack.predict_raw(mfe, raw_imgs_dataloader)
        
        for pert_name, perturbation in all_perturbations.items():
            pert_dataloader = get_images_dataloader(DATA_PATH, 128, transforms=get_images_transforms(perturbation))
            print(f'Getting perturbated predictions for {model_name} with perturbation `{pert_name}`...')
            pert_predictions = adv_attack.predict_raw(mfe, pert_dataloader)
            
            fooling_rate = AdversarialAttack.fooling_rate(
                initial_predictions['predictions'],
                pert_predictions['predictions']
            )
            print(f'Got {fooling_rate} fooling_rate for perturbation `{pert_name}` when evaluating on {model_name}')
            results[model_name][pert_name] = fooling_rate

    return results


def configure_constants(args):
    global LABELS_PATH
    global DATA_PATH
    global idx2label
    if args.path is not None:
        DATA_PATH = args.path
    if args.labels is not None:
        LABELS_PATH = args.path
    idx2label = get_idx2label_map(LABELS_PATH)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path for validating images')
    parser.add_argument('--labels', help='Path for labels.json map file')
    args = parser.parse_args(argv)

    configure_constants(args)

    results = run_all_experiments()

    os.makedirs('exps_results/generalization_exp', exist_ok=True)
    with open('./exps_results/generalization_exp/results.json', 'w') as f:
        json.dumps(results, f, indent=4)

if __name__ == '__main__':
    main(sys.argv[1:])
