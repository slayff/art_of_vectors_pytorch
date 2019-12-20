from art_of_vectors import (
    AdversarialAttack,
    ModelFeatureExtracter
)

from art_of_vectors.dataset_utils import (
    fix_seed,
    get_images_dataloader,
    get_images_transforms,
    normalize_image
)

import torch
import torchvision

import os
import sys
from time import time
import json

import argparse

from matplotlib import pyplot as plt


IMAGES_PATH = './data'
EXP_PATH = 'exps_results/batch_size_exp'
BATCH_SIZE = 1
LAYER_FOR_EXTRACTION_NUM = 9  # block2_pool
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_GRID = range(10, 101, 10)


def make_exp():
    # preparing model
    raw_transforms = get_images_transforms()
    raw_dataloader = get_images_dataloader(IMAGES_PATH, BATCH_SIZE, transforms=raw_transforms)

    model = torchvision.models.vgg19(pretrained=True)
    layer_to_extract_from = model.features[LAYER_FOR_EXTRACTION_NUM]
    mfe = ModelFeatureExtracter(model, layer_to_extract_from).to(DEVICE)

    input_img = next(iter(raw_dataloader))['image'].to(DEVICE)
    input_shape = input_img.shape[1:]
    output_shape = mfe.extract_layer_output(input_img).shape[1:]

    # running experiment
    fix_seed(999)

    fooling_rates = []
    perturbations = []
    for batch_sz in BATCH_GRID:
        print(f'Trying to attack with batch {batch_sz}')
        start = time()

        raw_dataloader = get_images_dataloader(IMAGES_PATH, batch_sz, transforms=raw_transforms)
        adv_attack = AdversarialAttack(input_shape, output_shape, device=DEVICE, verbose=1)

        adv_attack.fit(mfe, raw_dataloader)

        pert = adv_attack.get_perturbation().cpu()
        perturbations.append(pert)

        pert_transforms = get_images_transforms(perturbation=pert)
        pert_dataloader = get_images_dataloader(IMAGES_PATH, 128, transforms=pert_transforms)
        raw_dataloader_big = get_images_dataloader(IMAGES_PATH, 128, transforms=raw_transforms)

        print('Trying to evaluate raw')
        raw_pred = adv_attack.predict_raw(mfe, raw_dataloader_big)
        print('Trying to evaluate perturbed')
        pert_pred = adv_attack.predict_raw(mfe, pert_dataloader)

        fooling_rate = adv_attack.fooling_rate(raw_pred['predictions'], pert_pred['predictions'])
        fooling_rates.append(fooling_rate)

        print(f'Ended attacking with batch {batch_sz}, fooling rate {fooling_rate}, time spent {(time() - start) / 60} mins')
        print()

    # saving experiment results
    plt.plot(BATCH_GRID, fooling_rates)
    plt.grid(b=True)
    plt.xlabel('batch size')
    plt.ylabel('fooling rate')
    plt.title('fooling rate dependency by batch size')
    plt.savefig(EXP_PATH + '/fooling_rate_dependency.png', dpi=200)

    fig, ax = plt.subplots(1, 3, figsize=(13, 13))
    fig.tight_layout()
    for i, idx in enumerate(range(0, len(BATCH_GRID), 4)):
        ax[i].imshow(normalize_image(perturbations[idx]))
        ax[i].set_axis_off()
        ax[i].set_title(f'Batch size {BATCH_GRID[idx]}')
    fig.savefig(EXP_PATH + '/sample_perturbations.png', dpi=200)

    with open(EXP_PATH + '/exp_results', 'w') as f:
        json.dump({'fooling_rates': fooling_rates, 'batch_grid': list(BATCH_GRID)}, f)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path for validating images')
    args = parser.parse_args(argv)

    global IMAGES_PATH
    if args.path is not None:
        IMAGES_PATH = args.path

    os.makedirs(EXP_PATH, exist_ok=True)

    make_exp()


if __name__ == '__main__':
    main(sys.argv)
