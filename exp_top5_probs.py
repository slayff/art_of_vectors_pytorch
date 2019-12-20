import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torchvision
import torchvision.models
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from art_of_vectors import AdversarialAttack, ModelFeatureExtracter
from art_of_vectors.dataset_utils import (fix_seed, get_idx2label_map,
                                          get_images_dataloader,
                                          get_images_transforms,
                                          normalize_image)


DATA_PATH = './data'
LABELS_PATH = './labels.json'
idx2label = None


def make_random_samples_set(size=16):
    fix_seed(1324)
    dataset_files = os.listdir(DATA_PATH)[64:]
    random_filenames = np.random.choice(dataset_files, size=size, replace=False)
    os.makedirs('./random_samples', exist_ok=True)

    for fn in random_filenames:
        subprocess.check_call(['cp', os.path.join(DATA_PATH, fn), './random_samples'])


def create_adversarial_attack(mfe, q=10, device=torch.device('cpu'), verbose=1):
    mfe.to(device)
    train_dataloader = get_images_dataloader(DATA_PATH, 1, transforms=get_images_transforms())

    input_img = next(iter(train_dataloader))['image'].to(device)
    input_shape = input_img.shape[1:]
    output_shape = mfe.extract_layer_output(input_img).shape[1:]
    
    return AdversarialAttack(input_shape, output_shape, q=q, pm_maxiter=20, device=device, verbose=verbose)


def get_perturbation_with_norm(model, layer, seed=0, batch_size=64, adv_norm=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mfe = ModelFeatureExtracter(model, layer)
    adv_attack = create_adversarial_attack(mfe, device=device)

    fix_seed(seed)
    train_dataloader = get_images_dataloader(DATA_PATH, batch_size, transforms=get_images_transforms())
    print('Generating perturbation...')
    adv_attack.fit(mfe, train_dataloader)
    print('Done power method!')

    return adv_attack.get_perturbation(adv_norm=adv_norm).cpu()


def make_plot(preds, grid, directory):
    top5_preds = defaultdict(list)
    for i, pred_row in enumerate(preds):
        top5 = sorted(pred_row.tolist())[::-1][:5]
        for i, val in enumerate(top5):
            top5_preds[i].append(val)

    plt.figure(figsize=(12, 7), dpi=200)
    plt.xlabel("$\\|\\| \\varepsilon \\|\\|_{\\infty}$")
    plt.ylabel("$p(x)$")
    plt.title('Top-5 probability values w.r.t. $\\infty$-norm of perturbation. Used VGG-19 and block1_pool')
    for i, values in top5_preds.items():
        plt.plot(grid, values, label=f'Top-{i+1}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'top5_probs_plot.jpg'), dpi=200)


def run_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torchvision.models.vgg19(pretrained=True)
    layer = model.features[4]

    perturbation = get_perturbation_with_norm(model, layer, adv_norm=1)

    pert_norm_grid = np.linspace(0, 30, 192)

    print('Generating imgs with perturbation...')
    all_perturbed_imgs = []
    for norm_value in pert_norm_grid:
        dl = get_images_dataloader(
            './random_samples/',
            16,
            transforms=get_images_transforms(perturbation * norm_value)
        )
        img = next(iter(dl))['image'][-2]
        all_perturbed_imgs.append(img.unsqueeze(0))
    
    all_perturbed_imgs = torch.cat(all_perturbed_imgs)

    print('Evaluating...')
    mfe = ModelFeatureExtracter(model, layer).to(device)
    logits = mfe(all_perturbed_imgs.to(device))
    probs = torch.softmax(logits, -1)
    
    exp_dir_name = 'exps_results/top5_probs_exp'
    os.makedirs(exp_dir_name, exist_ok=True)
    make_plot(probs, pert_norm_grid, exp_dir_name)
    
    for i, img in enumerate(all_perturbed_imgs[::32]):
        plt.axis('off')
        plt.imshow(normalize_image(img))
        plt.savefig(os.path.join(exp_dir_name, f'img_with_inf_norm_{pert_norm_grid[i * 32]}.jpg'))


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

    make_random_samples_set()
    os.makedirs('exps_results/top5_probs_exp', exist_ok=True)

    run_experiment()


if __name__ == '__main__':
    main(sys.argv[1:])
