import argparse
import json
import os
import subprocess
import sys

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


def run_experiment(adv_attack, mfe, batch_size=64, seed=0, info_msg=''):
    if info_msg:
        print(info_msg)

    fix_seed(seed)

    train_dataloader = get_images_dataloader('./data', batch_size, transforms=get_images_transforms())
    print('Starting power method...')
    adv_attack.fit(mfe, train_dataloader)
    print('Done power method!')

    generated_pert = adv_attack.get_perturbation().cpu()
    
    pert_imgs_dataloader = get_images_dataloader('./data', 128, transforms=get_images_transforms(generated_pert))

    print('Starting predicting classes of perturbated images...')
    pert_ans = adv_attack.predict_raw(mfe, pert_imgs_dataloader)
    print('Done predicting!')

    eigen_value = adv_attack.power_method.eigen_val

    print('Done with experiment')
    print('='*50)

    return dict(perturbation=generated_pert, eigen_value=eigen_value, perturbated_answers=pert_ans)


def make_random_samples_set(size=16):
    fix_seed(1324)
    dataset_files = os.listdir('./data')[64:]
    random_filenames = np.random.choice(dataset_files, size=size, replace=False)
    os.makedirs('./random_samples', exist_ok=True)
    
    for fn in random_filenames:
        subprocess.check_call(['cp', os.path.join('data', fn), './random_samples'])


def get_model_predictions_on_samples(mfe):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = get_images_dataloader('./random_samples/', 1, transforms=get_images_transforms())
    answers = {}
    for batch in dataloader:
        logits = mfe(batch['image'].to(device))
        probs = torch.softmax(logits, -1)
        
        prob, target_class = map(lambda x: x.item(), torch.max(probs, dim=-1))
        answers[batch['name'][0]] = {
            'prediction': idx2label[target_class],
            'class_id': target_class,
            'probability': prob
        }

    return answers


def evaluate_perturbation_on_samples(mfe, perturbation, directory):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p = perturbation.permute(1, 2, 0).numpy() 
    mx = p.max()
    mn = p.min()
    plt.imshow((p - mn) / (mx - mn))
    plt.savefig(os.path.join(directory, 'perturbation.jpg'), dpi=200)

    dataloader = get_images_dataloader('./random_samples/', 1, transforms=get_images_transforms(perturbation))

    answers = {}
    for batch in dataloader:
        logits = mfe(batch['image'].to(device))
        probs = torch.softmax(logits, -1)

        prob, target_class = map(lambda x: x.item(), torch.max(probs, dim=-1))
        answers[batch['name'][0]] = {
            'prediction': idx2label[target_class],
            'class_id': target_class,
            'probability': prob
        }

        img_pert = batch['image'][0]
        mx = img_pert.max()
        mn = img_pert.min()
        img_pert = (img_pert - mn) / (mx - mn)
        plt.axis('off')
        plt.imshow(img_pert.permute(1, 2, 0))
        plt.savefig(os.path.join(directory, batch['name'][0]), dpi=200)

    with open(os.path.join(directory, 'model_ans.json'), 'w') as f:
        json.dump(answers, f, indent=4)


def create_adversarial_attack(mfe, q=10, device=torch.device('cpu'), verbose=1):
    train_dataloader = get_images_dataloader('./data/', 1, transforms=get_images_transforms())

    input_img = next(iter(train_dataloader))['image'].to(device)
    input_shape = input_img.shape[1:]
    output_shape = mfe.extract_layer_output(input_img).shape[1:]

    return AdversarialAttack(input_shape, output_shape, q=q, pm_maxiter=20, device=device, verbose=verbose)


def run_all_experiments_with_model(model, layers, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Started exp with {model_name}')

    init_mfe = ModelFeatureExtracter(model, list(layers.values())[0]).to(device)

    raw_imgs_dataloader = get_images_dataloader('./data/', 128, transforms=get_images_transforms())

    model_initial_predictions = create_adversarial_attack(init_mfe, device=device).predict_raw(init_mfe, raw_imgs_dataloader)

    exp_results = {}
    for layer_name, layer in layers.items():
        msg = f'Running {model_name} exp with {layer_name} layer'
        mfe = ModelFeatureExtracter(model, layer)
        adv_attack = create_adversarial_attack(mfe, device=device)
        res = run_experiment(adv_attack, mfe, info_msg=msg)

        exp_results[layer_name] = res

        fooling_rate = AdversarialAttack.fooling_rate(
            model_initial_predictions['predictions'],
            res['perturbated_answers']['predictions']
        )
        print(f'Fooling rate is {fooling_rate}')

        print('Evaluating on samples...')
        exp_dir_name = f'./exps_results/{model_name}_{layer_name}_exp'
        os.makedirs(exp_dir_name, exist_ok=True)
        evaluate_perturbation_on_samples(mfe, res['perturbation'], exp_dir_name)
        
        init_answers = get_model_predictions_on_samples(mfe)
        with open(os.path.join(exp_dir_name, 'initial_predictions.json'), 'w') as f:
            json.dump(init_answers, f, indent=4)
        print(f'Done evaluating. See {exp_dir_name} for samples and model predictions')
        print("="*50)

    print(f'Done exp with {model_name}')


def run_exp_with_vgg16():
    model = torchvision.models.vgg16(pretrained=True)
    vgg16_layers_mapping = {
        'block1_conv1': model.features[0],
        'block1_pool': model.features[4],
        'block2_conv1': model.features[5],
        'block2_pool': model.features[9],
        'block3_conv1': model.features[10],
        'block3_pool': model.features[16]
    }
    run_all_experiments_with_model(model, vgg16_layers_mapping, 'vgg16')


def run_exp_with_vgg19():
    model = torchvision.models.vgg19(pretrained=True)
    vgg19_layers_mapping = {
        'block1_conv1': model.features[0],
        'block1_pool': model.features[4],
        'block2_conv1': model.features[5],
        'block2_pool': model.features[9],
        'block3_conv1': model.features[10],
        'block3_pool': model.features[18]
    }
    run_all_experiments_with_model(model, vgg19_layers_mapping, 'vgg19')


def run_exp_with_resnet50():
    model = torchvision.models.resnet50(pretrained=True)
    resnet50_layers_mapping = {
        'conv1': model.conv1,
        'pool1': model.maxpool,
        'block1_0_bn1': model.layer1[0].bn1,
        'block1_2_conv1': model.layer1[2].conv1,
        'block2': model.layer2,
        'block3': model.layer3
    }
    run_all_experiments_with_model(model, resnet50_layers_mapping, 'resnet50')


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
    os.makedirs('exps_results', exist_ok=True)
    run_exp_with_vgg16()
    run_exp_with_vgg19()
    run_exp_with_resnet50()


if __name__ == '__main__':
    main(sys.argv[1:])
