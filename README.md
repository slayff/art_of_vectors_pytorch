# art_of_vectors_pytorch


This repository contains PyTorch implementation of 
[Art of singular vectors and universal adversarial perturbations](https://arxiv.org/pdf/1709.03582.pdf) paper.

* `art_of_vectors` directory contains the core library you may use on your own

* `exp_pert_samples.py` is the code to reproduce experiment with three models: VGG-16, VGG-19, ResNet-50.
6 perturbations is constructed for each model, using 6 pre-defined layers. 
You are required to load validation data from ILSVRC 2012 and index-to-label mapping for the dataset (provided in repo).

Usage:
```
python3 exp_pert_samples.py --path [path to directory with images] --labels [path to labels.json file]
```
Script will create `exps_results` folder where you can find all the results in corresponding subfolders (named by the model and layer name) - perturbated images, visualization of generated perturbation, model predictions.

* `exp_generalization.py` is the code to reproduce experiment with generalization of perturbations across neworks. Script will create `exps_results/generalization_exp` folder with `results.json`, containing information about fooling rates. The first level keys indicate the model used in evaluation step, the second level keys indicate the model FOR which the perturbation was constructed. Thus value of `result[a][b]` is the fooling rate of model `a` when perturbation was constructed for model `b`. _Usage follows the previous experiment example_.

* `exp_top5_probs.py` is the code to reproduce experiment with plotting the dependency between top5-probabilities predicted by model on single image and infinity-norm of generated perturbation, applied to the image. VGG-19 model and block1_pool layer activations were used to construct perturbation. Script will create `exps_results/top5_probs_exp` folder with plot and sample image with different perturbations applied.
_Usage follows the previous experiment example_.