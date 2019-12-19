import torch
from tqdm import tqdm

from .power_method import PowerMethod, JacobianOperator


class AdversarialAttack:
    def __init__(
            self,
            input_shape,
            output_shape,
            p=float('inf'),
            q=10,
            pm_maxiter=20,
            device=torch.device('cpu'),
            verbose=0
    ):
        self.input_shape = input_shape
        self.input_dim = torch.prod(torch.tensor(input_shape)).item()
        self.hidden_dim = torch.prod(torch.tensor(output_shape)).item()
        self.power_method = PowerMethod(p, q, maxiter=pm_maxiter, device=device, verbose=verbose)
        self.device = device
        self.verbose = verbose

    def fit(self, mfe, img_iter, n_batches=1):
        mfe.to(self.device)
        mfe.eval()

        for i, img_data in enumerate(img_iter):
            if i >= n_batches:
                break
               
            if self.verbose:
                print(f'Running power method on batch #{i}')

            img_batch = img_data['image'].to(self.device)
            jac = JacobianOperator(img_batch, mfe, self.input_dim, self.hidden_dim, self.device)
            self.power_method.fit(jac)

    def predict_raw(self, mfe, img_iter):
        mfe.to(self.device)
        mfe.eval()

        probs = []
        preds = []
        for i, img_data in enumerate(tqdm(img_iter, disable=not self.verbose)):
            img_batch = img_data['image'].to(self.device)
            probabilities = torch.softmax(mfe(img_batch), dim=-1)
            cur_probs, cur_preds = torch.max(probabilities, dim=-1)
            
            probs.append(cur_probs)
            preds.append(cur_preds)
        
        return dict(predictions=torch.cat(preds, dim=0), probabilities=torch.cat(probs, dim=0))

    def get_perturbation(self, adv_norm=10):
        return self.power_method.get_perturbation(self.input_shape, adv_norm=adv_norm)

    @staticmethod
    def fooling_rate(model_raw_pred, model_pert_pred):
        return (model_raw_pred != model_pert_pred).float().mean().item()

