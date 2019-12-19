import torch

from .power_method import PowerMethod, JacobianOperator


class AdversarialAttack:
    def __init__(
            self,
            model_feature_extracter,
            input_shape,
            output_shape,
            p=float('inf'),
            q=10,
            pm_maxiter=20,
            device=torch.device('cpu'),
            verbose=0
    ):
        self.mfe = model_feature_extracter
        self.input_shape = input_shape
        self.input_dim = torch.prod(torch.tensor(input_shape)).item()
        self.hidden_dim = torch.prod(torch.tensor(output_shape)).item()
        self.power_method = PowerMethod(p, q, maxiter=pm_maxiter, device=device)
        self.device = device
        self.verbose = verbose

    def fit(self, img_iter, n_batches=1):
        self.mfe.to(self.device)
        self.mfe.eval()

        for i, img_data in enumerate(img_iter):
            if i >= n_batches:
                break

            if self.verbose > 0:
                print('Batch', i)

            img_batch = img_data['image'].to(self.device)
            jac = JacobianOperator(img_batch, self.mfe, self.input_dim, self.hidden_dim, self.device)
            self.power_method.fit(jac)

    @staticmethod
    def fooling_rate(model_raw_pred, model_pert_pred):
        return (model_raw_pred != model_pert_pred).float().mean().item()

    def predict_raw(self, img_iter):
        self.mfe.to(self.device)
        self.mfe.eval()

        outs = []
        for i, img_data in enumerate(img_iter):
            if self.verbose > 0:
                print('Batch', i)

            img_batch = img_data['image'].to(self.device)
            outs.append(torch.argmax(self.mfe(img_batch), dim=-1))

        return torch.cat(outs, dim=0)

    def get_perturbation(self, adv_norm=10):
        return self.power_method.get_perturbation(self.input_shape, adv_norm=adv_norm)
