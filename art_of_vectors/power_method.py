import torch
import torch.nn as nn
from torch import autograd


class JacobianOperator:
    def __init__(self, img_batch, model_feature_extractor, input_dim, hidden_dim, device=torch.device('cpu')):
        self.img_batch = img_batch
        self.mfe = model_feature_extractor
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def _matvec_grad(self, img, vec):
        w = torch.zeros(self.hidden_dim, requires_grad=True).to(self.device)
        matvec_transposed = self._matvec_T_grad(img, w)
        dotproduct = torch.matmul(matvec_transposed.flatten(), vec.flatten())
        return autograd.grad(dotproduct, w)[0]

    def _matvec_T_grad(self, img, vec):
        img.requires_grad = True
        layer_output = self.mfe.extract_layer_output(img)
        dotproduct = torch.matmul(layer_output.flatten(), vec.flatten())
        return autograd.grad(dotproduct, img, create_graph=True)[0]

    def matvec(self, vec):
        output = []
        for img in self.img_batch:
            output.append(self._matvec_grad(img.unsqueeze(0), vec).detach())
        return torch.cat(output)

    def matvec_T(self, vec):
        output = torch.zeros(self.input_dim).to(self.device)
        vec_flatten = vec.reshape(-1, self.hidden_dim)
        for img, vec in zip(self.img_batch, vec_flatten):
            output += self._matvec_T_grad(img.unsqueeze(0), vec).detach().flatten()
        return output


class PowerMethod:
    def __init__(self, p=2, q=2, maxiter=20, device=torch.device('cpu'), verbose=0):
        self.p = p
        self.q = q
        self.maxiter = maxiter
        self.device = device
        self.verbose = verbose

        self.eigen_vec = None
        self.eigen_val = None

    @staticmethod
    def _psi(x, p):
        return torch.sign(x) * torch.abs(x) ** (p - 1)

    def _power_method(self, jac):
        v = self.eigen_vec
        if self.eigen_vec is None:
            v = torch.zeros(jac.input_dim).to(self.device)
            nn.init.normal_(v, std=0.2)
            v = v / torch.norm(v, p=self.p)

        p2 = 1.0 / (1.0 - 1.0 / self.p)

        for i in range(self.maxiter):
            Jv = jac.matvec(v)
            v = self._psi(jac.matvec_T(self._psi(Jv, self.q)), p2)
            v = v / torch.norm(v, p=self.p)

            if self.verbose > 0:
                s = torch.norm(jac.matvec(v), p=self.q)
                print(f'iteration {i} of PowerMethod, eigen value {s}')

        s = torch.norm(jac.matvec(v), p=self.q)
        return v.detach(), s.item()

    def fit(self, jac):
        self.eigen_vec, self.eigen_val = self._power_method(jac)

    def get_eigen_vec(self):
        return self.eigen_val

    def get_eigen_val(self):
        return self.eigen_val

    def get_perturbation(self, shape, adv_norm=10):
        return ((adv_norm / 255) * self.eigen_vec / torch.norm(self.eigen_vec, p=self.p)).reshape(shape).detach()
