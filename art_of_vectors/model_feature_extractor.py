import torch
from torch import nn


class ModelFeatureExtracter(nn.Module):
    def __init__(self, model, layer):
        super().__init__()
        self.model = model
        self.layer = layer
        self.model.eval()

    def extract_layer_output(self, x):
        outputs = []

        def extracting_hook(module, input, output):
            outputs.append(output)

        hook_handle = self.layer.register_forward_hook(extracting_hook)

        self.model(x)

        hook_handle.remove()
        return outputs[-1]

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
