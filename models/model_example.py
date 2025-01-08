from .build import MODELS
import torch.nn as nn


@MODELS.register_module()
class model_example(nn.Module):
    def __init__(self, config):
        super(model_example, self).__init__()
        self.config = config
        self.model = nn.Linear(1024 * 14, 1024 * 14)

    def forward(self, x):
        loss_dict = {}
        x = x.reshape(-1)
        out = self.model(x)
        mse_loss = nn.MSELoss()
        loss1 = mse_loss(out, x)
        loss_dict["loss1"] = loss1
        return loss_dict