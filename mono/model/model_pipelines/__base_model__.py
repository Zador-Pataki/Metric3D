import torch
import torch.nn as nn
from mono.utils.comm import get_func


class BaseDepthModel(nn.Module):
    def __init__(self, cfg, **kwargs) -> None:
        super(BaseDepthModel, self).__init__()
        model_type = cfg.model.type
        self.depth_model = get_func('mono.model.model_pipelines.' + model_type)(cfg)

    def forward(self, data):
        output = self.depth_model(**data)

        return output

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data)
        return out