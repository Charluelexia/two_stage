import efficientnet_pytorch.model
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch import nn

class HookTool:
    def __init__(self):
        self.feat = None

    def extract_feat(self, module, fea_in, fea_out):
        self.feat = fea_out


def get_feats_by_hook(model, module, list_feat_index):
    list_hook = []
    counter = 0
    for n, m in model.named_modules():
        if isinstance(m, module):
            if counter in list_feat_index:
                hook = HookTool()
                m.register_forward_hook(hook.extract_feat)
                list_hook.append(hook)
            counter += 1

    return list_hook

class FeatureExtractor(nn.Module):
    pass