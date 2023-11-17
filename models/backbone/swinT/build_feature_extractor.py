import os
import torch
import torch.nn as nn
from collections import OrderedDict

from models.backbone.swinT.swin_transformer import SwinTransformer, SwinTransformerBlock

PATH_CWD = os.path.dirname(os.path.abspath(__file__))
try:

    # kernel_path = os.path.abspath(os.path.join('..'))
    # sys.path.append(kernel_path)
    from models.backbone.swinT.kernels.window_process.window_process import WindowProcess, WindowProcessReverse
    # from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


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
    def __init__(self, model: nn.Module, state_dict, list_feat_index, frozen_stages=4):
        super(FeatureExtractor, self).__init__()

        new_state_dict = OrderedDict()
        model_dict = model.state_dict()
        for k, v in state_dict.items():
            if k in model_dict:
                name = k
                new_state_dict[name] = v
            else:
                print("Mismatch encountered while loading pre-training weights, skip:", k)
        model.load_state_dict(new_state_dict)

        self.extractor = model
        self.list_index_feat = list_feat_index

        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def forward(self, x):
        list_hook_feat = get_feats_by_hook(self.extractor, SwinTransformerBlock, self.list_index_feat)
        y = self.extractor(x)
        # Do not like manual operation
        output1 = list_hook_feat[0].feat  # .view(-1, 56, 56, 96).permute(0, 3, 1, 2)
        output2 = list_hook_feat[1].feat  # .view(-1, 28, 28, 192).permute(0, 3, 1, 2)
        output3 = list_hook_feat[2].feat  # .view(-1, 14, 14, 384).permute(0, 3, 1, 2)
        output4 = list_hook_feat[3].feat  # .view(-1, 7, 7, 768).permute(0, 3, 1, 2)
        return [output1, output2, output3, output4]  # [hook_feat.feat for hook_feat in list_hook_feat]

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.extractor.patch_embed.eval()
            for param in self.extractor.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.extractor.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.extractor.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(FeatureExtractor, self).train(mode)
        self._freeze_stages()


def build_feature_extractor_S(frozen_stages=4):
    model = SwinTransformer(embed_dim=96,
                            depths=[2, 2, 18, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            drop_path_rate=0.2,
                            num_classes=0
                            )
    pretrained_weight = torch.load(os.path.join(PATH_CWD, "swin_small_patch4_window7_224_22k.pth"))["model"]
    FE_S = FeatureExtractor(model, pretrained_weight, list_feat_index=[1, 3, 21, 23], frozen_stages=frozen_stages)
    return FE_S


def build_feature_extractor_B(frozen_stages=4):
    model = SwinTransformer(embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=7,
                            drop_path_rate=0.2,
                            num_classes=0
                            )
    pretrained_weight = torch.load(os.path.join(PATH_CWD, "swin_base_patch4_window7_224_22k.pth"))["model"]
    FE_S = FeatureExtractor(model, pretrained_weight, list_feat_index=[1, 3, 21, 23], frozen_stages=frozen_stages)
    return FE_S


if __name__ == '__main__':
    import time

    dummy_x = torch.rand(8, 3, 224, 224)
    t = time.time()
    fe = build_feature_extractor_B()
    fe.train()
    list_output = fe(dummy_x)
    fe.eval()
    print(time.time() - t)
    for output in list_output:
        print(output.shape)
