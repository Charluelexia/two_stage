import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.GRA_NCD import NeighborConnectionDecoder, ReverseStage, CrossUpdateBlock
from models.backbone.swinT.swin_transformer import BasicLayer
from models.backbone.swinT.build_feature_extractor import build_feature_extractor_S, build_feature_extractor_B
from models.modules.RFB import RFB_modified


class PatchRestore(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = RFB_modified(dim, dim // 2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.reduction(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        # ---- Partial Decoder ----
        self.in_channels = in_channels
        self.NCD = NeighborConnectionDecoder(in_channels[1:])
        # # ---- Cross Update ----
        self.cub4 = CrossUpdateBlock(in_channels[3])
        self.cub3 = CrossUpdateBlock(in_channels[2])
        self.cub2 = CrossUpdateBlock(in_channels[1])
        self.cub1 = CrossUpdateBlock(in_channels[0])

    def forward(self, x):
        # Receptive Field Block (enhanced)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4, x3, x2)

        # ---- Cross Update 4 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        cu4_feat = self.cub4(x4, guidance_g.detach())

        # ---- Cross Update 3 ----
        guidance_5 = F.interpolate(cu4_feat, scale_factor=2, mode='bilinear')
        cu3_feat = self.cub3(x3, guidance_5)  #

        # ---- Cross Update 2 ----
        guidance_4 = F.interpolate(cu3_feat, scale_factor=2, mode='bilinear')
        cu2_feat = self.cub2(x2, guidance_4.detach())

        # ---- Cross Update 1 ----
        guidance_3 = F.interpolate(cu2_feat, scale_factor=2, mode='bilinear')
        cu1_feat = self.cub1(x1, guidance_3.detach())

        """
        S_g: b, 1, 28, 28
        cu4_feat: b, 1, 7, 7
        cu3_feat: b, 1, 14, 14
        cu2_feat: b, 1, 28, 28
        cu1_feat: b, 1, 56, 56
        """
        return [S_g, cu4_feat, cu3_feat, cu2_feat, cu1_feat]


class Detector(nn.Module):
    def __init__(self, feature_extractor, in_channels, depth=[0, 0, 0, 0]):
        super(Detector, self).__init__()
        self.in_channels = in_channels
        self.feature_extractor = feature_extractor
        for i, in_channel in enumerate(in_channels):
            ir = (7 * (2 ** (len(in_channels) - i - 1)))
            self.__setattr__(f"neck_stage{i}",
                             BasicLayer(in_channels[i], input_resolution=(ir, ir), depth=depth[i],
                                        num_heads=4 * (2 ** i), window_size=7, drop=0.1, attn_drop=0.1, drop_path=0.2,
                                        downsample=PatchRestore))
        self.decoder = Decoder([in_channel // 2 for in_channel in in_channels])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.feature_extractor(x)
        list_stage_feat = [getattr(self, f"neck_stage{j}")(x[j]) for j in range(0, len(self.in_channels))]
        x = self.decoder(list_stage_feat)
        return x

    def forward_for_classifier(self, x):
        B, C, H, W = x.shape
        x = self.feature_extractor(x)
        list_stage_feat = [getattr(self, f"neck_stage{j}")(x[j]) for j in range(0, len(self.in_channels))]
        list_stage_result = self.decoder(list_stage_feat)
        return x, list_stage_result

    def get_training_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())


def build_detector_S(frozen_stages=4, return_dim=False, neck_depth=[2, 2, 2, 2]):
    embed_dim = 96
    feature_extractor = build_feature_extractor_S(frozen_stages)
    model = Detector(feature_extractor, [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8], depth=neck_depth)
    if return_dim:
        return model, embed_dim
    else:
        return model


def build_detector_B(frozen_stages=4, return_dim=False):
    embed_dim = 128
    feature_extractor = build_feature_extractor_B(frozen_stages)
    model = Detector(feature_extractor, [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8])
    if return_dim:
        return model, embed_dim
    else:
        return model


if __name__ == '__main__':
    import time

    dummy_x = torch.rand(1, 3, 224, 224).to("cuda:0")
    model = build_detector_S().cuda()
    ti = time.time()
    """dummy_x0 = torch.rand(1, 48, 56, 56)
    dummy_x1 = torch.rand(1, 96, 28, 28)
    dummy_x2 = torch.rand(1, 192, 14, 14)
    dummy_x3 = torch.rand(1, 384, 7, 7)
    model = _Decoder([48, 96, 192, 384])
    a = model([dummy_x0, dummy_x1, dummy_x2, dummy_x3])"""

    a = model(dummy_x)
    print(time.time() - ti)
    pass
