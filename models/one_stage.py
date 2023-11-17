import os

import torch
import torch.nn as nn

from models.detector import build_detector_S, build_detector_B
from models.modules.RFB import RFB_modified, RFB3D_modified, BasicConv2d

PATH_CWD = os.path.dirname(os.path.abspath(__file__))


class TAC(nn.Module):
    """Temporal artifact classifier"""

    def __init__(self, num_class, in_channels, mid_channel=16):
        super().__init__()

        self.bconv1_x = BasicConv2d(in_channels[0], in_channels[0] // 2, 3, padding=1)
        self.bconv1_y = BasicConv2d(1, mid_channel, 3, padding=1)
        self.bconv2_x = BasicConv2d(in_channels[1], in_channels[1] // 2, 3, padding=1)
        self.bconv2_y = BasicConv2d(1, mid_channel, 3, padding=1)
        self.bconv3_x = BasicConv2d(in_channels[2], in_channels[2] // 2, 3, padding=1)
        self.bconv3_y = BasicConv2d(1, mid_channel, 3, padding=1)
        self.bconv4_x = BasicConv2d(in_channels[3], in_channels[3] // 2, 3, padding=1)
        self.bconv4_y = BasicConv2d(1, mid_channel, 3, padding=1)

        self.down_sample = nn.MaxPool2d(3, stride=2, padding=1)

        self.fusion1 = self.get_fusion_sequential(in_channels[1] // 2 + in_channels[0] // 2 + 2 * mid_channel,in_channels[0])
        self.fusion2 = self.get_fusion_sequential(in_channels[2] // 2 + in_channels[0] + mid_channel, in_channels[1])
        self.fusion3 = self.get_fusion_sequential(in_channels[3] // 2 + in_channels[1] + mid_channel, in_channels[2])
        self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_channels[2], num_class)

    @staticmethod
    def get_fusion_sequential(in_channel, out_channel):
        return nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 5), padding=(1, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 3), padding=(2, 1)),
            BasicConv2d(out_channel, out_channel, 3, padding=(5, 5), dilation=(5, 5))
        )

    def forward(self, x, y):
        assert isinstance(x, list), f"The input type of x should be list"
        assert isinstance(y, list), "The input type of y should be list"
        f1, f2, f3, f4 = x
        y1, y2, y3, y4 = y

        f1 = torch.cat([self.bconv1_x(f1 + f1 * torch.sigmoid(y1)), self.bconv1_y(y1)], dim=1)
        f2 = torch.cat([self.bconv2_x(f2 + f2 * torch.sigmoid(y2)), self.bconv2_y(y2)], dim=1)
        f3 = torch.cat([self.bconv3_x(f3 + f3 * torch.sigmoid(y3)), self.bconv3_y(y3)], dim=1)
        f4 = torch.cat([self.bconv4_x(f4), self.bconv4_y(y4)], dim=1)

        f1 = self.down_sample(f1)
        f2 = self.down_sample(self.fusion1(torch.cat([f1, f2], dim=1)))
        f3 = self.down_sample(self.fusion2(torch.cat([f2, f3], dim=1)))
        f4 = self.fusion3(torch.cat([f3, f4], dim=1))
        x = self.ave_pool(f4)
        b, c, h, w = x.shape
        x=x.view([b,c])
        y = self.fc(x)
        return y


class Classifier(nn.Module):
    def __init__(self, detector, num_class, in_channels, mid_channel=16):
        super(Classifier, self).__init__()
        self.detector = detector
        self.tac = TAC(num_class=num_class, mid_channel=mid_channel, in_channels=in_channels)


    def forward(self, x):
        b, c, h, w = x.shape
        list_dtc_feat, list_dtc_result = self.detector.forward_for_classifier(x)

        list_dtc_result_tac = list_dtc_result[1:][::-1]

        list_dtc_feat_re = []
        for i, dtc_result in enumerate(list_dtc_result_tac):
            fr = list_dtc_result_tac[i].shape[-1]
            list_dtc_feat_re.append(list_dtc_feat[i].reshape([b, fr, fr, -1]).permute(0, 3, 1, 2))

        y = self.tac(list_dtc_feat_re, list_dtc_result_tac)
        return list_dtc_result,y

    def get_training_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())



def build_onestage_S(neck_depth):
    detector, dim = build_detector_S(frozen_stages=4, return_dim=True,neck_depth=neck_depth)
    classifier = Classifier(detector, num_class=2, in_channels=[dim, dim * 2, dim * 4, dim * 8])
    return classifier

"""def build_classifier_B(neck_depth, path_checkpoint_dtc=None):
    detector, dim = build_detector_B(frozen_stages=4, return_dim=True,neck_depth=neck_depth)
    if path_checkpoint_dtc == "" or not os.path.exists(path_checkpoint_dtc) or path_checkpoint_dtc is None:
        print(f"detector is not loaded to trained weights: {path_checkpoint_dtc}")
    else:
        checkpoint_detector = torch.load(path_checkpoint_dtc)
        detector.load_state_dict(checkpoint_detector['model_state_dict'])
        print("detector weights is successfully loaded, the best metric is", checkpoint_detector['metric_best'])
    classifier = Classifier(detector, num_class=2, in_channels=[dim, dim * 2, dim * 4, dim * 8])
    return classifier"""


if __name__ == '__main__':
    import time

    ti = time.time()
    """tem = 9
    dummy_x = []
    for i in range(tem):
        dummy_x += [torch.randn(1, 1, 28, 28), torch.randn(1, 1, 7, 7), torch.randn(1, 1, 14, 14),
                    torch.randn(1, 1, 28, 28), torch.randn(1, 1, 56, 56), torch.randn(1, 1, 224, 224)]
    model = NSCC(2)"""

    dummy_x = torch.rand(8, 3, 224, 224).to("cuda:0")
    model = build_onestage_S([4,4,2,2],
        path_checkpoint_dtc=r"D:\MyWorkPlace\python\ProjectZZA\tools\detection\2023-10-18-16\checkpoint_best.pth").to(
        "cuda:0")
    a = model(dummy_x)
    print(time.time() - ti)
    pass
