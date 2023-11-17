import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import roc_auc_score


def hybrid_stage_loss(list_output, mask):
    list_loss = []
    for output in list_output:
        b, c, h, w = output.shape
        _mask = F.interpolate(mask, output.shape[2:], mode='bilinear')
        list_loss.append(hybrid_loss(output, _mask))
    return sum(list_loss)


def hybrid_loss(pred, targ):
    B, C, H, W = pred.shape
    assert pred.shape == targ.shape
    pred = pred.float()
    targ = targ.float()
    # adaptive weighting masks
    weit = 1 + 5 * torch.abs(F.avg_pool2d(targ, kernel_size=(31, 31), stride=1, padding=(15, 15)) - targ)
    weit = torch.ones_like(weit)
    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, targ, reduction='none')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(B, C, 1, 1).repeat(1, 1, H, W)
    phiFM = pred - mpred

    mmask = targ.mean(dim=(2, 3)).view(B, C, 1, 1).repeat(1, 1, H, W)
    phiGT = targ - mmask
    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    # weighted iou loss function
    inter = ((pred * targ) * weit).sum(dim=(2, 3))
    union = ((pred + targ) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    # C represents the prediction mask for different stages, which is calculated separately using sum
    loss = (eloss + wiou + wbce).sum(dim=1).mean()
    return loss


def hybrid_weighted_loss(pred, targ, diff):
    diff = torch.mul(diff / 255, targ)

    B, C, H, W = pred.shape
    assert pred.shape == targ.shape
    pred = pred.float()
    targ = targ.float()
    # adaptive weighting masks
    weit = 1 + 5 * torch.abs(F.avg_pool2d(targ, kernel_size=(31, 31), stride=1, padding=(15, 15)) - targ)
    weit = weit + diff
    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, targ, reduction='none')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(B, C, 1, 1).repeat(1, 1, H, W)
    phiFM = pred - mpred

    mmask = targ.mean(dim=(2, 3)).view(B, C, 1, 1).repeat(1, 1, H, W)
    phiGT = targ - mmask
    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    # weighted iou loss function
    inter = ((pred * targ) * weit).sum(dim=(2, 3))
    union = ((pred + targ) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    # C represents the prediction mask for different stages, which is calculated separately using sum
    loss = (eloss + wiou + wbce).sum(dim=1).mean()
    return loss


def cal_tversky(pred, targ, smooth=1e-5, alpha=0.5, beta=0.5):
    assert pred.shape == targ.shape
    pred = torch.sigmoid(pred)
    pred_flat = pred.reshape(-1)
    target_flat = targ.reshape(-1)

    TP = (pred_flat * target_flat).sum()
    FP = ((1 - target_flat) * pred_flat).sum()
    FN = (target_flat * (1 - pred_flat)).sum()

    tversky_coeff = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return tversky_coeff.item()


def cal_aac(logits, true_labels):
    aac = 0
    prediction = logits.argmax(-1).cpu().tolist()
    true_labels = true_labels.cpu().tolist()
    aac += metrics.accuracy_score(true_labels, prediction)
    return aac


def cal_auc(logits, label):
    """
    计算AUC（Area Under the Curve）。

    Args:
    logits (Tensor): 模型的输出 logits。
    label (Tensor): 真实标签，通常是一个二元标签的张量。

    Returns:
    auc (float): 计算得到的AUC值。
    """
    # 将logits转换为概率值，使用sigmoid函数
    probabilities = torch.softmax(logits,dim=1)[:,1]

    # 获取概率值的NumPy数组
    probabilities = probabilities.cpu().numpy()
    label = label.cpu().numpy()

    # 计算AUC
    auc = roc_auc_score(label, probabilities)

    return auc


if __name__ == '__main__':
    import torch.nn

    """input=torch.triu(torch.ones((16, 4, 224, 224)), diagonal=0)"""
    list_output = [torch.randn(1, 1, 56, 56), torch.randn(1, 1, 7, 7), torch.randn(1, 1, 14, 14),
                   torch.randn(1, 1, 28, 28), torch.randn(1, 1, 56, 56), torch.randn(1, 1, 224, 224)]
    mask = torch.eye(224, 224).view([1, 1, 224, 224])
    loss = hybrid_stage_loss(list_output, mask)
    print(loss)
    """a = torch.randn((8, 5, 8, 224, 224))
    b = (1 - torch.eye(224).expand(8, 1, 8, -1, -1) - 0.1) * 1
    c = 1 - torch.eye(224).expand(8, 1, 8, -1, -1)
    d = torch.zeros((8, 1, 8, 224, 224))
    e = torch.ones((8, 1, 8, 224, 224))
    f = torch.triu(e, diagonal=0)
    print(hybrid_loss(f, f*10))
    print(cal_tversky(f, f*10))"""
