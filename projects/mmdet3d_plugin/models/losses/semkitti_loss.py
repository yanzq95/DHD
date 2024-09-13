import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast

def inverse_sigmoid(x, sign='A'):
    x = x.to(torch.float32)
    while x >= 1-1e-5:
        x = x - 1e-5

    while x< 1e-5:
        x = x + 1e-5

    return -torch.log((1 / x) - 1)

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    with autocast(False):
        return (
            F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, 'A'), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'B'), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec, 'C'), torch.ones_like(spec))
        )


def sem_scal_loss(pred_, ssc_target, ignore_index=255):
    # Get softmax probabilities
    with autocast(False):
        pred = F.softmax(pred_, dim=1)      # (B, n_class, Dx, Dy, Dz)
        loss = 0
        count = 0
        mask = ssc_target != ignore_index
        n_classes = pred.shape[1]
        begin = 0
        for i in range(begin, n_classes-1):
            # Get probability of class i
            p = pred[:, i]      # (B, Dx, Dy, Dz)

            # Remove unknown voxels
            target_ori = ssc_target     # (B, Dx, Dy, Dz)
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p)+ 1e-5)
                    loss_precision = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(precision, 'D'), torch.ones_like(precision)
                        )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target) +1e-5)
                    # loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

                    loss_recall = F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'E'), torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target) +  1e-5
                    )

                    loss_specificity = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(specificity, 'F'), torch.ones_like(specificity)
                        )
                    loss_class += loss_specificity
                loss += loss_class
                # print(i, loss_class, loss_recall, loss_specificity)
        l = loss/count
        if torch.isnan(l):
            from IPython import embed
            embed()
            exit()
        return l


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    # from IPython import embed
    # embed()
    # exit()
    with autocast(False):
        loss = criterion(pred, target.long())

    return loss

def vel_loss(pred, gt):
    with autocast(False):
        return F.l1_loss(pred, gt)



def geo_scal_loss_with_mask(pred, ssc_target, camera_mask, ignore_index=255, non_empty_idx=0):

    # pred: [B,18,200,200,16]
    # ssc_target:[B,200,200,16]
    # non_empty_idx: 17

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, non_empty_idx]  # [B,200,200,16] 选取了pred中“free”类的概率
    nonempty_probs = 1 - empty_probs      # [B,200,200,16] 选取了pred中非“free”类的概率

    camera_mask = camera_mask.bool ()

    # Remove unknown voxels
    mask = (ssc_target != ignore_index)&camera_mask   # [B,200,200,16]
    nonempty_target = ssc_target != non_empty_idx  # [B,200,200,16]
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]  # [Bx200x200x16]
    empty_probs = empty_probs[mask]  # [Bx200x200x16]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    with autocast(False):
        return (
            F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, 'A'), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'B'), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec, 'C'), torch.ones_like(spec))
        )

def sem_scal_loss_with_mask(pred_, ssc_target,camera_mask,ignore_index=255):
    # Get softmax probabilities
    with autocast(False):

        pred = F.softmax(pred_, dim=1)      # (B*Dx*Dy*Dz, n_class)
        loss = 0
        count = 0
        camera_mask = camera_mask.bool()  # (B*Dx*Dy*Dz, )
        mask = (ssc_target != ignore_index) & camera_mask
        n_classes = pred.shape[1]
        begin = 0
        for i in range(begin, n_classes-1):
            # Get probability of class i
            p = pred[:, i]      # (B, Dx, Dy, Dz)

            # Remove unknown voxels
            target_ori = ssc_target     # (B, Dx, Dy, Dz)
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p)+ 1e-5)
                    loss_precision = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(precision, 'D'), torch.ones_like(precision)
                        )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target) +1e-5)
                    # loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

                    loss_recall = F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'E'), torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target) +  1e-5
                    )

                    loss_specificity = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(specificity, 'F'), torch.ones_like(specificity)
                        )
                    loss_class += loss_specificity
                loss += loss_class
                # print(i, loss_class, loss_recall, loss_specificity)
        l = loss/count
        if torch.isnan(l):
            from IPython import embed
            embed()
            exit()
        return l
