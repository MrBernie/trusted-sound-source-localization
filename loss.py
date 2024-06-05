# contains the loss functions for the model

import torch
from torch.functional import F
import numpy as np

# loss function
def KL(self, alpha, c):
    beta = torch.ones((1, c)).to(alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0),
                    dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_kl_loss(self, p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)  # [48, 180]
    A = torch.sum(label * (torch.digamma(S) -
                    torch.digamma(alpha)), dim=1, keepdim=True)
    # annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    annealing_coef = torch.tensor(
        0.01, dtype=torch.float32).to(self.device)
    B = annealing_coef * self.KL(alp, c)
    # return (A + B)
    return A

def ce_loss_uncertainty(self, pred_batch=None, gt_batch=None, current_epoch=None):
    """ 
            Function: ce loss for uncertainty
            Args:
                    pred_batch: doa
                    gt_batch: dict{'doa'}
            Returns:
                    loss
    """
    # self.log("NO.CURRENT_EPOCH",current_epoch, sync_dist=True,on_epoch=True)
    nb, nt, _ = pred_batch.shape
    pred_batch = pred_batch.reshape(nb*nt, -1)  # [48, 180]

    gt_doa = gt_batch['doa'] * 180 / np.pi
    gt_doa = gt_doa[:, :, 1, :].type(
        torch.LongTensor).to(self.device)  # [2, 24, 1]
    gt_doa = gt_doa.reshape(nb*nt)  # [48]

    # obtain evidence
    # evidence = F.relu(pred_batch) # obtain evidence
    evidence = torch.exp(torch.clamp(pred_batch, -10, 10))

    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    U = 180 / S
    edl_loss = self.ce_kl_loss(
        gt_doa,
        alpha,
        180,
        current_epoch,
        self.lamdba_epochs,
    )
    loss_all = torch.mean(edl_loss)
    return loss_all, evidence, U

def ce_loss(self, pred_batch=None, gt_batch=None):
    """ 
            Function: ce loss
            Args:
                    pred_batch: doa
                    gt_batch: dict{'doa'}
            Returns:
                    loss
    """
    pred_doa = pred_batch
    gt_doa = gt_batch['doa'] * 180 / np.pi
    gt_doa = gt_doa[:, :, 1, :].type(torch.LongTensor).to(self.device)
    nb, nt, _ = pred_doa.shape
    pred_doa = pred_doa
    loss = torch.nn.functional.cross_entropy(
        pred_doa.reshape(nb*nt, -1), gt_doa.reshape(nb*nt))
    return loss