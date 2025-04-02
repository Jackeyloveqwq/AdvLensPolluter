from .base import BaseAttacker
from torch.optim import Optimizer
import torch


class OptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)
        self.optimizer = None

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def attack_loss(self, clean_bboxes_batch, output_clean_batch, adv_bboxes_batch, output_adv_batch):
        self.optimizer.zero_grad()
        loss = self.loss_func(clean_bboxes=clean_bboxes_batch, output_clean=output_clean_batch,
                                     adv_bboxes=adv_bboxes_batch, output_adv=output_adv_batch)
        target_conf_loss = loss['target_conf_loss']
        disappear_loss = loss['disappear_loss']
        untarget_iou_loss = loss['untarget_iou_loss']
        total_loss = 0.85 * target_conf_loss + 0.10 * untarget_iou_loss
        if not torch.isnan(disappear_loss):
            total_loss = total_loss + 0.05 * disappear_loss
        output = {'total_loss': total_loss, 'target_conf_loss': target_conf_loss, 'disappear_loss': disappear_loss,
                  'untarget_iou_loss': untarget_iou_loss}

        return output