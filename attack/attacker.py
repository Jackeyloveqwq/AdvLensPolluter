import torch
import os
from utils.det_utils import plot_boxes_cv2, inter_nms
from utils.convertor import FormatConverter
from detlib.utils import init_detectors
from scripts.dict import get_attack_method, loss_dict
from attack.uap import AdvImgObject


class UniversalAttacker(object):
    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.class_names = cfg.all_class_names
        self.attack_list = cfg.attack_list
        self.vlogger = None
        self.detectors = init_detectors(cfg_det=cfg.DETECTOR)
        self.adv_img_obj = AdvImgObject(cfg, device)

    def init_attacker(self):
        cfg = self.cfg.ATTACKER
        loss_fn = loss_dict[cfg.LOSS_FUNC]
        self.attacker = get_attack_method(cfg.METHOD)(
            loss_func=loss_fn, norm='L_infty', device=self.device, cfg=cfg, detector_attacker=self)

    def plot_boxes(self, img_tensor, boxes, save_path=None, save_name=None):
        """Plot detected boxes on images.
        """
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, save_name)
        img = FormatConverter.tensor2numpy_cv2(img_tensor.cpu().detach())
        plot_box = plot_boxes_cv2(img, boxes.cpu().detach().numpy(), self.class_names,savename=save_name)
        return plot_box

    def merge_batch(self, all_preds, preds):
        if all_preds is None:
            return preds
        for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
            if pred.shape[0]:
                pred = pred.to(all_pred.device)
                all_preds[i] = torch.cat((all_pred, pred), dim=0)
                continue
            all_preds[i] = all_pred if all_pred.shape[0] else pred
        return all_preds

    def detect_bbox(self, img_tensor_batch, detectors=None):
        if detectors is None:
            detectors = self.detectors
        all_preds = None
        for detector in detectors:
            preds = detector(img_tensor_batch.to(detector.device))['bbox_array']
            all_preds = self.merge_batch(all_preds, preds)
        # nms among detectors
        if len(detectors) > 1: all_preds = inter_nms(all_preds)
        return all_preds

    def attack(self, img_tensor_batch, mode):
        detectors_loss = []
        self.attacker.begin_attack()
        if mode == 'optim':
            for detector in self.detectors:
                loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
                detectors_loss.append(loss)
        self.attacker.end_attack()
        return torch.tensor(detectors_loss).mean()