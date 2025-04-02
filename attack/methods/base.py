import torch
from torch.optim.optimizer import Optimizer


class BaseAttacker(Optimizer):
    def __init__(self, loss_func, norm: str, cfg, device: torch.device, detector_attacker):
        defaults = dict(lr=cfg.STEP_LR)
        params = [p for p in detector_attacker.adv_img_obj.MudSpot.parameters()]
        super().__init__(params, defaults)

        self.loss_func = loss_func
        self.cfg = cfg
        self.detector_attacker = detector_attacker
        self.device = device
        self.norm = norm
        self.max_iters = cfg.MAX_EPOCH
        self.attack_class = cfg.ATTACK_CLASS

    def logger(self, detector, adv_tensor_batch, bboxes, loss_dict):
        vlogger = self.detector_attacker.vlogger
        if vlogger:
            vlogger.note_loss(loss_dict['total_loss'], loss_dict['target_conf_loss'],loss_dict['disappear_loss'], loss_dict['untarget_iou_loss'])
            if vlogger.iter % 5 == 0:
                vlogger.write_tensor(adv_tensor_batch[0], 'adv tensor')
                plotted = self.detector_attacker.plot_boxes(adv_tensor_batch[0], bboxes[0])
                vlogger.write_cv2(plotted, f'{detector.name}')

    def non_targeted_attack(self, clean_tensor_batch, detector):
        losses = []
        adv_tensor_batch = self.detector_attacker.adv_img_obj.generate_adv_tensor_batch(clean_tensor_batch, init_mode='mud_spot')
        # print(adv_tensor_batch.requires_grad)
        clean_bboxes_batch, output_clean_batch = detector(clean_tensor_batch).values()
        adv_bboxes_batch, output_adv_batch = detector(adv_tensor_batch).values()

        detector.zero_grad()
        loss_dict = self.attack_loss(clean_bboxes_batch=clean_bboxes_batch, output_clean_batch=output_clean_batch,
                                     adv_bboxes_batch=adv_bboxes_batch, output_adv_batch=output_adv_batch)
        loss = loss_dict['total_loss']
        loss.backward()
        losses.append(float(loss))

        self.logger(detector, adv_tensor_batch, adv_bboxes_batch, loss_dict)
        return torch.tensor(losses).mean()

    def attack_loss(self, clean_bboxes_batch, output_clean_batch, adv_bboxes_batch, output_adv_batch):
        obj_loss = self.loss_func(clean_bboxes=clean_bboxes_batch, output_clean=output_clean_batch,
                                     adv_bboxes=adv_bboxes_batch, output_adv=output_adv_batch)
        loss = obj_loss
        output = {'loss': loss, 'det_loss': obj_loss}
        return output

    def begin_attack(self):
        pass

    def end_attack(self):
        pass
