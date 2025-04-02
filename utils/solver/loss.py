import torch
import numpy as np


def calculate_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union_area = b1_area + b2_area - inter_area
    iou = inter_area / union_area
    return iou


def custom_attack_loss(**kwargs):
    clean_bboxes_batch = kwargs['clean_bboxes']
    output_adv_batch = kwargs['output_adv']
    adv_bboxes_batch = kwargs['adv_bboxes']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_class = 11
    conf_thres = 0.25
    image_size = [352, 640]

    def target_conf_loss(output_adv_batch):
        conf_every_candi = output_adv_batch[:, :, 5:] * output_adv_batch[:, :, 4:5]
        conf, index = conf_every_candi.max(2, keepdim=False)
        all_target_conf = conf_every_candi[:, :, target_class]
        over_thres_target_conf = all_target_conf[conf > conf_thres]
        zeros = torch.zeros(over_thres_target_conf.size()).to(output_adv_batch.device)
        zeros.requires_grad = True
        diff1 = torch.maximum(over_thres_target_conf - conf_thres, zeros)
        mean_conf = torch.sum(diff1, dim=0) / (output_adv_batch.size()[0] * output_adv_batch.size()[1])
        return mean_conf

    def disappear_loss(output_patch):
        def xywh2xyxy(x):
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y
        t_loss = 0.0
        not_nan_count = 0
        xc_patch = output_patch[..., 4] > conf_thres
        for (i, infer) in enumerate(output_patch):
            x1 = infer[xc_patch[i]]
            x2 = x1[:, 5:] * x1[:, 4:5]
            target_boxes = x1[x2[:, target_class] > conf_thres]
            if target_boxes.size(0) == 0:
                continue
            box_x1 = xywh2xyxy(target_boxes[:, :4])
            bboxes_x1_wh = xywh2xyxy(box_x1)[:, 2:]
            bboxes_x1_area = bboxes_x1_wh[:, 0] * bboxes_x1_wh[:, 1]
            img_loss = bboxes_x1_area.mean() / (image_size[0] * image_size[1])
            if not torch.isnan(img_loss):
                t_loss += img_loss
                not_nan_count += 1
        if not_nan_count == 0:
            t_loss_f = torch.tensor(float('nan'))
        else:
            t_loss_f = t_loss / not_nan_count
        return t_loss_f

    def untarget_iou_loss(clean_bboxes_batch, adv_bboxes_batch):
        batch_loss = []
        for clean_bboxes, adv_bboxes in zip(clean_bboxes_batch, adv_bboxes_batch):
            if clean_bboxes.ndim == 1:
                clean_bboxes = clean_bboxes.view(0, 6)
            if adv_bboxes.ndim == 1:
                adv_bboxes = adv_bboxes.view(0, 6)
            clean_bboxes_filtered = clean_bboxes[clean_bboxes[:, 5] != target_class]
            for clean_bbox in clean_bboxes_filtered:
                clean_class = clean_bbox[5]
                clean_xyxy = torch.stack([clean_bbox])
                clean_xyxy_out = clean_xyxy.to(device)
                adv_xyxy = adv_bboxes[adv_bboxes[:, 5].view(-1) == clean_class]
                adv_xyxy_out = adv_xyxy.to(device)
                if len(clean_xyxy_out) != 0 and len(adv_xyxy_out) != 0:
                    target = calculate_iou(adv_xyxy_out, clean_xyxy_out)
                    if len(target) != 0:
                        target_m, _ = target.max(dim=0)
                        batch_loss.append(target_m)
            one = torch.tensor(1.0).to(device)
            if len(batch_loss) == 0:
                return one
            return (one - torch.stack(batch_loss).mean())

    target_conf_loss = target_conf_loss(output_adv_batch)
    disappear_loss = disappear_loss(output_adv_batch)
    untarget_iou_loss = untarget_iou_loss(clean_bboxes_batch, adv_bboxes_batch)
    loss = {'target_conf_loss': target_conf_loss, 'disappear_loss': disappear_loss, 'untarget_iou_loss': untarget_iou_loss}
    return loss
