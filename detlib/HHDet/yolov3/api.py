import torch
from .PyTorch_YOLOv3.pytorchyolo.models import load_model
from .PyTorch_YOLOv3.pytorchyolo.utils.utils import rescale_boxes, non_max_suppression
from ...base import DetectorBase


class HHYolov3(DetectorBase):
    def __init__(self,
                 name, cfg,
                 input_tensor_height_size=352,
                 input_tensor_width_size=640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_height_size, input_tensor_width_size, device)
        self.imgsz = (input_tensor_height_size, input_tensor_width_size)
        self.target = None

    def requires_grad_(self, state: bool):
        self.detector.module_list.requires_grad_(state)
    
    def load(self, model_weights, detector_config_file=None):
        self.detector = load_model(model_path=detector_config_file, weights_path=model_weights).to(self.device)
        self.eval()

    def __call__(self, batch_tensor: torch.tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor) # torch.tensor([1, num, classes_num+4+1])
        preds = non_max_suppression(detections_with_grad, self.conf_thres, self.iou_thres)
        obj_confs = detections_with_grad[:, :, 4]
        cls_max_ids = detections_with_grad[:, :, 5]

        bbox_array = []
        for i, pred in enumerate(preds):
            box = rescale_boxes(pred, self.imgsz, self.ori_size).clone()
            box[:, [0, 2]] = box[:, [0, 2]] / self.ori_size[1]  # w
            box[:, [1, 3]] = box[:, [1, 3]] / self.ori_size[0]  # h
            torch.clamp(box[:, :4], min=0, max=1)
            bbox_array.append(box)

        # output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        output = {'bbox_array': bbox_array, 'output_after_model': detections_with_grad}
        return output