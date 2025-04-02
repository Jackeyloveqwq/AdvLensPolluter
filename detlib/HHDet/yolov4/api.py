from .Pytorch_YOLOv4.tool.utils import *
from .Pytorch_YOLOv4.tool.torch_utils import *
from .Pytorch_YOLOv4.tool.darknet2pytorch import Darknet
from ...base import DetectorBase


class HHYolov4(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_height_size=352,
                 input_tensor_width_size = 640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_height_size, input_tensor_width_size, device)
        self.test = 0

    def requires_grad_(self, state: bool):
        assert self.detector
        self.detector.models.requires_grad_(state)

    def load(self, model_weights, detector_config_file=None):
        self.detector = Darknet(detector_config_file).to(self.device)
        self.detector.load_weights(model_weights)
        self.eval()

    # def __call__(self, batch_tensor, **kwargs):
    #     detections_with_grad = self.detector(batch_tensor)
    #     bbox_array = post_processing(batch_tensor, self.conf_thres, self.iou_thres, detections_with_grad)
    #     for i, pred in enumerate(bbox_array):
    #         pred = torch.Tensor(pred).to(self.device)
    #         if len(pred) != 0:
    #             pred[:, :4] = torch.clamp(pred[:, :4], min=0, max=1)
    #         bbox_array[i] = pred # shape([1, 6])
    #     # output: [ [batch, num, 1, 4], [batch, num, num_classes] ]
    #     # v4's confs is the combination of obj conf & cls conf
    #     confs = detections_with_grad[1]
    #     # print(confs.shape)
    #     # cls_max_ids = torch.argmax(confs, dim=2)
    #     # print(cls_max_ids.shape)
    #     max_confs = torch.max(confs, dim=2)[0]
    #     # output = {'bbox_array': bbox_array, 'obj_confs': max_confs, "cls_max_ids": None}
    #     output = {'bbox_array': bbox_array, 'obj_confs': max_confs}
    #     return output

    def __call__(self, batch_tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor)
        bbox_tensor = detections_with_grad[0]  # shape (batch_size, num_detections, 1, 4)
        confs_tensor = detections_with_grad[1]  # shape (batch_size, num_detections, 80)
        # Remove the singleton dimension in bbox_tensor and concatenate the tensors
        bbox_tensor = bbox_tensor.squeeze(2)  # shape (batch_size, num_detections, 4)
        # Generate object confidence tensor (assuming object confidence is all ones)
        obj_conf = torch.ones((bbox_tensor.size(0), bbox_tensor.size(1), 1),
                              device=self.device)  # shape (batch_size, num_detections, 1)
        # Concatenate bbox_tensor and obj_conf along the last dimension
        bbox_with_conf = torch.cat((bbox_tensor, obj_conf), dim=2)  # shape (batch_size, num_detections, 5)
        # Concatenate bbox_with_conf and confs_tensor along the last dimension
        output_after_model = torch.cat((bbox_with_conf, confs_tensor), dim=2)  # shape (batch_size, num_detections, 85)
        bbox_array = post_processing(batch_tensor, self.conf_thres, self.iou_thres, detections_with_grad)
        for i, pred in enumerate(bbox_array):
            pred = torch.Tensor(pred).to(self.device)
            if len(pred) != 0:
                pred[:, :4] = torch.clamp(pred[:, :4], min=0, max=1)
            bbox_array[i] = pred  # shape([1, 6])

        max_confs = torch.max(confs_tensor, dim=2)[0]
        output = {'bbox_array': bbox_array, 'output_after_model': output_after_model}
        return output