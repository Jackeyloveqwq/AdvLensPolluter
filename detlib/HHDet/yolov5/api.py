import torch
# load from YOLOV5
from .yolov5.utils.general import non_max_suppression, scale_coords
from .yolov5.models.experimental import attempt_load  # scoped to avoid circular import
from .yolov5.models.yolo import Model
from ...base import DetectorBase


class HHYolov5(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_height_size=352,
                 input_tensor_width_size=640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_height_size, input_tensor_width_size, device)
        self.imgsz = (input_tensor_width_size, input_tensor_height_size)
        self.stride, self.pt = None, None

    def load_(self, model_weights, **args):
        w = str(model_weights[0] if isinstance(model_weights, list) else model_weights)
        self.detector = attempt_load(model_weights if isinstance(model_weights, list) else w,
                                     map_location=self.device, inplace=False)
        self.eval()
        self.stride = max(int(self.detector.stride.max()), 32)  # model stride
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names

    def load(self, model_weights, **args):
        model_config = args['model_config']
        # Create model
        self.detector = Model(model_config).to(self.device)
        self.detector.load_state_dict(torch.load(model_weights, map_location=self.device)['model'].float().state_dict())
        self.eval()

    def check_input_data(self, batch_tensor):
        if torch.isnan(batch_tensor).any() or torch.isinf(batch_tensor).any():
            raise ValueError("Input batch_tensor contains NaN or Inf values")

    def check_output_data(self, output_adv_batch):
        if torch.isnan(output_adv_batch).any() or torch.isinf(output_adv_batch).any():
            raise ValueError("Output from the model contains NaN or Inf values")

    def __call__(self, batch_tensor, **kwargs):
        # detections_all = self.detector(batch_tensor, augment=False, visualize=False)
        self.check_input_data(batch_tensor)
        detections_with_grad = self.detector(batch_tensor, augment=False, visualize=False)[0]
        self.check_output_data(detections_with_grad)
        preds = non_max_suppression(detections_with_grad, self.conf_thres, self.iou_thres) # [batch, num, 6] e.g., [1, 22743, 1, 4]
        # print(preds)
        # cls_max_ids = None
        for pred in preds:
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                raise ValueError("Post-processed prediction contains NaN or Inf values")
        bbox_array = []
        for pred in preds:
            box = scale_coords(batch_tensor.shape[-2:], pred, self.ori_size)  # 将检测框按照原图大小缩放
            box[:, [0,2]] /= self.ori_size[1]  # 将检测框的坐标归一化
            box[:, [1,3]] /= self.ori_size[0]
            bbox_array.append(box)

        # [batch, num, num_classes] e.g.,[1, 22743, 80]
        # obj_confs = detections_with_grad[:, :, 4:5]  # 提取检测结果中的物体置信度
        # class_confs = detections_with_grad[:, :, 5:]  # 提取检测结果中的类别置信度
        # output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, 'class_confs': class_confs}
        output = {'bbox_array': bbox_array, 'output_after_model': detections_with_grad}
        return output

