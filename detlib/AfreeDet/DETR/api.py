import torch
from .DETR.detr import Detection_Transformers


class DETR:
    def __init__(self, name, cfg, input_tensor_size=800,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.confidence = 0.5
        self.detector = Detection_Transformers()

    def __call__(self, batch_tensor, **kwargs):
        outputs = self.detector(batch_tensor)
        orig_target_sizes = torch.stack(
            [torch.tensor([self.input_tensor_size, self.input_tensor_size])] * len(batch_tensor), dim=0).to(self.device)
        results = self.postprocessors['bbox'](outputs, orig_target_sizes)

        bbox_array = []

        for res in results:
            if len(res) > 0:
                boxes = res['boxes']
                scores = res['scores']
                labels = res['labels']

                for box, score, label in zip(boxes, scores, labels):
                    if score >= self.confidence:
                        box = box.cpu().numpy()
                        x1, y1, x2, y2 = box / self.input_tensor_size
                        bbox_array.append([x1, y1, x2, y2, score.cpu().numpy(), label.cpu().numpy()])

        return bbox_array
