import numpy as np
from torch.utils.tensorboard import SummaryWriter
import subprocess
import time
from .. import FormatConverter


class VisualBoard:
    def __init__(self, name=None, start_iter=0, new_process=False, optimizer=None):
        if new_process:
            subprocess.Popen(['tensorboard', '--logdir=runs'])
        time_str = time.strftime("%m-%d-%H%M%S")
        if name is not None:
            self.writer = SummaryWriter(f'runs/{name}')
        else:
            self.writer = SummaryWriter(f'runs/{time_str}')

        self.iter = start_iter
        self.optimizer = optimizer
        self.init_loss()

    def __call__(self, epoch, iter):
        self.iter = iter
        self.writer.add_scalar('misc/epoch', epoch, self.iter)
        if self.optimizer:
            self.writer.add_scalar('misc/learning_rate', self.optimizer.param_groups[0]["lr"], self.iter)

    def write_scalar(self, scalar, name):
        try:
            scalar = scalar.detach().cpu().numpy()
        except:
            scalar = scalar
        self.writer.add_scalar(name, scalar, self.iter)

    def write_tensor(self, img, name):
        self.writer.add_image('attack/'+name, img.detach().cpu(), self.iter)

    def write_cv2(self, img, name):
        img = FormatConverter.bgr_numpy2tensor(img)[0]
        self.writer.add_image(f'attack/{name}', img, self.iter)

    def write_ep_loss(self, ep_loss):
        self.writer.add_scalar('loss/target_conf_loss', np.array(self.target_conf_loss).mean(), self.iter)
        self.writer.add_scalar('loss/untarget_iou_loss', np.array(self.untarget_iou_loss).mean(), self.iter)
        self.writer.add_scalar('loss/total_loss', np.array(self.total_loss).mean(), self.iter)
        self.writer.add_scalar('loss/disappear_loss', np.array(self.disappear_loss).mean(), self.iter)
        self.init_loss()

    def init_loss(self):
        self.total_loss = []
        # self.obj_loss = []
        self.target_conf_loss = []
        self.untarget_iou_loss = []
        self.disappear_loss = []

    def note_loss(self, total_loss, target_conf_loss, disappear_loss, untarget_iou_loss):
        self.total_loss.append(total_loss.detach().cpu().numpy())
        self.target_conf_loss.append(target_conf_loss.detach().cpu().numpy())
        self.untarget_iou_loss.append(untarget_iou_loss.detach().cpu().numpy())
        self.disappear_loss.append(disappear_loss.detach().cpu().numpy())