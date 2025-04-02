import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from natsort import natsorted
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DetDataset(Dataset):
    def __init__(self, images_path, input_size, is_augment=False, return_img_name=False, step=10):
        self.images_path = images_path
        # self.imgs = os.listdir(images_path)[::step]
        self.imgs = os.listdir(images_path)
        self.input_size = input_size
        self.n_samples = len(self.imgs)
        self.transform = transforms.Compose([])
        if is_augment:
            self.transform = self.transform_fn
        # self.ToTensor = transforms.Compose([
        #     transforms.Resize(self.input_size),
        #     transforms.ToTensor()
        # ])
        self.ToTensor = transforms.ToTensor()
        self.return_img_name = return_img_name

    def transform_fn(self, img, p_aug=0.5):
        """This is for random preprocesser augmentation of p_aug probability
        :param img:
        :param p_aug: probability to augment preprocesser.
        :return:
        """
        gate = torch.tensor([0]).bernoulli_(p_aug)
        if gate.item() == 0: return img
        img_t = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(5),
        ])(img)

        return img_t

    def pad_scale(self, img):
        """Padding the img to a square-shape to avoid stretch from the Resize op.
        :param img:
        :return:
        """
        w, h = img.size
        if w == h:
            return img

        pad_size = int((w - h) / 2)
        if pad_size < 0:
            pad = (abs(pad_size), 0)
            side_len = h
        else:
            side_len = w
            pad = (0, pad_size)

        padded_img = Image.new('RGB', (side_len, side_len), color=(127, 127, 127))
        padded_img.paste(img, pad)
        return padded_img

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_path, self.imgs[index])
        # image = Image.open(img_path).convert('RGB')
        # image = self.transform(image)
        # image = self.pad_scale(image)
        image = Image.open(img_path).convert('RGB')
        new_shape = (640, 352)
        img = image.resize(new_shape, Image.BICUBIC)

        if self.return_img_name:
            return self.ToTensor(img), self.imgs[index]

        return self.ToTensor(img)

    def __len__(self):
        return self.n_samples


class DetDatasetLab(Dataset):
    """This is a Dataset with preprocesser label loaded."""
    def __init__(self, images_path, lab_path, input_size):
        self.img_path = images_path
        self.lab_path = lab_path
        self.labs = natsorted(filter(lambda p: p.endswith('.txt'), os.listdir(lab_path)))
        self.input_size = input_size
        self.max_n_labels = 10
        self.ToTensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

    def pad_img(self, img, lab):
        """Padding the img to a square-shape and rescale the labels.
        :param img:
        :param lab:
        :return:
        """
        w, h = img.size
        if w == h:
            return img

        pad_size = int((w - h) / 2)
        if pad_size < 0:
            pad_size = abs(pad_size)
            pad = (pad_size, 0)
            side_len = h
            lab[:, [1, 3]] = (lab[:, [1, 3]] * w + pad_size) / h
        else:
            side_len = w
            lab[:, [2, 4]] = (lab[:, [2, 4]] * h + pad_size) / w
            pad = (0, pad_size)

        padded_img = Image.new('RGB', (side_len, side_len), color=(127, 127, 127))
        padded_img.paste(img, pad)

        return padded_img, lab

    def batchify_lab(self, lab):
        """Padding to batchify the lab in length of (self.max_n_labels).
        :param lab:
        :return:
        """
        lab = torch.cat(
            (lab[:, 1:], torch.ones(len(lab)).unsqueeze(1), torch.zeros(len(lab)).unsqueeze(1)),
            1
        )
        pad_size = self.max_n_labels - lab.shape[0]
        if (pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=0)
        else:
            padded_lab = lab
        return padded_lab

    def __getitem__(self, index):
        lab_path = os.path.join(self.lab_path, self.labs[index])
        img_path = os.path.join(self.img_path, self.labs[index].replace('txt', 'png'))

        lab = np.loadtxt(lab_path) if os.path.getsize(lab_path) else np.zeros(5)
        lab = torch.from_numpy(lab).float()
        if lab.dim() == 1:
            lab = lab.unsqueeze(0)
        lab = lab[:self.max_n_labels]

        image = Image.open(img_path).convert('RGB')
        image, lab = self.pad_img(image, lab)

        return self.ToTensor(image), self.batchify_lab(lab)

    def __len__(self):
        return len(self.labs)


def check_valid(name: str):
    """To check if the file name is of a valid image format.
    :param name: file name
    :return: Boolean
    """
    return name.endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))


def dataLoader(data_root, lab_root=None, input_size=None, batch_size=1, is_augment=False,
               shuffle=False, pin_memory=False, num_workers=16, sampler=None, return_img_name=False):
    if input_size is None:
        input_size = [416, 416]
    if lab_root is None:
        data_set = DetDataset(data_root, input_size, is_augment=is_augment, return_img_name=return_img_name)
    else:
        data_set = DetDatasetLab(data_root, lab_root, input_size)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    return data_loader