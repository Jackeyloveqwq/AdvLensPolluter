import torch
import torch.nn as nn
import torchvision.transforms as transforms


class AdvImgObject:
    def __init__(self, cfg, device, adv_param_path=None):
        self.cfg = cfg
        self.device = device
        self.Raindrop = Raindrop()
        self.MudSpot = MudSpot()
        self.Stain = Stain()
        self.Frost = Frost()
        self.Condensation = Condensation()
        if adv_param_path is not None:
            self.load_adv_param_file_raindrop(adv_param_path)
        self.adv_tensor_batch = None

    def load_adv_param_file_raindrop(self, adv_param_path):
        print(f'从以下路径加载Raindrop模型参数: {adv_param_path}')
        state_dict = torch.load(adv_param_path, map_location=self.device)
        self.Raindrop.load_state_dict(state_dict)
        self.Raindrop.to(self.device)
        self.Raindrop.eval()
        print('成功加载Raindrop模型参数并设置为eval模式.')

    def load_adv_param_file_mudspot(self, adv_param_path):
        print(f'从以下路径加载MudSpot模型参数: {adv_param_path}')
        state_dict = torch.load(adv_param_path, map_location=self.device)
        self.MudSpot.load_state_dict(state_dict)
        self.MudSpot.to(self.device)
        self.MudSpot.eval()
        print('成功加载MudSpot模型参数并设置为eval模式.')

    def load_adv_param_file_stain(self, adv_param_path):
        print(f'从以下路径加载stain模型参数: {adv_param_path}')
        state_dict = torch.load(adv_param_path, map_location=self.device)
        self.Stain.load_state_dict(state_dict)
        self.Stain.to(self.device)
        self.Stain.eval()
        print('成功加载stain模型参数并设置为eval模式.')

    def load_adv_param_file_frost(self, adv_param_path):
        print(f'从以下路径加载frost模型参数: {adv_param_path}')
        state_dict = torch.load(adv_param_path, map_location=self.device)
        self.Frost.load_state_dict(state_dict)
        self.Frost.to(self.device)
        self.Frost.eval()
        print('成功加载frost模型参数并设置为eval模式.')

    def load_adv_param_file_condensation(self, adv_param_path):
        print(f'从以下路径加载condensation模型参数: {adv_param_path}')
        state_dict = torch.load(adv_param_path, map_location=self.device)
        self.Condensation.load_state_dict(state_dict)
        self.Condensation.to(self.device)
        self.Condensation.eval()
        print('成功加载condensation模型参数并设置为eval模式.')

    def generate_adv_tensor_batch(self, clean_img_batch, init_mode):
        if init_mode.lower() == 'raindrop':
            self.adv_tensor_batch = self.Raindrop(clean_img_batch).to(self.device)

        elif init_mode.lower() == 'mud_spot':
            self.adv_tensor_batch = self.MudSpot(clean_img_batch).to(self.device)

        elif init_mode.lower() == 'stain':
            self.adv_tensor_batch = self.Stain(clean_img_batch).to(self.device)

        elif init_mode.lower() == 'frost':
            self.adv_tensor_batch = self.Frost(clean_img_batch).to(self.device)

        elif init_mode.lower() == 'condensation':
            self.adv_tensor_batch = self.Condensation(clean_img_batch).to(self.device)

        else:
            raise ValueError('Unsupported init_mode: ' + init_mode)

        return self.adv_tensor_batch


class Raindrop(nn.Module):
    def __init__(self, device='cuda', num_drops=12, min_radius=40, max_radius=50):
        super(Raindrop, self).__init__()
        self.device = device
        self.num_drops = num_drops
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.positions = torch.tensor(
            [[0.51, -0.12], [-0.43, -0.12], [-0.86, -0.54], [0.38, -0.64], [0.77, -0.66], [-0.26, -0.64],
             [-0.10, -0.17], [-0.77, 0.02], [0.19, -0.14], [-0.58, -0.61], [0.07, -0.68], [0.82, -0.18]], device=self.device)
        self.radius = nn.Parameter(torch.rand(num_drops, device=device) * (max_radius - min_radius) + min_radius,
                                   requires_grad=True)
        self.blur_radius = nn.Parameter(torch.rand(num_drops, device=device) * 20.0 + 5.0, requires_grad=True)
        self.beta = 1.8

    def forward(self, img_tensor_batch):
        batch_size, _, height, width = img_tensor_batch.shape
        adv_tensor_batch = img_tensor_batch.clone()
        self.positions.data.clamp_(-1, 1)
        self.radius.data.clamp_(self.min_radius, self.max_radius)
        for i in range(batch_size):
            image = img_tensor_batch[i].clone()
            drops = []
            for j in range(self.num_drops):
                x0 = (self.positions[j, 0] + 1) / 2 * width
                y0 = (self.positions[j, 1] + 1) / 2 * height
                width_radius = self.radius[j]
                height_radius = width_radius * 1.2
                new_drop = (x0, y0, width_radius, height_radius)
                drops.append(new_drop)
                blur_radius = self.blur_radius[j]
                image = self.apply_gaussian_blur_with_mask(image, height, width, new_drop[0], new_drop[1],
                                                           new_drop[2], new_drop[3], blur_radius)
            adv_tensor_batch[i] = image.to(self.device)
        return adv_tensor_batch

    def apply_gaussian_blur_with_mask(self, image, height, width, x0, y0, width_radius, height_radius, blur_radius):
        mask = self.create_ellipse_mask(height, width, x0, y0, width_radius, height_radius)
        raindrop_area = image * mask.unsqueeze(0)
        kernel_size = int(2 * blur_radius.item()) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        transform = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=blur_radius.item())
        raindrop_area_blurred = transform(raindrop_area.unsqueeze(0)).squeeze(0)
        adv_img = raindrop_area_blurred * mask.unsqueeze(0) + image * (1 - mask.unsqueeze(0))
        return adv_img

    def create_ellipse_mask(self, height, width, x0, y0, width_radius, height_radius):
        hv, wv = torch.meshgrid(torch.arange(0, height, device=self.device), torch.arange(0, width, device=self.device))
        hv, wv = hv.float(), wv.float()
        d = ((hv - y0) ** 2 / height_radius ** 2 + (wv - x0) ** 2 / width_radius ** 2)
        ellipse_mask = torch.exp(-d ** self.beta + 1e-10)
        ellipse_mask = torch.clamp(ellipse_mask, 0, 1)
        return ellipse_mask


class Condensation(nn.Module):
    def __init__(self, device='cuda', num_drops=10, min_radius=60, max_radius=80):
        super(Condensation, self).__init__()
        self.device = device
        self.num_drops = num_drops
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.positions = nn.Parameter(torch.tensor([[0.07, -0.68], [-0.83, -0.12], [0.57, -0.09], [-0.12, -0.19], [-0.47, -0.12],
                                       [0.83, -0.45], [0.42, -0.62], [-0.31, -0.66], [-0.71, -0.64], [0.20, -0.14]],
                                      device=self.device), requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([68.9755, 66.3124, 78.3028, 64.0275, 64.5939, 64.1804, 61.6286, 73.2401, 66.8596, 59.9934], device=self.device), requires_grad=True)
        self.blur_radius = torch.tensor([11.3535, 17.9381,  5.7966, 10.8586,  5.5301, 15.9075, 12.3225, 13.4871, 6.6639, 9.5413], device=self.device)
        self.beta = 1.8

    def forward(self, img_tensor_batch):
        batch_size, _, height, width = img_tensor_batch.shape
        adv_tensor_batch = img_tensor_batch.clone()
        self.positions.data.clamp_(-1, 1)
        self.radius.data.clamp_(self.min_radius, self.max_radius)
        for i in range(batch_size):
            image = img_tensor_batch[i].clone()
            drops = []
            for j in range(self.num_drops):
                x0 = (self.positions[j, 0] + 1) / 2 * width
                y0 = (self.positions[j, 1] + 1) / 2 * height
                width_radius = self.radius[j]
                height_radius = width_radius * 0.8
                new_drop = (x0, y0, width_radius, height_radius)
                drops.append(new_drop)
                blur_radius = self.blur_radius[j]
                image = self.apply_gaussian_blur_with_mask(image, height, width, new_drop[0], new_drop[1],
                                                           new_drop[2], new_drop[3], blur_radius)
            adv_tensor_batch[i] = image.to(self.device)
        return adv_tensor_batch

    def apply_gaussian_blur_with_mask(self, image, height, width, x0, y0, width_radius, height_radius, blur_radius):
        mask = self.create_ellipse_mask(height, width, x0, y0, width_radius, height_radius)
        raindrop_area = image * mask.unsqueeze(0)
        kernel_size = int(2 * blur_radius.item()) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        transform = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=blur_radius.item())
        raindrop_area_blurred = transform(raindrop_area.unsqueeze(0)).squeeze(0)
        adv_img = raindrop_area_blurred * mask.unsqueeze(0) + image * (1 - mask.unsqueeze(0))
        return adv_img

    def create_ellipse_mask(self, height, width, x0, y0, width_radius, height_radius):
        hv, wv = torch.meshgrid(torch.arange(0, height, device=self.device), torch.arange(0, width, device=self.device))
        hv, wv = hv.float(), wv.float()
        d = ((hv - y0) ** 2 / height_radius ** 2 + (wv - x0) ** 2 / width_radius ** 2)
        ellipse_mask = torch.exp(-d ** self.beta + 1e-10)
        ellipse_mask = torch.clamp(ellipse_mask, 0, 1)
        return ellipse_mask


class MudSpot(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(MudSpot, self).__init__()
        self.device = device
        self.means = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.stds = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.betas = torch.tensor([1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8], device=self.device)
        self.alphas = nn.Parameter(torch.tensor([0.9635, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9939, 0.6556], device=self.device), requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([0.6130, 1.0000, 1.0000, 0.7213, 1.0000, 1.0000, 1.0000, 0.9963, 1.0000, 0.6706], device=self.device), requires_grad=True)
        self.centers = torch.tensor([
            [0.54, 0.16], [0.08, 0.44], [0.78, 0.46], [0.44, 0.41], [0.27, 0.44], [0.91, 0.27], [0.71, 0.19],
            [0.35, 0.17], [0.14, 0.18], [0.60, 0.43]], device=self.device)
        self.colors = torch.tensor([
            [0.54, 0.49, 0.40], [0.54, 0.49, 0.40], [0.54, 0.49, 0.40], [0.54, 0.49, 0.40], [0.54, 0.49, 0.40],
            [0.54, 0.49, 0.40], [0.54, 0.49, 0.40], [0.54, 0.49, 0.40], [0.54, 0.49, 0.40], [0.54, 0.49, 0.40]], device=self.device)

    def forward(self, img_tensor_batch):
        batch_size, _, height, width = img_tensor_batch.shape
        adv_tensor_batch = img_tensor_batch.clone()
        for i in range(batch_size):
            adv_tensor = img_tensor_batch[i].clone()
            for idx in range(self.centers.shape[0]):
                mask = self.create_irregular_mask(height, width, self.centers[idx][0] * width,
                                                  self.centers[idx][1] * height,
                                                  self.radius[idx], self.betas[idx])
                normalized_color = self.normalize_color(self.colors[idx], self.means, self.stds)
                adv_tensor = self.create_adv_img(adv_tensor, mask, normalized_color, self.alphas[idx])
            adv_tensor_batch[i] = adv_tensor

        adv_tensor_batch = torch.clamp(adv_tensor_batch, 0.0, 1.0)
        return adv_tensor_batch

    @staticmethod
    def normalize_color(color, means, stds):
        return (color - means) / stds

    def create_irregular_mask(self, height, width, x_center, y_center, radius, beta):
        hv, wv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        hv, wv = hv.type(torch.FloatTensor), wv.type(torch.FloatTensor)
        hv, wv = hv.to(self.device), wv.to(self.device)
        d = ((hv - y_center) ** 2 + (wv - x_center) ** 2) / (radius * 90) ** 2
        circle_mask = torch.exp(- d ** beta + 1e-10)
        return circle_mask

    def create_adv_img(self, img_tensor, mask, color, alpha):
        alpha_tile = alpha * mask.unsqueeze(0)
        color_tile = color.view(3, 1, 1).expand(3, mask.shape[0], mask.shape[1])
        adv_img_tensor = (1. - alpha_tile) * img_tensor + alpha_tile * color_tile
        return adv_img_tensor


class Stain(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Stain, self).__init__()
        self.device = device
        self.means = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.stds = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.betas = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], device=self.device)
        self.alphas = nn.Parameter(torch.tensor([0.8481, 1.0000, 0.5854, 0.7475, 1.0000, 1.0000, 1.0000, 0.9803, 1.0000, 1.0000], device=self.device), requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([0.3043, 0.7687, 0.5005, 0.7680, 1.0000, 1.0000, 1.0000, 0.6267, 1.0000, 0.6254], device=self.device), requires_grad=True)
        self.centers = torch.tensor([
            [0.54, 0.16], [0.08, 0.44], [0.78, 0.46], [0.44, 0.41], [0.27, 0.44],
            [0.91, 0.27], [0.71, 0.19], [0.35, 0.17], [0.14, 0.18], [0.60, 0.43] ], device=self.device)
        self.colors = torch.tensor([
            [0.62, 0.59, 0.54], [0.62, 0.59, 0.54], [0.62, 0.59, 0.54], [0.62, 0.59, 0.54],
            [0.62, 0.59, 0.54], [0.62, 0.59, 0.54], [0.62, 0.59, 0.54], [0.62, 0.59, 0.54],
            [0.62, 0.59, 0.54], [0.62, 0.59, 0.54]], device=self.device)

    def forward(self, img_tensor_batch):
        batch_size, _, height, width = img_tensor_batch.shape
        adv_tensor_batch = img_tensor_batch.clone()
        for i in range(batch_size):
            adv_tensor = img_tensor_batch[i].clone()
            for idx in range(self.centers.shape[0]):
                mask = self.create_irregular_mask(height, width, self.centers[idx][0] * width,
                                                  self.centers[idx][1] * height,
                                                  self.radius[idx], self.betas[idx])
                normalized_color = self.normalize_color(self.colors[idx], self.means, self.stds)
                adv_tensor = self.create_adv_img(adv_tensor, mask, normalized_color, self.alphas[idx])
            adv_tensor_batch[i] = adv_tensor

        adv_tensor_batch = torch.clamp(adv_tensor_batch, 0.0, 1.0)
        return adv_tensor_batch

    @staticmethod
    def normalize_color(color, means, stds):
        return (color - means) / stds

    def create_irregular_mask(self, height, width, x_center, y_center, radius, beta):
        hv, wv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        hv, wv = hv.type(torch.FloatTensor), wv.type(torch.FloatTensor)
        hv, wv = hv.to(self.device), wv.to(self.device)
        d = ((hv - y_center) ** 2 + (wv - x_center) ** 2) / (radius * 100) ** 2
        circle_mask = torch.exp(- d ** beta + 1e-10)
        return circle_mask

    def create_adv_img(self, img_tensor, mask, color, alpha):
        alpha_tile = alpha * mask.unsqueeze(0)
        color_tile = color.view(3, 1, 1).expand(3, mask.shape[0], mask.shape[1])
        adv_img_tensor = (1. - alpha_tile) * img_tensor + alpha_tile * color_tile
        return adv_img_tensor


class Frost(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Frost, self).__init__()
        self.device = device
        self.means = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.stds = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.betas = torch.tensor([1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8], device=self.device)
        self.alphas = nn.Parameter(torch.tensor([1.0000, 1.0000, 0.3342, 0.9720, 1.0000, 1.0000, 1.0000, 0.7338, 1.0000, 1.0000], device=self.device), requires_grad=True)
        self.radius = nn.Parameter(torch.tensor([0.9325, 1.0000, 0.7548, 0.7765, 0.9944, 0.8690, 0.7821, 0.7697, 1.0000, 0.9314], device=self.device), requires_grad=True)
        self.centers = torch.tensor([
            [0.54, 0.16], [0.08, 0.44], [0.78, 0.46], [0.44, 0.41], [0.27, 0.44],
              [0.91, 0.27], [0.71, 0.19], [0.35, 0.17], [0.14, 0.18], [0.60, 0.43]], device=self.device)
        self.colors = torch.tensor([
            [0.71, 0.68, 0.63], [0.71, 0.68, 0.63], [0.71, 0.68, 0.63], [0.71, 0.68, 0.63], [0.71, 0.68, 0.63],
        [0.71, 0.68, 0.63], [0.71, 0.68, 0.63], [0.71, 0.68, 0.63], [0.71, 0.68, 0.63], [0.71, 0.68, 0.63]], device=self.device)

    def forward(self, img_tensor_batch):
        batch_size, _, height, width = img_tensor_batch.shape
        adv_tensor_batch = img_tensor_batch.clone()
        for i in range(batch_size):
            adv_tensor = img_tensor_batch[i].clone()
            for idx in range(self.centers.shape[0]):
                mask = self.create_irregular_mask(height, width, self.centers[idx][0] * width,
                                                  self.centers[idx][1] * height,
                                                  self.radius[idx], self.betas[idx])
                normalized_color = self.normalize_color(self.colors[idx], self.means, self.stds)
                adv_tensor = self.create_adv_img(adv_tensor, mask, normalized_color, self.alphas[idx])
            adv_tensor_batch[i] = adv_tensor

        adv_tensor_batch = torch.clamp(adv_tensor_batch, 0.0, 1.0)
        return adv_tensor_batch

    @staticmethod
    def normalize_color(color, means, stds):
        return (color - means) / stds

    def create_irregular_mask(self, height, width, x_center, y_center, radius, beta):
        hv, wv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        hv, wv = hv.type(torch.FloatTensor), wv.type(torch.FloatTensor)
        hv, wv = hv.to(self.device), wv.to(self.device)
        d = ((hv - y_center) ** 2 + (wv - x_center) ** 2) / (radius * 100) ** 2
        circle_mask = torch.exp(- d ** beta + 1e-10)
        return circle_mask

    def create_adv_img(self, img_tensor, mask, color, alpha):
        alpha_tile = alpha * mask.unsqueeze(0)
        color_tile = color.view(3, 1, 1).expand(3, mask.shape[0], mask.shape[1])
        adv_img_tensor = (1. - alpha_tile) * img_tensor + alpha_tile * color_tile
        return adv_img_tensor
