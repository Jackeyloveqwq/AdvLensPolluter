import torch
import os
import numpy as np
from tqdm import tqdm
from utils.plot import VisualBoard
from utils.loader import dataLoader
from utils.parser import logger
from scripts.dict import scheduler_factory, optim_factory
from attack.attacker import UniversalAttacker
from utils.parser import ConfigParser


def init(detector_attacker: UniversalAttacker, cfg: ConfigParser, data_root: str, args: object = None, log: bool = True):
    if log: logger(cfg, args)
    data_sampler = None
    data_loader = dataLoader(data_root,
                             input_size=cfg.DETECTOR.INPUT_SIZE, is_augment=cfg.DATA.AUGMENT,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler, shuffle=True)
    detector_attacker.init_attacker()

    vlogger = None
    if log and args and not args.debugging:
        vlogger = VisualBoard(name=args.board_name, new_process=args.new_process, optimizer=detector_attacker.attacker)
        detector_attacker.vlogger = vlogger

    return data_loader, vlogger


def train_uap(cfg: ConfigParser,
              detector_attacker: UniversalAttacker,
              save_name: str,
              args: object = None,
              data_root: str = '/home/ubuntu/WeatherJammer/data/BDD100K_stopsign/images/train'
              ):
    def get_iter():
        return (epoch - 1) * len(data_loader) + index

    data_loader, vlogger = init(detector_attacker, cfg, args=args, data_root=data_root)

    optimizer = optim_factory[cfg.ATTACKER.METHOD](detector_attacker.adv_img_obj.MudSpot.parameters(), cfg.ATTACKER.STEP_LR)
    detector_attacker.attacker.set_optimizer(optimizer)
    scheduler = scheduler_factory[cfg.ATTACKER.LR_SCHEDULER](optimizer)

    loss_array = []
    if args.save_process:
        args.save_path += '/mudspot/'
        os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(1, cfg.ATTACKER.MAX_EPOCH + 1):
        ep_loss = 0
        for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            if vlogger: vlogger(epoch, get_iter())
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)
            loss = detector_attacker.attack(img_tensor_batch, mode='optim')
            ep_loss += loss
            optimizer.step()

            if vlogger: vlogger.write_ep_loss(ep_loss)
            loss_array.append(float(ep_loss))
        ep_loss /= len(data_loader)
        scheduler.step(ep_loss=ep_loss, epoch=epoch)

        # if vlogger: vlogger.write_ep_loss(ep_loss)
        # loss_array.append(float(ep_loss))
        if epoch % 2 == 0 or epoch == cfg.ATTACKER.MAX_EPOCH:
            train_params = detector_attacker.adv_img_obj.MudSpot.state_dict()
            for name, param in train_params.items():
                    train_params[name] = torch.clamp(param, min=0, max=1)
            torch.save(train_params, os.path.join(args.save_path, 'mud_params_7_12_yolov5.pth'))
            print('Saving train parameters to', os.path.join(args.save_path, 'mud_params_7_12_yolov5.pth'))

    np.save(os.path.join(args.save_path, save_name + '-loss.npy'), loss_array)


if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default='train_demo.yaml',
                        help="A relative path of the .yaml proj config file.")
    parser.add_argument('-n', '--board_name', type=str, default='7_12',
                        help="Name of the Tensorboard as well as the patch name.")
    parser.add_argument('-d', '--debugging', action='store_true',
                        help="Will not start tensorboard process if debugging=True.")
    parser.add_argument('-s', '--save_path', type=str, default='results/7_12/',
                        help="Path to save the adversarial patch.")
    parser.add_argument('-np', '--new_process', action='store_true', default=False,
                        help="Start new TensorBoard server process.")
    parser.add_argument('-sp', '--save_process', action='store_true', default=False,
                        help="Save patches from intermediate epoches.")

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cfg = './configs/' + args.cfg
    save_adv_params_name = args.cfg.split('/')[-1].split('.')[0] if args.board_name is None else args.board_name

    cfg = ConfigParser(args.cfg)
    detector_attacker = UniversalAttacker(cfg, device)
    cfg.show_class_label(cfg.attack_list)
    train_uap(cfg, detector_attacker, save_adv_params_name, args)