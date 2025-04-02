from utils.solver import *
from attack.methods import OptimAttacker
from utils.solver.loss import *

scheduler_factory = {
    'plateau': PlateauLR,
    'cosine': CosineLR,
    'ALRS': ALRS,
    'ALRS_LowerTV': ALRS_LowerTV
}

optim_factory = {
    'optim': lambda params, lr: torch.optim.Adam(params, lr=lr, amsgrad=True),  # default
    'optim-adam': lambda params, lr: torch.optim.Adam(params, lr=lr, amsgrad=True),
    'optim-sgd': lambda params, lr: torch.optim.SGD(params, lr=lr * 100),
}

attack_method_dict = {
    "": None,
    "optim": OptimAttacker
}

loss_dict = {
    '': None,
    'custom-attack': custom_attack_loss,
}


def get_attack_method(attack_method: str):
    if 'optim' in attack_method:
        return attack_method_dict['optim']
    return attack_method_dict[attack_method]


MAP_PATHS = {'attack-img': 'attack-imgs',          # visualization of the detections on adversarial samples.
             'clean-img': 'clean-imgs',            # visualization of the detections on clean samples.
             'det-lab': 'det-labels',       # Detections on clean samples.
             'attack-lab': 'attack-labels', # Detections on adversarial samples.
             'det-res': 'det-res',          # statistics
             'ground-truth': 'ground-truth'
             }
