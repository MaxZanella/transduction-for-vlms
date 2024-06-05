import torch
import torchvision.transforms as transforms
from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .imagenet_a import ImageNetA
from .imagenet_v2 import ImageNetV2
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .utils import *

dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "fgvc_aircraft": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                "imagenet_a": ImageNetA,
                "imagenet_v2": ImageNetV2,
                "imagenet_r": ImageNetR,
                "imagenet_sketch": ImageNetSketch,
                }


def get_all_dataloaders(cfg, preprocess):
    dataset_name = cfg['dataset']

    if dataset_name.startswith('imagenet'):
        dataset = dataset_list[dataset_name](cfg['root_path'], cfg['shots'], preprocess=preprocess, train_preprocess=None, test_preprocess=None, load_cache=cfg['load_cache'], load_pre_feat=cfg['load_pre_feat'])
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
        train_loader = None
        val_loader = None
        if cfg['shots'] > 0:
            train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)
            val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=64, num_workers=8, shuffle=False)

    else:
        dataset = dataset_list[dataset_name](cfg['root_path'], cfg['shots'])
        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess,
                                       shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess,
                                        shuffle=False)
        train_loader = None
        if cfg['shots'] > 0:

            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=preprocess,
                                                   is_train=False, shuffle=False)

    return train_loader, val_loader, test_loader, dataset


