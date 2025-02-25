from __future__ import division
import os
import logging
import time
import glob
import datetime
import argparse
import numpy as np
from scipy.io import loadmat, savemat

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from arch_unet import UNet
from est_unet import est_UNet
import utils as util
from nutils.gutils import chen_estimate, gat, inv_gat, norm_tensor, denorm_tensor
from collections import OrderedDict

from torch.cuda.amp import autocast, GradScaler
from loss_visible import calc_mu_sigma, calc_noise_var, likelihood_loss

parser = argparse.ArgumentParser()
parser.add_argument(
    "--noisetype",
    type=str,
    default="real",
    choices=[
        "real",
        "gauss25",
        "gauss5_50",
        "poisson30",
        "poisson5_50",
        "pg30_25",
        "pg5_50_5_50",
        "pg0.01+0.0002",
        "pg0.01+0.02",
        "pg0.05+0.02"
    ],
)
parser.add_argument("--resume", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--ckpt_est", type=str)
parser.add_argument("--data_dir", type=str, default="../dataset/train/SIDD_Medium_Raw_noisy_sub512")
parser.add_argument("--val_dirs", type=str, default="../dataset/validation")
parser.add_argument("--save_model_path", type=str, default="./experiments/sidd")
parser.add_argument("--log_name", type=str, default="vmap_bbi_unet_raw_13rf21")
parser.add_argument("--gpu_devices", default="0", type=str)
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--n_feature", type=int, default=48)
parser.add_argument("--n_channel", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.5e-4)
parser.add_argument("--w_decay", type=float, default=1e-8)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--n_epoch", type=int, default=100)
parser.add_argument("--n_snapshot", type=int, default=1)
parser.add_argument("--batchsize", type=int, default=4)
parser.add_argument("--patchsize", type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=3.0)
parser.add_argument("--increase_ratio", type=float, default=21.0)
parser.add_argument("--w_est", type=float, default=0.0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
operation_seed_counter = 0
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_devices
if opt.parallel:
    device_ids = list(range(len(opt.gpu_devices.split(','))))

util.set_random_seed(0)
torch.set_num_threads(6)

# config loggers. Before it, the log will not work
opt.save_path = os.path.join(opt.save_model_path, opt.log_name, systime)
os.makedirs(opt.save_path, exist_ok=True)
util.setup_logger(
    "train",
    opt.save_path,
    "train_" + opt.log_name,
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("train")


def save_network(network, epoch, name):
    save_path = os.path.join(opt.save_path, "models")
    os.makedirs(save_path, exist_ok=True)
    model_name = "epoch_{}_{:03d}.pth".format(name, epoch)
    save_path = os.path.join(save_path, model_name)
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
    logger.info("Checkpoint saved to {}".format(save_path))


def load_network(load_path, network, strict=True):
    assert load_path is not None
    logger.info("Loading model from [{:s}] ...".format(load_path))
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith("module."):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)
    return network


def save_state(epoch, optimizer, scheduler):
    """Saves training state during training, which will be used for resuming"""
    save_path = os.path.join(opt.save_path, "training_states")
    os.makedirs(save_path, exist_ok=True)
    state = {
        "epoch": epoch,
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_filename = "{}.state".format(epoch)
    save_path = os.path.join(save_path, save_filename)
    torch.save(state, save_path)


def resume_state(load_path, optimizer, scheduler):
    """Resume the optimizers and schedulers for training"""
    resume_state = torch.load(load_path)
    epoch = resume_state["epoch"]
    resume_optimizer = resume_state["optimizer"]
    resume_scheduler = resume_state["scheduler"]
    optimizer.load_state_dict(resume_optimizer)
    scheduler.load_state_dict(resume_scheduler)
    return epoch, optimizer, scheduler


def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = "epoch_{}_{:03d}.pth".format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print("Checkpoint saved to {}".format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith("gauss"):
            self.params = [
                float(p) / 255.0 for p in style.replace("gauss", "").split("_")
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith("poisson"):
            self.params = [float(p) for p in style.replace("poisson", "").split("_")]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"
        elif style.startswith("pg"):
            self.params = [float(p) for p in style.replace("pg", "").split("_")]
            if len(self.params) == 2:
                self.params[1] = self.params[1] / 255.0
                self.style = "pg_fix"
            elif len(self.params) == 4:
                self.params[2] = self.params[2] / 255.0
                self.params[3] = self.params[3] / 255.0
                self.style = "pg_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = (
                torch.rand(size=(shape[0], 1, 1, 1), device=x.device)
                * (max_std - min_std)
                + min_std
            )
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = (
                torch.rand(size=(shape[0], 1, 1, 1), device=x.device)
                * (max_lam - min_lam)
                + min_lam
            )
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "pg_fix":
            lam, std = self.params
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)

            # # Gaussian Approximation
            # sigma = torch.sqrt(x / lam + std**2)
            # noise = torch.cuda.FloatTensor(shape, device=x.device)
            # torch.normal(mean=0.0,
            #              std=sigma,
            #              generator=get_generator(),
            #              out=noise)
            # out = x + noise

            # Poisson Gaussian Noise
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0, std=std, generator=get_generator(), out=noise)
            out = noised + noise
            return out
        elif self.style == "pg_range":
            min_lam, max_lam, min_std, max_std = self.params
            lam = (
                torch.rand(size=(shape[0], 1, 1, 1), device=x.device)
                * (max_lam - min_lam)
                + min_lam
            )
            std = (
                torch.rand(size=(shape[0], 1, 1, 1), device=x.device)
                * (max_std - min_std)
                + min_std
            )
            # Poisson Gaussian Noise
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            out = noised + noise
            return out

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "pg_fix":
            lam, std = self.params

            # # Gaussian Approximation
            # sigma = np.sqrt(x / lam + std**2)
            # noise = np.array(np.random.normal(size=shape) * sigma,
            #                 dtype=np.float32)
            # out = x + noise

            # Poisson Gaussian Noise
            noised = np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
            noise = np.array(np.random.normal(size=shape) * std, dtype=np.float32)
            out = noised + noise
            return out
        elif self.style == "pg_range":
            min_lam, max_lam, min_std, max_std = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            # Poisson Gaussian Noise
            noised = np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
            noise = np.array(np.random.normal(size=shape) * std, dtype=np.float32)
            out = noised + noise
            return out


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


def generate_mask(img, width=4, mask_type="random"):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask = torch.zeros(
        size=(n * h // width * w // width * width**2,),
        dtype=torch.int64,
        device=img.device,
    )
    idx_list = torch.arange(0, width**2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(
        size=(n * h // width * w // width,), dtype=torch.int64, device=img.device
    )

    if mask_type == "random":
        torch.randint(
            low=0,
            high=len(idx_list),
            size=(n * h // width * w // width,),
            device=img.device,
            generator=get_generator(device=img.device),
            out=rd_idx,
        )
    elif mask_type == "batch":
        rd_idx = torch.randint(
            low=0,
            high=len(idx_list),
            size=(n,),
            device=img.device,
            generator=get_generator(device=img.device),
        ).repeat(h // width * w // width)
    elif mask_type == "all":
        rd_idx = torch.randint(
            low=0,
            high=len(idx_list),
            size=(1,),
            device=img.device,
            generator=get_generator(device=img.device),
        ).repeat(n * h // width * w // width)
    elif "fix" in mask_type:
        index = mask_type.split("_")[-1]
        index = torch.from_numpy(np.array(index).astype(np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(
        start=0,
        end=n * h // width * w // width * width**2,
        step=width**2,
        dtype=torch.int64,
        device=img.device,
    )

    mask[rd_pair_idx] = 1

    mask = depth_to_space(
        mask.type_as(img)
        .view(n, h // width, w // width, width**2)
        .permute(0, 3, 1, 2),
        block_size=width,
    ).type(torch.int64)

    return mask


def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n * c, 1, h, w), kernel, stride=1, padding=1
    )

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv


class Masker(object):
    def __init__(self, width=4, mode="interpolate", mask_type="all"):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == "interpolate":
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type="fix_{}".format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print("fetch {} samples for training".format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx : xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy : yy + self.patch, :]
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


class DataLoader_SIDD_Medium_Raw(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_SIDD_Medium_Raw, self).__init__()
        self.data_dir = data_dir
        # get images path
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print("fetch {} samples for training".format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = loadmat(fn)["x"]
        # random crop
        H, W = im.shape
        CSize = 256
        rnd_h = np.random.randint(0, max(0, H - CSize))
        rnd_w = np.random.randint(0, max(0, W - CSize))
        im = im[rnd_h : rnd_h + CSize, rnd_w : rnd_w + CSize]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def get_SIDD_validation(dataset_dir):
    val_data_dict = loadmat(
        os.path.join(dataset_dir, "ValidationNoisyBlocksRaw.mat"))
    val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']
    val_data_dict = loadmat(
        os.path.join(dataset_dir, 'ValidationGtBlocksRaw.mat'))
    val_data_gt = val_data_dict['ValidationGtBlocksRaw']
    num_img, num_block, _, _ = val_data_gt.shape
    return num_img, num_block, val_data_noisy, val_data_gt


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(target, ref):
    """
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def calculate_psnr(target, ref, data_range=255.0):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr


# Training Set
TrainingDataset = DataLoader_SIDD_Medium_Raw(opt.data_dir)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

# Validation Set
valid_dict = {
    "SIDD_Val": get_SIDD_validation(opt.val_dirs)
}

# Masker
masker = Masker(width=4, mode="interpolate", mask_type="all")

# Network
if opt.n_channel == 4:
    out_chn = 14
elif opt.n_channel == 1:
    out_chn = 2

network = UNet(
    in_channels=opt.n_channel, out_channels=out_chn, wf=opt.n_feature
)
estimator = est_UNet(
    num_classes=3, in_channels=opt.n_channel, depth=3
)
if opt.parallel:
    network = torch.nn.DataParallel(network, device_ids=device_ids)
    estimator = torch.nn.DataParallel(estimator, device_ids=device_ids)
network = network.cuda()
estimator = estimator.cuda()

# about training scheme
num_epoch = opt.n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(
    [{"params": network.parameters()}, 
     {"params": estimator.parameters(), "lr":opt.lr*0.5}],
    lr=opt.lr,
    weight_decay=opt.w_decay,
)
scheduler = lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[
        int(20 * ratio),
        int(40 * ratio),
        int(60 * ratio),
        int(80 * ratio),
    ],
    gamma=opt.gamma,
)
print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

# Resume and load pre-trained model
epoch_init = 1
if opt.resume is not None:
    epoch_init, optimizer, scheduler = resume_state(opt.resume, optimizer, scheduler)
    epoch_init += 1
if opt.checkpoint is not None:
    network = load_network(opt.checkpoint, network, strict=True)
if opt.ckpt_est is not None:
    estimator = load_network(opt.ckpt_est, estimator, strict=True)

# temp
if opt.checkpoint is not None and opt.resume is None:
    epoch_init = 40
    for i in range(1, epoch_init):
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
        logger.info("----------------------------------------------------")
        logger.info("==> Resuming Training with learning rate:{}".format(new_lr))
        logger.info("----------------------------------------------------")

print("init finish")

if opt.noisetype in ["gauss25", "poisson30"]:
    Thread1 = 0.8
    Thread2 = 1.0
else:
    Thread1 = 0.4
    Thread2 = 1.0

Thread1 = 0.8
Thread2 = 1.0

if "gauss" in opt.noisetype:
    noise_type = "gauss"
elif "poisson" in opt.noisetype:
    noise_type = "poisson"
elif "pg" in opt.noisetype:
    noise_type = "pg"
elif "real" in opt.noisetype:
    noise_type = "pg"

Lambda1 = opt.Lambda1
Lambda2 = opt.Lambda2
increase_ratio = opt.increase_ratio

clip_grad_E = 1e1
clip_grad_D = 1e1
if not epoch_init == 1:
    clip_grad_E = 1e1
    clip_grad_D = 2e0
param_E = [x for name, x in estimator.named_parameters()]
param_D = [x for name, x in network.named_parameters()]
for epoch in range(epoch_init, opt.n_epoch + 1):
    grad_norm_E = 0
    grad_norm_D = 0
    cnt = 0
    for param_group in optimizer.param_groups:
        current_lr = param_group["lr"]
        print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    # g25, p30: 1_1-2; frange-10
    # g5-50 | p5-50 | raw; 1_1-2; range-10
    Lambda = epoch / opt.n_epoch
    if Lambda <= Thread1:
        beta = Lambda2
    elif Thread1 <= Lambda <= Thread2:
        beta = Lambda2 + (Lambda - Thread1) * (increase_ratio - Lambda2) / (
            Thread2 - Thread1
        )
    else:
        beta = increase_ratio
    alpha = Lambda1

    network.train()
    estimator.train()
    for ii, noisy in enumerate(TrainingLoader):
        st = time.time()
        noisy = noisy.cuda()
        # pack raw data
        noisy = space_to_depth(noisy, 2)
        n, c, h, w = noisy.shape

        optimizer.zero_grad()

        # estimator
        est_map = estimator(noisy)
        est_map = torch.abs(est_map) + 1e-5
        est_map = torch.mean(est_map, dim=[2, 3], keepdim=True)
        est_sigma_x = est_map[:, 0:1, :, :]
        est_sigma_e = est_map[:, 1:2, :, :]
        est_sigma = torch.sqrt((est_sigma_x + beta**2*est_sigma_e) / (1 + beta)**2)
        est_alpha = est_map[:, 2:3, :, :]
        est_map = torch.cat((est_sigma, est_alpha), dim=1)
        gat_noisy = gat(noisy, est_sigma, est_alpha)

        gat_noisy_sub = torch.cat((
            gat_noisy[:,:,0:3*h//4,0:3*w//4], 
            gat_noisy[:,:,0:3*h//4,w//4:w],
            gat_noisy[:,:,h//4:h,0:3*w//4],
            gat_noisy[:,:,h//4:h,w//4:w]), dim=1)
        try:
            est_val_sub = chen_estimate(gat_noisy_sub)
            est_val = chen_estimate(gat_noisy)
        except Exception:
            continue
        print(f"est_val: {est_val[0].data}")
        est_diff = (est_val - 1.0)**2
        est_diff_sub = (est_val_sub - 1.0)**2
        est_diff = 1.5 * est_diff.repeat(1,4) + est_diff_sub
        loss_est = torch.mean(est_diff)
        loss_est = opt.w_est * loss_est

        # # 0.0rf0.01 from 41 to 100
        # border = int(40 * ratio)
        # if epoch<=border:
        #     loss_est = 0.0 * loss_est
        # else:
        #     # fix pattern
        #     loss_est = opt.w_est * loss_est
        #     # # range pattern
        #     # factor = (epoch-border)/(opt.n_epoch-border)
        #     # w_est = (1-factor)*0.0+factor*opt.w_est 
        #     # loss_est = w_est * loss_est

        # dn
        net_input, mask = masker.train(noisy)
        dn_output = (network(net_input) * mask).view(n, -1, out_chn, h, w).sum(dim=1)
        
        # exp
        with torch.no_grad():
            exp_output = network(noisy)

        mu_x, sigma_x = calc_mu_sigma(noisy, dn_output)
        mu_e, sigma_e = calc_mu_sigma(noisy, exp_output)

        # # sub
        # mu_m = mu_x + (mu_e - mu_x.detach()) * beta / (1 + beta) 

        # ordivk
        k = 2.  # 1. || 2.
        mu_m = mu_x / k + mu_x.detach() * (1 - 1 / k) + (mu_e - mu_x.detach()) * beta / (1 + beta)       

        sigma_m = (sigma_x + beta**2 * sigma_e) / (1 + beta)**2
        sigma_n = calc_noise_var(est_map, mu_m.detach(), noise_type)

        diff = mu_x - noisy
        exp_diff = mu_e - noisy
        mid_diff = mu_m - noisy

        try:
            loss_brv = likelihood_loss(noisy, mu_m, sigma_m, sigma_n, train=True)
        except Exception:
            continue
        loss_brv = torch.mean(loss_brv)
        loss_all = loss_brv + loss_est

        loss_all.backward()
        # clip the gradient norm of estimator and network
        total_norm_E = nn.utils.clip_grad_norm_(param_E, clip_grad_E)
        total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)
        if torch.isnan(total_norm_D) or torch.isnan(total_norm_E):
            continue
        grad_norm_E = (grad_norm_E*(ii/(ii+1)) + total_norm_E/(ii+1))
        grad_norm_D = (grad_norm_D*(ii/(ii+1)) + total_norm_D/(ii+1))
        optimizer.step()
        logger.info(
            "{:04d} {:05d} diff={:.6f}, exp_diff={:.6f}, mid_diff={:.6f}, Lambda={}, Loss_est={:.6f}, Loss_brv={:.6f}, Loss_All={:.6f}, Time={:.4f}, GE:{:.1e}/{:.1e}, GD:{:.1e}/{:.1e}\n sigma_x:{:.8f}, sigma_e:{:.8f}, sigma:{:.8f}, alpha:{:.8f}".format(
                epoch,
                ii,
                torch.mean(diff**2).item(),
                torch.mean(exp_diff**2).item(),
                torch.mean(mid_diff**2).item(),
                Lambda,
                loss_est.item(),
                loss_brv.item(),
                loss_all.item(),
                time.time() - st,
                clip_grad_E,
                total_norm_E,
                clip_grad_D,
                total_norm_D,
                torch.sqrt(est_sigma_x).squeeze().cpu().data.numpy()[0],
                torch.sqrt(est_sigma_e).squeeze().cpu().data.numpy()[0],
                est_sigma.squeeze().cpu().data.numpy()[0],
                est_alpha.squeeze().cpu().data.numpy()[0],
            )
        )

    scheduler.step()
    clip_grad_E = min(clip_grad_E, grad_norm_E)
    clip_grad_D = min(clip_grad_D, grad_norm_D)

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
        network.eval()
        estimator.eval()
        # save checkpoint
        save_network(network, epoch, "model")
        save_network(estimator, epoch, "est")
        save_state(epoch, optimizer, scheduler)
        # validation
        save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)
        np.random.seed(101)

        for valid_name, valid_data in valid_dict.items():
            avg_psnr_mu_e = []
            avg_ssim_mu_e = []
            avg_psnr_mu_m = []
            avg_ssim_mu_m = []
            avg_psnr_pme_e = []
            avg_ssim_pme_e = []
            avg_psnr_pme_m = []
            avg_ssim_pme_m = []
            save_dir = os.path.join(validation_path, valid_name)
            os.makedirs(save_dir, exist_ok=True)
            num_img, num_block, valid_noisy, valid_gt = valid_data
            for idx in range(num_img):
                for idy in range(num_block):
                    im = valid_gt[idx, idy][:, :, np.newaxis]
                    noisy_im = valid_noisy[idx, idy][:, :, np.newaxis]

                    origin255 = im.copy() * 255.0
                    origin255 = origin255.astype(np.uint8)
                    noisy255 = noisy_im.copy() * 255.0
                    noisy255 = noisy255.astype(np.uint8)

                    transformer = transforms.Compose([transforms.ToTensor()])
                    noisy_im = transformer(noisy_im)
                    noisy_im = torch.unsqueeze(noisy_im, 0)
                    noisy_im = noisy_im.cuda()
                    # pack raw data
                    noisy_im = space_to_depth(noisy_im, block_size=2)
                    with torch.no_grad():
                        n, c, h, w = noisy_im.shape
                        # # estimator
                        # est_map = estimator(noisy_im)
                        # est_map = torch.abs(est_map) + 1e-5
                        # est_map = torch.mean(est_map, dim=[2, 3], keepdim=True)
                        # dn
                        net_input, mask = masker.train(noisy_im)
                        dn_output = (network(net_input) * mask).view(n, -1, out_chn, h, w).sum(dim=1)
                        # exp
                        exp_output = network(noisy_im)

                        mu_x, sigma_x = calc_mu_sigma(noisy_im, dn_output)
                        mu_e, sigma_e = calc_mu_sigma(noisy_im, exp_output)
                        mu_m = (mu_x + beta * mu_e) / (1 + beta)

                        # unpack raw data
                        mu_e = depth_to_space(mu_e, block_size=2)
                        mu_m = depth_to_space(mu_m, block_size=2)

                    mu_e = mu_e.permute(0, 2, 3, 1)
                    mu_m = mu_m.permute(0, 2, 3, 1)

                    mu_e = mu_e.cpu().data.clamp(0, 1).numpy().squeeze(0)
                    mu_m = mu_m.cpu().data.clamp(0, 1).numpy().squeeze(0)

                    mu255_e = np.clip(mu_e * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    mu255_m = np.clip(mu_m * 255.0 + 0.5, 0, 255).astype(np.uint8)

                    # calculate psnr
                    # mu_e
                    psnr_mu_e = calculate_psnr(
                        im.astype(np.float32), mu_e.astype(np.float32)
                    )
                    avg_psnr_mu_e.append(psnr_mu_e)
                    ssim_mu_e = calculate_ssim(
                        (im*255.0).astype(np.float32), (mu_e*255.0).astype(np.float32)
                    )
                    avg_ssim_mu_e.append(ssim_mu_e)
                    # mu_m
                    psnr_mu_m = calculate_psnr(
                        im.astype(np.float32), mu_m.astype(np.float32)
                    )
                    avg_psnr_mu_m.append(psnr_mu_m)
                    ssim_mu_m = calculate_ssim(
                        (im*255.0).astype(np.float32), (mu_m*255.0).astype(np.float32)
                    )
                    avg_ssim_mu_m.append(ssim_mu_m)

                    # visualization
                    save_path = os.path.join(
                        save_dir,
                        "{}_{:03d}-{:03d}-{:03d}_clean.png".format(valid_name, idx, idy, epoch),
                    )
                    Image.fromarray(origin255.squeeze()).save(save_path)
                    save_path = os.path.join(
                        save_dir,
                        "{}_{:03d}-{:03d}-{:03d}_noisy.png".format(valid_name, idx, idy, epoch),
                    )
                    Image.fromarray(noisy255.squeeze()).save(save_path)
                    save_path = os.path.join(
                        save_dir,
                        "{}_{:03d}-{:03d}-{:03d}_mu_e.png".format(valid_name, idx, idy, epoch),
                    )
                    Image.fromarray(mu255_e.squeeze()).save(save_path)
                    save_path = os.path.join(
                        save_dir,
                        "{}_{:03d}-{:03d}-{:03d}_mu_m.png".format(valid_name, idx, idy, epoch),
                    )
                    Image.fromarray(mu255_m.squeeze()).save(save_path)

            avg_psnr_mu_e = np.array(avg_psnr_mu_e)
            avg_psnr_mu_e = np.mean(avg_psnr_mu_e)
            avg_ssim_mu_e = np.mean(avg_ssim_mu_e)

            avg_psnr_mu_m = np.array(avg_psnr_mu_m)
            avg_psnr_mu_m = np.mean(avg_psnr_mu_m)
            avg_ssim_mu_m = np.mean(avg_ssim_mu_m)

            log_path = os.path.join(validation_path, "A_log_{}.csv".format(valid_name))
            with open(log_path, "a") as f:
                f.writelines(
                    "epoch:{},mu_e:{:.6f}/{:.6f}, mu_m:{:.6f}/{:.6f}\n".format(
                        epoch,
                        avg_psnr_mu_e,
                        avg_ssim_mu_e,
                        avg_psnr_mu_m,
                        avg_ssim_mu_m,
                    )
                )
