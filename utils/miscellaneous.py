import argparse

from torchvision import transforms
import numpy as np
import torch

from metadata import Net_MEAN, Net_STD, mobilenet_v2_MEAN, mobilenet_v2_STD


def get_transforms(args):
    transform = {}
    if args.model_name == "Net":
        MEAN = Net_MEAN
        STD = Net_STD
    elif args.model_name == "mobilenet_v2":
        MEAN = mobilenet_v2_MEAN
        STD = mobilenet_v2_STD
    else:
        raise NotImplementedError

    if args.use_data_aug:
        transform["train"] = transforms.Compose([
            #transforms.RandomApply(
            #    torch.nn.ModuleList([
            #        transforms.Grayscale(num_output_channels=3),
            #    ]),
            #    p=0.1
            #),
            #transforms.RandomApply(
            #    torch.nn.ModuleList([
            #        transforms.ColorJitter(brightness=.1, hue=.1),
            #    ]),
            #    p=0.1
            #),
            #transforms.RandomApply(
            #    torch.nn.ModuleList([
            #        transforms.GaussianBlur((5, 9), sigma=(0.1, 5)),
            #    ]),
            #    p=0.1
            #),
            transforms.RandomApply(
                torch.nn.ModuleList([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ColorJitter(brightness=.1, hue=.1),
                    transforms.GaussianBlur((5, 9), sigma=(0.1, 5)),
                ]),
                p=0.1
            ),

            transforms.ToTensor(),
            transforms.Normalize(
                MEAN, STD
            )
        ])
    else:
        transform["train"] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                MEAN, STD
            )
        ])

    transform["val"] = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            MEAN, STD
        )
    ])

    transform["test"] = transform["val"]

    return transform
    

def hcrop(img, crop_ratio):
    assert type(img) is np.ndarray
    assert len(img.shape) == 3
    assert img.shape[-1] in [1, 3]
    # img.shape (H, W, C)

    idx_crop = int(img.shape[0] * crop_ratio)
    img = img[idx_crop:, :, :]
    return img


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--num_epochs", type=int, default=250)
    parser.add_argument("--feat_vect_dim", type=int, default=48048)
    parser.add_argument("--no_global_avg_pool", action="store_true")
    parser.add_argument("--model_name", type=str, default="Net")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--crop_ratio", type=float, default=0.35)
    parser.add_argument("--weighted_sampler", action="store_true")
    parser.add_argument("--use_data_aug", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    assert args.train_split > 0 and args.train_split < 1
    args.val_split = 1 - args.train_split

    assert args.crop_ratio >= 0 and args.crop_ratio < 1

    return args