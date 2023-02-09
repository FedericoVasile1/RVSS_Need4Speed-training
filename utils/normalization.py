import os
import sys
import argparse

from torchvision import transforms
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.getcwd())
from training.steerDS import SteerDataSet

def main(args):
    transform = {}
    transform["train"] = transforms.Compose([
        transforms.ToTensor(),

    ])
    ds = SteerDataSet(
        os.path.join(os.getcwd(), "data_train"),
        args.crop_ratio,
        transform,
        ".jpg",
    )
    ds.phase = "train"

    print("The dataset contains %d images " % len(ds))

    trainset, _ = random_split(
        ds, [args.train_split, args.val_split], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in trainloader:
        data = data["image"]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    
    print("mean: ", mean)
    print("std: ", std)


if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "RVSS_Need4Speed-training"
    assert launch_dir == expected

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--crop_ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=float, default=10)

    args = parser.parse_args()

    assert args.train_split > 0 and args.train_split < 1
    args.val_split = 1 - args.train_split


    main(args)
