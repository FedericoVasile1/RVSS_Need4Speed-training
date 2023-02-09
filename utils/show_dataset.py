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
    if args.phase == "train":
        transform[args.phase] = transforms.Compose([

            transforms.ToTensor(),
        ])
    elif args.phase == "val":
        transform[args.phase] = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.phase == "test":
        transform[args.phase] = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError

    base_folder = "data_test" if args.phase == "test" else "data_train"
    ds = SteerDataSet(
        os.path.join(os.getcwd(), base_folder),
        args.crop_ratio,
        transform,
        ".jpg",
    )
    ds.phase = args.phase

    print("The dataset contains %d images " % len(ds))

    if args.phase != "test":
        trainset, valset = random_split(
            ds, [args.train_split, args.val_split], 
            generator=torch.Generator().manual_seed(args.seed)
        )
        loader = DataLoader(
            trainset if args.phase=="train" else valset, 
            batch_size=1, 
            shuffle=True if args.phase=="train" else False
        )

    for data in loader:
        images, labels = data["images"], data["steering"]
        print(data.shape)



if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "RVSS_Need4Speed-training"
    assert launch_dir == expected

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--crop_ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=float, default=10)
    parser.add_argument("--phase", type=str, default="train")

    args = parser.parse_args()

    assert args.train_split > 0 and args.train_split < 1
    args.val_split = 1 - args.train_split

    assert args.phase in ["train", "val", "test"]

    main(args)
