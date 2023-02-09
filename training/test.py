from inspect import istraceback
import os
import time
import argparse
import sys

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from training.net import load_model
from training.steerDS import SteerDataSet


def main(args):
    transform = {}
    if args.model_name == "Net":
        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]
        transform["test"] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                MEAN, STD
            )
        ])
    elif args.model_name == "mobilenet_v2":
        transform["test"] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise NotImplementedError

    # Load data
    ds = SteerDataSet(
        os.path.join(os.getcwd(), "data_test"),
        args.crop_ratio,
        transform,
        ".jpg",
    )
    print("The dataset contains %d images " % len(ds))

    testloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create network
    net = load_model(args.model_name, args.feat_vect_dim, args.use_avg_pool)

    checkpoint = torch.load(
        os.path.join(os.getcwd(), "runs", args.logdir, "best_model.pth")
    )
    net.load_state_dict(checkpoint["state_dict"])

    net.to(device)

    dataloader = {}
    dataloader["test"] = testloader

    # Test NN
    test_acc = []

    phases = ["test"]
    for p in phases:

        is_training = True if p == "train" else False

        net.train(is_training)
        dataloader[p].dataset.dataset.phase = p
        with torch.set_grad_enabled(is_training):

            for i, data in enumerate(dataloader[p], start=1):
                inputs, labels = data['image'].to(device), data['steering'].to(device)
                b_size = inputs.shape[0]

                # forward + backward + optimize
                outputs = net(inputs).squeeze()
                if not len(outputs.shape):
                    outputs = outputs.unsqueeze(dim=0)

                outputs = outputs.argmax(dim=1)
                test_acc += (outputs == labels).tolist()

    if (sum(accuracy_epoch["val"]) / len(accuracy_epoch["val"])) > test_acc:
        test_acc = sum(accuracy_epoch["val"]) / len(accuracy_epoch["val"])
        epoch_best_model = epoch
        torch.save(
            {
                "epoch": epoch,
                "loss_clipped": test_acc,
                "model_state_dict": net.state_dict(),
                "crop_ratio": args.crop_ratio,
            },
            os.path.join(writer.log_dir, "best_model.pth")
        )

    # TODO create test.py testing pipeline
    # TODO create script to compute mean and std training set
    # TODO create script visualize data from dataloader
    # TODO put avg pooling in the model
    # TODO put augmentation e.g. brightness, blur etc
    # TODO decrease learning rate if val loss does not decrease
    # TODO early stopping

    print('Finished Training')
    print("Best validation accuracy is {:.4f} obtained at epoch {}"
          .format(test_acc, epoch_best_model))


if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "RVSS_Need4Speed-training"
    assert launch_dir == expected

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--feat_vect_dim", type=int, default=48048)
    parser.add_argument("--use_avg_pool", action="store_true")
    parser.add_argument("--min_max_boundaries", type=str, default="-0.5,0.5")
    parser.add_argument("--model_name", type=str, default="Net")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--crop_ratio", type=float, default=0.35)
    args = parser.parse_args()

    assert args.train_split > 0 and args.train_split < 1
    args.val_split = 1 - args.train_split

    assert len(args.min_max_boundaries.split(",")) == 2
    args.min_bound = args.min_max_boundaries.split(",")[0]
    args.max_bound = args.min_max_boundaries.split(",")[1]
    assert args.min_bound < args.max_bound

    assert args.crop_ratio >= 0 and args.crop_ratio < 1

    main(args)