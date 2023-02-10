import os
import argparse
import sys

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import classification_report

sys.path.append(os.getcwd())
from training.net import load_model
from training.steerDS import SteerDataSet
from metadata import CLASSES_LIST
from utils.miscellaneous import define_args, get_transforms


def main(args):
    transform = get_transforms(args)

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

    # Load checkpoint
    checkpoint = torch.load(os.path.join(os.getcwd(), args.checkpoint))
    net.load_state_dict(checkpoint["model_state_dict"])

    net.to(device)

    phases = ["test"]

    dataloader = {}
    dataloader["test"] = testloader

    test_acc = []
    total_preds = []
    total_labels = []

    # Test NN
    for p in phases:

        is_training = True if p == "train" else False

        net.train(is_training)
        dataloader[p].dataset.phase = p
        with torch.set_grad_enabled(is_training):

            for i, data in enumerate(dataloader[p], start=1):
                inputs, labels = data['image'].to(device), data['steering'].to(device)
                b_size = inputs.shape[0]

                # forward
                outputs = net(inputs).squeeze()
                if not len(outputs.shape):
                    outputs = outputs.unsqueeze(dim=0)

                outputs = outputs.argmax(dim=1)
                test_acc += (outputs == labels).tolist()

                total_preds += outputs.tolist()
                total_labels += labels.tolist()

    print(sum(test_acc) / len(test_acc))

    # TODO
    print(
        classification_report(
            total_labels, 
            total_preds, 
            target_names=[str(c_l) for c_l in CLASSES_LIST]
        )
    )

if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "RVSS_Need4Speed-training"
    assert launch_dir == expected

    args = define_args()

    main(args)