import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import cv2
import numpy as np

sys.path.append(os.getcwd())
from utils.miscellaneous import define_args, get_transforms
from training.steerDS import SteerDataSet
from metadata import Net_MEAN, Net_STD, mobilenet_v2_MEAN, mobilenet_v2_STD, CLASSES_LIST


def main(args):
    transform = get_transforms(args)

    if args.model_name == "Net":
        MEAN = Net_MEAN
        STD = Net_STD
    elif args.model_name == "mobilenet_v2":
        MEAN = mobilenet_v2_MEAN
        STD = mobilenet_v2_STD
    else:
        raise NotImplementedError

    inv_MEAN = [-m for m in MEAN]
    inv_STD = [1/s for s in STD]
    invTrans = transforms.Compose([
        transforms.Normalize(
            mean=[ 0., 0., 0. ], std=inv_STD
        ),
        transforms.Normalize(
            mean=inv_MEAN, std=[1., 1., 1.]
        ),
    ])

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
        image, label = data["image"], data["steering"]
        image = invTrans(image)
        image = image.squeeze(0)
        image *= 255
        image = torch.permute(image, (1, 2, 0))
        image = image.numpy().astype("uint8")

        # redraw the cropped part as black, just for sake of visualization
        if args.crop_ratio > 0:
            h_no_crop = image.shape[0] / (1 - args.crop_ratio)
            offset = h_no_crop - image.shape[0]
            h_black = np.zeros(
                (int(offset), image.shape[1], image.shape[2]), dtype=np.uint8
            ) 
            image = np.concatenate((h_black, image), axis=0)

        label = CLASSES_LIST[label.squeeze().item()]
        cv2.putText(
            image, 
            str(label),
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1
        )

        cv2.imshow("image", image)
        if cv2.waitKey(0) == ord("q"):
            exit(0)


if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "RVSS_Need4Speed-training"
    assert launch_dir == expected

    # add a custom argument, specific only of this script 
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="")
    temp_args, unknown = parser.parse_known_args()
    assert temp_args.phase in ["train", "test"]
    
    # Remove --phase argument to make the define_args above working
    subtract = 0
    for idx in range(len(sys.argv)):
        val = sys.argv[idx - subtract]
        if val in []:
            # only argument (e.g. boolean)
            del sys.argv[idx - subtract]
            subtract += 1
        elif val in ['--phase']:
            # argument + parameter
            del sys.argv[idx - subtract]
            del sys.argv[idx - subtract]
            subtract += 2

    args = define_args()

    args.phase = temp_args.phase

    args.batch_size = 1

    main(args)
