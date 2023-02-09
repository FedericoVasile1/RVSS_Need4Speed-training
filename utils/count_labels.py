import torch
import os
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append(os.getcwd())
from training.steerDS import SteerDataSet
from torchvision import transforms
from metadatas import CLASSES_LIST


transform = {}
transform["train"] = transforms.Compose([
        transforms.ToTensor(),
    ])
ds = SteerDataSet(
    os.path.join(os.getcwd(), "data_train"),
    0.35,
    transform,
    ".jpg",
)
ds.phase = "train"

print("The dataset contains %d images " % len(ds))

trainset, valset = random_split(
        ds, [0.7, 0.3],
        generator=torch.Generator().manual_seed(10)
    )


trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
dict_classes = {c_l: 0 for c_l in CLASSES_LIST}

for data in trainloader:
    label = data['steering']
    label = CLASSES_LIST[label.item()]

    dict_classes[label] += 1

print(dict_classes)
