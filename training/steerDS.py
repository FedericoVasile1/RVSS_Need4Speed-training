import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from glob import glob
from os import path
from PIL import Image

from metadata import CLASSES_LIST
from utils.miscellaneous import hcrop
import random

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder, crop_ratio, transform, img_ext = ".jpg", subsample=False, val_list=None):
        self.root_folder = root_folder
        self.crop_ratio = crop_ratio
        assert type(transform) is dict
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(
            path.join(self.root_folder, "*", "*" + self.img_ext)
        )
        # set this in the training loop, at each epoch for each phase
        self.phase = None
        if subsample:
            assert val_list is not None
            self.filenames = [filename for filename in self.filenames if filename not in val_list]
            random.shuffle(self.filenames)
            filenames_dict = {}
            classes_cnt = {i:0 for i in range(len(CLASSES_LIST))}
            # Count classes occurrences
            for filename in self.filenames:
                steering = np.float32(filename.split("/")[-1].split(self.img_ext)[0][6:])
                steering = round(steering.item(), 1)
                idx_steering = CLASSES_LIST.index(steering)
                classes_cnt[idx_steering] += 1
            # Filter filenames
            max_img_per_class = min(classes_cnt.values())
            classes_cnt_included = {i:0 for i in range(len(CLASSES_LIST))}
            filenames_filtered = []
            for filename in self.filenames:
                steering = np.float32(filename.split("/")[-1].split(self.img_ext)[0][6:])
                steering = round(steering.item(), 1)
                idx_steering = CLASSES_LIST.index(steering)
                if classes_cnt_included[idx_steering] < max_img_per_class:
                    classes_cnt_included[idx_steering] += 1
                    filenames_filtered.append(filename)
            self.filenames = filenames_filtered

        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        assert self.phase in ["train", "val", "test"]

        f = self.filenames[idx]        
        img = cv2.imread(f)
        
        if self.crop_ratio > 0:
            img = hcrop(img, self.crop_ratio)

        steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        steering = np.float32(steering)        

        if self.phase == "train":
            if torch.randint(0, 2, (1,)).item():
                img = cv2.flip(img, 1)
                steering *= np.array(-1, dtype=steering.dtype)

        steering = round(steering.item(), 1)

        idx_steering = CLASSES_LIST.index(steering)

        img = Image.fromarray(img)
        img = self.transform[self.phase](img)   
        
        sample = {"image":img , "steering":idx_steering}        
        
        return sample


def test():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = SteerDataSet("/home/iiticublap205/RVSS_Need4Speed/on_laptop/data",".jpg",transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)
    for S in ds_dataloader:
        im = S["image"]    
        y  = S["steering"]
        
        print(im.shape)
        print(y)
        break



if __name__ == "__main__":
    test()
