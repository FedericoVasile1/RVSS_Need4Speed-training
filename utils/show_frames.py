import argparse
from email.mime import image
import os
import glob

import cv2 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="")
    args = parser.parse_args()

    images = glob.glob(os.path.join(os.getcwd(), args.dataset_path, "*.jpg"))

    for idx in range(len(images)):
        img_path = images[idx]
        img_name = os.path.basename(img_path)
        img_seq, img_label = img_name[:6], img_name[6:]
        img_label = img_label[:-4]  # remove .jpg

        new_img_path = img_path.replace(img_name, img_seq)

        images[idx] = new_img_path

    images.sort()

    for partial_img_path in images:
        img_path = glob.glob(partial_img_path + "*.jpg")
        assert len(img_path) == 1
        img_path = img_path[0]
        
        label = os.path.basename(img_path)
        label = label[6:]   # remove image sequence number
        label = label[:-4]

        img = cv2.imread(img_path)
        cv2.putText(
            img, 
            label,
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1
        )

        cv2.imshow("image", img)
        print(os.path.basename(img_path))
        if cv2.waitKey(0) == ord("q"):
            exit(0)