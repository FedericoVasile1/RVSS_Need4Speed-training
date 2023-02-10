import cv2 
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, default="")
    args = parser.parse_args()

    