import os
from PIL import Image
import numpy as np
import time
import json
import argparse
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--config", type=str, default="extract_frames_train_config.json")
    args = parser.parse_args()

    f = open(args.config)
    configs = json.load(f)
    f.close()

    count_video = 0
    directory = "code/train/frames"
    labels = ["real", "fake"]

    for label in labels:
        img_directory = os.path.join(directory, label)
        for root, dirs, files in os.walk(img_directory):
            print(count_video)
            for filename in files:
                count_video += 1
                
                base_name = root[23:] 
                # print(base_name)
                img_path = os.path.join(root, filename)
                # print(img_path)
                img =cv2.imread(img_path)
                img = cv2.resize(img, (224,224))

                frame_path = os.path.join("code\\train_resize\\frames", label)
                frame_path = os.path.join(frame_path, base_name)
                os.makedirs(frame_path, exist_ok=True)
                frame_path = os.path.join(frame_path, filename)
                # img.save(frame_path)
                cv2.imwrite(frame_path, img)

