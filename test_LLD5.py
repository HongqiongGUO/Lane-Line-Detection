import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from LLD5_Lane_Detection_Pipeline import lane_finding_pipeline

for image_path in list(os.listdir('CarND-LaneLines-P1/test_images')):
    fig = plt.figure(figsize=(20, 10))
    image = mpimg.imread(f'CarND-LaneLines-P1/test_images/{image_path}')
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(image)
    ax.set_title("Input Image")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])
    plt.imshow(lane_finding_pipeline(image))
    ax.set_title("Output Image [Lane Line Detected]")
    plt.show()