import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from helper import get_hog_features, draw_boxes, slide_window


cars = glob.glob('data/vehicles/*/*')
notcars = glob.glob('data/non-vehicles/*/*')

% matplotlib inline


def show_car_samples():
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()
    for i in np.arange(8):
        img = cv2.imread(cars[np.random.randint(0, len(cars))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].set_title('car', fontsize=10)
        axs[i].imshow(img)
    for i in np.arange(8, 16):
        img = cv2.imread(notcars[np.random.randint(0, len(notcars))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].set_title('not car', fontsize=10)
        axs[i].imshow(img)


show_car_samples()


def show_hog_sample():
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()
    i = 0
    car_img = cv2.imread(cars[np.random.randint(0, len(cars))])
    car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
    axs[i].set_title('car', fontsize=16)
    axs[i].imshow(car_img)
    i += 1
    car_features, car_hog_img = get_hog_features(
        car_img[:, :, 2], 16, 16, 2, vis=True, feature_vec=True)
    axs[i].set_title('car', fontsize=16)
    axs[i].imshow(car_hog_img, cmap='gray')
    i += 1
    notcar_img = cv2.imread(notcars[np.random.randint(0, len(cars))])
    notcar_img = cv2.cvtColor(notcar_img, cv2.COLOR_BGR2RGB)
    axs[i].set_title('not car', fontsize=16)
    axs[i].imshow(notcar_img)
    i += 1
    notcar_features, notcar_hog_img = get_hog_features(
        notcar_img[:, :, 2], 16, 16, 2, vis=True, feature_vec=True)
    axs[i].set_title('not car hog', fontsize=16)
    axs[i].imshow(notcar_hog_img, cmap='gray')


show_hog_sample()


def show_sliding_window():
    img = mpimg.imread('test_images/test1.jpg')
    bbox_list = []
    for scale in [1.0, 1.2, 1.5, 1.7, 2.0]:
        ystart = 400
        ystop = 540 + int(scale * 60)
        xy_window = (int(80 * scale), int(80 * scale))
        windows = slide_window(img, x_start_stop=[250, None], y_start_stop=[ystart, ystop],
                               xy_window=xy_window, xy_overlap=(0.5, 0.5))
        bbox_list += windows
    draw_img = draw_boxes(img, bbox_list)
    fig, axs = plt.subplots(2, 1, figsize=(16, 10))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()
    i = 0
    axs[i].set_title('car', fontsize=16)
    axs[i].imshow(img)
    i += 1
    axs[i].set_title('{} windows to search'.format(len(bbox_list)), fontsize=16)
    axs[i].imshow(draw_img)


show_sliding_window()
