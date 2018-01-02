import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import time
from helper import single_img_features, slide_window, add_heat, apply_threshold
from helper import draw_labeled_bboxes, draw_boxes, convert_color
from helper import bin_spatial, color_hist, get_hog_features
from scipy.ndimage.measurements import label
from keras.models import load_model
from moviepy.editor import VideoFileClip


svm_fid = open('svm.pkl', 'rb')
color_space = pickle.load(svm_fid)
orient = pickle.load(svm_fid)
pix_per_cell = pickle.load(svm_fid)
cell_per_block = pickle.load(svm_fid)
hog_channel = pickle.load(svm_fid)
spatial_size = pickle.load(svm_fid)
hist_bins = pickle.load(svm_fid)
spatial_feat = pickle.load(svm_fid)
hist_feat = pickle.load(svm_fid)
hog_feat = pickle.load(svm_fid)
svc = pickle.load(svm_fid)
X_scaler = pickle.load(svm_fid)

model = load_model('cnn.h5')


def search_windows(img, windows, clf):
    bbox_list = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        if clf == 'svm':
            features = single_img_features(test_img,
                                           color_space=color_space,
                                           spatial_size=spatial_size,
                                           hist_bins=hist_bins,
                                           orient=orient,
                                           pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel,
                                           spatial_feat=spatial_feat,
                                           hist_feat=hist_feat,
                                           hog_feat=hog_feat)
            test_features = X_scaler.transform(np.array(features).reshape(1, -1))
            prediction = svc.predict(test_features)
            if prediction[0] == 1:
                bbox_list.append(window)
        elif clf == 'cnn':
            test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
            test_img = test_img.astype(np.float32) - 0.5
            prediction = model.predict(np.array([test_img]), batch_size=1)
            if prediction[0][0] < 0.5:
                bbox_list.append(window)
        else:
            pass
    return bbox_list


def find_cars(img, xstart, ystart, ystop, scale):
    bbox_list = []
    img_tosearch = img[ystart:ystop, xstart:, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
    elif hog_channel == 0:
        ch1 = ctrans_tosearch[:, :, 1]
    elif hog_channel == 1:
        ch2 = ctrans_tosearch[:, :, 1]
    else:
        ch3 = ctrans_tosearch[:, :, 1]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    # nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    if hog_channel == 'ALL':
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    elif hog_channel == 0:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    elif hog_channel == 1:
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            elif hog_channel == 0:
                hog_features = hog1[ypos:ypos + nblocks_per_window,
                                    xpos:xpos + nblocks_per_window].ravel()
            elif hog_channel == 1:
                hog_features = hog2[ypos:ypos + nblocks_per_window,
                                    xpos:xpos + nblocks_per_window].ravel()
            else:
                hog_features = hog3[ypos:ypos + nblocks_per_window,
                                    xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            if spatial_feat:
                spatial_features = bin_spatial(subimg, size=spatial_size)
            else:
                spatial_features = []

            if hist_feat:
                hist_features = color_hist(subimg, nbins=hist_bins)
            else:
                hist_features = []

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox_list.append(((xbox_left + xstart, ytop_draw + ystart),
                                  (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))
    return bbox_list


def process_test_image(img, method='svm', threshold=1, fast_find=False):
    total_bbox_list = []
    img = img.astype(np.float32) / 255
    for scale in [1.0, 1.2, 1.5, 1.7, 2.0]:
        ystart = 400
        ystop = 540 + int(scale * 60)
        if fast_find and method == 'svm':
            bbox_list = find_cars(img, 250, ystart, ystop, scale)
        else:
            xy_window = (int(80 * scale), int(80 * scale))
            windows = slide_window(img, x_start_stop=[250, None], y_start_stop=[
                                   ystart, ystop], xy_window=xy_window, xy_overlap=(0.5, 0.5))
            bbox_list = search_windows(img, windows, method)
        total_bbox_list += bbox_list
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, total_bbox_list)
    heat = apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    return draw_labeled_bboxes(np.copy(img), labels), heatmap, total_bbox_list


orig_imgs = []
draw_imgs = []
heatmaps = []
total_bbox_lists = []


def run_test_images():
    % matplotlib inline
    for i in range(1, 7):
        img = mpimg.imread('test_images/test%d.jpg' % i)
        orig_imgs.append(img)
        t = time.time()
        draw_img, heatmap, total_bbox_list = process_test_image(img)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to process image...')
        draw_imgs.append(draw_img)
        heatmaps.append(heatmap)
        total_bbox_lists.append(total_bbox_list)


# run_test_images()


def show_draw_imgs():
    fig, axs = plt.subplots(2, 3, figsize=(24, 10))
    fig.subplots_adjust(hspace=.2, wspace=.1)
    axs = axs.ravel()
    for i in range(0, 6):
        axs[i].set_title('test{}.jpg'.format(i + 1), fontsize=16)
        img = draw_boxes(orig_imgs[i], total_bbox_lists[i])
        axs[i].imshow(img)


# show_draw_imgs()


def show_output_imgs():
    fig, axs = plt.subplots(2, 3, figsize=(24, 10))
    fig.subplots_adjust(hspace=.2, wspace=.1)
    axs = axs.ravel()
    for i in range(0, 6):
        axs[i].set_title('test{}.jpg'.format(i + 1), fontsize=16)
        axs[i].imshow(draw_imgs[i])


# show_output_imgs()


def show_heatmaps():
    fig, axs = plt.subplots(6, 2, figsize=(15, 20))
    fig.subplots_adjust(hspace=.2, wspace=.1)
    axs = axs.ravel()
    for i in range(0, 6):
        axs[i * 2].axis('off')
        axs[i * 2].set_title('test{}.jpg'.format(i + 1), fontsize=16)
        img = draw_boxes(orig_imgs[i], total_bbox_lists[i])
        axs[i * 2].imshow(img)
        axs[i * 2 + 1].axis('off')
        axs[i * 2 + 1].set_title('test{}.jpg - heatmap'.format(i), fontsize=16)
        axs[i * 2 + 1].imshow(heatmaps[i], cmap='hot')


# show_heatmaps()


def show_lables():
    fig, axs = plt.subplots(6, 2, figsize=(15, 20))
    fig.subplots_adjust(hspace=.2, wspace=.1)
    axs = axs.ravel()
    for i in range(0, 6):
        axs[i * 2].axis('off')
        axs[i * 2].set_title('test{}.jpg'.format(i + 1), fontsize=16)
        axs[i * 2].imshow(draw_imgs[i])
        labels = label(heatmaps[i])
        axs[i * 2 + 1].axis('off')
        axs[i * 2 + 1].set_title('test{}.jpg - {} car found'.format(i, labels[1]), fontsize=16)
        axs[i * 2 + 1].imshow(labels[0], cmap='gray')


# show_lables()


class SingletonDecorator:
    def __init__(self, kclass):
        self.kclass = kclass
        self.instance = None

    def __call__(self, *args, **kwds):
        if self.instance is None:
            self.instance = self.kclass(*args, **kwds)
        return self.instance


class Frames():
    def __init__(self, n):
        self.n = n
        self.frames = []

    def add_bbox_list(self, bbox_list):
        self.frames.append(bbox_list)
        if len(self.frames) > self.n:
            self.frames.pop(0)

    def get_accumulated_bbox_list(self):
        total_bbox_list = []
        for frame in self.frames:
            total_bbox_list += frame
        return total_bbox_list

    def get_num_frames(self):
        return len(self.frames)


Frames = SingletonDecorator(Frames)


num_frames = 10
g_frames = Frames(num_frames)
method = 'cnn'
fast_find = False


def process_image(img):
    total_bbox_list = []
    test_img = img.astype(np.float32) / 255
    for scale in [1.0, 1.2, 1.5, 1.7, 2.0]:
        ystart = 400
        ystop = 500 + int(scale * 60)
        if fast_find and method == 'svm':
            bbox_list = find_cars(test_img, 250, ystart, ystop, scale)
        else:
            xy_window = (int(80 * scale), int(80 * scale))
            windows = slide_window(test_img, x_start_stop=[250, None], y_start_stop=[
                                   ystart, ystop], xy_window=xy_window, xy_overlap=(0.5, 0.5))
            bbox_list = search_windows(test_img, windows, method)
        total_bbox_list += bbox_list
    g_frames.add_bbox_list(total_bbox_list)
    all_bbox_list = g_frames.get_accumulated_bbox_list()
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, all_bbox_list)
    heat = apply_threshold(heat, 2 * num_frames)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    return draw_labeled_bboxes(np.copy(img), labels)


def process_video():
    video_output = 'project_video_output.mp4'
    video_input = VideoFileClip('project_video.mp4')
    processed_video = video_input.fl_image(process_image)
    % time processed_video.write_videofile(video_output, audio=False)


def process_test_video():
    video_output = 'test_video_output.mp4'
    video_input = VideoFileClip('test_video.mp4')
    processed_video = video_input.fl_image(process_image)
    % time processed_video.write_videofile(video_output, audio=False)


process_video()
