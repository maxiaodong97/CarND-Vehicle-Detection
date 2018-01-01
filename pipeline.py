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
            test_img = test_img.astype(np.float32) / 255
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
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            test_img = test_img.astype(np.float32) / 255 - 0.5
            prediction = model.predict(np.array([test_img]), batch_size=1)
            if prediction[0][0] < 0.5:
                bbox_list.append(window)
        else:
            pass
    return bbox_list


def find_cars(img, ystart, ystop, scale):
    bbox_list = []
    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

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

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction[0] == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox_list.append(((xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return bbox_list


def process_test_image(img, method='svm', threshold=1, fast_find=False):
    total_bbox_list = []
    for scale in [1.0, 1.5, 2.0]:
        ystart = 380
        ystop = 550 + int(scale * 64)
        if fast_find and method == 'svm':
            bbox_list = find_cars(img, ystart, ystop, scale)
        else:
            xy_window = (int(64 * scale), int(64 * scale))
            windows = slide_window(img, y_start_stop=[ystart, ystop], xy_window=xy_window)
            bbox_list = search_windows(img, windows, method)
        total_bbox_list += bbox_list
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, total_bbox_list)
    heat = apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    return draw_labeled_bboxes(np.copy(img), labels), heatmap, total_bbox_list


def run_test_images():
    % matplotlib inline
    for i in range(1, 7):
        img = mpimg.imread('test_images/test%d.jpg' % i)
        t = time.time()
        draw_img, heatmap, total_bbox_list = process_test_image(img, method='cnn', threshold=2)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to process image...')
        fig = plt.figure(figsize=(16, 32))
        box_img = draw_boxes(img, total_bbox_list)
        plt.subplot(i * 100 + 31)
        plt.imshow(box_img)
        plt.title('Car with Boxes')
        plt.subplot(i * 100 + 32)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(i * 100 + 33)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()


run_test_images()


def process_image(img):
    total_bbox_list = []
    for scale in [1.0, 1.5, 2.0]:
        ystart = 380
        ystop = 550 + int(scale * 64)
        xy_window = (int(64 * scale), int(64 * scale))
        windows = slide_window(img, y_start_stop=[ystart, ystop], xy_window=xy_window)
        bbox_list = search_windows(img, windows, 'cnn')
        total_bbox_list += bbox_list
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, total_bbox_list)
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    return draw_labeled_bboxes(np.copy(img), labels)


def process_video():
    video_output = 'project_video_output.mp4'
    video_input = VideoFileClip('project_video.mp4')
    processed_video = video_input.fl_image(process_image)
    % time processed_video.write_videofile(video_output, audio=False)
