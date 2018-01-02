import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
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
        ystop = 500 + int(scale * 60)
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


% matplotlib inline

video_input = VideoFileClip('project_video.mp4')

video_input.save_frame('test_images/test7.jpg', t=48)

img = mpimg.imread('test_images/test7.jpg')
draw_img, heatmap, total_bbox_list = process_test_image(img)
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=.2, wspace=.1)
axs.set_title('test7.jpg', fontsize=16)
img = draw_boxes(img, total_bbox_list)
axs.imshow(img)


# show_draw_imgs()
