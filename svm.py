import matplotlib.image as mpimg
import numpy as np
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from helper import single_img_features

cars = glob.glob('data/vehicles/*/*.png')
notcars = glob.glob('data/non-vehicles/*/*.png')

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 13  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off


def extract_features(imgs):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        img_features = single_img_features(image,
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
        features.append(img_features)
    return features


t = time.time()
car_features = extract_features(cars)
notcar_features = extract_features(notcars)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract HOG features...')
X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=1)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
svc = LinearSVC()
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
CM = confusion_matrix(y_test, svc.predict(X_test))
print('False positive {:.2%}'.format(CM[0][1] / len(y_test)))
print('False negative {:.2%}'.format(CM[1][0] / len(y_test)))

with open('svm.pkl', 'wb') as fid:
    pickle.dump(color_space, fid)
    pickle.dump(orient, fid)
    pickle.dump(pix_per_cell, fid)
    pickle.dump(cell_per_block, fid)
    pickle.dump(hog_channel, fid)
    pickle.dump(spatial_size, fid)
    pickle.dump(hist_bins, fid)
    pickle.dump(spatial_feat, fid)
    pickle.dump(hist_feat, fid)
    pickle.dump(hog_feat, fid)
    pickle.dump(svc, fid)
    pickle.dump(X_scaler, fid)
