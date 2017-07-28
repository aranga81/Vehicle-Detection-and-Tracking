# Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import time
import pickle

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


############### LOAD TRAINING DATA #####################
vehicle_images = glob.glob('vehicles/**/*.png')
non_vehicle_images = glob.glob('non-vehicles/**/*.png')
print(len(vehicle_images), len(non_vehicle_images))

############### VISUALIZE TRAINING DATA #####################

fig, axs = plt.subplots(4,4, figsize=(16, 12))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()
for i in np.arange(16):
    img = cv2.imread(vehicle_images[np.random.randint(0,len(vehicle_images))])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axs[i].axis('off')
    axs[i].imshow(img)
plt.show()

fig, axs = plt.subplots(4,4, figsize=(16, 12))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()
for i in np.arange(16):
    img = cv2.imread(non_vehicle_images[np.random.randint(0,len(non_vehicle_images))])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axs[i].axis('off')
    axs[i].imshow(img)
plt.show()


############### SPATIAL BINNING #####################

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features

############### HISTOGRAM OF COLORS #####################

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

############### HOG FEATURE EXTRACTION #####################

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features

# Generate a random index to look at a car image
ind = np.random.randint(0, len(vehicle_images))
ind1 = np.random.randint(0, len(non_vehicle_images))
# Read in the image
image = mpimg.imread(vehicle_images[ind])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

image1 = mpimg.imread(non_vehicle_images[ind1])
gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
_, hog_image = get_hog_features(gray, orient,
                        pix_per_cell, cell_per_block,
                        vis=True, feature_vec=False)

_, hog_image1 = get_hog_features(gray1, orient,
                        pix_per_cell, cell_per_block,
                        vis=True, feature_vec=False)

bin_feat = bin_spatial(image, size=(32, 32))

# Plot the examples
fig = plt.figure()
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(222)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
plt.subplot(223)
plt.imshow(image1, cmap='gray')
plt.title('non vehicle')
plt.subplot(224)
plt.imshow(hog_image1, cmap='gray')
plt.title('HOG non vehicle visualization')
plt.show()

fig = plt.figure()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.plot(bin_feat)
plt.title('spatial bin features')
plt.show()


############### EXTRACT AND COMBINE ALL FEATURES #####################

def extract_features(imgs, cspace='RGB', spatial_size=(32,32), hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_binning=True,
                     color_hist_feat=True, hog_feat=True):
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_binning == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if color_hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        # get_hog_features() with vis=False, feature_vec=True
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


############### CALL extract_features() for vehicles/nonvehicles #####################

colorspace = 'YUV'
orient = 9
pixpcell = 8
cellpblock = 2
color_channels = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_binning = True
hist_feat = True
hog_feat = True

vehicle_features = extract_features(vehicle_images, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pixpcell, cell_per_block=cellpblock, hog_channel=color_channels, spatial_binning=spatial_binning,
                     color_hist_feat=hist_feat, hog_feat=hog_feat)
non_vehicle_features = extract_features(non_vehicle_images, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pixpcell, cell_per_block=cellpblock, hog_channel=color_channels, spatial_binning=spatial_binning,
                     color_hist_feat=hist_feat, hog_feat=hog_feat)


features = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
features_scaler = StandardScaler().fit(features)

features_scaled = features_scaler.transform(features)
labels = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

print('length of total features:', len(features_scaled))

indx = np.random.randint(0, len(vehicle_images))
fig = plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(mpimg.imread(vehicle_images[indx]))
plt.title('Original Image')
plt.subplot(132)
plt.plot(features[indx])
plt.title('Raw Features')
plt.subplot(133)
plt.plot(features_scaled[indx])
plt.title('Normalized Features')
fig.tight_layout()
plt.show()


############### TRAINING AND TEST DATA SPLIT #####################

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=rand_state)

print('length of training dataset:', X_train.shape[0])
print('length of test dataset:', X_test.shape[0])
print('length of features:', X_train.shape[1])

X_train, y_train = shuffle(X_train, y_train, random_state=14)

############### TRAIN A LINEARSVC CLASSIFIER #####################

svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

exit()

############### SLIDING WINDOW SEARCH #####################

def find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, cspace, hog_channel, spatial_size,
              hist_bins, features_scaler):

    rectangles = []
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]

    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(image)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = features_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                rectangles.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))

    return rectangles


############### DRAW BOXES #####################

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

############### TEST IMAGE #####################

test_img = mpimg.imread('./test_images/test1.jpg')

ystart = 400
ystop = 656
scale = 1.5

rectangles = find_cars(test_img, ystart, ystop, scale, svc, orient, pixpcell, cellpblock, colorspace, color_channels,
                       spatial_size, hist_bins, features_scaler)

test_img_boxes = draw_boxes(test_img, rectangles)
print('done')
plt.imshow(test_img_boxes)
plt.show()

############### SLIDING WINDOW SELECTION #####################

rects = []

ystart = 400
ystop = 600
scale = 2.0
rects.append(find_cars(test_img, ystart, ystop, scale, svc, orient, pixpcell, cellpblock, colorspace, color_channels,
                       spatial_size, hist_bins, features_scaler))
ystart = 440
ystop = 660
scale = 2.0
rects.append(find_cars(test_img, ystart, ystop, scale, svc, orient, pixpcell, cellpblock, colorspace, color_channels,
                       spatial_size, hist_bins, features_scaler))

rectangles = [item for sublist in rects for item in sublist]
test_img_rects = draw_boxes(test_img, rectangles, color='random', thick=2)
plt.figure(figsize=(10, 10))
plt.imshow(test_img_rects)
plt.show()


############### MULTI SCALE WINDOWS #####################

def multiscale_windows(img):

    ystart = [400, 420, 440, 460]
    ystop = [500, 540, 560, 600, 660]
    scale = [1.0, 1.5, 2.0, 3.0]
    colorspace = 'YUV'
    orient = 9
    pixpcell = 8
    cellpblock = 2
    color_channels = 'ALL'
    spatial_size = (32, 32)
    hist_bins = 32

    test_img = img

    rectangles = []

    rectangles.append(find_cars(test_img, ystart[0], ystop[0], scale[0], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))

    rectangles.append(find_cars(test_img, ystart[1], ystop[1], scale[0], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))



    rectangles.append(find_cars(test_img, ystart[0], ystop[1], scale[1], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))

    rectangles.append(find_cars(test_img, ystart[1], ystop[2], scale[1], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))


    rectangles.append(find_cars(test_img, ystart[0], ystop[2], scale[2], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))

    rectangles.append(find_cars(test_img, ystart[1], ystop[3], scale[2], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))

    rectangles.append(find_cars(test_img, ystart[2], ystop[4], scale[2], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))


    rectangles.append(find_cars(test_img, ystart[0], ystop[3], scale[3], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))

    rectangles.append(find_cars(test_img, ystart[2], ystop[4], scale[3], svc, orient, pixpcell,
                                cellpblock, colorspace, color_channels, spatial_size, hist_bins, features_scaler))


    rect = [item for sublist in rectangles for item in sublist]
    return rect

rectangles = multiscale_windows(test_img)
test_img_multiple_windows = draw_boxes(test_img, rectangles)

plt.imshow(test_img_multiple_windows)
plt.show()

############### HEAT MAP #####################

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)
    # Return the image
    return img

heat_img = np.zeros_like(test_img[:, :, 0])
heat = add_heat(heat_img, rectangles)

heat = apply_threshold(heat, 1)
heatmap = np.clip(heat, 0, 255)

labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(test_img), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()


############### PROCESS PIPELINE #####################

def process(img):

    rectangles = multiscale_windows(img)

    if len(rectangles) > 0:
        track.track_rect(rectangles)

    heatmap = np.zeros_like(img[:, :, 0])

    for rect in track.cars:
        heatmap = add_heat(heatmap, rect)

    heatmap = apply_threshold(heatmap, 1 + len(track.cars) // 2)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img


############### VEHICLE TRACKING CLASS #####################

class Vehicle_tracking():
    def __init__(self):
        self.cars =  []

    def track_rect(self, rects):
        self.cars.append(rects)
        if len(self.cars) > 15:
            # throw out oldest rectangle set(s)
            self.cars = self.cars[len(self.cars) - 15:]



############### Test Images #####################

test_images = glob.glob('./test_images/test*.jpg')

fig, axs = plt.subplots(3, 2, figsize=(16, 14))
fig.subplots_adjust(hspace =.004, wspace=.002)
axs = axs.ravel()

for i, img in enumerate(test_images):
    track = Vehicle_tracking()
    axs[i].imshow(process(mpimg.imread(img)))
    axs[i].axis('off')
plt.show()


############### Test Video #####################
track = Vehicle_tracking()

test_output = 'test_video_output1.mp4'
clip_test = VideoFileClip('test_video.mp4')
clip_test_out = clip_test.fl_image(process)
clip_test_out.write_videofile(test_output, audio=False)


############### Project Video #####################
track = Vehicle_tracking()

test_output1 = 'project_video_output1.mp4'
clip_test1 = VideoFileClip('project_video.mp4')
clip_test_out1 = clip_test1.fl_image(process)
clip_test_out1.write_videofile(test_output1, audio=False)