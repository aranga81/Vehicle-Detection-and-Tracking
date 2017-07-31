# Vehicle-Detection-and-Tracking
Pipeline to detect and track cars - Using linear SVM for classification 

## Project Overview:

•	Performed a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and also appended the binned color transform features and trained a classifier Linear SVM classifier

•	Implemented  a sliding-window technique and use your trained classifier to search for vehicles in images.

•	Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. Estimate a bounding box for vehicles detected.

Pipeline and all the code in the following description is included in “Vehicle_Detection.py”.

## DATA VISUALIZATION:
- Training data contained cars and non-car images appropriately labelled as provided by Udacity classroom. Below shown are some randomly sampled vehicle and non-vehicle images from the dataset:
#### Vehicles...!!
![car](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/vehicle_images.png)

#### Non Vehicles...!!
![noncar](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/non_vehicle_images.png)

## Feature Extraction..!!!

### Color Histogram and Spatial Binning:

Performed spatial binning and also evaluated histogram of colors for all the input training images:
![sample](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/spatial_binning.png)

### [Histogram oriented gradients](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) (HOG) features:
Below are some random images from the vehicles and non-vehicles dataset along with the HOG visualization.
![HOG](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/HOG_visualization.png)

Thus the final pipeline refers to the function “extract_features()” in the file which reads in images and extracts all the three i.e. HOG, spatial information and color features. These are features are then concatenated and appended to the final “features” output.
Following the feature extraction step I normalized all the features using sklearn.preprocessing – StandardScaler().fit method.
Below plot summarizes the features before and after normalization:

![features](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/features_visualization.png)

## Classification - Linear SVM ...!!!
As I extract and normalize all the features, I then label the dataset 1 for cars and 0 for non-cars.
Then using the sklearn – train_test_split method, I split out 20% of the data as test data from training dataset.
The final training features and labels are shuffled and fed into a linear support vector machine classifier (SVM). 

## SLIDING WINDOW SEARCH:
The function find_cars() in the pipeline code combines all the spatial, color and HOG features with a sliding window search, and this specific function subsamples the features in a given window and runs them through the classifier to predict if there is a car in the window.
In case the classifier predicts a car in the window the function returns a rectangle object corresponding to that window and all others that it searches through.
These rectangle object returns from the sliding window search are used to draw and track the cars in the final test image and video pipelines.

### Multi-scale Window search:
To better cover the complete road ahead and search for cars many configurations for window sizes and positions have to be included. Hence I included the “multiscale_windows()” function which inturn calls the find_cars() function multiple times – each time covering different start and stop Y coordinated and different scaled windows to search for cars.

Different configurations explored and included in the function are shown below:

Sample test image with small scaled windows to track the far aways cars.

Y_start = 400 y_stop = 480
Y_start = 440 y_stop = 500 (overall window to increase confidence)
Scale = 1.0
![search](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/multiple_windowsearch_01.png)

Y_start = 400 y_stop = 480
Y_start = 440 y_stop = 500 (overall window to increase confidence)
Scale = 1.5 (medium sized)
![search](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/multiple_windowsearch_02.png)

Y_start = 400 y_stop = 480
Y_start = 440 y_stop = 500 (overall window to increase confidence)
Scale = 2.5 (large window search)
![search](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/multiple_windowsearch_03.png)

Final Test image with all the rectangles plotted after completing search through all the given window sizes and positions.
![search](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/multiple_windowsearch.png)

Finally after finishing the sliding window search on multi scale windows – I included the logic to eliminate any false positives. A true positive is typically accompanied by several positive detections, while false positives are typically accompanied by only one or two detections, a combined heat map and threshold is used to differentiate the two.
This is done by using the add_heat() & apply_threshold() functions in the pipeline where I iterate through all the rectangle boxes and add a +1 for all the pixels inside each box. This way the areas covered by more rectangles are assigned more heat and hence differentiating a false positive detections. Then the apply_threshold() function is included to reject any such false detections below a threshold and boxes are drawn against the most confident detections.

The image below shows the final boxes drawn on the left with the corresponding heat map.
![heat map](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/heatmap.png)

## RESULTS FOR TEST IMAGES:
![Test images](https://raw.github.com/aranga81/Vehicle-Detection-and-Tracking/master/output_images/test_images_output.png)

## test video and project video output included in the repository....!!!

### Next steps:
The problems in this project are mainly choosing the parameters for HOG feature extractions and also choosing the window sizes and best combinations for sliding window search. I would surely work on developing higher accuracy classifier and also optimize the window search to better detect and track the cars.
Pipeline would fail probably due to unfiltered shadowing or cars or other vehicles that the classifier will not be able to identify.
Also opposite end traffic is sometimes an issue and it will be more robust if the window search is limited to more specific zones.
Determine vehicle motion – develop a motion model and predict its location in next frame using kalman filtering technique for better tracking.
More better window searching technique – using convolutions or optimized window sizes & overlaps.












