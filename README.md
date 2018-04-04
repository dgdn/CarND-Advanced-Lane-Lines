## **Advanced Lane Finding Project**

In this project, my goal is to write a software pipeline to identify the lane boundaries in a video

---

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image20]: ./test_images/test3.jpg "Road Transformed"
[image21]: ./output_images/undistort.png "Undistorted"
[image3]: ./output_images/threshold.jpg "Binary Example"
[image4]: ./output_images/warp.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

The code for this step is contained in lines 10 of the file called `pipeline.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

![alt text][image1]

### Pipeline (single images)

#### 1. Apply distortion correction

I applied distortion correction to the image using the `cv2.undistort()` function which takes in the camera matrix and distortion coeffcients. Here are the two images that demonstrate the distortion correction effect, the left is the original image and the right is the undistorted image.

![alt text][image20] ![alt text][image21]

#### 2. Use color transforms, gradients to create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image. I converted the image to HLS space and extracted the S channel as our thresholded target, this resulted in robust edge detection in varying degrees of daylight and shadow. Only with color threshold it is not enough, we also need to apply Sobel in x orientation which is suitable for detect vertical edges like lanes. Combining his these two threshold transform, we can get a thresholded binary image with lanes on it. The code for this transform includes a function called `threshold()`, which appears in lines 43 through 80 in the file `pipline.py`. Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective transform

In order to calcalate the radius of curvature of the lane, we need to apply perspective transform to the lane, which will give us a bird eye view of the lane. To get perspective transform matrix, we need to provide 4 points located in the original image and 4 corresponding points in the destination image. We chose the image that contains straight line as our original image, this is easy for us to pick source and destination points accurately. After selecting these points, we can use `cv2.getPerspectiveTransform(src, dst)` to get the perspective transform `M` and the inverse perspective tranform matrix `Minv`. The inverse perpective transform is used for warping the the eye bird lanes back onto the original image. The code for my perspective transform includes a function called `warpe()`, which inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Detect lane pixels and fit

Given the warped lane image, we should take approaches to detect the pixels of the left and right lane. Overall, we use the window search method to detect the lane pixel from bottom to top. First we need to locate the start position of both lanes. I first take a histogram along all columns in the lower half of the image. I did not include the upper half because we know that lane start position is located in the bottom and we do not want the upper halft part of the pixel affect the position. We choose the peak of histogram value as the start position of lane. After knowing the start position, we started to use a window to search the lane pixel from bottom to top. If the number of pixels located in the window is greater than some threshold, we adjust the horizontal position of the window to be the average horizontal position of the all pixels within current window. In this searching process, we gradually collected all the pixels for the left and right lanes. Finally we fit the lane pixels with 2nd order polynomial kinda. The code for this process is handled in the function called `search_lane()`, which appears in line 83 through 156.

Here is image demonstrate the detected lane pixels and the fitted line:

![alt text][image5]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center

Now we have obtained the line of lane in the pixel world. In order to calculate the radius of curvature in real world space, we should first figure out the conversion between these two space based on datas measured on real road. Using the conversion between meter and pixel, we transformed the lane points in pixel space to world space. Then we fitted the world space poinits to get the line measured in meter. Finally We used these two lines to calculate the radius of curvature using the known curve formula. I implemented did this in a function called `calculate_curvature()`, which appears in lines 208 through 224. In the below image, we can see that the radius of curvature is `1136.6m` and the vehicle is `0.09m` left of center.

#### 6. Warp the detected lane boundaries back onto the original image

Using the inverse projective matrix `Minv`, we can easily warp the lane boundaries back onto the original image, such that the lane area is identified clearly. I implemented this step in lines 190 through 206 in my code in `pipeline.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

We can easily process video stream using the `pipline()` whithout adding more fancy stuff. Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

Here I want to discussion some potential problems. One of problem is that the pipeline will not work in the tricky circumstance where mulptiple vertical lines appeared in the ROI. The pipeline may pick the wrong line as the lane, this is because our pipeline always pick the one that has peak histogram value as the start position of the lane. To deal with this, we may introduce more advanced algorithm to identify the true lane among multiple candidates. Maybe we could consider the color of the lines or the distance symetry of the lanes as ways to filter out the interfered lines.