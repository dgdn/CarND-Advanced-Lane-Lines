## **Advanced Lane Finding Project**

The goals / steps of this project are the following:

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
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

The code for this step is contained in lines #10 of the file called `pipeline.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

### Pipeline (single images)

#### 1. Apply distortion correction

I applied distortion correction to the image using the `cv2.undistort()` function which will take in the camera matrix and distortion coeffcients. Here are the two images that demonstrate the distortion correction effect, the left is the origin image and the right is the undistort image.

![alt text][image1] ![alt text][image2]

#### 2. Use color transforms, gradients to create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image. For color transforms, I transform the color mode to HLS, which may result in robost extracting line, especially the light is dark. Only with color thresold is not enough,we should also apply edge dection using Sobel. I applied Sobel on x direction, because this is suit for detect vertical edges .The code for this transform includes a function called `threshold()`, which appears in lines 43 through 80 in the file `pipline.py`. Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective transform

In order to calcalate the curature of the lane, we need to apply perspective transform to the lane, which will give us a bird eye view of the lane. At first, we need to figure out the transform matrix which will be used to transform other images. We use a image with straight lane, this is easy for us to fidn out of source and destionation points. Using the points, we can use `cv2.getPerspectiveTransform(src,dst)` to get the perspective transform and the inver perspective tranform matrix. The invers perpective transform is used for peject the eye bird lane to the origin image.The code for my perspective transform includes a function called `warpe()`, which inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

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

Given the warped lane image, we should take approch to detect the left and right lane pixel. Overall, we use a window search method to detect the lane pixel from bottom to top. First we need to locate the start position of left and right lane. We compute the historm of the image in the half bottom part. We don't compute historm for entire image because we know that lane start position is located in the bottom and we want the upper part of the pixel to affect the position. We choose the position with biggest hisgram value as the start position of lane. Known the start position, we start to use a square window to search the lane pixel from bottom to top. If the number of pixels locate in the window is greater than some thresold, we adjust the position to be the average location of the pixel. In this searching process, we gradually collect all the pixels for the left and right lanes. Finally we fit the lane pixels with 2nd order polynomial kinda. The code for this process is handled in the function called `search_lane()`, which appears in line 83 through 156.

Here is image demonstrate the detected lane pixels and the fitted line:

![alt text][image5]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center

Now we have abtain the line of lane in the pixel world. In order to calculate the radius curvature in world space, we should first figure out the relation between this two space based on the data measured in real road. Prevously, we get the fit line in pixel space, we can not simply scale appropriately to get the world space fit line. Instaed, we need to fit the world space poinits to get the line. We then use this two line to calculate the curveture using the known curve formula. I implemented did this in a function called `calculateCurvature()`, which appears in lines 208 through 224.

#### 6. Warp the detected lane boundaries back onto the original image

Using the inverse perjected matrix, we can easily warp the lane boundaries back onto the original image, such that the lane area is identified clearly. I implemented this step in lines 190 through 206 in my code in `pipeline.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

We can easily process video stream using the `pipline()` whithout adding more fancy stuff. Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you  to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The pipeline will not in some tricky circunstances. When beside the lane, there is another lane is much the same with the real lane, the pipeline may pick the wrong line as the lane, this is because our althoitm pick the biggest higroam value as the start position of the lane. Arbitary line that share the charistric with the real lane will be picked as the real lane.
To deal with this, we should not simply selecte only lane with the largest hisgram, if encounter two or more candidate lines as lane, we should use other method to select the best lane such as based on their color or the symetry to the other side lane.