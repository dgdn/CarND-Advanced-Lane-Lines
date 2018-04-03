import cv2
import pickle
import sys
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

def calibrateCamera(nx, ny, img_path, img_size):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:ny,0:nx].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space.
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(img_path+'/*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        print('processing calibrate image: ' + fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (ny,nx), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    #calibrate camera to get camera matrix and distortion coefficients
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def threshold(img, thresh_min, thresh_max, s_thresh_min, s_thresh_max):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # only for debug use
    if False:
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 3, figsize=(20,10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap="gray")

    return combined_binary

def warp(combined_binary, M):
    return cv2.warpPerspective(combined_binary, M, combined_binary.shape[::-1])

def search_lane(binary_warped):

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, leftx, lefty, rightx, righty

def fast_search_lane(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def map_lane(left_fitx, right_fitx, ploty, Minv, binary_warped, img):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

def calculateCurvature(left_fitx, right_fitx, ploty):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


def findPerspectiveM(img_size):
    # Important!! size is (w, h)  img_size is (h, w)
    size = img_size[::-1]
    src = np.float32(
        [[(size[0] / 2) - 60, size[1] / 2 + 100],
        [((size[0] / 6) - 10), size[1]],
        [(size[0] * 5 / 6) + 60, size[1]],
        [(size[0] / 2 + 55), size[1] / 2 + 100]])
    dst = np.float32(
        [[(size[0] / 4), 0],
        [(size[0] / 4), size[1]],
        [(size[0] * 3 / 4), size[1]],
        [(size[0] * 3 / 4), 0]])

    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src) 

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

def pipeline(img):

    global pipeline_args
    mtx = pipeline_args["mtx"]
    dist = pipeline_args["dist"]
    M = pipeline_args["M"]
    Minv = pipeline_args["Minv"]

    undistorted = undistort(img, mtx, dist)

    thresh_min = 18
    thresh_max = 100
    s_thresh_min = 170
    s_thresh_max = 255
    thresholded = threshold(undistorted, thresh_min, thresh_max, s_thresh_min, s_thresh_max)

    binary_warped = warp(thresholded, M)

    left_fitx, right_fitx, ploty = search_lane(binary_warped)

    left_curvature, right_curvature = calculateCurvature(left_fitx, right_fitx, ploty)

    projected =  map_lane(left_fitx, right_fitx, ploty, Minv, binary_warped, undistorted)

    tx = 100
    ty = 100
    text = "Left Curvature:{.2f}m Right Curvature:{.2f}".format(left_curvature, right_curvature)
    final = cv2.putText(projected, text, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    return final

pipeline_args = dict()

def main():
    img_size = (720, 1280)

    if os.path.isfile('calibrate.pickle'):
        data = pickle.load(open('calibrate.pickle', 'rb'))
        mtx = data['mtx']
        dist = data['dist']
    else:
        mtx, dist = calibrateCamera(6, 9, 'camera_cal', img_size)
        pickle.dump({"mtx": mtx, "dist": dist}, open('calibrate.pickle', 'wb'))

    M, Minv = findPerspectiveM(img_size)

    pipeline_args["mtx"] = mtx
    pipeline_args["dist"] = dist
    pipeline_args["M"] = M
    pipeline_args["Minv"] = Minv

    easy_output = 'project_video_output.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    easy_clip = clip1.fl_image(pipeline)
    easy_clip.write_videofile(easy_output, audio=False)

def test():
    img_size = (720, 1280)
    if os.path.isfile('calibrate.pickle'):
        data = pickle.load(open('calibrate.pickle', 'rb'))
        mtx = data['mtx']
        dist = data['dist']
    else:
        mtx, dist = calibrateCamera(6, 9, 'camera_cal', img_size)
        pickle.dump({"mtx": mtx, "dist": dist}, open('calibrate.pickle', 'wb'))

    M, Minv = findPerspectiveM(img_size)
    pipeline_args["mtx"] = mtx
    pipeline_args["dist"] = dist
    pipeline_args["M"] = M
    pipeline_args["Minv"] = Minv

    img = cv2.imread('test_images/test3.jpg') 
    result = pipeline(img)
    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()