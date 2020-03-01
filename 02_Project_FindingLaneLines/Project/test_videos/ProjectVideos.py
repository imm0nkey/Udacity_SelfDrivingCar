import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Grayscale the image
def covert_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray


# Define a kernel size and apply Gaussian smoothing
def covert_gaussian(image):
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blur_gray


# Define our parameters for Canny and apply
def covert_edges(image):
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


# Next we'll create a masked edges image using cv2.fillPoly(), this time we are defining a four sided polygon to mask
def covert_masked(image):
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    imshape = image.shape
    # vertices = np.array([[(0, 650), (640, 420), (680, 420), (imshape[1], 650)]], dtype=np.int32)
    # vertices = np.array([[(0, imshape[0]), (480, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    vertices = np.array([[(0, imshape[0]), (800, 800), (1120, 800), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(image, mask)
    return mask, masked_edges


# Define the Hough transform parameters and make a blank the same size as our image to draw on
# Run Hough on edge detected image
def covert_hough(image, original_image):
    rho = 2
    theta = np.pi/180
    threshold = 10
    min_line_length = 40
    max_line_gap = 20
    line_image = np.copy(original_image)*0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    if lines is None:
        return False
    else:
        for line in lines:  # Iterate over the output "lines" and draw lines on the black
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        combo = cv2.addWeighted(original_image, 0.8, line_image, 1, 0)  # Draw the lines on the edge image
        return combo


# Read in
videoCapture = cv2.VideoCapture("20191121test_3.MOV")
frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print(range(int(frames)))
fps = 25
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_writer = cv2.VideoWriter(filename='20191121test_3/result.avi', fourcc=fourcc, fps=fps, frameSize=(1920, 1080))
for i in range(int(frames)):
    ret, frame = videoCapture.read()
    if frame is None:
        break
    else:
        # b, g, r = cv2.split(frame)
        # frame_new = cv2.merge([r, g, b])
        image_gray = covert_gray(frame)
        # plt.imshow(frame)
        # plt.show()
        image_blur_gray = covert_gaussian(image_gray)
        image_edges = covert_edges(image_blur_gray)
        area_masked, image_masked = covert_masked(image_edges)
        # plt.imshow(image_masked)
        # plt.show()
        mask_3 = np.dstack((area_masked, area_masked, area_masked))
        combo_image_mask = cv2.addWeighted(frame, 0.8, mask_3, 0.2, 0)
        # plt.imshow(combo_image_mask)
        # plt.show()
        image_hough = covert_hough(image_masked, frame)
        if image_hough is False:
            cv2.imwrite("20191121test_3/%d.jpg" % i, frame)
            cv2.waitKey(100)
            video_writer.write(frame)
            print('***' + str(i) + '/' + str(int(frames)))
        else:
            r, g, b = cv2.split(image_hough)
            image_hough_new = cv2.merge([b, g, r])
            # plt.imshow(image_hough)
            # plt.show()
            cv2.imwrite("20191121test_3/%d.jpg" % i, image_hough_new)
            cv2.waitKey(100)
            video_writer.write(image_hough_new)
            print(str(i) + '/' + str(int(frames)))
video_writer.release()
videoCapture.release()
