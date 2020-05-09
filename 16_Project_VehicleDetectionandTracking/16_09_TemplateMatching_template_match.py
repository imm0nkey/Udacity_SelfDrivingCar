import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')
# image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['coutout1.jpg', 'coutout2.jpg', 'coutout3.jpg',
            'coutout4.jpg', 'coutout5.jpg', 'coutout6.jpg']

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    bbox_list = []
    imcopy = np.copy(img)
    for temp_name in template_list:
        # Read in templates one by one
        temp = cv2.imread(temp_name)
        # Use cv2.matchTemplate() to
        result = cv2.matchTemplate(imcopy, temp, method=cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        top_left = maxLoc
        w, h = temp.shape[1],temp.shape[0]
        bottom_right = (maxLoc[0]+w,maxLoc[1]+h)

        bbox_list.append(top_left, bottom_right)
    # Iterate through template list
    # Read in templates one by one
    # Use cv2.matchTemplate() to search the image
    #     using whichever of the OpenCV search methods you prefer
    # Use cv2.minMaxLoc() to extract the location of the best match
    # Determine bounding box corners for the match
    # Return the list of bounding boxes
    return bbox_list


bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()
