import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from cv2 import cv2

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape)==3:
        channel_count = img.shape[2]
        match_mask_color = (255,) * channel_count
    else:
        match_mask_color = 255 
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_image = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
    # Return the modified image.
    return img


image = mpimg.imread('snapshot/b8f3506c-3f65-11eb-9798-16f63a1aa8c9.jpg')
height, width, channel = image.shape
region_of_interest_vertices = [
    (0, height),
    (width / 2, (height / 2) - 30),
    (width, height),
]

cropped_image = region_of_interest(image,np.array([region_of_interest_vertices], np.int32),)

gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 50, 200)

cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32)
)

lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)

line_image = draw_lines(image, lines)

plt.figure()
plt.imshow(line_image)
plt.show()