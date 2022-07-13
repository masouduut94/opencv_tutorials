import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 Reading image, and checking its dimensions
img = cv2.imread('../assets/IMAGES/woman-boat.jpg')
h, w, _ = img.shape
print(f'Width X Height:')
print(f'{w} X {h}')

# cv2.imshow("original img", img)
# k = cv2.waitKey(0)  # 1, 2, 14

# 2. Resizing image with custom size
new_img1 = cv2.resize(img, (2000, 1500))
h, w, _ = new_img1.shape

print(f'Width X Height:')
print(f'{w} X {h}')

# cv2.imshow("new_img1", img)
# k = cv2.waitKey(0)  # 1, 2, 14

# 2. Resizing image with fx and fy.
new_img2 = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
h, w, _ = new_img2.shape
print(f'Width X Height:')
print(f'{w} X {h}')

# cv2.imshow("new_img2", new_img2)
# k = cv2.waitKey(0)  # 1, 2, 14

# 3. Convert image color space
"""
YUV: YUV color space was particularly useful when color was first 
     introduced to television as it could decode and display both 
     color and black and white.
     
BGR: Default Opencv image color space. cv2.imshow default color projection
     is set to BGR. (matplotlib.pyplot.imshow defaults on RGB!)

HSV: Abbreviated for Hue-Saturation-Value, We can use this format to 
     Increase/Decrease intensity to the color space easily. 
"""
codes = {
    'grayscale': cv2.COLOR_BGR2GRAY,
    'rgb': cv2.COLOR_BGR2RGB,
    'hsv': cv2.COLOR_BGR2HSV,
    'yuv': cv2.COLOR_BGR2YUV

}

# GrayScale
grayscale = cv2.cvtColor(new_img2, codes['grayscale'])

# cv2.imshow("grayscale", grayscale)
# k = cv2.waitKey(0)  # 1, 2, 14

# RGB ColorSpace
rgb = cv2.cvtColor(new_img2, codes['rgb'])

# 4. Numpy Operations (This method replaces the R channel with B channel)
new_color2 = new_img2[..., ::-1]

# cv2.imshow("numpy Operation color", new_color2)
# k = cv2.waitKey(0)  # 1, 2, 14

# 5. Splitting and Merging channels (Replacing R and B channels)

R, G, B = cv2.split(new_img2)
merged = cv2.merge([B, G, R])

cv2.imshow("Original Image", new_img2)
cv2.imshow("Merged Channels", merged)
k = cv2.waitKey(0)

# 6. writing images to a destination
cv2.imwrite("../assets/OUTPUT/IMAGES/gray.png", grayscale)
