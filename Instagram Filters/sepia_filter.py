# Sepia Filter-like Effect
import cv2
import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def apply_sepia(image, display=True):
    """
    This function will create instagram Sepia filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Sepia filter applied.
    """

    # Convert the image into float type to prevent loss during operations.
    image_float = np.array(image, dtype=np.float64)

    # Manually transform the image to get the idea of exactly whats happening.
    ##################################################################################################

    # Split the blue, green, and red channel of the image.
    blue_channel, green_channel, red_channel = cv2.split(image_float)

    # Apply the Sepia filter by perform the matrix multiplication between
    # the image and the sepia matrix.
    output_blue = (red_channel * .272) + (green_channel * .534) + (blue_channel * .131)
    output_green = (red_channel * .349) + (green_channel * .686) + (blue_channel * .168)
    output_red = (red_channel * .393) + (green_channel * .769) + (blue_channel * .189)

    # Merge the blue, green, and red channel.
    output_image = cv2.merge((output_blue, output_green, output_red))

    ##################################################################################################

    # OR Either create this effect by using OpenCV matrix transformation function.
    ##################################################################################################

    # Get the sepia matrix for BGR colorspace images.
    sepia_matrix = np.matrix([[.272, .534, .131],
                              [.349, .686, .168],
                              [.393, .769, .189]])

    # Apply the Sepia filter by perform the matrix multiplication between
    # the image and the sepia matrix.
    # output_image = cv2.transform(src=image_float, m=sepia_matrix)

    ##################################################################################################

    # Set the values > 255 to 255.
    output_image[output_image > 255] = 255

    # Convert the image back to uint8 type.
    output_image = np.array(output_image, dtype=np.uint8)

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Input Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
    # Otherwise.
    else:

        # Return the output image.
        return output_image




