import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Warm Filter
# Construct a lookuptable for increasing pixel values.
# We are giving y values for a set of x values.
# And calculating y for [0-255] x values accordingly to the given range.
increase_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 155, 255])(range(256))

# Similarly construct a lookuptable for decreasing pixel values.
decrease_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 45, 95, 255])(range(256))

# Display the first 10 mappings from the constructed tables.
print(f'First 10 elements from the increase table: \n {increase_table[:10]}\n')
print(f'First 10 elements from the decrease table:: \n {decrease_table[:10]}')


# #######################  Warm Filter


def apply_warm(image, display=False):
    """
    This function will create instagram Warm filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Warm filter applied.
    """

    # Split the blue, green, and red channel of the image.
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Increase red channel intensity using the constructed lookuptable.
    red_channel = cv2.LUT(red_channel, increase_table).astype(np.uint8)

    # Decrease blue channel intensity using the constructed lookuptable.
    blue_channel = cv2.LUT(blue_channel, decrease_table).astype(np.uint8)

    # Merge the blue, green, and red channel.
    output_image = cv2.merge((blue_channel, green_channel, red_channel))

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


