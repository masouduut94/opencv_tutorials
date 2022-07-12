import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_sharpening(image, display=True):
    """
    This function will create the Sharpening filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Sharpening filter applied. 
    """

    # Get the kernel required for the sharpening effect.
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9.2, -1],
                                  [-1, -1, -1]])

    # Apply the sharpening filter on the image.
    output_image = cv2.filter2D(src=image, ddepth=-1,
                                kernel=sharpening_kernel)

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


if __name__ == '__main__':
    image = cv2.imread('media/sample13.jpg')
    apply_sharpening(image)
