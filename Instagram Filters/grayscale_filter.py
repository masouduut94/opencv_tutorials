import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_grayscale(image, display=True):
    """
    This function will create instagram Grayscale filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Grayscale filter applied.
    """

    # Convert the image into the grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Merge the grayscale (one-channel) image three times to make it a three-channel image.
    output_image = cv2.merge((gray, gray, gray))

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
    # Read a sample image| and apply Grayscale filter on it.
    image = cv2.imread('media/sample7.jpg')
    apply_grayscale(image)


