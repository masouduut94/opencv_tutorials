import cv2
import matplotlib.pyplot as plt


def apply_invert(image, display=True):
    """
    This function will create the Invert filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Invert filter applied.
    """

    # Apply the Invert Filter on the image. 
    output_image = cv2.bitwise_not(image)

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
    # Read a sample image and apply invert filter on it.
    image = cv2.imread('media/sample16.jpg')
    apply_invert(image)
