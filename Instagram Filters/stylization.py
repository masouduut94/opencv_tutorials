import cv2
import matplotlib.pyplot as plt


def apply_stylization(image, display=False):
    """
    This function will create instagram cartoon-paint filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the cartoon-paint filter applied.
    """

    # Apply stylization effect on the image.
    output_image = cv2.stylization(image, sigma_s=15, sigma_r=0.55)

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
    # Read a sample image and apply Stylization filter on it.
    image = cv2.imread('media/sample16.jpg')
    apply_stylization(image)

