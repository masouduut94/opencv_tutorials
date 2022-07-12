
import cv2
import matplotlib.pyplot as plt


def apply_detail_enhancing(image, display=True):
    """
    This function will create the HDR filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the HDR filter applied. 
    """

    # Apply the detail enhancing effect by enhancing the details of the image.
    output_image = cv2.detailEnhance(image, sigma_s=15, sigma_r=0.15)

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
    # Read another sample image and apply Detail Enhancing filter on it.
    image = cv2.imread('media/sample15.jpg')
    apply_detail_enhancing(image)

