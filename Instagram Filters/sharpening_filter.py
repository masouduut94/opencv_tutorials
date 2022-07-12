import cv2
import numpy as np


def apply_sharpening(image):
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

    return output_image


if __name__ == '__main__':
    img = cv2.imread("../assets/IMAGES/woman-boat.jpg")
    img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    new_img = apply_sharpening(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
