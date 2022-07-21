import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

from assets import AssetMeta


def apply_gotham(image):
    """
    This function will create instagram Gotham filter like effect on an image.
    Args:
        image:   The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Gotham filter applied.
    """

    # Split the blue, green, and red channel of the image.
    midtone_contrast_increase = UnivariateSpline(x=[0, 25, 51, 76, 102, 128, 153, 178, 204, 229, 255],
                                                 y=[0, 13, 25, 51, 76, 128, 178, 204, 229, 242, 255])(range(256))

    # Construct a lookuptable for increasing lowermid pixel values.
    lowermids_increase = UnivariateSpline(
        x=[0, 16, 32, 48, 64, 80, 96, 111, 128, 143, 159, 175, 191, 207, 223, 239, 255],
        y=[0, 18, 35, 64, 81, 99, 107, 112, 121, 143, 159, 175, 191, 207, 223, 239, 255])(
        range(256))

    # Construct a lookuptable for decreasing uppermid pixel values.
    uppermids_decrease = UnivariateSpline(
        x=[0, 16, 32, 48, 64, 80, 96, 111, 128, 143, 159, 175, 191, 207, 223, 239, 255],
        y=[0, 16, 32, 48, 64, 80, 96, 111, 128, 140, 148, 160, 171, 187, 216, 236, 255])(
        range(256))

    blue_channel, green_channel, red_channel = cv2.split(image)

    # Boost the mid-tone red channel contrast using the constructed lookuptable.
    red_channel = cv2.LUT(red_channel, midtone_contrast_increase).astype(np.uint8)

    # Boost the Blue channel in lower-mids using the constructed lookuptable.
    blue_channel = cv2.LUT(blue_channel, lowermids_increase).astype(np.uint8)

    # Decrease the Blue channel in upper-mids using the constructed lookuptable.
    blue_channel = cv2.LUT(blue_channel, uppermids_decrease).astype(np.uint8)

    # Merge the blue, green, and red channel.
    output_image = cv2.merge((blue_channel, green_channel, red_channel))

    return output_image


if __name__ == '__main__':
    meta = AssetMeta()
    img = cv2.imread(meta.IMG_NORMAL[0])
    img = cv2.resize(img, (960, 540))
    new_img = apply_gotham(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
