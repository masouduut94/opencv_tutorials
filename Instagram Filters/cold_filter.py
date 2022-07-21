import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from assets import AssetMeta


def apply_cold(image):
    """
    This function will create instagram Cold filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Cold filter applied.
    """
    increase_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 155, 255])(range(256))

    # Similarly construct a lookuptable for decreasing pixel values.
    decrease_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 45, 95, 255])(range(256))

    # Split the blue, green, and red channel of the image.
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Decrease red channel intensity using the constructed lookup_table.
    red_channel = cv2.LUT(red_channel, decrease_table).astype(np.uint8)

    # Increase blue channel intensity using the constructed lookup_table.
    blue_channel = cv2.LUT(blue_channel, increase_table).astype(np.uint8)

    # Merge the blue, green, and red channel.
    output_image = cv2.merge((blue_channel, green_channel, red_channel))

    return output_image


if __name__ == '__main__':
    meta = AssetMeta()
    img = cv2.imread(meta.IMG_NORMAL[0])
    img = cv2.resize(img, (960, 540))
    new_img = apply_cold(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
