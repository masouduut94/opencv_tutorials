# Sepia Filter-like Effect
import cv2
import numpy as np

from assets import AssetMeta


def apply_sepia(image):
    """
    This function will create instagram Sepia filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
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

    return output_image


if __name__ == '__main__':
    meta = AssetMeta()
    img = cv2.imread(meta.IMG_NORMAL[0])
    img = cv2.resize(img, (960, 540))
    new_img = apply_sepia(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

