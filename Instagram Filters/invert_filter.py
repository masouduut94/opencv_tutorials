import cv2

from assets import AssetMeta


def apply_invert(image):
    """
    This function will create the Invert filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
    Returns:
        output_image: A copy of the input image with the Invert filter applied.
    """

    # Apply the Invert Filter on the image. 
    output_image = cv2.bitwise_not(image)
    return output_image


if __name__ == '__main__':
    # Read a sample image and apply invert filter on it.
    meta = AssetMeta()
    img = cv2.imread(meta.IMG_NORMAL[0])
    img = cv2.resize(img, (960, 540))
    new_img = apply_invert(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
