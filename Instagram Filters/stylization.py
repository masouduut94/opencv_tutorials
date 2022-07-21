import cv2

from assets import AssetMeta


def apply_stylization(image):
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

    return output_image


if __name__ == '__main__':
    meta = AssetMeta()
    img = cv2.imread(meta.IMG_NORMAL[0])
    img = cv2.resize(img, (960, 540))
    new_img = apply_stylization(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

