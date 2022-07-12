
import cv2


def apply_detail_enhancing(image):
    """
    This function will create the HDR filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
    Returns:
        output_image: A copy of the input image with the HDR filter applied. 
    """

    # Apply the detail enhancing effect by enhancing the details of the image.
    output_image = cv2.detailEnhance(image, sigma_s=15, sigma_r=0.15)

    # Check if the original input image and the output image are specified to be displayed.
    return output_image


if __name__ == '__main__':
    img = cv2.imread("../assets/IMAGES/woman-boat.jpg")
    img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    new_img = apply_detail_enhancing(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
