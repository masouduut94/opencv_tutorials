import cv2


def apply_grayscale(image):
    """
    This function will create instagram Grayscale filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Grayscale filter applied.
    """

    # Convert the image into the grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Merge the grayscale (one-channel) image three times to make it a three-channel image.
    output_image = cv2.merge((gray, gray, gray))

    # Check if the original input image and the output image are specified to be displayed.
    return output_image


if __name__ == '__main__':
    img = cv2.imread("../assets/IMAGES/woman-boat.jpg")
    img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    new_img = apply_grayscale(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


