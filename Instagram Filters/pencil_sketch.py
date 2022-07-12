import cv2


def apply_pencil_sketch(image):
    """
    This function will create instagram Pencil Sketch filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
    Returns:
        output_image: A copy of the input image with the Pencil Sketch filter applied. 
    """

    # Apply Pencil Sketch effect on the image.
    gray_sketch, color_sketch = cv2.pencilSketch(image, sigma_s=20, sigma_r=0.5, shade_factor=0.02)

    return color_sketch


if __name__ == '__main__':
    img = cv2.imread("../assets/IMAGES/woman-boat.jpg")
    img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    new_img = apply_pencil_sketch(img.copy())
    cv2.imshow("result", new_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
