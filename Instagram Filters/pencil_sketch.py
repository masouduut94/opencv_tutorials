import cv2
import matplotlib.pyplot as plt


def apply_pencil_sketch(image, display=True):
    """
    This function will create instagram Pencil Sketch filter like effect on an image.
    Args:
        image:  The image on which the filter is to be applied.
        display: A boolean value that is if set to true the function displays the original image,
                 and the output image, and returns nothing.
    Returns:
        output_image: A copy of the input image with the Pencil Sketch filter applied. 
    """

    # Apply Pencil Sketch effect on the image.
    gray_sketch, color_sketch = cv2.pencilSketch(image, sigma_s=20, sigma_r=0.5, shade_factor=0.02)

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(131)
        plt.imshow(image[:, :, ::-1])
        plt.title("Input Image")
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(color_sketch[:, :, ::-1])
        plt.title("ColorSketch Image")
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(gray_sketch, cmap='gray')
        plt.title("GraySketch Image")
        plt.axis('off')

    # Otherwise.
    else:

        # Return the output image.
        return color_sketch
