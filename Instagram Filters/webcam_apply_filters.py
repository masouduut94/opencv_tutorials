"""
Source: https://bleedai.com/designing-advanced-image-filters-in-opencv-creating-instagram-filters-pt-3%E2%81%843/#pt1

"""

import cv2
import numpy as np
import pygame
import matplotlib.pyplot as plt
from cold_filter import apply_cold
from warm_filter import apply_warm
from sepia_filter import apply_sepia
from gotham_filter import apply_gotham
from invert_filter import apply_invert
from stylization import apply_stylization
from grayscale_filter import apply_grayscale
from pencil_sketch import apply_pencil_sketch
from sharpening_filter import apply_sharpening
from detail_enhancing import apply_detail_enhancing


def apply_selected_filter(image, filter_applied):
    """
    This function will apply the selected filter on an image.
    Args:
        image:          The image on which the selected filter is to be applied.
        filter_applied: The name of the filter selected by the user.
    Returns:
        output_image: A copy of the input image with the selected filter applied.
    """

    # Check if the specified filter to apply, is the Warm filter.
    if filter_applied == 'Warm':

        # Apply the Warm Filter on the image.
        output_image = apply_warm(image)

    # Check if the specified filter to apply, is the Cold filter.
    elif filter_applied == 'Cold':

        # Apply the Cold Filter on the image.
        output_image = apply_cold(image)

    # Check if the specified filter to apply, is the Gotham filter.
    elif filter_applied == 'Gotham':

        # Apply the Gotham Filter on the image.
        output_image = apply_gotham(image)

    # Check if the specified filter to apply, is the Grayscale filter.
    elif filter_applied == 'Grayscale':

        # Apply the Grayscale Filter on the image.
        output_image = apply_grayscale(image)

        # Check if the specified filter to apply, is the Sepia filter.
    if filter_applied == 'Sepia':

        # Apply the Sepia Filter on the image.
        output_image = apply_sepia(image)

    # Check if the specified filter to apply, is the Pencil Sketch filter.
    elif filter_applied == 'Pencil Sketch':

        # Apply the Pencil Sketch Filter on the image.
        output_image = apply_pencil_sketch(image)

    # Check if the specified filter to apply, is the Sharpening filter.
    elif filter_applied == 'Sharpening':

        # Apply the Sharpening Filter on the image.
        output_image = apply_sharpening(image)

    # Check if the specified filter to apply, is the Invert filter.
    elif filter_applied == 'Invert':

        # Apply the Invert Filter on the image.
        output_image = apply_invert(image)

    # Check if the specified filter to apply, is the Detail Enhancing filter.
    elif filter_applied == 'Detail Enhancing':

        # Apply the Detail Enhancing Filter on the image.
        output_image = apply_detail_enhancing(image)

    # Check if the specified filter to apply, is the Stylization filter.
    elif filter_applied == 'Stylization':

        # Apply the Stylization Filter on the image.
        output_image = apply_stylization(image)

    # Return the image with the selected filter applied.`
    return output_image


def mouse_callback(event, x, y, flags, userdata):
    """
    This function will update the filter to apply on the frame and capture images based on different mouse events.
    Args:
        event:    The mouse event that is captured.
        x:        The x-coordinate of the mouse pointer position on the window.
        y:        The y-coordinate of the mouse pointer position on the window.
        flags:    It is one of the MouseEventFlags constants.
        userdata: The parameter passed from the `cv2.setMouseCallback()` function.
    """
    #  Access the filter applied, and capture image state variable.
    global filter_applied, capture_image

    # Check if the left mouse button is pressed.
    if event == cv2.EVENT_LBUTTONDOWN:

        # Check if the mouse pointer is over the camera icon ROI.
        if y >= (frame_height - 10) - camera_icon_height and \
                (frame_width // 2 - camera_icon_width // 2) <= x <= (frame_width // 2 + camera_icon_width // 2):

            # Update the image capture state to True.
            capture_image = True

        # Check if the mouse pointer y-coordinate is over the filters ROI.
        elif y <= 10 + preview_height:

            # Check if the mouse pointer x-coordinate is over the Warm filter ROI.
            if (int(frame_width // 11.6) - preview_width // 2) < x <= (
                    int(frame_width // 11.6) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Warm.
                filter_applied = 'Warm'

            # Check if the mouse pointer x-coordinate is over the Cold filter ROI.
            elif (int(frame_width // 5.9) - preview_width // 2) < x < (
                    int(frame_width // 5.9) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Cold.
                filter_applied = 'Cold'

            # Check if the mouse pointer x-coordinate is over the Gotham filter ROI.
            elif (int(frame_width // 3.97) - preview_width // 2) < x < (
                    int(frame_width // 3.97) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Gotham.
                filter_applied = 'Gotham'

            # Check if the mouse pointer x-coordinate is over the Grayscale filter ROI.
            elif (int(frame_width // 2.99) - preview_width // 2) < x <= (
                    int(frame_width // 2.99) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Grayscale.
                filter_applied = 'Grayscale'

            # Check if the mouse pointer x-coordinate is over the Sepia filter ROI.
            elif (int(frame_width // 2.395) - preview_width // 2) < x < (
                    int(frame_width // 2.395) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Sepia.
                filter_applied = 'Sepia'

            # Check if the mouse pointer x-coordinate is over the Normal filter ROI.
            elif (int(frame_width // 2) - preview_width // 2) < x < (
                    int(frame_width // 2) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Normal.
                filter_applied = 'Normal'

            # Check if the mouse pointer x-coordinate is over the Pencil Sketch filter ROI.
            elif (frame_width // 1.715 - preview_width // 2) < x <= (
                    frame_width // 1.715 - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Pencil Sketch.
                filter_applied = 'Pencil Sketch'

            # Check if the mouse pointer x-coordinate is over the Sharpening filter ROI.
            elif (int(frame_width // 1.501) - preview_width // 2) < x < (
                    int(frame_width // 1.501) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Sharpening.
                filter_applied = 'Sharpening'

            # Check if the mouse pointer x-coordinate is over the Invert filter ROI.
            elif (int(frame_width // 1.335) - preview_width // 2) < x < (
                    int(frame_width // 1.335) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Invert.
                filter_applied = 'Invert'

            # Check if the mouse pointer x-coordinate is over the Detail Enhancing filter ROI.
            elif (int(frame_width // 1.202) - preview_width // 2) < x < (
                    int(frame_width // 1.202) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Detail Enhancing.
                filter_applied = 'Detail Enhancing'

            # Check if the mouse pointer x-coordinate is over the Stylization filter ROI.
            elif (int(frame_width // 1.094) - preview_width // 2) < x < (
                    int(frame_width // 1.094) - preview_width // 2) + preview_width:

                # Update the filter applied variable value to Stylization.
                filter_applied = 'Stylization'


# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Create a named resizable window.
cv2.namedWindow('Instagram Filters', cv2.WINDOW_NORMAL)

# Attach the mouse callback function to the window.
cv2.setMouseCallback('Instagram Filters', mouse_callback)

# Initialize a variable to store the current applied filter.
filter_applied = 'Normal'

# Initialize a variable to store the copies of the frame
# with the filters applied.
filters = None

# Initialize the pygame modules and load the image-capture music file.
pygame.init()
pygame.mixer.music.load("C:/Users/masoud/PycharmProjects/opencv_tutorials/assets/SOUNDS/mixkit-arcade-game-jump-coin-216.wav")

# Initialize a variable to store the image capture state.
capture_image = False

# Initialize a variable to store a camera icon image.
camera_icon = None

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly then
    # continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Check if the filters variable doesnot contain the filters.
    if not filters:
        # Update the filters variable to store a dictionary containing multiple
        # copies of the frame with all the filters applied.
        filters = {'Normal': frame.copy(), 'Warm': apply_warm(frame),
                   'Cold': apply_cold(frame),
                   'Gotham': apply_gotham(frame),
                   'Grayscale': apply_grayscale(frame),
                   'Sepia': apply_sepia(frame),
                   'Pencil Sketch': apply_pencil_sketch(frame),
                   'Sharpening': apply_sharpening(frame),
                   'Invert': apply_invert(frame),
                   'Detail Enhancing': apply_detail_enhancing(frame),
                   'Stylization': apply_stylization(frame)}

    # Initialize a list to store the previews of the filters.
    filters_previews = []

    # Iterate over the filters dictionary.
    for filter_name, filtered_frame in filters.items():

        # Check if the filter we are iterating upon, is applied.
        if filter_applied == filter_name:

            # Set color to green.
            # This will be the border color of the filter preview.
            # And will be green for the filter applied and white for the other filters.
            color = (0, 255, 0)

        # Otherwise.
        else:

            # Set color to white.
            color = (255, 255, 255)

        # Make a border around the filter we are iterating upon.
        filter_preview = cv2.copyMakeBorder(src=filtered_frame, top=100, bottom=100,
                                            left=10, right=10, borderType=cv2.BORDER_CONSTANT,
                                            value=color)

        # Resize the preview to the 1/12th of its current width and height.
        filter_preview = cv2.resize(filter_preview, (frame_width // 12, frame_height // 12))

        # Append the filter preview into the list.
        filters_previews.append(filter_preview)

    # Get the new height and width of the previews.
    preview_height, preview_width, _ = filters_previews[0].shape

    # Check if any filter is selected.
    if filter_applied != 'Normal':
        # Apply the selected Filter on the frame.
        frame = apply_selected_filter(frame, filter_applied)

    # Check if the image capture state is True.
    if capture_image:
        # Capture an image and store it in the disk.
        cv2.imwrite('Captured_Image.png', frame)

        # Display a black image.
        cv2.imshow('Instagram Filters', np.zeros((frame_height, frame_width)))

        # Play the image capture music to indicate that an image is captured and wait for 100 milliseconds.
        pygame.mixer.music.play()
        cv2.waitKey(100)

        # Display the captured image.
        plt.close()
        plt.figure(figsize=[10, 10])
        plt.imshow(frame[:, :, ::-1])
        plt.title("Captured Image")
        plt.axis('off')

        # Update the image capture state to False.
        capture_image = False

    # Check if the camera icon variable doesnot contain the camera icon image.
    if not camera_icon:
        # Read a camera icon png image with its blue, green, red, and alpha channel.
        camera_iconBGRA = cv2.imread('../assets/IMAGES/woman-boat.jpg', cv2.IMREAD_UNCHANGED)

        # Resize the camera icon image to the 1/12th of the frame width,
        # while keeping the aspect ratio constant.
        camera_iconBGRA = cv2.resize(camera_iconBGRA,
                                     (frame_width // 12,
                                      int(((frame_width // 12) / camera_iconBGRA.shape[1]) * camera_iconBGRA.shape[0])))

        # Get the new height and width of the camera icon image.
        camera_icon_height, camera_icon_width, _ = camera_iconBGRA.shape

        # Get the first three-channels (BGR) of the camera icon image.
        camera_iconBGR = camera_iconBGRA[:, :, :-1]

        # Get the alpha channel of the camera icon.
        camera_icon_alpha = camera_iconBGRA[:, :, -1]

    # Get the region of interest of the frame where the camera icon image will be placed.
    frame_roi = frame[(frame_height - 10) - camera_icon_height: (frame_height - 10),(frame_width // 2 - camera_icon_width // 2): (frame_width // 2 - camera_icon_width // 2) + camera_icon_width]

    # Overlay the camera icon over the frame by updating the pixel values of the frame
    # at the indexes where the alpha channel of the camera icon image has the value 255.
    frame_roi[camera_icon_alpha == 255] = camera_iconBGR[camera_icon_alpha == 255]

    # Overlay the resized preview filter images over the frame by updating
    # its pixel values in the region of interest.
    #######################################################################################

    # Overlay the Warm Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 11.6) - preview_width // 2): (int(frame_width // 11.6) - preview_width // 2) + preview_width] = filters_previews[1]

    # Overlay the Cold Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 5.9) - preview_width // 2): (int(frame_width // 5.9) - preview_width // 2) + preview_width] = filters_previews[2]

    # Overlay the Gotham Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 3.97) - preview_width // 2): (int(frame_width // 3.97) - preview_width // 2) + preview_width] = filters_previews[3]

    # Overlay the Grayscale Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 2.99) - preview_width // 2): (int(frame_width // 2.99) - preview_width // 2) + preview_width] = filters_previews[4]

    # Overlay the Sepia Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 2.395) - preview_width // 2): (int(frame_width // 2.395) - preview_width // 2) + preview_width] = filters_previews[5]

    # Overlay the Normal frame (no filter) preview on the frame.
    frame[10: 10 + preview_height, (frame_width // 2 - preview_width // 2): (frame_width // 2 - preview_width // 2) + preview_width] = filters_previews[0]

    # Overlay the Pencil Sketch Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 1.715) - preview_width // 2): (int(frame_width // 1.715) - preview_width // 2) + preview_width] = filters_previews[6]

    # Overlay the Sharpening Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 1.501) - preview_width // 2): (int(frame_width // 1.501) - preview_width // 2) + preview_width] = filters_previews[7]

    # Overlay the Invert Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 1.335) - preview_width // 2): (int(frame_width // 1.335) - preview_width // 2) + preview_width] = filters_previews[8]

    # Overlay the Detail Enhancing Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 1.202) - preview_width // 2): (int(frame_width // 1.202) - preview_width // 2) + preview_width] = filters_previews[9]

    # Overlay the Stylization Filter preview on the frame.
    frame[10: 10 + preview_height, (int(frame_width // 1.094) - preview_width // 2): (int(frame_width // 1.094) - preview_width // 2) + preview_width] = filters_previews[10]

    #######################################################################################

    # Display the frame.
    cv2.imshow('Instagram Filters', frame)

    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed and break the loop.
    if k == 27:
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()
