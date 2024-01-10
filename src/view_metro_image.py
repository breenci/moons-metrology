import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
import sys
import numpy as np


def add_brightness_mask(img_arr, sat_low, sat_high, colors=[[0, 0, 255], [255, 0, 0]]):
    """Add a brightness mask to a grayscale image array"""

    # make the image 3 channel
    arr_3_channel = np.dstack((img_arr, img_arr, img_arr))

    # mask pixels below brightness threshold
    sat_mask_low = img_arr < sat_low

    # color pixels below brightness threshold with the first color
    arr_3_channel[sat_mask_low] = colors[0]

    # mask pixels above brightness threshold
    sat_mask_high = img_arr > sat_high

    # color pixels above brightness threshold with the second color
    arr_3_channel[sat_mask_high] = colors[1]

    return arr_3_channel


def plot_images(image_files, sat_low=80, sat_high=200):
    num_images = len(image_files)

    # If there's only one image, create a single subplot
    if num_images == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5), sharex=True, sharey=True)
    else:
        # Create a subplot grid with shared axes
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5), sharex=True, sharey=True)

    for i, image_path in enumerate(image_files):
        # Read the image and convert to grayscale
        img = Image.open(image_path).convert('L')

        # convert to numpy array
        img_arr = np.array(img)

        # add brightness mask
        arr_3_channel = add_brightness_mask(img_arr, sat_low, sat_high)

        # convert back to PIL image
        img_3_channel = Image.fromarray(arr_3_channel)

        # Plot the image on the corresponding subplot
        if num_images == 1:
            axes.imshow(img_3_channel, cmap='gray')
            axes.axis('off')  # Turn off axis labels
        else:
            axes[i].imshow(img_3_channel, cmap='gray')
            axes[i].axis('off')  # Turn off axis labels

    # Add a common title and show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot grayscale image from JPEG files.')
    parser.add_argument('image_paths', nargs='+', help='Paths to grayscale JPEG image files')
    parser.add_argument('-l', '--low', help='Lower brightness threshold for mask',
                        default=80, type=int)
    parser.add_argument('-u', '--upper', help='Upper brightness threshold for mask',
                        default=200, type=int)

    args = parser.parse_args()

    # Check if the provided files exist
    for image_path in args.image_paths:
        if not os.path.isfile(image_path):
            print(f"Error: File '{image_path}' not found.")
            sys.exit(1)

    # Plot the grayscale images
    plot_images(args.image_paths, args.low, args.upper)


if __name__ == "__main__":
    main()