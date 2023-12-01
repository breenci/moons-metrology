import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def plot_grayscale_image(image_path):
    # Read the image using matplotlib's imread function
    img = mpimg.imread(image_path)

    # Check if the image is grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert the RGB image to grayscale
        img = img.mean(axis=-1)

    # Plot the image
    plt.imshow(img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')  # Turn off axis labels
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot grayscale image from a JPEG file.')
    parser.add_argument('image_path', help='Path to the grayscale JPEG image file')

    args = parser.parse_args()

    # Check if the provided file exists
    if not os.path.isfile(args.image_path):
        print(f"Error: File '{args.image_path}' not found.")
        sys.exit(1)

    # Plot the grayscale image
    plot_grayscale_image(args.image_path)

if __name__ == "__main__":
    main()