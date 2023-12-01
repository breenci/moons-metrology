import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fn = 'data/testing_301123/IMG_0013.JPG'

img = mpimg.imread(fn)
img_plot = plt.imshow(img, cmap='grey')
plt.show()