import matplotlib.pyplot as plt
import numpy as np


def plot_helper(image):
    # assumes a 1-channel PyTorch tensor image
    image = np.squeeze(image.numpy())
    image = image[0] + 1j * image[1]

    plt.imshow(np.absolute(image))
    plt.gray()
    plt.show()
