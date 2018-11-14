"""
Process an image that we can pass to our networks.
"""
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

def process_image(image, target_shape, chanel_3d=False):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image[0], target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    if chanel_3d:
        # TODO YUV
        x = (img_arr[:, :, 1] / 255.).astype(np.float32)
    else:
        x = (img_arr / 255.).astype(np.float32)

    return x

if __name__ == '__main__':
    process_image(r'd:\datasetConvert\300VW_Dataset_2015_12_14\114\0\001788.jpg', (32,32,3))