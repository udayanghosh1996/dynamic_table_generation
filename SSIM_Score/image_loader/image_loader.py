from skimage import io
from skimage.util import img_as_float
import os


def load_image(image_name):
    image_path = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), 'Dataset'), 'val'), 'images'),
                              image_name)
    image = io.imread(image_path)
    return img_as_float(image)
