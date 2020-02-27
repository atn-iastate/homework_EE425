# from hw2a_An_Adam import GDAModel, LogisticModel
import gzip
import numpy as np
import shutil
from mlxtend.data import loadlocal_mnist


class minst_file_info:

    def __init__(self, mb=0, no_of_images=0, nrow=0, ncol=0):
        self.magic_number = mb
        self.no_of_images = no_of_images
        self.nrow = nrow
        self.ncol = ncol

    def get_mnist_file_info(self, buffer):

        info = np.frombuffer(buffer.read(16), dtype=np.uint8)  # the first 16 bytes contain the file's information
        convert_vector = np.asarray([1, 256, 256**2, 256**3])
        self.magic_number = np.sum(np.dot())


def main():

    files = {
        "test_images": "../mnist/t10k-images-idx3-ubyte.gz",
        "test_labels": "../mnist/t10k-labels-idx1-ubyte.gz",
        "train_images": "../mnist/train-images-idx3-ubyte.gz",
        "train_labels": "../mnist/train-labels-idx1-ubyte.gz"
    }

    with gzip.open(files['train_images'], 'rb') as train_images, \
        gzip.open(files['train_labels'], 'rb') as train_labels, \
        gzip.open(files['test_images'], 'rb') as test_images, \
        gzip.open(files['test_labels'], 'rb') as test_labels:
        buf = train_images.read(4)
        print(buf)
        c = np.frombuffer(buf, dtype=np.uint8)
        buf = train_images.read(4)
        b = np.frombuffer(buf, dtype=np.uint8)
        print(c)
        print(b)

if __name__ == "__main__":
    main()

