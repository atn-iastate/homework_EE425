from hw2.hw2a import GDAModel, LogisticModel, compute_accuracy
import gzip
import numpy as np


class MnistFileInfo:

    def __init__(self, mb=0, no_of_images=0, height=0, width=0, file_type="images"):
        self.magic_number = mb
        self.no_of_images = no_of_images
        self.height = height
        self.width = width
        self.file_type = file_type

    def get_mnist_file_info(self, buffer, file_type="images"):

        self.file_type = file_type
        convert_vector = np.asarray([256**3, 256**2, 256, 1])

        if self.file_type == "images":
            info = np.frombuffer(buffer.read(16), dtype=np.uint8)  # the first 16 bytes contain the file's information
            self.magic_number = np.dot(info[0:4], convert_vector)
            self.no_of_images = np.dot(info[4:8], convert_vector)
            self.height = np.dot(info[8:12], convert_vector)
            self.width = np.dot(info[12:16], convert_vector)

        if self.file_type == "labels":
            info = np.frombuffer(buffer.read(8), dtype=np.uint8)  # the first 16 bytes contain the file's information
            self.magic_number = np.dot(info[0:4], convert_vector)
            self.no_of_images = np.dot(info[4:8], convert_vector)

    def get_bytes(self):
        '''Get the number of bytes containing data'''
        if self.file_type == "images":
            return self.no_of_images * self.height * self.width

        if self.file_type == "labels":
            return self.no_of_images

    def get_dimension(self):
        '''Get the dimension of the data to be reshape in numpy'''
        if self.file_type == "images":
            return self.no_of_images, self.height * self.width

        if self.file_type == "labels":
            return self.no_of_images


def to_numpy_dataframe(bytestream, bytestream_info):
    '''Convert the byte stream to a numpy array based on the corresponding information matrix'''

    all_bytes = np.frombuffer(bytestream.read(bytestream_info.get_bytes()), dtype=np.uint8)
    data_frame = np.asarray(all_bytes).reshape(bytestream_info.get_dimension())

    return data_frame


def remove_zero_columns(numpy_array):
    zeros_index = []
    for i in range(numpy_array.shape[1]):
        if np.sum(numpy_array[:, i]) == 0:
            zeros_index.append(i)

    return np.delete(numpy_array, zeros_index, 1), zeros_index


def remove_middle_rows(numpy_array):
    remove_index = []



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

        # Getting the information header of each file
        train_images_info = MnistFileInfo()
        train_images_info.get_mnist_file_info(train_images)

        train_labels_info = MnistFileInfo()
        train_labels_info.get_mnist_file_info(train_labels, file_type="labels")

        test_images_info = MnistFileInfo()
        test_images_info.get_mnist_file_info(test_images)

        test_labels_info = MnistFileInfo()
        test_labels_info.get_mnist_file_info(test_labels, file_type="labels")

        # convert the bytestream to numpy arrays
        train_images = to_numpy_dataframe(train_images, train_images_info)
        train_labels = to_numpy_dataframe(train_labels, train_labels_info)
        test_images = to_numpy_dataframe(test_images, test_images_info)
        test_labels = to_numpy_dataframe(test_labels, test_labels_info)

        # remove all zeros columns in images files
        train_images, removed_cols = remove_zero_columns(train_images)

        test_images = np.delete(test_images, removed_cols, 1)

        model = GDAModel.gda_estimate(train_images, train_labels)
        model.predict(test_images)


        a = np.arange(0,4,1).reshape(2, 2)
        print(a)
        # testing
        print(train_images.shape)
        print(train_labels.shape)
        print(test_images.shape)
        print(test_labels.shape)

        # buf = train_images.read(4)
        # print(buf)
        # c = np.frombuffer(buf, dtype=np.uint8)
        # buf = train_images.read(4)
        # b = np.frombuffer(buf, dtype=np.uint8)
        # print(c)
        # print(b)
        # convert_vector = np.asarray([256**3, 256**2, 256, 1])

if __name__ == "__main__":
    main()

