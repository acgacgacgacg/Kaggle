# Created on Wed May 31 14:48:46 2017
# Updated on 2017/11/18 by Wu
# @author: Frederik Kratzert


"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
Dataset = tf.data.Dataset

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, data_path, img_list, label_list, mode, batch_size,
                 num_classes, shuffle=True, buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            data_path: Path of the data folder. Must end with '/'.
            img_list: A python list of image file names.
            label_list: A python list of labels.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.img_list = img_list
        self.label_list = label_list
        self.num_classes = num_classes
        self.data_path = data_path

        # number of samples in the dataset
        self.data_size = len(self.label_list)

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_list, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.label_list, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)

        elif mode == 'validation':
            data = data.map(self._parse_function_validation, num_parallel_calls=8)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(self.data_path + filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [180, 180])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
        img_bgr = tf.transpose(img_bgr, [2, 0, 1])

        return tf.image.convert_image_dtype(img_bgr, dtype='float32'), one_hot

    def _parse_function_validation(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(self.data_path + filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [180, 180])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
        img_bgr = tf.transpose(img_bgr, [2, 0, 1])
        return tf.image.convert_image_dtype(img_bgr, dtype='float32'), one_hot
