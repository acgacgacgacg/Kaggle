# Created on Wed May 31 14:48:46 2017
# Updated on 2017/11/18 by Wu
# @author: Frederik Kratzert


"""Containes a helper class for image input pipelines in tensorflow."""
from datetime import datetime
import tensorflow as tf
import numpy as np
from math import ceil, floor
Dataset = tf.data.Dataset
Iterator = tf.data.Iterator


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
IMAGE_SIZE = 180


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, file_name, mode, batch_size,
                 num_classes, shuffle=True, buffer_size=1000, channel_first=False):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            file_name: A tf_record file containing images and labels.
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
        self.file_name = file_name
        self.num_classes = num_classes
        self.channel_first = channel_first

        # create dataset
        data = tf.data.TFRecordDataset(
            self.file_name, compression_type='ZLIB', buffer_size=100 * 1024)

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)

        elif mode == 'validation':
            data = data.map(self._parse_function_validation,
                            num_parallel_calls=8)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _parse_function_train(self, tf_record):
        # For data augmentation
        def get_translate_image(image):
            index = np.random.randint(0, 4)
            s = floor(0.2 * IMAGE_SIZE)
            cl = ceil(0.8 * IMAGE_SIZE)
            starts = np.array([[0, 0], [s, 0], [0, s], [0, 0]], dtype=np.int32)
            sizes = np.array([[IMAGE_SIZE, cl], [cl, IMAGE_SIZE],
                              [IMAGE_SIZE, cl], [cl, IMAGE_SIZE]])
            if index == 0:  # Translate left 20 percent
                h_start, w_start = starts[index]
                height, width = sizes[index]
                cropped_image = tf.image.crop_to_bounding_box(
                    image, h_start, w_start, height, width)
                image = tf.image.pad_to_bounding_box(
                    cropped_image, h_start, w_start, IMAGE_SIZE, IMAGE_SIZE)
            return image
        """Input parser for samples of the training set."""
        features = tf.parse_single_example(
            tf_record,
            # Defaults are not specified since both keys are required.
            features={
                'category_id': tf.FixedLenFeature([], tf.int64),
                'product_id': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a float32 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.image.decode_jpeg(features['img_raw'], tf.float32)
        # product_id = tf.cast(features['product_id'], tf.int32)
        label = tf.cast(features['category_id'], tf.int32)

        image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
        # image = tf.reshape(image, [180, 180, 3])
        one_hot = tf.one_hot(label, self.num_classes)

        """
        Dataaugmentation comes here.
        """
        if np.random.randint(0, 9) <= 2:
            # random crop
            if np.random.randint(0, 9) <= 1:
                image = tf.image.central_crop(image, 0.7)
                image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])

            # random translation
            if np.random.randint(0, 9) <= 1:
                image = get_translate_image(image)

            if np.random.randint(0, 9) <= 1:
                image = tf.contrib.image.rotate(
                    image, np.random.randint(0, 180))

            if np.random.randint(0, 9) <= 1:
                image = tf.image.random_flip_left_right(image)

            if np.random.randint(0, 9) <= 1:
                image = tf.image.random_flip_up_down(image)

            if np.random.randint(0, 9) <= 1:
                imshape = [IMAGE_SIZE, IMAGE_SIZE]
                vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                                      (0.43 * imshape[1], 0.32 * imshape[0]),
                                      (0.56 * imshape[1], 0.32 * imshape[0]),
                                      (0.85 * imshape[1], 0.99 * imshape[0])]], dtype=np.int32)
                image = tf.contrib.image.transform(image, vertices)

        img_centered = tf.subtract(image, IMAGENET_MEAN) / 255.
        img_cropped = tf.random_crop(img_centered, [160, 160, 3])
        if self.channel_first:
            # RGB -> BGR
            img_bgr = img_cropped[:, :, ::-1]
            img_bgr = tf.transpose(img_bgr, [2, 0, 1])
        else:
            img_bgr = img_cropped

        return img_bgr, one_hot

    def _parse_function_validation(self, tf_record):
        """Input parser for samples of the training set."""
        features = tf.parse_single_example(
            tf_record,
            # Defaults are not specified since both keys are required.
            features={
                # 'category_id': tf.FixedLenFeature([], tf.int64),
                'product_id': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a float32 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.image.decode_jpeg(features['img_raw'], tf.float32)
        product_id = tf.cast(features['product_id'], tf.int32)
        # label = tf.cast(features['category_id'], tf.int32)

        image = tf.image.resize_images(image, [180, 180])
        # one_hot = tf.one_hot(label, self.num_classes)

        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(image, IMAGENET_MEAN) / 255.

        if self.channel_first:
            # RGB -> BGR
            img_bgr = img_centered[:, :, ::-1]
            img_bgr = tf.transpose(img_bgr, [2, 0, 1])
        else:
            img_bgr = img_centered

        return img_bgr, product_id
        # , one_hot


def test():
    path = 'F:/ML/CDiscount Image Classification/output_file.tfrecords'
    tr_data = ImageDataGenerator(path, 'training', 64, 5270)
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    x, y = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data.data)
    op = tf.reduce_sum(tf.multiply(x, x))

    with tf.Session() as sess:
        sess.run(training_init_op)
        for i in range(10000):
            s = sess.run(op)
            if i % 100 == 0:
                print(datetime.now(), s)

# from PIL import Image
# from io import BytesIO
# # test()
# tfrecords_filename = '/home/aldin/Desktop/output_file.tfrecords'
# opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
# record_iterator = tf.python_io.tf_record_iterator(
#     path=tfrecords_filename, options=opts)
# img_string = ""
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)

#     img_string = example.features.feature['img_raw'].bytes_list.value[0]
#     product_id = example.features.feature['product_id'].int64_list.value[0]
#     print(product_id)
#     im = Image.open(BytesIO(img_string))
#     im.show()
#     break
