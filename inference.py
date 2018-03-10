# import os
import csv
import bson
import io
import numpy as np
import tensorflow as tf
from datetime import datetime
from resnet_model import *
from skimage.data import imread
from PIL import Image

def optimistic_restore(session, save_file):
    '''
      restore weights from checkpoint as many as possible
    '''
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    # print(restore_vars)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

IMAGENET_MEAN = [123.68, 116.779, 103.939]
# Paths
path = '/home/aldin/Desktop/'
checkpoint_path = path + 'checkpoints/'
# example_path = '/home/aldin/Desktop/train_example.bson'
example_path = '/media/aldin/2EA2E320A2E2EAF3/ML/CDiscount Image Classification/train.bson'

# Transfer No Category.
int_to_category = dict()
with open(path + 'category_names.csv') as csvfile:
    i = 0
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        int_to_category[i] = row['category_id']
        i += 1

num_classes = 5270

# Link variable to model output
x = tf.placeholder(tf.float32, [1, 3, 180, 180])
logits = inference(x, False, num_classes=num_classes)
prob = tf.nn.softmax(logits)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
f = open(path + 'submission.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(['_id', 'category_id'])

with tf.Session() as sess:
    # Load the pretrained weights into the non-trainable layer
    # When execute first time
    # optimistic_restore(sess, checkpoint_path + 'ResNet-L50.ckpt')
    # Else
    latest = tf.train.latest_checkpoint(checkpoint_path)
    if latest is not None:
        print("resume", latest)
        saver.restore(sess, latest)

    print("{} Start predicting...".format(datetime.now()))
    # test_data = bson.decode_file_iter(open(example_path, 'rb'))

    # cnt = 0
    # img1 = None
    # img2 = None

    # for c, d in enumerate(test_data):
    #     product_id = d['_id']
    #     probs = np.zeros(5270, dtype=np.float32)
    #     for e, pic in enumerate(d['imgs']):
    #         img = Image.open(io.BytesIO(pic['picture']))
    #         # img.show()
    #         # img = (imread(io.BytesIO(pic['picture'])).astype(
    #         #     np.float32) - IMAGENET_MEAN) / 255.
    #         # img = img[:, :, ::-1]
    #         img1 = imread(io.BytesIO(pic['picture'])).astype(np.float32)
    #         break
    #         probs += sess.run(prob, feed_dict={x: np.expand_dims(
    #             np.transpose(img, (2, 0, 1)), axis=0)}).reshape(-1)
    #     break
    #     category = int_to_category[np.argmax(probs)]
    #     writer.writerow([product_id, category])
    #     print(np.max(probs))
    #     cnt += 1
    #     if cnt % 1000 == 0:
    #         print(datetime.now(), cnt)
    #         break

    tfrecords_filename = '/home/aldin/Desktop/output_file.tfrecords'
    opts = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.ZLIB)
    record_iterator = tf.python_io.tf_record_iterator(
        path=tfrecords_filename, options=opts)
    img_string = ""
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = example.features.feature['img_raw'].bytes_list.value[0]
        product_id = example.features.feature['product_id'].int64_list.value[0]
        print(product_id)
        im = Image.open(io.BytesIO(img_string))
        img = (np.asarray(im, dtype=np.float32) - IMAGENET_MEAN) / 255.
        img = img[:, :, ::-1]
        probs = sess.run(prob, feed_dict={x: np.expand_dims(
            np.transpose(img, (2, 0, 1)), axis=0)}).reshape(-1)
        print(np.max(probs))
