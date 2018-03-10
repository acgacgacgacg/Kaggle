# import os
import csv
# import bson
# import io
import numpy as np
import tensorflow as tf
from datetime import datetime
from resnet_model import *
from datagenerator_tfrecord import ImageDataGenerator
Iterator = tf.data.Iterator
# from skimage.data import imread
# from PIL import Image


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
checkpoint_path = 'checkpoints/'
# example_path = '/home/aldin/Desktop/train_example.bson'
example_path = '/home/aldin/Desktop/test.tfrecords'

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
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(example_path,
                                 mode='validation',
                                 batch_size=512,
                                 num_classes=num_classes,
                                 shuffle=False)
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    x, product_id = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)

logits = inference(x, False, num_classes=num_classes)
prob = tf.nn.softmax(logits)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
f = open(path + 'submission.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(['_id', 'category_id', 'prob'])

pids = []
submission = open(path + 'submission2.csv', 'r')
reader = csv.DictReader(submission, delimiter=',')
for row in reader:
    pids.append(row['_id'])

print('pids finished')

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
    sess.run(training_init_op)
    cnt = 0
    while True:
        try:
            probs, ids = sess.run([prob, product_id])
            # ids = sess.run(product_id)
            # print(probs.shape[0])
            for i in range(len(ids)):
                writer.writerow(
                    [ids[i], int_to_category[np.argmax(probs[i])], np.max(probs[i])])

            cnt += len(ids)
            if cnt % 1024 * 8 == 0:
                print(cnt)

            # break
        except tf.errors.OutOfRangeError:
            break
