import csv
import numpy as np
import tensorflow as tf
from datetime import datetime
from datagenerator_tfrecord import ImageDataGenerator
from keras import backend as K
from keras.applications.xception import Xception
from keras.layers import Dense
Iterator = tf.data.Iterator

sess = tf.Session()

K.set_session(sess)


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
                                 batch_size=256,
                                 num_classes=num_classes,
                                 shuffle=False)
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    x, product_id = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)


model = Xception(include_top=False, input_tensor=x,
                 input_shape=(180, 180, 3), pooling='avg', classes=None)
prob = Dense(5270, activation='softmax')(model.output)


# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
f = open(path + 'submission.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(['_id', 'category_id', 'prob'])

with sess.as_default():
    latest = tf.train.latest_checkpoint(checkpoint_path)
    if latest is not None:
        print("resume", latest)
        saver.restore(sess, latest)

    print("{} Start predicting...".format(datetime.now()))

    sess.run(training_init_op)
    cnt = 0
    while True:
        try:
            probs, ids = sess.run([prob, product_id])

            for i in range(len(ids)):
                writer.writerow(
                    [ids[i], int_to_category[np.argmax(probs[i])], np.max(probs[i])])

            cnt += len(ids)
            if cnt % (1024 * 8) == 0:
                print(cnt)

            # break
        except tf.errors.OutOfRangeError:
            break
