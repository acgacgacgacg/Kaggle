# import os
# import csv
# import numpy as np
import tensorflow as tf
from datetime import datetime
# from datagenerator import ImageDataGenerator
from datagenerator_tfrecord import ImageDataGenerator
from resnet_model import *
Iterator = tf.data.Iterator


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


# Paths
# path = '/media/aldin/2EA2E320A2E2EAF3/ML/CDiscount Image Classification/'

filewriter_path = 'summary/'
checkpoint_path = 'checkpoints/'
# example_path = path + 'example/'
example_path = '/home/aldin/Desktop/output_file.tfrecords'

# learning_rate = 0.01
num_epochs = 20
batch_size = 128
percent_for_train = 0.98
_WEIGHT_DECAY = 1e-4
_MOMENTUM = 0.9


num_classes = 5270
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(example_path,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(example_path,
                                  mode='validation',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)
    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    x, y = iterator.get_next()


# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# Get the number of training/validation steps per epoch
# train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
# val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
train_batches_per_epoch = 189435 // 2

# Op for training
# Link variable to model output
logits = inference(x, True, num_classes=num_classes)

# Added 2017/12/15 confirm prob
op_probs = tf.nn.softmax(logits)

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = loss(logits, y)

# Train op
with tf.name_scope("train"):
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 256, the learning rate should be 0.1.
    initial_learning_rate = 0.001 * batch_size / 256
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
    boundaries = [
        int(train_batches_per_epoch * epoch) for epoch in [5, 6, 7, 60]]
    values = [
        initial_learning_rate * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create optimizer and apply gradient descent to the trainable　variables
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars=grads)

    # Set variables to train
    # train_op = optimizer.minimize(loss, var_list=tf.get_collection(
    #     tf.GraphKeys.TRAINABLE_VARIABLES, "fc"))
    # train_op = optimizer.minimize(
    # loss, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss, global_step, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add summary
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()
# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)


# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    print('Initializing finished')

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    # When execute first time
    optimistic_restore(sess, checkpoint_path + 'ResNet-L50.ckpt')
    # Else
    # latest = tf.train.latest_checkpoint(checkpoint_path)
    if latest is not None:
        print("resume", latest)
        saver.restore(sess, latest)

    print("{} Start training...".format(datetime.now()))
    print('train_batches_per_epoch:', train_batches_per_epoch)
    for epoch in range(10, num_epochs):
        # Initialize training iterator each epoch
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):
            # Construct a op list
            sess.run(train_op)
            probs = sess.run(op_probs)
            print(np.max(probs, axis=-1))

            if step % 100 == 0:
                # Create summary every step_size steps
                o = sess.run([loss, accuracy])
                writer.add_summary(
                    sess.run(merged_summary_op), epoch * train_batches_per_epoch + step)
                format_str = ('step %d, loss = %.2f, accuracy = %.2f')
                print(datetime.now(), format_str % (step, o[0], o[1]))

            # if step % 1000 == 1:
            #     print("{} Saving checkpoint of model...".format(datetime.now()))
            #     # save checkpoint of the model
            #     checkpoint_name = checkpoint_path + \
            #         'model_epoch' + str(epoch) + '.ckpt'
            #     save_path = saver.save(sess, checkpoint_name)
            #     print("{} Model checkpoint saved at {}".format(datetime.now(),
            #                                                    checkpoint_name))

        # Validate the model on the entire validation set
        # print("{} Start validation".format(datetime.now()))
        # sess.run(validation_init_op)
        # test_acc = 0.
        # test_count = 0
        # for _ in range(val_batches_per_epoch):
        #     acc = sess.run(accuracy)
        #     test_acc += acc
        #     test_count += 1
        # test_acc /= test_count
        # print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
        #                                                test_acc))
