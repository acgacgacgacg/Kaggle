
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras import backend as K
from datetime import datetime

sess = tf.Session()

from keras import backend as K
K.set_session(sess)


# In[2]:


from datagenerator_tfrecord import ImageDataGenerator
Iterator = tf.data.Iterator

filewriter_path = 'summary/'
checkpoint_path = 'checkpoints/'
example_path = '/home/aldin/Desktop/train.tfrecords'

# learning_rate = 0.01
num_epochs = 20
batch_size = 64
percent_for_train = 0.98
_WEIGHT_DECAY = 1e-4
_MOMENTUM = 0.9
num_classes = 5270

with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(example_path,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    # val_data = ImageDataGenerator(example_path,
    #                               mode='validation',
    #                               batch_size=batch_size,
    #                               num_classes=num_classes,
    #                               shuffle=False)
    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    x, y = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
# validation_init_op = iterator.make_initializer(val_data.data)


# In[10]:


from keras.applications.xception import Xception
from keras.layers import Flatten, Dense

model = Xception(include_top=False, weights='imagenet', input_tensor=x,
                 input_shape=(160, 160, 3), pooling='avg', classes=None)

# model.summary()


# In[16]:


from keras.models import Model
from keras.layers import Flatten, Dense

for layer in model.layers:
    layer.trainable = False

# X = Flatten()(model.output)
# X = Dense(5270, activation='relu')(X)
logits = Dense(5270, activation='softmax')(model.output)
# new_model = Model(input = model.input, output = logits)
# new_model.summary()


# In[15]:


from keras.objectives import categorical_crossentropy

loss = tf.reduce_mean(categorical_crossentropy(y, logits))

# val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
train_batches_per_epoch = 189435 // 2

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
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars=grads)

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

with sess.as_default():
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    print('Initializing finished')

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    latest = tf.train.latest_checkpoint(checkpoint_path)
    if latest is not None:
        print("resume", latest)
        saver.restore(sess, latest)

    print("{} Start training...".format(datetime.now()))
    print('train_batches_per_epoch:', train_batches_per_epoch)
    for epoch in range(1, 10):
        # Initialize training iterator each epoch
        sess.run(training_init_op)
        step = 0
        while True:
            try:
                # Construct a op list
                sess.run(train_op)
                step += 1
                if step % 100 == 0:
                    # Create summary every step_size steps
                    o = sess.run([loss, accuracy])
                    # writer.add_summary(
                    #     sess.run(merged_summary_op), epoch * train_batches_per_epoch + step)
                    format_str = ('step %d, loss = %.2f, accuracy = %.2f')
                    print(datetime.now(), format_str % (step, o[0], o[1]))

                if step % 1000 == 0:
                    print("{} Saving checkpoint of model...".format(datetime.now()))
                    # save checkpoint of the model
                    checkpoint_name = checkpoint_path + \
                        'model_epoch' + str(epoch) + '.ckpt'
                    save_path = saver.save(sess, checkpoint_name)
                    print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                                   checkpoint_name))

            except tf.errors.OutOfRangeError:
                break
