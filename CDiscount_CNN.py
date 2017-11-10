
# coding: utf-8

# In[25]:


import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from PIL import Image   # or, whatever image library you prefer
import csv
import sys
import pandas as pd
import numpy as np

path = 'F:/ML/CDiscount Image Classification/'
filewriter_path = path + 'summary/'
checkpoint_path = path + 'checkpoints/'
display_step = 100
# Transfer Category_id to integer
category_to_int = dict()
with open(path+'category_names.csv') as csvfile:
    i = 0
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        category_to_int[row['category_id']] = i
        i += 1

num_category = len(category_to_int)


# In[2]:


# creates a dictionary with key indexing item _id and values (offset, length)
import struct
from tqdm import tqdm_notebook

num_dicts = 7069896 # according to data page

IDS_MAPPING = {}

length_size = 4 # number of bytes decoding item length

with open(path+'train.bson', 'rb') as f, tqdm_notebook(total=num_dicts) as bar:
    item_data = []
    offset = 0
    while True:        
        bar.update()
        f.seek(offset)
        
        item_length_bytes = f.read(length_size)     
        if len(item_length_bytes) == 0:
            break                
        # Decode item length:
        length = struct.unpack("<i", item_length_bytes)[0]

        f.seek(offset)
        item_data = f.read(length)
        assert len(item_data) == length, "%i vs %i" % (len(item_data), length)
        
        # Check if we can decode
        item = bson.BSON.decode(item_data)
        
        IDS_MAPPING[item['_id']] = (offset, length)        
        offset += length            
            


# In[4]:


def get_item(item_id):
    assert item_id in IDS_MAPPING
    with open(path+'train.bson', 'rb') as f:
        offset, length = IDS_MAPPING[item_id]
        f.seek(offset)
        item_data = f.read(length)
        return bson.BSON.decode(item_data)
    


# In[58]:


import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import warnings
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
warnings.filterwarnings("ignore")
learn = tf.contrib.learn
slim = tf.contrib.slim

w, h, d = 180, 180, 3
batch_size = 128


# In[ ]:


#-----------------Read image from bson-------------------------#
# num_samples = len(IDS_MAPPING)
# ids = np.array(list(IDS_MAPPING.keys()))
# np.random.shuffle(ids)
# offset = int(num_samples*0.9)
# train_ids = ids[: offset]
# val_ids = ids[offset:]
#--------------------------------------------------------------#

#-----------------Read image from directory--------------------#
from os import listdir
from os.path import isfile, join
prod_to_category = np.load(path+'example/prod_to_category.npy')
image_files = [f for f in listdir(path+'example') ]
num_samples = len(image_files)
np.random.shuffle(image_files)
pointer = 0
offset = int(num_samples*0.9)
#--------------------------------------------------------------#


# In[81]:


def next_batch(batch_size, pointer):
    """
    This function gets the next n ( = batch_size) images from the path list
    and labels and loads the images into them into memory 
    """
    # Get next batch of image (path) and labels
    imgs_names = image_files[pointer : pointer + batch_size]
    labels = [int(prod_to_category[int(f[0])]) for f in imgs_names]

    # Read images
    images = np.ndarray([batch_size, h, w, d])
    for i in range(len(imgs_names)):
        img = np.asarray(Image.open(path + 'example/' + imgs_names[i]), dtype='float32')/255.-.05
        images[i] = img

    # Expand labels to one hot encoding
    one_hot_labels = np.zeros((batch_size, num_category))
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1

    #return array of images and labels
    return images, one_hot_labels



def train_input_fn():
    labels = np.zeros(1, dtype='int32')
    imgs = np.zeros((1, 180, 180, 3), dtype='float32')
    for idx in np.random.choice(train_ids, batch_size):
        item = get_item(idx)
        category_id = item['category_id']
        for e, pic in enumerate(item['imgs']):
            picture = np.asarray(Image.open(io.BytesIO(pic['picture']))).astype('float32') / 255. - 0.5
            labels = np.append(labels, category_to_int[str(category_id)])
            imgs = np.vstack((imgs, np.expand_dims(picture, axis=0)))

    # Expand labels to one hot encoding
    one_hot_labels = np.zeros((labels.shape[0]-1, num_category))
    for i in range(labels.shape[0]-1):
        one_hot_labels[i][labels[i+1]] = 1
            
    return imgs[1:], one_hot_labels


def val_input_fn():
    labels = np.zeros(1, dtype="int32")
    imgs = np.zeros((1, 180, 180, 3), dtype="float32")
    for idx in np.random.choice(val_ids, batch_size):
        item = get_item(idx)
        category_id = item['category_id']
        for e, pic in enumerate(item['imgs']):
            picture = np.asarray(Image.open(io.BytesIO(pic['picture']))).astype('float32') / 255. - 0.5
            labels = np.append(labels, category_to_int[str(category_id)])
            imgs = np.vstack((imgs, np.expand_dims(picture, axis=0)))

    # Expand labels to one hot encoding
    one_hot_labels = np.zeros((labels.shape[0]-1, num_category))
    for i in range(labels.shape[0]-1):
        one_hot_labels[i][labels[i+1]] = 1
            
    return imgs[1:], one_hot_labels


# In[87]:



train_input_fn()


# In[41]:


import os
from datetime import datetime
tf.reset_default_graph() 

x = tf.placeholder(shape=[None, h, w, d], dtype=tf.float32)
y = tf.placeholder(shape=[None, num_category], dtype=tf.int32)
net = slim.conv2d(x, 24, [5,5], scope='conv1')
net = slim.max_pool2d(net, [2,2], scope='pool1')
net = slim.conv2d(net, 48, [5,5], scope='conv2')
net = slim.max_pool2d(net, [2,2], scope='pool2')
net = slim.conv2d(net, 96, [5,5], scope='conv3')
net = slim.max_pool2d(net, [2,2], scope='pool3')
net = slim.flatten(net, scope='flatten')
net = slim.fully_connected(net, 64, scope='fc1')
logits = slim.fully_connected(net, num_category,
        activation_fn=None, scope='fc2')
prob = slim.softmax(logits)

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, logits))
    
# Train op
with tf.name_scope("train"):
    train_op = slim.optimize_loss(loss, slim.get_global_step(),
            learning_rate=0.001,
            optimizer='Adam')
    
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(len(train_ids) / batch_size).astype(np.int32)
val_batches_per_epoch = np.floor(len(val_ids) / batch_size).astype(np.int32)


# In[83]:


# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
    
# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Load checkpoint
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    last_model = ckpt.model_checkpoint_path
    saver.restore(sess, last_model)
    
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),filewriter_path))

    # Loop over number of epochs
    for epoch in range(500):
        step = 1
        while step < train_batches_per_epoch:
            # Get a batch of images and labels
            batch_xs, batch_ys = next_batch(batch_size, pointer)
            #update pointer
            pointer += batch_size
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys
                                          })

            # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys
                                                        })
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                writer.flush()
                print("step: ", step)
            step += 1

        # Validate the model on the entire validation set
        test_acc = 0.
        test_count = 0
        for _ in range(5):
            batch_tx, batch_ty = val_input_fn()
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty
                                                })
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{0} Validation Accuracy = {1:.4f}".format(datetime.now(), test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        #save checkpoint of the model
        checkpoint_name = checkpoint_path+'model_epoch'+str(epoch)+'.ckpt'
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


# In[ ]:




