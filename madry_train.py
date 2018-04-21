"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

txt = open("training_summary.txt", "w+")

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

# batch_size = config['training_batch_size']
batch_size = 1000

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

# Extract x_batch and y_batch
x_data = []
y_data = []

x_batch, y_batch = mnist.train.next_batch(10000)
for j in range(0,10):
  count = 0

  for i in range(1,10000):
    if y_batch[i]==j:
      x_data.append(x_batch[i])
      y_data.append(y_batch[i])
      count += 1
      if count == 10: # total number of examples per class
        print(str(count)+ " number of class "+str(j)+ " appended to x_batch")
        break;
print("Size of x_data = "+str(len(x_data)))
print("Size of y_data = "+str(len(y_data)))

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    #x_batch, y_batch = mnist.train.next_batch(batch_size)

    # for index in range(0,len(x_batch)):
    #   if(y_batch[index]!=6 and y_batch[index]!=9):
    #     k = np.random.randint(4) ## allow up to 270 degree rotation
    #     two_d = np.reshape(x_batch[index], (28, 28))
    #     two_d = np.rot90(two_d, k)
    #     x_batch[index] = np.reshape(two_d, (1, 784))[0]

    # Compute Adversarial Perturbations
    # start = timer()
    # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    # end = timer()
    # training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    # adv_dict = {model.x_input: x_batch_adv,
    #             model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      # adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    '.format(ii))
      txt.write('Step {}:    '.format(ii))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      txt.write('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      # print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      # txt.write('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0

    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=nat_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    # start = timer()
    sess.run(train_step, feed_dict=nat_dict)
    # end = timer()
    # training_time += end - start

txt.close()
