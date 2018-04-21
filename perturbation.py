import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from data import get_data_set
from model import model

# train_x, train_y, train_l = get_data_set()

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10
_ITERATION = 100000
_SAVE_PATH = "./tensorboard/cifar-10/"


# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
# optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)

# correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# tf.summary.scalar("Accuracy/train", accuracy)

# sess=tf.Session()
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('./modified-500.meta')
# saver.restore(sess, "./modified-500")

# # # Access saved Variables directly
# graph = tf.get_default_graph()
# w1 = graph.get_tensor_by_name("conv1/weights:0")
# # w2 = graph.get_tensor_by_name("conv2/weights:0")
# # w3 = graph.get_tensor_by_name("conv3/weights:0")
# print(w1.shape)
# print(type(w1))
# k = sess.run(w1)
# print(k)
# print(k[0].shape)
# print(type(k[0]))

# new_w = np.ones(k[0].shape, np.int32)
# # print(new_w)

# # x = w1 + new_w
# # print(sess.run(x))
# # print(k)
# # print(sess.run([w1]))
# # print("print w2 now.........")
# # print(sess.run([w2]))
# # print(sess.run([w3]))


# # assign_op = tf.assign(w1, tf.zeros_like(w1))
# assign_op = tf.assign(w1, w1+new_w)
# sess.run(assign_op)  # or `assign_op.op.run()`

# #Now, save the graph
# saver.save(sess, './modified',global_step=500)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
epsilon = [0.01, 0.05, 0.1, 0.3, 0.5]

# sess=tf.Session()    
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('./-7000.meta')
# saver.restore(sess, "./-7000")

# # Access saved Variables directly
# graph = tf.get_default_graph()
# w1 = graph.get_tensor_by_name("conv1/weights:0")
# intermediate = w1
# weight1 = sess.run([w1])
# np_arr = np.array(weight1)
# max_val = np.max(np_arr)
# delta = [max_val*ep for ep in epsilon]

# d = delta[0]/2
# rand = np.random.uniform(-d, d)

# print(type(w1))
# print(w1.shape)
# print(tf.train.latest_checkpoint('./'))
# print(w1)
# print(weight1)
# print(type(weight1))
# print(np_arr)
# print(np.max(np_arr))
# print(delta[0])
# print (rand)

epsilon = [0.01, 0.05, 0.1, 0.2, 0.3]
sess=tf.Session() 
directory = './overfit_delta'
subfolder = '/modified'
y = []
num_perturbation = 50

for d in range(0,len(epsilon)):
	y = []
	for i in range(1, num_perturbation+1):
		saver = tf.train.import_meta_graph('./-7000.meta')
		saver.restore(sess, "./-7000")
		graph = tf.get_default_graph()
		w1 = graph.get_tensor_by_name("conv1/weights:0")
		weight1 = sess.run([w1])
		np_arr = np.array(weight1)
		max_val = np.max(np_arr)
		delta = [max_val*ep for ep in epsilon]

		d_half = float(delta[d]/2)
		perturbation = np.random.uniform(-d_half, d_half, w1.shape)
		assign_op = tf.assign(w1, w1 + perturbation)
		sess.run(assign_op)  # or `assign_op.op.run()`
		saver.save(sess, directory+str(epsilon[d])+subfolder+str(epsilon[d]), global_step = i)

		# find the test accuracy after the perturbation
		y.append(prediction(saver, sess, directory+str(epsilon[d])+subfolder+str(epsilon[d])+'-'+str(i)))

	print("average of test accuracy for perturbation ("+str(epsilon[d]*100) ") = " + str(sum(y)/num_perturbation))
	plt.figure(d)
	plt.plot(x,y,'r-')
	plt.xlabel('Trials')
	plt.ylabel('Test Accuracy')
	plt.title('The Effect of Perturbing Overfit model by Delta = ' + str(epsilon[index]*100)+'%')
	plt.show()
sess.close()