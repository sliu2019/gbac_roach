import tensorflow as tf
import IPython
import numpy as np
# mine
from utils import *

# mnist = tf.keras.datasets.mnist

# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0


# IPython.embed()

##### Get data #########################

# Make sure you set k!
k = 8
meta_batch_size = 100 
load_data =  True

if load_data:
	# Redo
	inputs = np.load("/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/k_terrain_classify_data/inputs.npy")
	outputs = np.load("/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/k_terrain_classify_data/outputs.npy")

	inputs_val = np.load("/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/k_terrain_classify_data/inputs_val.npy")
	outputs_val = np.load("/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/k_terrain_classify_data/outputs_val.npy")

else:
	# Training data
	dataX=[]
	dataX_full=[] #this is just for your personal use for forwardsim (for debugging)
	dataY=[]
	dataZ=[]

	# Validation data
	dataX_val = []
	dataX_full_val=[]
	dataY_val=[]
	dataZ_val=[]

	agg_itr = 0
	training_ratio = 0.9
	use_one_hot = False
	use_camera = False
	cheaty_training = False
	state_representation = "exclude_x_y"

	dataX_curr, dataY_curr, dataZ_curr, dataX_curr_full = getDataFromDisk(agg_itr, 'terrain_types', 
																			use_one_hot, use_camera, 
																			cheaty_training, state_representation)

	for i in range(len(dataX_curr)):
		taski_num_rollout = len(dataX_curr[i])
		print("taski_num_rollout: ", taski_num_rollout)
		dataX.append(dataX_curr[0][:int(taski_num_rollout*training_ratio)])

		dataX_full.append(dataX_curr_full[0][:int(taski_num_rollout*training_ratio)])
		dataY.append(dataY_curr[0][:int(taski_num_rollout*training_ratio)])
		dataZ.append(dataZ_curr[0][:int(taski_num_rollout*training_ratio)])

		dataX_val.append(dataX_curr[0][int(taski_num_rollout*training_ratio):])
		dataX_full_val.append(dataX_curr_full[0][int(taski_num_rollout*training_ratio):])
		dataY_val.append(dataY_curr[0][int(taski_num_rollout*training_ratio):])
		dataZ_val.append(dataZ_curr[0][int(taski_num_rollout*training_ratio):])


	#concatenate state and action --> inputs
	#IPython.embed()
	dataX = np.array(dataX)
	dataY = np.array(dataY)
	dataZ = np.array(dataZ)

	dataX_val = np.array(dataX_val)
	dataY_val = np.array(dataY_val)
	dataZ_val = np.array(dataZ_val)

	# Make sure you standardize here!
	x_mean = np.mean(dataX, axis=(0, 1, 2)) 
	y_mean = np.mean(dataY, axis=(0, 1, 2)) 
	z_mean = np.mean(dataZ, axis=(0, 1, 2)) 

	x_std = np.std(dataX, axis=(0, 1, 2)) 
	y_std = np.std(dataY, axis=(0, 1, 2)) 
	z_std = np.std(dataZ, axis=(0, 1, 2))

	dataX = (dataX - x_mean)/x_std
	dataY = (dataY - y_mean)/y_std
	dataZ = (dataZ - z_mean)/z_std

	dataX_val = (dataX_val - x_mean)/x_std
	dataY_val = (dataY_val - y_mean)/y_std
	dataZ_val = (dataZ_val - z_mean)/z_std
	##################################################################
	concat = np.concatenate((dataX, dataY, dataZ), axis=3)
	inputs = np.empty((0, k, concat.shape[2]))
	outputs = np.empty((0))

	for taski in range(concat.shape[0]):
	    for rollouti in range(concat.shape[1]):
	    	for windowi in range(concat.shape[2]-k):
		    	k_batch = concat[taski][rollouti][windowi: windowi+k]
	    		inputs = np.append(inputs, np.expand_dims(k_batch, axis=0), axis=0)
	    		outputs = np.append(outputs, taski)
	    		#IPython.embed()
	      
	#IPython.embed()
	####################################################################
	concat_val = np.concatenate((dataX_val, dataY_val, dataZ_val), axis=3)
	inputs_val = np.empty((0, k, concat_val.shape[2]))
	outputs_val = np.empty((0))

	for taski in range(concat_val.shape[0]):
	    for rollouti in range(concat_val.shape[1]):
	    	for windowi in range(concat_val.shape[2]-k):
		    	k_batch = concat_val[taski][rollouti][windowi: windowi+k]
	    		inputs_val = np.append(inputs_val, np.expand_dims(k_batch, axis=0), axis=0)
	    		outputs_val = np.append(outputs_val, taski)
	####################################################################

	### Saving, since making the data takes forever ### 
	np.save("/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/k_terrain_classify_data/inputs.npy", inputs)
	np.save("/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/k_terrain_classify_data/outputs.npy", outputs)

	np.save("/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/k_terrain_classify_data/inputs_val.npy", inputs_val)
	np.save("/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/k_terrain_classify_data/outputs_val.npy", outputs_val)

# Modify input
# inputs[:,:,24:] = inputs[:,:, 24:] - inputs[:,:,:24]
# inputs = inputs[:,:,24:]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape = (k, inputs.shape[2])),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(inputs, outputs, batch_size = meta_batch_size, epochs=5)
model.evaluate(inputs_val, outputs_val)

