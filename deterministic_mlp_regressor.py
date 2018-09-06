import tensorflow as tf
from sandbox.ignasi.maml.utils import normalize
import numpy as np
from tensorflow.python.platform import flags
from utils import *

class DeterministicMLPRegressor(object):
    def __init__(self, dim_input, dim_output, dim_hidden=(64, 64), dim_conv1d=(8, 8, 8), nonlinearity = "relu", norm='None', dim_obs=None,
                 dim_bias=0, multi_input=0, tf_datatype=tf.float32, seed=None, weight_initializer = "truncated_normal"):

        # dims
        self.multi_input = multi_input
        if self.multi_input:
            self.dim_input = dim_input * self.multi_input
        else:
            self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.dim_conv1d = dim_conv1d
        self.dim_obs = dim_obs
        self.dim_bias = dim_bias

        # initializers
        self.weight_initializer_name = weight_initializer
        if weight_initializer == "xavier":
            self.weight_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=seed, dtype=tf_datatype)
            #self.bias_initializer = tf.zeros
            self.bias_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=seed, dtype=tf_datatype)
        elif weight_initializer == "truncated_normal":
            self.weight_initializer = tf.truncated_normal
            self.bias_initializer = tf.zeros

        self.seed = seed
        #IPython.embed() # check seed
        #self.xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=seed, dtype=tf_datatype) ## 
        self.mean_initializer = np.zeros
        self.std_initializer = np.ones
        self.is_rnn = False
        self.is_oneshot = False

        # placeholders and variables
        self.tf_datatype = tf_datatype
        self.inputs = tf.placeholder(tf.float32, shape=(None, None), name='test_inputs')
        self._update_params = self._update_params_data_dist(dim_input, dim_output)
        self.weights, self._inputs, self._output, self.output = None, None, None, None
        self._weights = None

        self.norm = norm

        if nonlinearity == "relu":
            self.activation = tf.nn.relu
        elif nonlinearity == "tanh":
            self.activation = tf.nn.tanh
        elif nonlinearity == "leaky_relu":
            self.activation = tf.nn.leaky_relu
        elif nonlinearity == "elu":
            self.activation = tf.nn.elu
        elif nonlinearity == "sigmoid":
            self.activation = tf.nn.sigmoid

    def construct_fc_weights(self, meta_loss=False):
        self.weights = self._construct_fc_weights(meta_loss=meta_loss)
        # define the forward pass
        self._inputs = (self.inputs - self._x_mean_var) / self._x_std_var
        self._output = self.forward_fc(self._inputs, self.weights)
        self.output = self._output * self._y_std_var + self._y_mean_var
        self._weights = dict([(k, tf.placeholder(tf.float32, name=('ph_'+k))) for k in self.weights.keys()])
        self._set_weights()

    def _construct_fc_weights(self, meta_loss):
        weights = dict()
        if self.dim_bias > 0:
            weights['bias'] = tf.Variable(self.bias_initializer([self.dim_bias]), name = 'bias')
            #weights['bias'] = tf.Variable(self.xavier_initializer([self.dim_bias]), name='bias')

        # the 1st hidden layer
        if self.weight_initializer_name == "truncated_normal":
            weights['W0'] = tf.Variable(self.weight_initializer([self.dim_input + self.dim_bias, self.dim_hidden[0]], stddev=0.01, seed=self.seed), name = 'W0')
        elif self.weight_initializer_name == "xavier":
            weights['W0'] = tf.Variable(self.weight_initializer([self.dim_input + self.dim_bias, self.dim_hidden[0]]), name='W0')
        weights['b0'] = tf.Variable(self.bias_initializer([self.dim_hidden[0]]), name = 'b0')
        #weights['W0'] = tf.Variable(self.xavier_initializer([self.dim_input + self.dim_bias, self.dim_hidden[0]]), name='W0')
        #weights['b0'] = tf.Variable(self.xavier_initializer([self.dim_hidden[0]]), name='b0')

        # intermediate hidden layers
        for i in range(0, len(self.dim_hidden)):
            if self.weight_initializer_name == "truncated_normal":
                weights['W' + str(i+1)] = tf.Variable(self.weight_initializer([self.dim_hidden[i], self.dim_hidden[i]], stddev=0.01, seed=self.seed), name='W'+str(i+1))
            elif self.weight_initializer_name == "xavier":
                weights['W' + str(i+1)] = tf.Variable(self.weight_initializer([self.dim_hidden[i], self.dim_hidden[i]]), name='W'+str(i+1))
            weights['b' + str(i+1)] = tf.Variable(self.bias_initializer([self.dim_hidden[i]]), name='b'+str(i+1))
            #weights['W' + str(i)] = tf.Variable(self.xavier_initializer([self.dim_hidden[i - 1], self.dim_hidden[i]]), name='W'+str(i))
            #weights['b' + str(i)] = tf.Variable(self.xavier_initializer([self.dim_hidden[i]]), name='b'+str(i))

        if self.weight_initializer_name == "truncated_normal":
            weights['W' + str(len(self.dim_hidden)+1)] = tf.Variable(self.weight_initializer([self.dim_hidden[-1], self.dim_output], stddev=0.01, seed=self.seed),  name='W' + str(len(self.dim_hidden)+1))
        elif self.weight_initializer_name == "xavier":
            weights['W' + str(len(self.dim_hidden)+1)] = tf.Variable(self.weight_initializer([self.dim_hidden[-1], self.dim_output]),  name='W' + str(len(self.dim_hidden)+1))
        weights['b' + str(len(self.dim_hidden)+1)] = tf.Variable(self.bias_initializer([self.dim_output]), name='b' + str(len(self.dim_hidden)+1))
        #weights['W' + str(len(self.dim_hidden))] = tf.Variable(self.xavier_initializer([self.dim_hidden[-1], self.dim_output]), name='W' + str(len(self.dim_hidden)))
        #weights['b' + str(len(self.dim_hidden))] = tf.Variable(self.xavier_initializer([self.dim_output]),name='b' + str(len(self.dim_hidden)))
        
        if meta_loss:
            for i in range(len(self.dim_conv1d)):
                if self.weight_initializer_name == "truncated_normal":
                    weights['W_1d_conv' + str(i)] = tf.Variable(self.weight_initializer([self.dim_conv1d[i], self.dim_output, self.dim_output], stddev=0.01, seed=self.seed), name='W_1d_conv' + str(i))
                elif self.weight_initializer_name == "xavier":
                    weights['W_1d_conv' + str(i)] = tf.Variable(self.weight_initializer([self.dim_conv1d[i], self.dim_output, self.dim_output]), name='W_1d_conv' + str(i))
                weights['b_1d_conv' + str(i)] = tf.Variable(self.bias_initializer([self.dim_output]), name='b_1d_conv' + str(i))
                #weights['W_1d_conv' + str(i)] = tf.Variable(self.xavier_initializer([self.dim_conv1d[i], self.dim_output, self.dim_output]), name='W_1d_conv' + str(i))
                #weights['b_1d_conv' + str(i)] = tf.Variable(self.xavier_initializer([self.dim_output]), name='b_1d_conv' + str(i))
        #IPython.embed()
        print(weights)
        return weights

    def forward_fc(self, inp, weights, reuse=False, meta_loss=False):
        # pass through intermediate hidden layers
        if self.dim_bias > 0:
            n = tf.shape(inp)[0]
            bias = tf.tile(tf.expand_dims(weights['bias'], 0), (tf.to_int32(n), 1))
            hidden = tf.concat([inp, bias], axis=-1)
        else:
            hidden = inp
        for i in range(0, len(self.dim_hidden)+1):
            hidden = normalize(tf.matmul(hidden, weights['W' + str(i)]) + weights['b' + str(i)],
                               activation=self.activation, reuse=reuse, scope=str(i), norm=self.norm)

        # pass through output layer
        #IPython.embed()
        out = tf.matmul(hidden, weights['W' + str(len(self.dim_hidden)+1)]) + weights[
            'b' + str(len(self.dim_hidden)+1)]
        if meta_loss:
            out = tf.reshape(out, [-1, out.get_shape().dims[-2].value, self.dim_output])
            for i in range(len(self.dim_conv1d) - 1):
                out = normalize(tf.nn.conv1d(out,  weights['W_1d_conv' + str(i)], 1,
                                'SAME', data_format='NHWC', name='conv1d'+str(i), use_cudnn_on_gpu=True) +
                                weights['b_1d_conv'+str(i)], activation=self.activation,
                                reuse=reuse, scope='conv1d'+str(i), norm=self.norm)
            out = tf.nn.conv1d(out,  weights['W_1d_conv' + str(len(self.dim_conv1d)-1)], 1,
                               'SAME', data_format='NHWC', name='conv1d'+str(len(self.dim_conv1d)-1), use_cudnn_on_gpu=True) +\
                  weights['b_1d_conv'+str(len(self.dim_conv1d)-1)]
        return out

    # run forward pass of NN using the given inputs
        #ANY FUNCTION that calls this one should give it
        #inputs of dims of what the NN was trained on
    def predict(self, *input_vals):
        sess = tf.get_default_session()
        feed_dict = dict(list(zip([self.inputs], input_vals)))
        return sess.run(self.output, feed_dict=feed_dict)

    def do_forward_sim(self, states, actions, state_representation):
        #SO FAR, this func only ever gets called on a single trajectory
        traj=[]
        obs = np.expand_dims(states[0], axis=0)
        traj.append(np.squeeze(obs))
        for h in range(len(states)):
            use_this = create_nn_input_using_staterep(obs, state_representation, multiple=True) #its single not multiple, but shape is [1x?] so need obs[:,2:]
            obs = self.predict(np.concatenate([use_this, np.expand_dims(actions[h], axis=0)], axis=1)) + obs
                #[1x s] + [1x a] --> [1x (s+a)]
                #output of predic = [1x s]
            traj.append(np.squeeze(obs))
        return traj

    # evaluate and return the NN's weights and mean/std vars
    def get_params(self):
        sess = tf.get_default_session()
        return sess.run(self.weights)

    def _set_weights(self):
        self.set_weights = [tf.assign(self.weights[k], self._weights[k]) for k in self.weights.keys()]

    # update the NN's weights and mean/std vars
    def set_params(self, params):
        sess = tf.get_default_session()
        weights = params
        feed_dict = dict([(self._weights[k], weights[k]) for k in weights.keys()])
        _ = sess.run(self.set_weights, feed_dict=feed_dict)

    # helper function to update the NN's mean/std vars
    def update_params_data_dist(self, x_mean=None, x_std=None, y_mean=None, y_std=None, nb=0):
        x_mean = np.zeros(1, self.dim_input) if x_mean is None else x_mean
        x_std = np.ones(1, self.dim_input) if x_std is None else x_std
        y_mean = np.zeros(1, self.dim_output) if y_mean is None else y_mean
        y_std = np.ones(1, self.dim_output) if y_std is None else y_std
        feed_dict = {self._x_mean_ph: x_mean,
                     self._x_std_ph: x_std,
                     self._y_mean_ph: y_mean,
                     self._y_std_ph: y_std,
                     self._nb_ph: nb}
        sess = tf.get_default_session()
        _ = sess.run(self._update_params, feed_dict=feed_dict)

    # initialize the NN's mean/std vars
    # and provide assign commands to be able to update them when desired
    def _update_params_data_dist(self, dim_input, dim_output):
        if self.multi_input:
         multi = self.multi_input
        else:
          multi = 1
        self._x_mean_var = tf.Variable(
            self.mean_initializer((1, multi * dim_input)), dtype=np.float32,
            trainable=False,
            name="x_mean",
        )
        self._x_std_var = tf.Variable(
            self.std_initializer((1, multi * dim_input)), dtype=np.float32,
            trainable=False,
            name="x_std",
        )
        self._y_mean_var = tf.Variable(
            self.mean_initializer((1, dim_output)), dtype=np.float32,
            trainable=False,
            name="y_mean",
        )
        self._y_std_var = tf.Variable(
            self.std_initializer((1, dim_output)), dtype=np.float32,
            trainable=False,
            name="y_std",
        )
        self._nt = tf.Variable(0., dtype=np.float32,
                               trainable=False,
                               name='nt')

        self._x_mean_ph = tf.placeholder(tf.float32,
                                         shape=(1, dim_input),
                                         name="x_mean_ph",
                                         )
        self._x_std_ph = tf.placeholder(tf.float32,
                                        shape=(1, dim_input),
                                        name="x_std_ph",
                                        )
        self._y_mean_ph = tf.placeholder(tf.float32,
                                         shape=(1, dim_output),
                                         name="y_mean_ph",
                                         )
        self._y_std_ph = tf.placeholder(tf.float32,
                                        shape=(1, dim_output),
                                        name="y_std_ph",
                                        )
        self._nb_ph = tf.placeholder(tf.float32, shape=(), name="nb")
        running_mean = lambda curr_value, new_value: (curr_value * self._nt
                                                      + new_value * self._nb_ph)/tf.add(self._nt, self._nb_ph)
        if self.multi_input:
            _update_params = [
              tf.assign(self._x_mean_var, running_mean(self._x_mean_var, tf.tile(self._x_mean_ph, (1, self.multi_input)))),
              tf.assign(self._x_std_var, running_mean(self._x_std_var, tf.tile(self._x_std_ph, (1, self.multi_input)))),
              tf.assign(self._y_mean_var, running_mean(self._y_mean_var, self._y_mean_ph)),
              tf.assign(self._y_std_var, running_mean(self._y_std_var, self._y_std_ph)),
              tf.assign(self._nt, tf.add(self._nt, self._nb_ph)),]
        else:
            _update_params = [
                tf.assign(self._x_mean_var, running_mean(self._x_mean_var, self._x_mean_ph)),
                tf.assign(self._x_std_var, running_mean(self._x_std_var, self._x_std_ph)),
                tf.assign(self._y_mean_var, running_mean(self._y_mean_var, self._y_mean_ph)),
                tf.assign(self._y_std_var, running_mean(self._y_std_var, self._y_std_ph)),
                tf.assign(self._nt, tf.add(self._nt, self._nb_ph)),]
        return _update_params


