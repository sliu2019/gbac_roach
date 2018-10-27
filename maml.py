""" Code for the MAML algorithm and network definitions. """
import numpy as np
import tensorflow as tf
from sandbox.ignasi.maml.utils import mse
import IPython


class MAML:
    def __init__(self, regressor, dim_input, dim_output, config):
        """ must call construct_model() after initializing MAML! """
        self.config = config
        self.train_config = config["training"]
        self.test_config = config["testing"]

        self.regressor = regressor
        self.regressor.construct_fc_weights(meta_loss=self.train_config['meta_loss'])
        self.forward = self.regressor.forward_fc
        
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.update_lr = self.test_config['update_lr']
        self.meta_lr = tf.placeholder_with_default(self.test_config['meta_lr'], ())
        self.num_updates = self.test_config['num_updates']
        self.num_sgd_steps = self.test_config['num_sgd_steps']
        self.k = self.test_config['update_batch_size']
        
        self.loss_func = mse
        self.regularization_weight = self.train_config['regularization_weight']
        
        self.meta_learn_lr = config["model"]["meta_learn_lr"]        
        self.learn_loss_weighting = config["model"]["learn_loss_weighting"]
        

    def construct_model(self, input_tensors=None, prefix='metatrain_'):

        #placeholders to hold the inputs/outputs
            # a: training data for inner gradient
            # b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(self.regressor.tf_datatype,
                          shape=(None, (self.test_config['update_batch_size']+self.num_sgd_steps-1), self.regressor.dim_input))
            self.inputb = tf.placeholder(self.regressor.tf_datatype,
                          shape=(None, self.test_config['update_batch_size'], self.regressor.dim_input))
            self.labela = tf.placeholder(self.regressor.tf_datatype,
                          shape=(None, (self.test_config['update_batch_size']+self.num_sgd_steps-1), self.regressor.dim_output))
            self.labelb = tf.placeholder(self.regressor.tf_datatype,
                          shape=(None, self.test_config['update_batch_size'], self.regressor.dim_output))
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        #placeholders to hold the preprocessed inputs/outputs (mean 0, std 1)
        inputa = (self.inputa - self.regressor._x_mean_var)/self.regressor._x_std_var
        inputb = (self.inputb - self.regressor._x_mean_var)/self.regressor._x_std_var
        labela = (self.labela - self.regressor._y_mean_var)/self.regressor._y_std_var
        labelb = (self.labelb - self.regressor._y_mean_var)/self.regressor._y_std_var

        with tf.variable_scope('model', reuse=None) as training_scope:

            #init vars
            num_updates = self.num_updates

            #define the weights to be the regressor weights... these are weights{0}
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.regressor.weights

            if self.learn_loss_weighting:
                self.loss_weighting = tf.Variable(initial_value= 1.0*tf.ones((self.regressor.dim_output,)), name='lossweighting', trainable=True)
            else:
                self.loss_weighting = None

            if self.meta_learn_lr:
                self.update_lr = dict([(k, tf.Variable(initial_value=self.update_lr * tf.ones([1]), name='lr_'+k, trainable=True)) for k,v in weights.items()])

            #########################################################
            ## Function to perform the inner gradient calculations
            #########################################################

            #this func is for a single task (we call this for each task)
            def task_metalearn(inp, reuse=True):

                # init vars
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                fast_weights = weights
                weights_norms = []

                ########################################################################
                ### calculate theta' using k pts (from a single task)
                ########################################################################
                
                ## loop where you do theta_i+1 = L(f_theta_i(inputa_chunk_i)), with chunks progressing by 1 point each time
                for i in range(self.num_sgd_steps):
                    
                    #f_theta_i(inputa_chunk_i)
                    task_outputa = self.forward(inputa[i:i + self.k], fast_weights, reuse=True, meta_loss=self.config['meta_loss'])  # only reuse on the first iter
                    #L(f_theta_i(inputa_chunk_i))
                    task_lossa = self.loss_func(task_outputa, labela[i:i+self.k], self.loss_weighting)

                    # weights{1} = use loss(f_weights{0}(inputa)) to take a gradient step on weights{0}
                    grads = tf.gradients(task_lossa, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    if self.meta_learn_lr:
                        theta_prime = fast_weights = dict(
                            zip(fast_weights.keys(), [fast_weights[key] - self.update_lr[key] * gradients[key] for key in fast_weights.keys()]))
                    else:
                        theta_prime = fast_weights = dict(
                            zip(fast_weights.keys(), [fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))

                ########################################################################
                ### calculate L of f(theta') on k pts (from a single task)
                ########################################################################

                # calculate output of f_weights{1}(inputb)
                
                output = self.forward(inputb, fast_weights, reuse=True, meta_loss=False)
                task_outputbs.append(output)

                # calculate loss(f_weights{1}(inputb))
                task_lossesb.append(self.loss_func(output, labelb, self.loss_weighting))

                #regularizer
                trainable_vars = tf.trainable_variables()
                non_bias_weights = [tf.nn.l2_loss(v) for v in trainable_vars if ('bias' not in v.name and 'b' not in v.name)]
                regularizer = self.regularization_weight*tf.add_n(non_bias_weights)
                weights_norms.append(regularizer)

                # if taking more inner-update gradient steps
                for j in range(num_updates - 1):
                    for i in range(self.num_sgd_steps):
                        loss = self.loss_func(self.forward(inputa[i:i+self.k], fast_weights, reuse=True, meta_loss=self.train_config['meta_loss']), labela[i:i+self.k], self.loss_weighting)

                        # weights{1} = use loss(f_weights{0}(inputa)) to take a gradient step on weights{0}
                        grads = tf.gradients(loss, list(fast_weights.values()))
                        gradients = dict(zip(fast_weights.keys(), grads))
                        if self.meta_learn_lr:
                            fast_weights = dict(
                                zip(fast_weights.keys(), [fast_weights[key] - self.update_lr[key] * gradients[key] for key in fast_weights.keys()]))
                        else:
                            fast_weights = dict(
                                zip(fast_weights.keys(), [fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))
                    # use loss(f_weights{i}(inputa)) to calculate f_weights{i+1}
                    #calculate f_weights{i+1}(inputb)
                    
                    output = self.forward(inputb, fast_weights, reuse=True, meta_loss=False)

                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb, self.loss_weighting))

                    trainable_vars = tf.trainable_variables()
                    non_bias_weights = [tf.nn.l2_loss(v) for v in trainable_vars if ('bias' not in v.name and 'b' not in v.name)]
                    regularizer = self.regularization_weight*tf.add_n(non_bias_weights)
                    weights_norms.append(regularizer)
                # task_outputa :    f_weights{0}(inputa)
                # task_outputbs :   [f_weights{1}(inputb), f_weights{2}(inputb), ...]
                # task_lossa :      loss(f_weights{0}(inputa))
                # task_lossesb :    [loss(f_weights{1}(inputb)), loss(f_weights{2}(inputb)), ...]

                ####################task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                list_of_theta = [self.weights['W0'],self.weights['W1'],self.weights['W2'],self.weights['b0'],self.weights['b1'],self.weights['b2']]
                list_of_gradient = [gradients['W0'],gradients['W1'],gradients['W2'],gradients['b0'],gradients['b1'],gradients['b2']]
                list_of_thetaPrime = [theta_prime['W0'],theta_prime['W1'],theta_prime['W2'],theta_prime['b0'],theta_prime['b1'],theta_prime['b2']]
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb, weights_norms, list_of_gradient] ###############, list_of_theta, self.update_lr, list_of_gradient, list_of_thetaPrime, weights_norms]
                return task_output

            # to initialize the batch norm vars
            # might want to combine this, and not run idx 0 twice.
            if self.regressor.norm is not 'None':
                _ = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            #########################################################
            ## Output of performing inner gradient calculations
            #########################################################

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates, [self.regressor.tf_datatype]*num_updates, [self.regressor.tf_datatype]*6]
            ####################out_dtype = [self.regressor.tf_datatype, [self.regressor.tf_datatype]*num_updates, self.regressor.tf_datatype, [self.regressor.tf_datatype]*num_updates, [self.regressor.tf_datatype]*6, self.regressor.tf_datatype, [self.regressor.tf_datatype]*6, [self.regressor.tf_datatype]*6, [self.regressor.tf_datatype]*num_updates]
            #out_dtype = [tf.int32, [tf.int32]*num_updates, tf.int32, [tf.int32]*num_updates, [tf.int32]*7, tf.int32, [tf.int32]*7, [tf.int32]*7]
            result = tf.map_fn(task_metalearn, elems=(inputa, inputb, labela, labelb), dtype=out_dtype, parallel_iterations=self.train_config['meta_batch_size'])

            #these return values are lists w a different entry for each task...
                #average over all tasks when doing the outer gradient update (metatrain_op)
            outputas, outputbs, lossesa, lossesb, norms_of_weights, list_of_gradients = result
            self.check_outputbs = outputbs
            self.norms_of_weights = norms_of_weights
            self.lossesb = lossesb
            self.lossesa = lossesa

            self.list_of_gradients = list_of_gradients

        ################################################
        ## Calculate preupdate and postupdate losses
        ################################################

        # assign vars
        self.outputas, self.outputbs = outputas, outputbs

        # avg loss(f_weights{0}(inputa))
        self.total_loss1 = tf.reduce_mean(lossesa)

        #############################################################################
        ###### YOU WILL NEED TO CHANGE THIS REGULARIZER WHEN NUM_UPDATES > 1 ########
        #############################################################################

        # The reg term needs to be specific to each num_update 

        if self.train_config['use_reg']:
            self.total_losses2 = [tf.reduce_mean(lossesb[j] + self.norms_of_weights[j]) for j in range(num_updates)]
        else: 
            self.total_losses2 = [tf.reduce_mean(lossesb[j]) for j in range(num_updates)]

        
        self.mse_loss = [tf.reduce_mean(lossesb[j]) for j in range(num_updates)]

        self.regularizer = tf.reduce_mean(self.norms_of_weights[num_updates-1])

        #########################################
        ## Define pretrain_op
        #########################################

        #UPDATE weights{0} using loss(f_weights{0}(inputa))
        #standard supervised learning

        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_loss1)
        fast_weights = weights
        inputa = inputa[0]
        labela = labela[0]

        ##############################################################
        test_optimizer = tf.train.MomentumOptimizer(self.update_lr, self.train_config['momentum'], use_nesterov = False)
        ##############################################################

        for i in range(self.num_sgd_steps): # This used to say multi-updates???
            task_outputa = self.forward(inputa[i:i+self.k], fast_weights, reuse=True, meta_loss=self.train_config['meta_loss'])
            # calculate loss(f_weights{0}(inputa))
            task_lossa = self.loss_func(task_outputa, labela[i:i+self.k], self.loss_weighting)
            # weights{1} = use loss(f_weights{0}(inputa)) to take a gradient step on weights{0}
            
            ##############################################################
            if self.train_config['use_momentum']:
                grads = test_optimizer.compute_gradients(task_lossa, var_list = list(fast_weights.values()))
                grads = [x[0] for x in grads]
            else:
                grads = tf.gradients(task_lossa, list(fast_weights.values()))

            ##############################################################
            gradients = dict(zip(fast_weights.keys(), grads))
            
            if self.meta_learn_lr:
                fast_weights = dict(
                            zip(fast_weights.keys(), [fast_weights[key] - self.update_lr[key] * gradients[key] for key in fast_weights.keys()]))
            else:
                fast_weights = dict(
                            zip(fast_weights.keys(), [fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))

        self.test_op = [tf.assign(v, fast_weights[k]) for k, v in weights.items()]

        #########################################
        ## Define metatrain_op
        #########################################

        #UPDATE weights{0} using loss(f_weights{last_step}(inputb))
            #each inner update was done on K points (to calculate each theta')
            #this outer update is done using metaBS*K points
        if self.train_config['optimizer'] == "adam":
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
        elif self.train_config['optimizer'] == "momentum":
            optimizer = tf.train.MomentumOptimizer(self.meta_lr, 0.9)

        ##################### #this is just one gradient step (ie normal training)
        ##################### #used for debugging other parts of code
        ##############self.gvs = gvs = optimizer.compute_gradients(self.total_loss1) 
        self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[self.train_config['num_updates']-1]) #this is real maml loss func (ie adapt)
        if self.train_config['use_clip']:
            self.gvs = gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]

        self.metatrain_op = optimizer.apply_gradients(gvs)

        ##############################################
        ## Summaries
        ##############################################

        tf.summary.scalar(prefix+'Pre-update loss', self.total_loss1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), self.total_losses2[j])


