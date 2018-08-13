""" Code for the MAML algorithm and network definitions. """
import numpy as np
import tensorflow as tf
from sandbox.ignasi.maml.utils import mse
import IPython


class MAML:
    def __init__(self, regressor, dim_input=1, dim_output=1, config={}):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = config['update_lr']
        self.meta_lr = tf.placeholder_with_default(config['meta_lr'], ())
        self.num_updates = config['num_updates']
        self.loss_func = mse
        self.regressor = regressor
        self.regressor.construct_fc_weights(meta_loss=config['meta_loss'])
        self.forward = self.regressor.forward_fc
        self.config = config
        self.multistep_loss = config['multistep_loss']
        self.multi_updates = config.get('multi_updates', 1)
        self.meta_learn_lr = config.get('meta_learn_lr', False)
        assert self.multi_updates > 0

    def construct_model(self, input_tensors=None, prefix='metatrain_'):

        #placeholders to hold the inputs/outputs
            # a: training data for inner gradient
            # b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32,
                          shape=(None, self.config['update_batch_size'], self.regressor.dim_input))
            self.inputb = tf.placeholder(tf.float32,
                          shape=(None, self.config['update_batch_size'], self.regressor.dim_input))
            self.labela = tf.placeholder(tf.float32,
                          shape=(None, self.config['update_batch_size'], self.regressor.dim_output))
            self.labelb = tf.placeholder(tf.float32,
                          shape=(None, self.config['update_batch_size'], self.regressor.dim_output))
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

            if self.meta_learn_lr:
                self.update_lr = dict([(k, tf.Variable(self.update_lr * tf.ones(tf.shape(v)),
                                                       name='lr_'+k)) for k,v in weights.items()])

            #########################################################
            ## Function to perform the inner gradient calculations
            #########################################################

            #this func is for a single task (we call this for each task)
            def task_metalearn(inp, reuse=True):

                # init vars
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                fast_weights = weights

                ########################################################################
                ### calculate theta' using k consecutive points... from a single task
                ########################################################################

                # calculate output of f_weights{0}(inputa)
                k = self.config['update_batch_size']//self.multi_updates
                for i in range(self.multi_updates):
                    task_outputa = self.forward(inputa[i*k:(i+1)*k], fast_weights, reuse=True, meta_loss=self.config['meta_loss'])  # only reuse on the first iter
                    # calculate loss(f_weights{0}(inputa))
                    task_lossa = self.loss_func(task_outputa, labela[i*k:(i+1)*k])

                    # weights{1} = use loss(f_weights{0}(inputa)) to take a gradient step on weights{0}
                    grads = tf.gradients(task_lossa, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    if self.meta_learn_lr:
                        fast_weights = dict(
                            zip(fast_weights.keys(), [fast_weights[key] - self.update_lr[key] * gradients[key] for key in fast_weights.keys()]))
                    else:
                        fast_weights = dict(
                            zip(fast_weights.keys(), [fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))

                ########################################################################
                ### calculate L of f(theta') on k consecutive points... for single task
                ########################################################################

                # calculate output of f_weights{1}(inputb)
                output = self.forward(inputb, fast_weights, reuse=True, meta_loss=False)
                task_outputbs.append(output)

                # calculate loss(f_weights{1}(inputb))
                task_lossesb.append(self.loss_func(output, labelb))

                # if taking more inner-update gradient steps
                for j in range(num_updates - 1):
                    for i in range(self.multi_updates):
                        loss = self.loss_func(self.forward(inputa[i * k:(i + 1) * k], fast_weights, reuse=True,
                                                           meta_loss=self.config['meta_loss']), labela[i * k:(i + 1) * k])  # only reuse on the first iter

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
                    task_lossesb.append(self.loss_func(output, labelb))
                # task_outputa :    f_weights{0}(inputa)
                # task_outputbs :   [f_weights{1}(inputb), f_weights{2}(inputb), ...]
                # task_lossa :      loss(f_weights{0}(inputa))
                # task_lossesb :    [loss(f_weights{1}(inputb)), loss(f_weights{2}(inputb)), ...]
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                return task_output

            # to initialize the batch norm vars
            # might want to combine this, and not run idx 0 twice.
            if self.regressor.norm is not 'None':
                _ = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            #########################################################
            ## Output of performing inner gradient calculations
            #########################################################

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            result = tf.map_fn(task_metalearn, elems=(inputa, inputb, labela, labelb), dtype=out_dtype,
                               parallel_iterations=self.config['meta_batch_size'])

            #these return values are lists w a different entry for each task...
                #average over all tasks when doing the outer gradient update (metatrain_op)
            outputas, outputbs, lossesa, lossesb = result

        ################################################
        ## Calculate preupdate and postupdate losses
        ################################################

        # assign vars
        self.outputas, self.outputbs = outputas, outputbs

        # avg loss(f_weights{0}(inputa))
        self.total_loss1 = total_loss1 = tf.reduce_mean(lossesa)

        # [avg loss(f_weights{1}(inputb)), avg loss(f_weights{2}(inputb)), ...]
            #this is the L of f(theta') on validation set, which is used by the (metatrain_op)

        #############################################################################
        ###### YOU WILL NEED TO CHANGE THIS REGULARIZER WHEN NUM_UPDATES > 1 ########
        #############################################################################
        # The reg term needs to be specific to each num_update 
        self.regularization_weight = self.config['regularization_weight']
        self.trainable_vars = tf.trainable_variables()
        self.regularizer = self.regularization_weight*tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_vars if ('bias' not in v.name and 'b' not in v.name)])
        #self.total_losses2 = [tf.reduce_mean(lossesb[j] + self.regularizer) for j in range(num_updates)]
        self.mse_loss = [tf.reduce_mean(lossesb[j]) for j in range(num_updates)]

        self.total_losses2 = [tf.reduce_mean(lossesb[j]) for j in range(num_updates)]
        #IPython.embed()

        #########################################
        ## Define pretrain_op
        #########################################

        #UPDATE weights{0} using loss(f_weights{0}(inputa))
        #standard supervised learning

        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
        k = self.config['update_batch_size'] // self.multi_updates
        fast_weights = weights
        inputa = inputa[0]
        labela = labela[0]
        for i in range(self.multi_updates):
            task_outputa = self.forward(inputa[i*k:(i+1)*k], fast_weights, reuse=True, meta_loss=self.config['meta_loss'])
            # calculate loss(f_weights{0}(inputa))
            task_lossa = self.loss_func(task_outputa, labela[i*k:(i+1)*k])
            # weights{1} = use loss(f_weights{0}(inputa)) to take a gradient step on weights{0}
            grads = tf.gradients(task_lossa, list(fast_weights.values()))
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

        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[self.config['num_updates']-1]) #this is real maml loss func (ie adapt)
        self.gvs = gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
        self.metatrain_op = optimizer.apply_gradients(gvs)

        ##################### #this is just one gradient step (ie normal training)
        ##################### #used for debugging other parts of code
        #####################self.gvs = gvs = optimizer.compute_gradients(self.total_loss1) 

        ##############################################
        ## Summaries
        ##############################################

        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), self.total_losses2[j])


