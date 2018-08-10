import numpy as np
import tensorflow as tf
import random
import pickle
import csv
import time
import os.path as osp
import os
import yaml
import rllab.misc.logger as logger
import IPython
from utils import *
import matplotlib.pyplot as plt

def train(inputs_full, outputs_full, curr_agg_iter, model, saver, sess, config, inputs_full_val, outputs_full_val):

  #get certain sections of vars from config file
  log_config = config['logging']
  train_config = config['training']
  batchsize = config['sampler']['batch_size']
  meta_bs = train_config['meta_batch_size']
  update_bs = train_config['update_batch_size']

  #init vars
  t0 = time.time()
  prelosses, postlosses,vallosses, traininglosses = [], [], [], []
  multitask_weights, reg_weights = [], []
  total_points_per_task = len(inputs_full[0])*len(inputs_full[0][0]) #numRollouts * pointsPerRollout

  #writer for log
  path = logger.get_snapshot_dir() 
  if log_config['log']:
      train_writer = tf.summary.FileWriter(path, sess.graph)

  #save all of the used config params to a file in the save directory
  yaml.dump(config, open(osp.join(path, 'saved_config.yaml'), 'w'))

  #do metatraining
  for training_epoch in range(config['sampler']['max_epochs']):

    print("\n\n************* TRAINING EPOCH: ", training_epoch)
    gradient_step=0
    while(gradient_step*update_bs < total_points_per_task):

      ####################################################
      ## randomly select batch of data, for 1 outer-gradient step
      ####################################################

      #full data is (many tasks, rollouts from that task, each rollout has acs/obs/etc.)
      #batch data should be (metaBS, 2K, dim)

      idxs_task=[]
      idxs_path=[]
      idxs_ts=[]
      for task_num in range(meta_bs):
        #which task
        which_task = np.random.randint(0,  len(inputs_full))
        idxs_task.append(which_task)
        #which rollout, from that task
        which_path = np.random.randint(0, len(inputs_full[which_task]))
        idxs_path.append(which_path)
        #which 2k consective datapoints, from that rollout
        i = np.random.randint(2*update_bs, len(inputs_full[which_task][which_path]))
        idxs_ts.append(np.arange(i - 2*update_bs, i))
      outputs_batch = np.array([outputs_full[task][path_num][ts] for task, path_num, ts in zip(idxs_task, idxs_path, idxs_ts)])
      inputs_batch = np.array([inputs_full[task][path_num][ts] for task, path_num, ts in zip(idxs_task, idxs_path, idxs_ts)])

      #use the 1st half as training data for inner-loop gradient
      inputa = inputs_batch[:, :update_bs, :]
      labela = outputs_batch[:, :update_bs, :]
      #use the 2nd half as test data for outer-loop gradient
      inputb = inputs_batch[:, update_bs:, :]
      labelb = outputs_batch[:, update_bs:, :]

      #############################
      ## run meta-training iteration
      #############################

      #each inner update is done on update_bs (k) points (to calculate each theta')
      #an outer update is done using metaBS*k points

      #which tensors to populate/execute during the sess.run call
      input_tensors = [model.metatrain_op]
      if gradient_step % log_config['print_itr'] == 0 or gradient_step % log_config['summary_itr'] == 0:
          input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[train_config['num_updates'] - 1]])

      #make the sess.run call to perform one metatraining iteration
      feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
      result = sess.run(input_tensors, feed_dict)
      #IPython.embed()
      #############################
      ## logging and saving
      #############################
      #print(result)
      if gradient_step % log_config['summary_itr'] == 0:
          prelosses.append(result[-2])
          if log_config['log']:
              train_writer.add_summary(result[1], gradient_step) ###is this a typo? should it be result[0]?
          postlosses.append(result[-1])
          #IPython.embed()

      if gradient_step % log_config['print_itr'] == 0:
          print_str = 'Gradient step ' + str(gradient_step)
          print_str += '   | Mean pre-losses: ' + str(np.mean(prelosses)) + '   | Mean post-losses: ' + str(np.mean(postlosses))
          print_str += '    | Time spent:   {0:.2f}'.format(time.time() - t0)
          print(print_str)
          t0 = time.time()
          prelosses, postlosses = [], []
      gradient_step += 1

    ####### Training loss #########
    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
    train_loss = sess.run(model.total_losses2[train_config['num_updates'] - 1], feed_dict)
    traininglosses.append(train_loss)

    ####### Validation loss ########
    ################################

    # Want to keep ts constant across rollouts
    inputa = inputs_full_val[:, :, :update_bs]
    inputb = inputs_full_val[:, :, update_bs:2*update_bs]
    labela = outputs_full_val[:, :, :update_bs]
    labelb = outputs_full_val[:, :, update_bs:2*update_bs]

    inputa = np.concatenate([inputa[i] for i in range(len(inputa))])
    inputb = np.concatenate([inputb[i] for i in range(len(inputb))])
    labela = np.concatenate([labela[i] for i in range(len(labela))])
    labelb = np.concatenate([labelb[i] for i in range(len(labelb))])

    #IPython.embed()

    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
    val_loss = sess.run(model.total_losses2[train_config['num_updates'] - 1], feed_dict)
    vallosses.append(val_loss)

    

  #save dynamics model
  name = 'model' + str(curr_agg_iter)
  print('Saving model at: ', osp.join(path, name))
  #IPython.embed()
  saver.save(sess,  osp.join(path, name))

  # Make validation plot
  x = np.arange(0, config['sampler']['max_epochs'])

  IPython.embed()
  plt.plot(x, traininglosses, color ='r', label="Training")
  plt.plot(x, vallosses, color='g', label="Validation, random")
  # plt.plot(x, v_mpc_loss_list, color='b', label="Validation, MPC")
  # plt.plot(x, v_rand_xy_loss_list, color='g', linestyle="--", label="Validation, random, xy")
  # plt.plot(x, v_mpc_xy_loss_list, color='b', linestyle="--", label="Validation, MPC, xy")
  plt.legend(loc="upper right", prop = {"size":6})
  plt.title("MPE")
  plt.savefig(osp.join(path, name) + "_mpe_graph.png")
  plt.show()