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

def train(inputs_full, outputs_full, curr_agg_iter, model, saver, sess, config):

  #get certain sections of vars from config file
  log_config = config['logging']
  train_config = config['training']
  batchsize = config['sampler']['batch_size']
  meta_bs = train_config['meta_batch_size']
  update_bs = train_config['update_batch_size']

  #init vars
  t0 = time.time()
  prelosses, postlosses = [], []
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

      #############################
      ## logging and saving
      #############################

      if gradient_step % log_config['summary_itr'] == 0:
          prelosses.append(result[-2])
          if log_config['log']:
              train_writer.add_summary(result[1], gradient_step) ###is this a typo? should it be result[0]?
          postlosses.append(result[-1])

      if gradient_step % log_config['print_itr'] == 0:
          print_str = 'Gradient step ' + str(gradient_step)
          print_str += '   | Mean pre-losses: ' + str(np.mean(prelosses)) + '   | Mean post-losses: ' + str(np.mean(postlosses))
          print_str += '    | Time spent:   {0:.2f}'.format(time.time() - t0)
          print(print_str)
          t0 = time.time()
          prelosses, postlosses = [], []
      gradient_step += 1

  #save dynamics model
  name = 'model' + str(curr_agg_iter)
  print('Saving model')
  saver.save(sess,  osp.join(path, name))