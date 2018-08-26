import numpy as np
import numpy.random as npr
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

def train(inputs_full, outputs_full, curr_agg_iter, model, saver, sess, config, inputs_full_val, outputs_full_val, inputs_full_val_onPol, outputs_full_val_onPol, save_path):

  #get certain sections of vars from config file
  log_config = config['logging']
  train_config = config['training']
  batchsize = config['sampler']['batch_size']
  meta_bs = train_config['meta_batch_size']
  update_bs = train_config['update_batch_size']
  curr_agg_iter = config['aggregation']['curr_agg_iter']

  #init vars
  t0 = time.time()
  prelosses, postlosses,vallosses, vallosses_onPol, metatrain_losses_withReg, metatrain_losses = [], [], [], [], [], []
  multitask_weights, reg_weights = [], []
  total_points_per_task = len(inputs_full[0])*len(inputs_full[0][0]) #numRollouts * pointsPerRollout
  
  all_indices = []
  for taski in range(len(inputs_full)):
    for rollouti in range(len(inputs_full[taski])):
      all_rollout_indices = [(taski, rollouti, x) for x in range(len(inputs_full[taski][rollouti]) - update_bs*2 + 1)]
      all_indices.extend(all_rollout_indices)

  all_indices = np.array(all_indices)
  #IPython.embed() # check all_indices, rand_indices shape, look inside inputs_full to see if neglected tasks occupy empty space
  
  #writer for log
  #path = save_path[:-len(save_path.split('/')[-1])]
  path = logger.get_snapshot_dir() ########################################################################################################################## remember to comment this out
  
  #IPython.embed()
  if log_config['log']:
      train_writer = tf.summary.FileWriter(path, sess.graph)

  #save all of the used config params to a file in the save directory
  yaml.dump(config, open(osp.join(path, 'saved_config.yaml'), 'w'))

  #do metatraining
  all_means_ratio=[]
  all_means_grads=[]
  all_means_weights=[]
  which_epochs=[]

  # Before training, save the weight and bias initialization in numpy form
  all_random_indices = np.empty((0, len(all_indices)))

  # DELETE AFTER 3:00 PM SATURDAY
  trainable_vars = sess.run(tf.trainable_variables())
  pickle.dump(trainable_vars, open(osp.join(path, 'weight_restore_before_training.yaml'), 'w'))


  for training_epoch in range(config['sampler']['max_epochs']):
    random_indices = npr.choice(len(all_indices), size=(len(all_indices)), replace=False) # shape: (x,)

    all_random_indices = np.append(all_random_indices, np.expand_dims(random_indices, axis=0), axis=0)

    print()
    print()
    print("************* TRAINING EPOCH: ", training_epoch)
    gradient_step=0
    while((gradient_step*meta_bs*update_bs) < total_points_per_task*4): # I feel like this should be different....

      ####################################################
      ## randomly select batch of data, for 1 outer-gradient step
      ####################################################

      random_indices_batch = random_indices[gradient_step*meta_bs: min((gradient_step+1)*meta_bs, len(random_indices)-1)]
      #IPython.embed()
      indices_batch = all_indices[random_indices_batch]
      
      inputs_batch = np.array([inputs_full[i[0]][i[1]][i[2]:i[2] + 2*update_bs] for i in indices_batch])
      outputs_batch = np.array([outputs_full[i[0]][i[1]][i[2]:i[2] + 2*update_bs] for i in indices_batch])

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

      if gradient_step % log_config['summary_itr'] == 0:
        prelosses.append(result[-2])
        if log_config['log']:
            train_writer.add_summary(result[1], gradient_step)
        postlosses.append(result[-1])

      if gradient_step % log_config['print_itr'] == 0:
        print_str = 'Gradient step ' + str(gradient_step)
        print_str += '   | Mean pre-losses: ' + str(np.mean(prelosses)) + '   | Mean post-losses: ' + str(np.mean(postlosses))
        print_str += '    | Time spent:   {0:.2f}'.format(time.time() - t0)
        print(print_str)
        if(train_config['use_reg']):
          print("Regularization loss: ", sess.run(model.regularizer, feed_dict))
        print("MSE loss: ", sess.run(model.mse_loss, feed_dict))

        t0 = time.time()
        prelosses, postlosses = [], []
      gradient_step += 1

    ################################
    ####### Training loss #########
    ################################

    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
    metatrain_loss, metatrain_loss_withReg = sess.run([model.mse_loss[train_config['num_updates'] - 1], model.total_losses2[train_config['num_updates'] - 1]], feed_dict)
    metatrain_losses_withReg.append(metatrain_loss_withReg)
    metatrain_losses.append(metatrain_loss)

    ################################
    ####### Validation loss ########
    ################################
    # Random
    # Want to keep ts constant across rollouts
    inputa = inputs_full_val[:, :, :update_bs] # Err ... it's on the same data everytime. WHich should be the case, no? 
    inputb = inputs_full_val[:, :, update_bs:2*update_bs]
    labela = outputs_full_val[:, :, :update_bs]
    labelb = outputs_full_val[:, :, update_bs:2*update_bs]

    inputa = np.concatenate([inputa[i] for i in range(len(inputa))])
    inputb = np.concatenate([inputb[i] for i in range(len(inputb))])
    labela = np.concatenate([labela[i] for i in range(len(labela))])
    labelb = np.concatenate([labelb[i] for i in range(len(labelb))])

    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
    val_loss = sess.run(model.mse_loss[train_config['num_updates'] - 1], feed_dict)
    vallosses.append(val_loss)

    print("Validation loss:", val_loss)

    # MPC
    if curr_agg_iter != 0:
      inputa = inputs_full_val_onPol[:, :, :update_bs] # Err ... it's on the same data everytime. WHich should be the case, no? 
      inputb = inputs_full_val_onPol[:, :, update_bs:2*update_bs]
      labela = outputs_full_val_onPol[:, :, :update_bs]
      labelb = outputs_full_val_onPol[:, :, update_bs:2*update_bs]

      inputa = np.concatenate([inputa[i] for i in range(len(inputa))])
      inputb = np.concatenate([inputb[i] for i in range(len(inputb))])
      labela = np.concatenate([labela[i] for i in range(len(labela))])
      labelb = np.concatenate([labelb[i] for i in range(len(labelb))])

      feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
      val_loss_onPol = sess.run(model.mse_loss[train_config['num_updates'] - 1], feed_dict)
      vallosses_onPol.append(val_loss_onPol)

      print("Validation loss (on policy):", val_loss_onPol)
    ################################
    ##### Save every 5 epochs ######
    ################################

    if training_epoch % 5 == 0:
      name = "model_aggIter" + str(curr_agg_iter) + "_epoch" + str(training_epoch)
      saver.save(sess,  osp.join(path, name))
      #print("SAVING TO ", osp.join(path, name))

      # Save
      np.save(path + "/all_random_indices_aggIter"+str(curr_agg_iter)+".npy", all_random_indices)
      np.save(path + "/all_indices_aggIter"+str(curr_agg_iter)+".npy", all_indices)
      np.save(path + '/debuggingLR_metatrain_losses_withReg_aggIter'+str(curr_agg_iter)+'.npy', metatrain_losses_withReg)
      np.save(path + '/debuggingLR_metatrain_losses_aggIter'+str(curr_agg_iter)+'.npy', metatrain_losses)
      np.save(path + '/debuggingLR_vallosses_aggIter'+str(curr_agg_iter)+'.npy', vallosses)
      np.save(path + '/debuggingLR_vallosses_onPol_aggIter'+str(curr_agg_iter)+'.npy', vallosses)

  ################################
  ####### Save at the end ########
  ################################
  # Save
  np.save(path + "/all_random_indices_aggIter"+str(curr_agg_iter)+".npy", all_random_indices)
  np.save(path + "/all_indices_aggIter"+str(curr_agg_iter)+".npy", all_indices)
  np.save(path + '/debuggingLR_metatrain_losses_withReg_aggIter'+str(curr_agg_iter)+'.npy', metatrain_losses_withReg)
  np.save(path + '/debuggingLR_metatrain_losses_aggIter'+str(curr_agg_iter)+'.npy', metatrain_losses)
  np.save(path + '/debuggingLR_vallosses_aggIter'+str(curr_agg_iter)+'.npy', vallosses)
  np.save(path + '/debuggingLR_vallosses_onPol_aggIter'+str(curr_agg_iter)+'.npy', vallosses)

  # Make validation plot
  x = np.arange(0, config['sampler']['max_epochs'])
  plt.figure()
  plt.plot(x, metatrain_losses_withReg, color ='r', label="metatrain_losses_withReg")
  plt.plot(x, metatrain_losses, 'r--', label="metatrain_losses")
  plt.plot(x, vallosses, color='g', label="Validation, random")
  if curr_agg_iter != 0:
    plt.plot(x, vallosses_onPol, color='b', label="Validation, on policy")
  plt.legend(loc="upper right", prop = {"size":6})
  plt.title("MPE")
  plt.savefig(path + "/mpe_graph_aggIter"+str(curr_agg_iter)+".png")
  plt.clf()
  #plt.show()