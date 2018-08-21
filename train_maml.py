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

def train(inputs_full, outputs_full, curr_agg_iter, model, saver, sess, config, inputs_full_val, outputs_full_val):

  #get certain sections of vars from config file
  log_config = config['logging']
  train_config = config['training']
  batchsize = config['sampler']['batch_size']
  meta_bs = train_config['meta_batch_size']
  update_bs = train_config['update_batch_size']

  #init vars
  t0 = time.time()
  prelosses, postlosses,vallosses, metatrain_losses_withReg, metatrain_losses = [], [], [], [], []
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
  path = logger.get_snapshot_dir() 
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
  trainable_vars = tf.trainable_variables()
  init_values = sess.run(trainable_vars)
  with open(osp.join(path, "weight_initializations.pickle"), "wb") as output_file:
    pickle.dump(init_values, output_file)

  ###############################################################################
  # t0_init_values = pickle.load(open("/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Monday/lr_10.0_hidden_layers_2_trial_6/weight_initializations.pickle", "r"))
  # for i in range(len(t0_init_values)):
  #   if not np.array_equal(t0_init_values[i], init_values[i]):
  #     print("weight init differ")
  #     IPython.embed()
  ###############################################################################

  all_random_indices = np.empty((0, len(all_indices)))
  for training_epoch in range(config['sampler']['max_epochs']):
    random_indices = npr.choice(len(all_indices), size=(len(all_indices)), replace=False) # shape: (x,)

    all_random_indices = np.append(all_random_indices, np.expand_dims(random_indices, axis=0), axis=0)

    print("\n\n************* TRAINING EPOCH: ", training_epoch)
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

      #####################################################################################################
      """weight_index = 1
      #hard_drive_path = "/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/MAML_roach_copy/lr_10_hidden_layers_trial2"
      hard_drive_path = path
      np.save(hard_drive_path + "/inputa_" + str(gradient_step) + ".npy", inputa)

      # t0_inputa = np.load("/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Monday/lr_10.0_hidden_layers_2_trial_6/inputa_" + str(gradient_step) + ".npy")
      # if not np.array_equal(inputa, t0_inputa):
      #   print("batch inputs differ")
      #   IPython.embed()
      # Save the data batch
      trainable_vars = tf.trainable_variables()
      weights = sess.run(trainable_vars)
      np.save(hard_drive_path + "/input_weight_layer_" + str(gradient_step) + ".npy", weights[weight_index])

      # t0_input_weight_layer = np.load("/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Monday/lr_10.0_hidden_layers_2_trial_6/input_weight_layer_" + str(gradient_step) + ".npy")
      # if not np.array_equal(weights[weight_index], t0_input_weight_layer):
      #   print("input layer weight matrix differs")
      #   IPython.embed()
      # Save the weights
      inner_losses, inner_gradients, outer_losses, outer_gradients = sess.run([model.lossesa, model.gradients_of_theta_multiple, model.lossesb, model.gvs], feed_dict)
      #IPython.embed()
      np.save(hard_drive_path + "/inner_losses_" + str(gradient_step) + ".npy", inner_losses)
      np.save(hard_drive_path + "/outer_losses_" + str(gradient_step) + ".npy", outer_losses)

      np.save(hard_drive_path + "/inner_gradients_" + str(gradient_step) + ".npy", inner_gradients[0]) 
      np.save(hard_drive_path + "/outer_gradients_" + str(gradient_step) + ".npy", outer_gradients[0])


      # t0_inner_losses = np.load("/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Monday/lr_10.0_hidden_layers_2_trial_6/inner_losses_" + str(gradient_step) + ".npy")    
      # t0_outer_losses = np.load("/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Monday/lr_10.0_hidden_layers_2_trial_6/outer_losses_" + str(gradient_step) + ".npy")    
      # t0_inner_gradients = np.load("/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Monday/lr_10.0_hidden_layers_2_trial_6/inner_gradients_" + str(gradient_step) + ".npy")    
      # t0_outer_gradients = np.load("/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Monday/lr_10.0_hidden_layers_2_trial_6/outer_gradients_" + str(gradient_step) + ".npy") 

      # if not np.array_equal(t0_inner_losses, inner_losses):
      #   print("inner losses differs")
      #   IPython.embed()
      # if not np.array_equal(t0_inner_gradients, inner_gradients[0]):
      #   print("inner gradients differs")
      #   IPython.embed()
      # if not np.array_equal(t0_outer_losses, outer_losses):
      #   print("outer losses differs")
      #   IPython.embed()
      # if not np.array_equal(t0_outer_gradients, outer_gradients[0]):
      #   print("outer_gradients differs")
      #   IPython.embed()

      # Save the model.losses and shit"""
      #####################################################################################################

      result = sess.run(input_tensors, feed_dict)
      #IPython.embed()
      #############################
      ## logging and saving
      #############################
      #print(result)
      if gradient_step % log_config['summary_itr'] == 0:
          prelosses.append(result[-2])
          if log_config['log']:
              train_writer.add_summary(result[1], gradient_step)
          postlosses.append(result[-1])
          #IPython.embed()

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

      if training_epoch%5==0:
        if(gradient_step==0):
          check_outputbs, theta_multiple, update_lr_multiple, gradients_of_theta_multiple, theta_prime_multiple = sess.run([model.check_outputbs, model.theta_multiple, model.update_lr_multiple, model.gradients_of_theta_multiple, model.theta_prime_multiple], feed_dict)

          #theta
          my_weights=[]
          for index in range(len(theta_multiple)):
            my_weights.append(theta_multiple[index][0])
          #gradients
          my_gradients=[]
          for index in range(len(gradients_of_theta_multiple)):
            my_gradients.append(gradients_of_theta_multiple[index][0])
          #calculate theta'
          calculated_theta_prime_one=[]
          calculated_theta_prime_1=[]
          calculated_theta_prime_01=[]
          calculated_theta_prime_001=[]
          for index in range(len(my_weights)):
            calculated_theta_prime_one.append(my_weights[index]-1.0*my_gradients[index])
            calculated_theta_prime_1.append(my_weights[index]-0.1*my_gradients[index])
            calculated_theta_prime_01.append(my_weights[index]-0.01*my_gradients[index])
            calculated_theta_prime_001.append(my_weights[index]-0.001*my_gradients[index])
          #back to dicts for forward func
          # calculated_theta_prime_one={'W0':calculated_theta_prime_one[0], 'W1':calculated_theta_prime_one[1], 'W2':calculated_theta_prime_one[2], 'b0':calculated_theta_prime_one[3], 'b1':calculated_theta_prime_one[4], 'b2':calculated_theta_prime_one[5], 'bias':calculated_theta_prime_one[6]}
          # calculated_theta_prime_1={'W0':calculated_theta_prime_1[0], 'W1':calculated_theta_prime_1[1], 'W2':calculated_theta_prime_1[2], 'b0':calculated_theta_prime_1[3], 'b1':calculated_theta_prime_1[4], 'b2':calculated_theta_prime_1[5], 'bias':calculated_theta_prime_1[6]}
          # calculated_theta_prime_01={'W0':calculated_theta_prime_01[0], 'W1':calculated_theta_prime_01[1], 'W2':calculated_theta_prime_01[2], 'b0':calculated_theta_prime_01[3], 'b1':calculated_theta_prime_01[4], 'b2':calculated_theta_prime_01[5], 'bias':calculated_theta_prime_01[6]}
          # calculated_theta_prime_001={'W0':calculated_theta_prime_001[0], 'W1':calculated_theta_prime_001[1], 'W2':calculated_theta_prime_001[2], 'b0':calculated_theta_prime_001[3], 'b1':calculated_theta_prime_001[4], 'b2':calculated_theta_prime_001[5], 'bias':calculated_theta_prime_001[6]}
          # #want to see if updateLR affects predictions or not
          # state = np.expand_dims(inputb[0][0], axis=0)
          # prediction_one = model.forward(state, calculated_theta_prime_one)
          # prediction_1 = model.forward(state, calculated_theta_prime_1)
          # prediction_01 = model.forward(state, calculated_theta_prime_01)
          # prediction_001 = model.forward(state, calculated_theta_prime_001)

          #live: to do, check the scaling factors between weights and gradients
          #IPython.embed()
          means_for_this_epoch_ratio=[]
          means_for_this_epoch_grads=[]
          means_for_this_epoch_weights=[]
          for index in range(len(my_gradients)):
            means_for_this_epoch_ratio.append(np.mean(np.abs(np.divide(my_gradients[index], my_weights[index]))))
            means_for_this_epoch_grads.append(np.mean(np.abs(my_gradients[index])))
            means_for_this_epoch_weights.append(np.mean(np.abs(my_weights[index])))
          all_means_ratio.append(means_for_this_epoch_ratio)
          all_means_grads.append(means_for_this_epoch_grads)
          all_means_weights.append(means_for_this_epoch_weights)
          which_epochs.append(training_epoch)

          #print("\n\nDEBUGGING ULR IN TRAIN MAML...")
          #IPython.embed()

      if training_epoch==(config['sampler']['max_epochs']-1):
        if(gradient_step==0):
          all_means_arr = np.array(all_means_ratio)

          plt.title('Ration of gradients/weights for a) WO b) W1 c) W2')
          plt.subplot(311)
          plt.plot(which_epochs, all_means_arr[:,0], '.')

          #plt.title('W1: gradients/weights')
          plt.subplot(312)
          plt.plot(which_epochs, all_means_arr[:,1], '.')

          #plt.title('W2: gradients/weights')
          plt.subplot(313)
          plt.plot(which_epochs, all_means_arr[:,2], '.')
          #plt.show()
          plt.savefig(path + "/ratioGradientsToWeights_graph.png")

          np.save(path + '/debuggingLR_gradMag.npy', all_means_grads)
          np.save(path + '/debuggingLR_weightMag.npy', all_means_weights)
          np.save(path + '/debuggingLR_rationMag.npy', all_means_ratio)

      gradient_step += 1

    ####### Training loss #########
    ################################

    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
    metatrain_loss, metatrain_loss_withReg = sess.run([model.mse_loss[train_config['num_updates'] - 1], model.total_losses2[train_config['num_updates'] - 1]], feed_dict)
    metatrain_losses_withReg.append(metatrain_loss_withReg)
    metatrain_losses.append(metatrain_loss)

    ####### Validation loss ########
    ################################

    # Want to keep ts constant across rollouts
    inputa = inputs_full_val[:, :, :update_bs] # Err ... it's on the same data everytime. WHich should be the case, no? 
    inputb = inputs_full_val[:, :, update_bs:2*update_bs]
    labela = outputs_full_val[:, :, :update_bs]
    labelb = outputs_full_val[:, :, update_bs:2*update_bs]

    inputa = np.concatenate([inputa[i] for i in range(len(inputa))])
    inputb = np.concatenate([inputb[i] for i in range(len(inputb))])
    labela = np.concatenate([labela[i] for i in range(len(labela))])
    labelb = np.concatenate([labelb[i] for i in range(len(labelb))])

    #IPython.embed()

    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
    val_loss = sess.run(model.mse_loss[train_config['num_updates'] - 1], feed_dict)
    vallosses.append(val_loss)

    print("Validation loss:", val_loss)

    ###### Saving epochs #####
    if training_epoch % 5 == 0:
      #IPython.embed()
      name = "model_epoch" + str(training_epoch)
      saver.save(sess,  osp.join(path, name))

  #save dynamics model
  # name = 'model' + str(curr_agg_iter)
  # print('Saving model at: ', osp.join(path, name))
  # #IPython.embed()
  # saver.save(sess,  osp.join(path, name))

  # Saving all the random indices
  np.save(path + "/all_random_indices.npy", all_random_indices)
  np.save(path + "/all_indices.npy", all_indices)
  # Make validation plot
  x = np.arange(0, config['sampler']['max_epochs'])

  np.save(path + '/debuggingLR_metatrain_losses_withReg.npy', metatrain_losses_withReg)
  np.save(path + '/debuggingLR_metatrain_losses.npy', metatrain_losses)
  np.save(path + '/debuggingLR_vallosses.npy', vallosses)

  print("before plot")
  #IPython.embed()
  plt.figure()
  plt.plot(x, metatrain_losses_withReg, color ='r', label="metatrain_losses_withReg")
  plt.plot(x, metatrain_losses, 'r--', label="metatrain_losses")
  plt.plot(x, vallosses, color='g', label="Validation, random")
  # plt.plot(x, v_mpc_loss_list, color='b', label="Validation, MPC")
  # plt.plot(x, v_rand_xy_loss_list, color='g', linestyle="--", label="Validation, random, xy")
  # plt.plot(x, v_mpc_xy_loss_list, color='b', linestyle="--", label="Validation, MPC, xy")
  plt.legend(loc="upper right", prop = {"size":6})
  plt.title("MPE")
  plt.savefig(path + "/mpe_graph.png")
  plt.clf()
  #plt.show()