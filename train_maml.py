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

def train(inputs_full, outputs_full, curr_agg_iter, model, saver, sess, config, inputs_full_val, outputs_full_val, inputs_full_val_onPol, outputs_full_val_onPol):

    #get certain sections of vars from config file
    log_config = config['logging']
    train_config = config['training']
    batchsize = config['sampler']['batch_size']
    meta_bs = train_config['meta_batch_size']
    update_bs = train_config['update_batch_size']
    curr_agg_iter = config['aggregation']['curr_agg_iter']
    num_sgd_steps = train_config['num_sgd_steps']

    #init vars
    t0 = time.time()
    prelosses, postlosses,vallosses, vallosses_onPol, metatrain_losses_withReg, metatrain_losses = [], [], [], [], [], []
    multitask_weights, reg_weights = [], []
    
    #count total datapoints
    #make list of all possible 2k chunks took look at
    total_datapoints = 0
    all_indices = []
    for taski in range(len(inputs_full)):
        for rollouti in range(len(inputs_full[taski])):
            all_rollout_indices = [(taski, rollouti, x) for x in range(len(inputs_full[taski][rollouti]) - 2*(update_bs+num_sgd_steps-1) + 1)]
            all_indices.extend(all_rollout_indices)
            total_datapoints += len(inputs_full[taski][rollouti])
    all_indices = np.array(all_indices)

    #writer for log
    path = logger.get_snapshot_dir()
    if log_config['log']:
        train_writer = tf.summary.FileWriter(path, sess.graph)

    #save all of the used config params to a file in the save directory
    yaml.dump(config, open(osp.join(path, 'saved_config.yaml'), 'w'))

    #init more vars
    all_means_ratio=[]
    all_means_grads=[]
    all_means_weights=[]
    which_epochs=[]
    all_random_indices = np.empty((0, len(all_indices)))

    # Form the validation data (random data)
    inputa_val_offpol = []
    inputb_val_offpol = []
    labela_val_offpol = []
    labelb_val_offpol = []
    for task_num in range(len(inputs_full_val)):
        for rollout_num in range(len(inputs_full_val[task_num])):
            for i in range(len(inputs_full_val[task_num][rollout_num]) - 2*(update_bs+num_sgd_steps-1) + 1):
                inputa_val_offpol.append(inputs_full_val[task_num][rollout_num][i : i + (update_bs+num_sgd_steps-1)])
                inputb_val_offpol.append(inputs_full_val[task_num][rollout_num][i + (update_bs + num_sgd_steps -1) : i + 2*(update_bs + num_sgd_steps -1)])
                labela_val_offpol.append(outputs_full_val[task_num][rollout_num][i : i + (update_bs + num_sgd_steps - 1)])
                labelb_val_offpol.append(outputs_full_val[task_num][rollout_num][i + (update_bs + num_sgd_steps -1) : i + 2*(update_bs  + num_sgd_steps -1)])
    inputa_val_offpol = np.array(inputa_val_offpol)
    inputb_val_offpol = np.array(inputb_val_offpol)
    labela_val_offpol = np.array(labela_val_offpol)
    labelb_val_offpol = np.array(labelb_val_offpol)

    # Form the validation data (onpol data)
    inputa_val_onpol = []
    inputb_val_onpol = []
    labela_val_onpol = []
    labelb_val_onpol = []
    if curr_agg_iter != 0:
        for task_num in range(len(inputs_full_val_onPol)):
            for rollout_num in range(len(inputs_full_val_onPol[task_num])):
                for i in range(len(inputs_full_val_onPol[task_num][rollout_num]) - 2*(update_bs+num_sgd_steps-1) + 1):
                    inputa_val_onpol.append(inputs_full_val_onPol[task_num][rollout_num][i : i + (update_bs+num_sgd_steps-1)])
                    inputb_val_onpol.append(inputs_full_val_onPol[task_num][rollout_num][i + (update_bs + num_sgd_steps -1) : i + 2*(update_bs + num_sgd_steps -1)])
                    labela_val_onpol.append(outputs_full_val_onPol[task_num][rollout_num][i : i + (update_bs + num_sgd_steps - 1)])
                    labelb_val_onpol.append(outputs_full_val_onPol[task_num][rollout_num][i + (update_bs + num_sgd_steps -1) : i + 2*(update_bs  + num_sgd_steps -1)])
    inputa_val_onpol = np.array(inputa_val_onpol)
    inputb_val_onpol = np.array(inputb_val_onpol)
    labela_val_onpol = np.array(labela_val_onpol)
    labelb_val_onpol = np.array(labelb_val_onpol)

    # EPOCHS OF METATRAINING
    for training_epoch in range(config['sampler']['max_epochs']):

        random_indices = npr.choice(len(all_indices), size=(len(all_indices)), replace=False) # shape: (x,)
        all_random_indices = np.append(all_random_indices, np.expand_dims(random_indices, axis=0), axis=0)

        print()
        print()
        print("************* TRAINING EPOCH: ", training_epoch)
        gradient_step=0
        while((gradient_step*meta_bs*update_bs) < total_datapoints): # I feel like this should be different....

            ####################################################
            ## randomly select batch of data, for 1 outer-gradient step
            ####################################################

            random_indices_batch = random_indices[gradient_step*meta_bs: min((gradient_step+1)*meta_bs, len(random_indices)-1)]
            indices_batch = all_indices[random_indices_batch]
            
            inputs_batch = np.array([inputs_full[i[0]][i[1]][i[2]:i[2] + (update_bs+num_sgd_steps-1) + update_bs] for i in indices_batch])
            outputs_batch = np.array([outputs_full[i[0]][i[1]][i[2]:i[2] + (update_bs+num_sgd_steps-1) + update_bs] for i in indices_batch])

            #use the 1st half as training data for inner-loop gradient

            inputa = inputs_batch[:, :update_bs+num_sgd_steps-1, :]
            labela = outputs_batch[:, :update_bs+num_sgd_steps-1, :]
            #use the 2nd half as test data for outer-loop gradient
            inputb = inputs_batch[:, update_bs+num_sgd_steps-1:, :]
            labelb = outputs_batch[:, update_bs+num_sgd_steps-1:, :]

            #############################
            ## run meta-training iteration
            #############################

            ##########TEMP
            '''print("INSIDE TRAIN MAML... check MPE of the model on sty data...")
            import IPython
            IPython.embed()
            k=2
            for rollout_num_temp in range(len(inputs_full[k])-20, len(inputs_full[k])):
                my_results=[]
                for rand_num in range(len(inputs_full[k][rollout_num_temp])-2*(update_bs+num_sgd_steps-1) + 1):
                    batchx = np.array(inputs_full[k][rollout_num_temp][rand_num:rand_num+ (update_bs+num_sgd_steps-1) + update_bs])
                    batchy = np.array(outputs_full[k][rollout_num_temp][rand_num:rand_num+ (update_bs+num_sgd_steps-1) + update_bs])

                    inputa = [batchx[:update_bs+num_sgd_steps-1, :]]
                    labela = [batchy[:update_bs+num_sgd_steps-1, :]]
                    inputb = [batchx[update_bs+num_sgd_steps-1:, :]]
                    labelb = [batchy[update_bs+num_sgd_steps-1:, :]]

                    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
                    result = sess.run([model.total_losses2[train_config['num_updates'] - 1]], feed_dict)
                    my_results.append(result[0])
                print("rollout ", rollout_num_temp, "mean min max std: ", np.mean(my_results), np.min(my_results), np.max(my_results), np.std(my_results))'''

            #each inner update is done on update_bs (k) points (to calculate each theta')
            #an outer update is done using metaBS*k points

            #which tensors to populate/execute during the sess.run call
            input_tensors = [model.metatrain_op]
            if gradient_step % log_config['print_itr'] == 0 or gradient_step % log_config['summary_itr'] == 0:
                    input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[train_config['num_updates'] - 1], model.lossesb])

            #make the sess.run call to perform one metatraining iteration
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
            result = sess.run(input_tensors, feed_dict)

            #############################
            ## logging and saving
            #############################

            if gradient_step % log_config['summary_itr'] == 0:
                prelosses.append(result[2])
                if log_config['log']:
                        train_writer.add_summary(result[1], gradient_step)
                postlosses.append(result[3])

            if gradient_step % log_config['print_itr'] == 0:
                print_str = 'Gradient step ' + str(gradient_step)
                print_str += '   | Mean pre-losses: ' + str(np.mean(prelosses)) + '   | Mean post-losses: ' + str(np.mean(postlosses))
                print_str += '    | Time spent:   {0:.2f}'.format(time.time() - t0)
                print(print_str)
                
                if(train_config['use_reg']):
                    print("Regularization loss: ", sess.run(model.regularizer, feed_dict))
                print("loss without reg: ", sess.run(model.mse_loss, feed_dict))

                t0 = time.time()
                prelosses, postlosses = [], []

                print(model.list_of_gradients)
                print(model.gvs)


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

        # Random (off-policy) data
        feed_dict = {model.inputa: inputa_val_offpol, model.inputb: inputb_val_offpol, model.labela: labela_val_offpol, model.labelb: labelb_val_offpol}
        val_loss = sess.run(model.mse_loss[train_config['num_updates'] - 1], feed_dict)
        vallosses.append(val_loss)
        print ''
        print ''
        print ''
        print("Validation loss:", val_loss)

        # MPC (on-policy) data
        if curr_agg_iter != 0:
            feed_dict = {model.inputa: inputa_val_onpol, model.inputb: inputb_val_onpol, model.labela: labela_val_onpol, model.labelb: labelb_val_onpol}
            val_loss_onPol = sess.run(model.mse_loss[train_config['num_updates'] - 1], feed_dict)
            vallosses_onPol.append(val_loss_onPol)

            print("Validation loss (on policy):", val_loss_onPol)

        ################################
        ##### Save every epoch #########
        ################################

        if(curr_agg_iter==0):
            save_freq = 5
        else:
            save_freq = 1

        if training_epoch % save_freq == 0:
            name = "model_aggIter" + str(curr_agg_iter) + "_epoch" + str(training_epoch)
            saver.save(sess,  osp.join(path, name))
            print("SAVING TO ", osp.join(path, name))

            # Save
            np.save(path + "/all_random_indices_aggIter"+str(curr_agg_iter)+".npy", all_random_indices)
            np.save(path + "/all_indices_aggIter"+str(curr_agg_iter)+".npy", all_indices)
            np.save(path + '/debuggingLR_metatrain_losses_withReg_aggIter'+str(curr_agg_iter)+'.npy', metatrain_losses_withReg)
            np.save(path + '/debuggingLR_metatrain_losses_aggIter'+str(curr_agg_iter)+'.npy', metatrain_losses)
            np.save(path + '/debuggingLR_vallosses_aggIter'+str(curr_agg_iter)+'.npy', vallosses)
            np.save(path + '/debuggingLR_vallosses_onPol_aggIter'+str(curr_agg_iter)+'.npy', vallosses)

            if(training_epoch>0):

                # Make validation plot
                x = np.arange(0, training_epoch+1)
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

    ################################
    ####### Save at the end ########
    ################################
    np.save(path + "/all_random_indices_aggIter"+str(curr_agg_iter)+".npy", all_random_indices)
    np.save(path + "/all_indices_aggIter"+str(curr_agg_iter)+".npy", all_indices)
    np.save(path + '/debuggingLR_metatrain_losses_withReg_aggIter'+str(curr_agg_iter)+'.npy', metatrain_losses_withReg)
    np.save(path + '/debuggingLR_metatrain_losses_aggIter'+str(curr_agg_iter)+'.npy', metatrain_losses)
    np.save(path + '/debuggingLR_vallosses_aggIter'+str(curr_agg_iter)+'.npy', vallosses)
    np.save(path + '/debuggingLR_vallosses_onPol_aggIter'+str(curr_agg_iter)+'.npy', vallosses)

 