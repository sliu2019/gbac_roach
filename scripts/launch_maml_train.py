import os.path as osp
import os
import sys
from six.moves import cPickle
import yaml
import time
import tensorflow as tf
import random
import argparse
import IPython
import pickle
import threading
import multiprocessing
import matplotlib.pyplot as plt
import copy 

# Our utilities
from gbac_roach.msg import velroach_msg
from utils import *
from roach_utils import *
from gbac_controller_node import GBAC_Controller

#rllab maml stuff that was moved to this repo, to allow changes
from train_maml import train
from maml import MAML
from deterministic_mlp_regressor import DeterministicMLPRegressor
from naive_mpc_controller import NaiveMPCController as Policy

#other rllab stuff
from rllab.misc.instrument import run_experiment_lite
from sandbox.ignasi.maml.utils import replace_in_dict
from rllab.misc.instrument import VariantGenerator

#####################################################################
#####################################################################

#TO DO
    #add on policy data to the validation

#BEFORE RUNNING
    #source workspace
    #source activate rllab3
    #python launch_maml_train.py

#MODELS USED RIGHT NOW FOR GBAC
    #/home/anagabandi/rllab-private/data/local/experiment/MAML_roach_copy/Wednesday_optimization/ulr_5_num_update_1/_ubs_8_ulr_2.0num_updates1_layers_1_x100_task_list_all/model_epoch45
    #/home/anagabandi/rllab-private/data/local/experiment/MAML_roach_copy/Wednesday_optimization/ulr_5_num_update_1/_ubs_8_ulr_2.0num_updates1_layers_1_x100_task_list_all/model_aggIter1_epoch45

#####################################################################
#####################################################################

def run(d):

    #restore old dynamics model
    train_now = False
        # IF TRUE saved the new training into "previous_dynamics_model"
    restore_previous = True

    learn_loss_weighting=False

    #aggiter0 for GBAC
    #previous_dynamics_model = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_7_optimization/_ubs_23_ulr_2.0num_updates2_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False/model_aggIter0_epoch45"

    #aggiter1 for GBAC
    #previous_dynamics_model = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_11_optimization/_ubs_23_ulr_2.0num_updates2_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False_agg_0.5/model_aggIter1_epoch23"

    #aggiter0 for NONGBAC
    previous_dynamics_model = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_11_optimization/_ubs_23_ulr_0.0num_updates1_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False/model_aggIter0_epoch45"

    ########################################################################
    
    desired_shape_for_rollout = "right"                     #straight, left, right, circle_left, zigzag, figure8
    save_rollout_run_num = 0
    rollout_save_filename= desired_shape_for_rollout + str(save_rollout_run_num)

    num_steps_per_rollout= 30  #150 turf right... 80 for straight, 270 for multi_terrain uturn......... 150 for slopes
    if (desired_shape_for_rollout=="figure8"):
        num_steps_per_rollout= 150 ###400
    elif (desired_shape_for_rollout=="zigzag"):
        num_steps_per_rollout= 150

    #settings
    cheaty_training = False
    use_one_hot = False #True
    use_camera = False #True
    playback_mode = False
    
    state_representation = "exclude_x_y" #["exclude_x_y", "all"]

    # Settings (generally, keep these to default)
    default_addrs= [b'\x00\x01']
    use_pid_mode = True      
    slow_pid_mode = True
    visualize_rviz=True   #turning this off can make things go faster
    visualize_True = True
    visualize_False = False
    noise_True = True
    noise_False = False
    make_aggregated_dataset_noisy = True
    make_training_dataset_noisy = True
    perform_forwardsim_for_vis= True
    print_minimal=False
    noiseToSignal = 0
    if(make_training_dataset_noisy):
        noiseToSignal = 0.01

    # Defining datatypes
    tf_datatype= tf.float32
    np_datatype= np.float32

    # Setting motor limits
    left_min = 1200
    right_min = 1200
    left_max = 2000
    right_max = 2000
    if(use_pid_mode):
      if(slow_pid_mode):
        left_min = 2*math.pow(2,16)*0.001
        right_min = 2*math.pow(2,16)*0.001
        left_max = 9*math.pow(2,16)*0.001
        right_max = 9*math.pow(2,16)*0.001
      else: #this hasnt been tested yet
        left_min = 4*math.pow(2,16)*0.001
        right_min = 4*math.pow(2,16)*0.001
        left_max = 12*math.pow(2,16)*0.001
        right_max = 12*math.pow(2,16)*0.001

    #vars from config
    config = d['config']
    curr_agg_iter = config['aggregation']['curr_agg_iter']
    save_dir = d['save_dir']
    print("\n\nSAVING EVERYTHING TO: ", save_dir)

    #make directories
    if not os.path.exists(save_dir + '/saved_rollouts'):
        os.makedirs(save_dir + '/saved_rollouts')
    if not os.path.exists(save_dir + '/saved_rollouts/'+rollout_save_filename+ '_aggIter' +str(curr_agg_iter)):
        os.makedirs(save_dir + '/saved_rollouts/'+rollout_save_filename+ '_aggIter' +str(curr_agg_iter))

    ######################################
    ######## GET TRAINING DATA ###########
    ######################################

    print("\n\nCURR AGGREGATION ITER: ", curr_agg_iter)
    
    # Training data
    # Random
    dataX=[]
    dataX_full=[] #this is just for your personal use for forwardsim (for debugging)
    dataY=[]
    dataZ=[]

    # Training data
    # MPC
    dataX_onPol=[]
    dataX_full_onPol=[]
    dataY_onPol=[]
    dataZ_onPol=[]

    # Validation data
    # Random
    dataX_val = []
    dataX_full_val=[]
    dataY_val=[]
    dataZ_val=[]

    # Validation data
    # MPC
    dataX_val_onPol = []
    dataX_full_val_onPol=[]
    dataY_val_onPol=[]
    dataZ_val_onPol=[]

    training_ratio = config['training']['training_ratio']
    for agg_itr_counter in range(curr_agg_iter+1):

        #getDataFromDisk should give (tasks, rollouts from that task, each rollout has its points)
        dataX_curr, dataY_curr, dataZ_curr, dataX_curr_full = getDataFromDisk(config['experiment_type'], 
                                                                            use_one_hot, use_camera, 
                                                                            cheaty_training, state_representation, agg_itr_counter, config_training=config['training'])

        if(agg_itr_counter==1):
            print("*********TRYING TO FIND THE WEIRD ROLLOUT...")
            for rollout in range(len(dataX_curr[2])):
                val=dataX_curr[2][rollout][:,4]
                if(np.any(val<0)):
                    dataX_curr[2][rollout] = dataX_curr[2][rollout+1]
                    dataY_curr[2][rollout] = dataY_curr[2][rollout+1]
                    dataZ_curr[2][rollout] = dataZ_curr[2][rollout+1]
                    print("FOUND IT!!!!!!! rollout number ", rollout)

        #import IPython
        #IPython.embed()
        #len(dataX_curr) = numtasks
        #len(dataX_curr[task]) = rollouts per task
        #len(dataX_curr[task][rollout]) = (steps in that rollout, dim)
        
        #random data
        #go from dataX_curr (tasks, rollouts, steps) --> to dataX (tasks, some rollouts, steps) and dataX_val (tasks, some rollouts, steps)
        if(agg_itr_counter==0):
            for task_num in range(len(dataX_curr)):
                taski_num_rollout = len(dataX_curr[task_num])
                print("task" + str(task_num) + "_num_rollouts: ", taski_num_rollout)

                #for each task, append something like (356, 48, 22) (numrollouts per task, num steps in that rollout, dim)
                dataX.append(dataX_curr[task_num][:int(taski_num_rollout*training_ratio)])
                dataX_full.append(dataX_curr_full[task_num][:int(taski_num_rollout*training_ratio)])
                dataY.append(dataY_curr[task_num][:int(taski_num_rollout*training_ratio)])
                dataZ.append(dataZ_curr[task_num][:int(taski_num_rollout*training_ratio)])

                dataX_val.append(dataX_curr[task_num][int(taski_num_rollout*training_ratio):])
                dataX_full_val.append(dataX_curr_full[task_num][int(taski_num_rollout*training_ratio):])
                dataY_val.append(dataY_curr[task_num][int(taski_num_rollout*training_ratio):])
                dataZ_val.append(dataZ_curr[task_num][int(taski_num_rollout*training_ratio):])

        #on-policy data
        #go from dataX_curr (tasks, rollouts, steps) --> to dataX_onPol (tasks, some rollouts, steps) and dataX_val_onPol (tasks, some rollouts, steps)
        elif(agg_itr_counter==1):

            for task_num in range(len(dataX_curr)):
                taski_num_rollout = len(dataX_curr[task_num])
                print("task" + str(task_num) + "_num_rollouts for onpolicy: ", taski_num_rollout)

                dataX_onPol.append(dataX_curr[task_num][:int(taski_num_rollout*training_ratio)])
                dataX_full_onPol.append(dataX_curr_full[task_num][:int(taski_num_rollout*training_ratio)])
                dataY_onPol.append(dataY_curr[task_num][:int(taski_num_rollout*training_ratio)])
                dataZ_onPol.append(dataZ_curr[task_num][:int(taski_num_rollout*training_ratio)])

                dataX_val_onPol.append(dataX_curr[task_num][int(taski_num_rollout*training_ratio):])
                dataX_full_val_onPol.append(dataX_curr_full[task_num][int(taski_num_rollout*training_ratio):])
                dataY_val_onPol.append(dataY_curr[task_num][int(taski_num_rollout*training_ratio):])
                dataZ_val_onPol.append(dataZ_curr[task_num][int(taski_num_rollout*training_ratio):])

        #on-policy data
        #go from dataX_curr (tasks, rollouts, steps) --> to ADDING ONTO dataX_onPol (tasks, some more rollouts than before, steps) and dataX_val_onPol (tasks, some more rollouts than before, steps)
        else:
            for task_num in range(len(dataX_curr)):

                taski_num_rollout = len(dataX_curr[task_num])
                print("task" + str(task_num) + "_num_rollouts for onpolicy: ", taski_num_rollout)

                dataX_onPol[task_num].extend(dataX_curr[task_num][:int(taski_num_rollout*training_ratio)])
                dataX_full_onPol[task_num].extend(dataX_curr_full[task_num][:int(taski_num_rollout*training_ratio)])
                dataY_onPol[task_num].extend(dataY_curr[task_num][:int(taski_num_rollout*training_ratio)])
                dataZ_onPol[task_num].extend(dataZ_curr[task_num][:int(taski_num_rollout*training_ratio)])

                dataX_val_onPol[task_num].extend(dataX_curr[task_num][int(taski_num_rollout*training_ratio):])
                dataX_full_val_onPol[task_num].extend(dataX_curr_full[task_num][int(taski_num_rollout*training_ratio):])
                dataY_val_onPol[task_num].extend(dataY_curr[task_num][int(taski_num_rollout*training_ratio):])
                dataZ_val_onPol[task_num].extend(dataZ_curr[task_num][int(taski_num_rollout*training_ratio):])

    #############################################################

    #import IPython
    #IPython.embed()

    #count number of random and onpol data points
    total_random_data = len(dataX)*len(dataX[1])*len(dataX[1][0]) # numSteps = tasks * rollouts * steps
    if(len(dataX_onPol)==0):
        total_onPol_data=0
    else:
        total_onPol_data = len(dataX_onPol)*len(dataX_onPol[0])*len(dataX_onPol[0][0]) #this is approximate because each task doesn't have the same num rollouts or the same num steps
    total_num_data = total_random_data +  total_onPol_data
    print()
    print()
    print("Number of random data points: ", total_random_data)
    print("Number of on-policy data points: ", total_onPol_data)
    print("TOTAL number of data points: ", total_num_data)

    #############################################################

    #combine random and onpol data into a single dataset for training
    ratio_new = config["aggregation"]["ratio_new"]
    num_new_pts = ratio_new*(total_random_data)/(1.0-ratio_new)
    if(len(dataX_onPol)==0):
        num_times_to_copy_onPol=0
    else:
        num_times_to_copy_onPol = int(num_new_pts/total_onPol_data)

    #copy all rollouts from each task of onpol data, and do this copying this many times
    for i in range(num_times_to_copy_onPol):
        for task_num in range(len(dataX_onPol)):
            for rollout_num in range(len(dataX_onPol[task_num])):
                dataX[task_num].append(dataX_onPol[task_num][rollout_num])
                dataX_full[task_num].append(dataX_full_onPol[task_num][rollout_num])
                dataY[task_num].append(dataY_onPol[task_num][rollout_num])
                dataZ[task_num].append(dataZ_onPol[task_num][rollout_num])
    print("num_times_to_copy_onPol: ", num_times_to_copy_onPol)

    ######## TO DO: comment the above back in
    '''dataX = dataX_onPol
    dataY = dataY_onPol
    dataZ = dataZ_onPol
    dataX_full = dataX_full_onPol'''

    #############################################################

    #make a list of all X,Y,Z so can take mean of them
    ## concatenate state and action --> inputs (for training)
    all_points_inp=[]
    all_points_outp=[]
    outputs = copy.deepcopy(dataZ)
    inputs = copy.deepcopy(dataX)
    for task_num in range(len(dataX)):
        for rollout_num in range(len(dataX[task_num])):

            #this will just be a big list of everything, so can take the mean
            input_pts = np.concatenate((dataX[task_num][rollout_num], dataY[task_num][rollout_num]), axis=1)
            output_pts = dataZ[task_num][rollout_num]

            #this will the concatenate thing for later
            inputs[task_num][rollout_num] = np.concatenate([dataX[task_num][rollout_num], dataY[task_num][rollout_num]], axis=1)

            all_points_inp.append(input_pts)
            all_points_outp.append(output_pts)
    all_points_inp=np.concatenate(all_points_inp)
    all_points_outp=np.concatenate(all_points_outp)

    ## concatenate state and action --> inputs (for validation)
    outputs_val = copy.deepcopy(dataZ_val)
    inputs_val = copy.deepcopy(dataX_val)
    for task_num in range(len(dataX_val)):
        for rollout_num in range (len(dataX_val[task_num])):
            #dataX[task_num][rollout_num] (steps x s_dim)
            #dataY[task_num][rollout_num] (steps x a_dim)
            inputs_val[task_num][rollout_num] = np.concatenate([dataX_val[task_num][rollout_num], dataY_val[task_num][rollout_num]], axis=1)

    ## concatenate state and action --> inputs (for validation onpol)
    outputs_val_onPol = copy.deepcopy(dataZ_val_onPol)
    inputs_val_onPol = copy.deepcopy(dataX_val_onPol)
    for task_num in range(len(dataX_val_onPol)):
        for rollout_num in range (len(dataX_val_onPol[task_num])):
            #dataX[task_num][rollout_num] (steps x s_dim)
            #dataY[task_num][rollout_num] (steps x a_dim)
            inputs_val_onPol[task_num][rollout_num] = np.concatenate([dataX_val_onPol[task_num][rollout_num], dataY_val_onPol[task_num][rollout_num]], axis=1)

    #############################################################
    
    #inputs should now be (tasks, rollouts from that task, [s,a])
    #outputs should now be (tasks, rollouts from that task, [ds])
    #IPython.embed()

    inputSize = inputs[0][0].shape[1]
    outputSize = outputs[1][0].shape[1]
    print("\n\nDimensions:")
    print("states: ", dataX[1][0].shape[1])
    print("actions: ", dataY[1][0].shape[1])
    print("inputs to NN: ", inputSize)
    print("outputs of NN: ", outputSize)

    #inputSize = 24
    #outputSize = 24

    #calc mean/std on full dataset
    '''if config["model"]["nonlinearity"] == "tanh":
        # Do you scale inputs to [-1, 1] and then standardize outputs?
        #IPython.embed()
        inputs_array = np.array(inputs)
        mean_inp = (inputs_array.max() + inputs_array.min())/2.0
        std_inp = inputs_array.max() - mean_inp

        mean_inp = mean_inp*np.ones((1, inputs_array.shape[3]))
        std_inp = std_inp*np.ones((1, inputs_array.shape[3]))
        #IPython.embed()

        mean_outp = np.expand_dims(np.mean(outputs,axis=(0,1,2)), axis=0)
        std_outp = np.expand_dims(np.std(outputs,axis=(0,1,2)), axis=0)
        #IPython.embed() # HOw should I expand_dims? # check that after the operation, all inputs do lie in this range
    elif config["model"]["nonlinearity"] == "sigmoid":
        # Do you scale inputs to [0, 1] and then standardize outputs?
        #IPython.embed()
        inputs_array = np.array(inputs)
        mean_inp = inputs_array.min()
        std_inp = inputs_array.max() - mean_inp

        mean_inp = mean_inp*np.ones((1, inputs_array.shape[3]))
        std_inp = std_inp*np.ones((1, inputs_array.shape[3]))

        #IPython.embed()

        mean_outp = np.expand_dims(np.mean(outputs,axis=(0,1,2)), axis=0)
        std_outp = np.expand_dims(np.std(outputs,axis=(0,1,2)), axis=0)
        #IPython.embed() # HOw should I expand_dims? # check that after the operation, all inputs do lie in this range
    else:  # for all the relu variants
        mean_inp = np.expand_dims(np.mean(inputs,axis=(0,1,2)), axis=0)
        std_inp = np.expand_dims(np.std(inputs,axis=(0,1,2)), axis=0)
        mean_outp = np.expand_dims(np.mean(outputs,axis=(0,1,2)), axis=0)
        std_outp = np.expand_dims(np.std(outputs,axis=(0,1,2)), axis=0)'''

    mean_inp = np.expand_dims(np.mean(all_points_inp, axis=0), axis=0)
    std_inp = np.expand_dims(np.std(all_points_inp, axis=0), axis=0)
    mean_outp = np.expand_dims(np.mean(all_points_outp, axis=0), axis=0)
    std_outp = np.expand_dims(np.std(all_points_outp, axis=0), axis=0)
    print("\n\nCalulated means and stds... ", mean_inp.shape, std_inp.shape, mean_outp.shape, std_outp.shape, "\n\n")

    ###########################################################
    ## CREATE regressor, policy, data generator, maml model
    ###########################################################

    # create regressor (NN dynamics model)
    regressor = DeterministicMLPRegressor(inputSize, outputSize, dim_obs=outputSize, tf_datatype=tf_datatype, seed=config['seed'],weight_initializer=config['training']['weight_initializer'], **config['model'])

    # create policy (MPC controller)
    policy = Policy(regressor, inputSize, outputSize, 
                    left_min, right_min, left_max, right_max, state_representation=state_representation,
                    visualize_rviz=config['roach']['visualize_rviz'], 
                    x_index=config['roach']['x_index'], 
                    y_index=config['roach']['y_index'], 
                    yaw_cos_index=config['roach']['yaw_cos_index'],
                    yaw_sin_index=config['roach']['yaw_sin_index'], 
                    **config['policy'])

    # create MAML model
        # note: this also constructs the actual regressor network/weights
    model = MAML(regressor, inputSize, outputSize, learn_loss_weighting, config=config['training'])
    model.construct_model(input_tensors=None, prefix='metatrain_')
    model.summ_op = tf.summary.merge_all()

    # GPU config proto
    gpu_device = 0
    gpu_frac = 0.4 #0.4 #0.8 #0.3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    config_2 = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    # saving
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.InteractiveSession(config=config_2)

    # initialize tensorflow vars
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    # set the mean/std of regressor according to mean/std of the data we have so far
    regressor.update_params_data_dist(mean_inp, std_inp, mean_outp, std_outp, total_num_data)

    ###########################################################
    ## TRAIN THE DYNAMICS MODEL
    ###########################################################

    #train on the given full dataset, for max_epochs
    if train_now:
        if(restore_previous):
            print("\n\nRESTORING PREVIOUS DYNAMICS MODEL FROM ", previous_dynamics_model, " AND CONTINUING TRAINING...\n\n")
            saver.restore(sess, previous_dynamics_model)
            
            """trainable_vars = tf.trainable_variables()
            weights = sess.run(trainable_vars)
            with open(osp.join(osp.dirname(previous_dynamics_model), "weights.pickle"), "wb") as output_file:
                pickle.dump(weights, output_file)"""
        #IPython.embed()
        np.save(save_dir + "/inputs.npy", inputs)
        np.save(save_dir + "/outputs.npy", outputs)
        np.save(save_dir + "/inputs_val.npy", inputs_val)
        np.save(save_dir + "/outputs_val.npy", outputs_val)
        # # mean_inp.shape, std_inp.shape, mean_outp.shape, std_outp.shape
        # np.save(save_dir + "/mean_inp.npy", mean_inp)
        # np.save(save_dir + "/std_inp.npy", std_inp)
        # np.save(save_dir + "/mean_outp.npy", mean_outp)
        # np.save(save_dir + "/std_outp.npy", std_outp)
        
        # REMOVE AFTER 3 PM TODAY

        trainable_vars = sess.run(tf.trainable_variables())
        #IPython.embed()
        pickle.dump(trainable_vars, open(osp.join(save_dir, 'weight_restore_in_launch.yaml'), 'w'))

        train(inputs, outputs, curr_agg_iter, model, saver, sess, config, inputs_val, outputs_val, inputs_val_onPol, outputs_val_onPol)
    else: 
        print("\n\nRESTORING A DYNAMICS MODEL FROM ", previous_dynamics_model)
        saver.restore(sess, previous_dynamics_model)

    #return
    #IPython.embed()
    #predicted_traj = regressor.do_forward_sim(dataX_full[0][0][27:45], dataY[0][0][27:45], state_representation)
    #np.save(save_dir + '/forwardsim_true.npy', dataX_full[0][7][27:45])
    #np.save(save_dir + '/forwardsim_pred.npy', predicted_traj)

    ###########################################################
    ## RUN THE MPC CONTROLLER
    ###########################################################

    #create controller node
    controller_node = GBAC_Controller(sess=sess, policy=policy, model=model,
                                    state_representation=state_representation, use_pid_mode=use_pid_mode, 
                                    default_addrs=default_addrs, update_batch_size=config['testing']['update_batch_size'], num_updates=config['testing']['num_updates'], de=config['testing']['dynamic_evaluation'], **config['roach'])

    #do 1 rollout
    print("\n\n\nPAUSING... right before a controller run... RESET THE ROBOT TO A GOOD LOCATION BEFORE CONTINUING...")
    #IPython.embed()
    resulting_x, selected_u, desired_seq, list_robot_info, list_mocap_info, old_saving_format_dict, list_best_action_sequences = controller_node.run(num_steps_per_rollout, desired_shape_for_rollout)
    
    #where to save this rollout
    pathStartName = save_dir + '/saved_rollouts/'+rollout_save_filename+ '_aggIter' +str(curr_agg_iter)
    print("\n\n************** TRYING TO SAVE EVERYTHING TO: ", pathStartName)

    #save the result of the run
    np.save(pathStartName + '/oldFormat_actions.npy', old_saving_format_dict['actions_taken'])
    np.save(pathStartName + '/oldFormat_desired.npy', old_saving_format_dict['desired_states'])
    np.save(pathStartName + '/oldFormat_executed.npy', old_saving_format_dict['traj_taken'])
    np.save(pathStartName + '/oldFormat_perp.npy', old_saving_format_dict['save_perp_dist'])
    np.save(pathStartName + '/oldFormat_forward.npy', old_saving_format_dict['save_forward_dist'])
    np.save(pathStartName + '/oldFormat_oldforward.npy', old_saving_format_dict['saved_old_forward_dist'])
    np.save(pathStartName + '/oldFormat_movedtonext.npy', old_saving_format_dict['save_moved_to_next'])
    np.save(pathStartName + '/oldFormat_desheading.npy', old_saving_format_dict['save_desired_heading'])
    np.save(pathStartName + '/oldFormat_currheading.npy', old_saving_format_dict['save_curr_heading'])
    np.save(pathStartName + '/list_best_action_sequences.npy', list_best_action_sequences)

    yaml.dump(config, open(osp.join(pathStartName, 'saved_config.yaml'), 'w'))

    #save the result of the run
    np.save(pathStartName + '/actions.npy', selected_u)
    np.save(pathStartName + '/states.npy', resulting_x)
    np.save(pathStartName + '/desired.npy', desired_seq)
    pickle.dump(list_robot_info,open(pathStartName + '/robotInfo.obj','w'))
    pickle.dump(list_mocap_info,open(pathStartName + '/mocapInfo.obj','w'))

    #stop roach
    print("killing robot")
    controller_node.kill_robot()

    return


def main(train, manual_edit_params, load_saved_params, saved_config_path, previous_dynamics_model):
    #################################
    ######## Set parameters #########
    #################################

    #read in default config
    config = yaml.load(open("../config.yaml"))

    if load_saved_params: 
        saved_config = yaml.load(open(saved_config_path, "r"))
        #IPython.embed()
        if train:
            # replcae training config only
            config["training"] = recursive_dict_merge(config["training"], saved_config["training"])
        else:
            # replace testing config only
            config["testing"] = recursive_dict_merge(config["testing"], saved_config["testing"])
            IPython.embed()

    vg = VariantGenerator()
    vg.add('config', [config])

    if manual_edit_params:
        vg.add('meta_batch_size', [64]) ######### 64
        vg.add('meta_lr', [0.001])
        vg.add('update_batch_size', [23]) #[8, 4, 16, 20]

        vg.add('max_runs_per_surface', [5]) #396
        vg.add('num_updates', [1]) 
        vg.add('update_lr', [0.0]) #[0.1, 1.0, 0.01]
            # in learn_inner_loss = True, then it seems like this being large is unstable when learning these
                #confirmed that if trainable=False on these though, then ulr=2 is same for learn_inner_loss True and learn_inner_loss False (this is just a sanity check)
            # this should be 2 for learn_inner_loss = False
            # this should be 0.1 for learn_inner_loss = True
        vg.add("task_list", [["turf", "styrofoam", "carpet"]]) #"all"
        vg.add('max_epochs', [50]) 
        vg.add('num_sgd_steps', [1]) #[1, 5, 10]

        # Aggregation
        vg.add('ratio_new', [0.9])
        vg.add('curr_agg_iter', [0]) #0, 1, 2, etc

        # Misc
        vg.add('horizon', [5]) #5, 10
        vg.add('use_reg', [True]) # This only changes the save filename! The config.yaml var needs to agree with this one if True ##################################################
        vg.add('seed', [0]) 
        vg.add('nonlinearity', ['relu'])
        if config['training']['use_reg']:
            vg.add('regularization_weight', [0.001]) #[0.001, 0.0], may need to be bigger since the number of scalar weights has increased
        vg.add('use_clip', [True])
        vg.add("weight_initializer", ["xavier"])
        vg.add("dim_hidden", [[500, 500]]) #[500,500]
        vg.add('optimizer', ["adam"])
        vg.add('dim_bias', [5])
        vg.add('use_momentum', [False])
        vg.add('learn_inner_loss', [False]) 

    for v in vg.variants():
        time.sleep(1.)
        if manual_edit_params:
            _v = v.copy(); del _v['config'], _v['_hidden_keys']
            v['config'] = replace_in_dict(v['config'], _v)

        if train:
            # Want the testing parameters to match the training parameters, so you can easily load this saved config for testing
            v['config']['testing'] = recursive_dict_merge(v['config']['testing'], v['config']['training'])

        # Example foldername if training
        #v['exp_name'] = "MAML_roach/9_11_optimization/" + "_ubs_" + str(v['config']['training']['update_batch_size']) + "_ulr_" + str(v['config']['training']['update_lr']) + "num_updates" + str(v['config']['training']['num_updates']) + "_layers_" + str(len(v['config']['model']['dim_hidden'])) + "_x" + str((v['config']['model']['dim_hidden'])[0]) + "_task_list_" + "_".join(v['config']['training']['task_list']) + "_mlr_" + str(v['config']['training']['meta_lr']) + "_mbs_" + str(v['config']['testing']['meta_batch_size']) + "_num-sgd-steps_" + str(v['config']['training']['num_sgd_steps']) + '_reg_weight_' + str(v['config']['training']['regularization_weight']) + "_dim_bias_" + str(v['config']['model']['dim_bias']) + "_metatrain_lr_" + str(v['config']['training']['learn_inner_loss'])

        # Example foldername if testing (i.e. you can place the rollouts in the same folder as the model used)
        v['exp_name'] = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_11_optimization/_ubs_23_ulr_0.0num_updates1_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False/video"
        run_experiment_lite(
            run,
            sync_s3_pkl=True,
            periodic_sync=True,
            variant=v,
            snapshot_mode="all",
            mode="local",
            use_cloudpickle=True,
            exp_name=v['exp_name'],
            use_gpu=False,
            pre_commands=[#"yes | pip install --upgrade pip",
                          "yes | pip install tensorflow=='1.4.1'",
                          "yes | pip install --upgrade cloudpickle"],
            seed=v['config']['seed']
        )

if __name__ == "__main__":
    """ NOTE TO USER: 
    The training and testing workflow is as follows:
    1) Specify if training or testing. Set train = False for testing. 
    2) Specify if manually editing parameters (for train/test) and/or loading a saved config (this will copy the saved config's training AND testing parameters onto the default config).

        For example: If I wanted to train with the same parameters that I used previously, except for batch_size, which I want to do hyperparameter search over, I would set train=True, manual_edit_params=True and load_saved_params=True and then specify the path to the saved config file. 
        This will just copy the saved training params. 

        Note: the config produced after training will have testing parameters equal to the training parameters, for ease of use.  

        Another example: If I want to test a trained model, I set load_saved_params=True and specify in the saved_config.yml output by training. This will copy its testing params. 
    3) Specify the name of the folder to save the trained model/testing data within the main function, as v["exp_name"]. If training, models, plotting, and formatted training data will be all be saved in that folder. If testing, then the rollout details will be saved in the folder/saved_rollouts.
    4) If testing, this is the model to load. If training and curr_agg_iter != 0 (in other words, we're doing data aggregation), then this is the model to do additional training on top of. 
    
    Most of the params should be left as default, but you may need to change serial_port in config.yaml.
    """
    
    train = False 
    manual_edit_params = False 
    load_saved_params = True
    saved_config_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_7_optimization/_ubs_23_ulr_2.0num_updates2_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False/paper/saved_rollouts/straight0_aggIter0/saved_config.yaml"
    previous_dynamics_model = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_11_optimization/_ubs_23_ulr_0.0num_updates1_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False/model_aggIter0_epoch45"

    
    main(train, manual_edit_params, load_saved_params, saved_config_path, previous_dynamics_model)
