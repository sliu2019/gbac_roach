#fine
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

#mine
from nn_dynamics_roach.msg import velroach_msg
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



#BEFORE RUNNING
    #source workspace
    #source activate rllab3
    #python launch_maml_train.py

def run(d):

    #restore old dynamics model
    old_exp_name = 'MAML_roach/terrain_types__max_epochs_5__meta_batch_size_40__batch_size_2000__update_batch_size_20__horizon_5'
    old_model_num = 0
    previous_dynamics_model = '/home/anagabandi/rllab-private/data/local/experiment/'+old_exp_name+'/model'+str(old_model_num)
    previous_dynamics_model = None

    num_steps_per_rollout=10
    desired_shape_for_rollout = "straight"                     #straight, left, right, circle_left, zigzag, figure8
    save_rollout_run_num = 0
    rollout_save_filename= desired_shape_for_rollout + str(save_rollout_run_num)

    #settings
    train_now = False
    model_path = ""
    cheaty_training = False
    use_one_hot = False #True
    use_camera = False #True
    playback_mode = False
    
    state_representation = "exclude_x_y" #["exclude_x_y", "all"]

    #don't change much
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

    #datatypes
    tf_datatype= tf.float32
    np_datatype= np.float32

    #motor limits
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
    curr_agg_iter = d['curr_agg_iter']
    save_dir = d['exp_name']
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

    dataX=[]
    dataX_full=[] #this is just for your personal use for forwardsim (for debugging)
    dataY=[]
    dataZ=[]
    agg_itr = 0
    for agg_itr in range(curr_agg_iter+1):
        #getDataFromDisk should give (tasks, rollouts from that task, each rollout has its points)
        dataX_curr, dataY_curr, dataZ_curr, dataX_curr_full = getDataFromDisk(agg_itr, config['experiment_type'], 
                                                                            use_one_hot, use_camera, 
                                                                            cheaty_training, state_representation)
        if(agg_itr==0):
            dataX= copy.deepcopy(dataX_curr)
            dataY= copy.deepcopy(dataY_curr)
            dataZ= copy.deepcopy(dataZ_curr)
            dataX_full= copy.deepcopy(dataX_curr_full)
        else:
            #combine these rollouts w previous rollouts, so everything is still organized by task
            for task_num in range(len(dataX)):
                for rollout_num in range(len(dataX_curr[task_num])):
                    dataX[task_num].append(dataX_curr[task_num][rollout_num])
                    dataY[task_num].append(dataY_curr[task_num][rollout_num])
                    dataZ[task_num].append(dataZ_curr[task_num][rollout_num])
                    dataX_full[task_num].append(dataX_curr_full[task_num][rollout_num])

    total_num_data = len(dataX)*len(dataX[0])*len(dataX[0][0]) # numSteps = tasks * rollouts * steps
    print("\n\nTotal number of data points: ", total_num_data)

    ## concatenate state and action --> inputs
    outputs = copy.deepcopy(dataZ)
    inputs = copy.deepcopy(dataX)
    for task_num in range(len(dataX)):
        for rollout_num in range (len(dataX[task_num])):
            #dataX[task_num][rollout_num] (steps x s_dim)
            #dataY[task_num][rollout_num] (steps x a_dim)
            inputs[task_num][rollout_num] = np.concatenate([dataX[task_num][rollout_num], dataY[task_num][rollout_num]], axis=1)
    
    #inputs should now be (tasks, rollouts from that task, [s,a])
    #outputs should now be (tasks, rollouts from that task, [ds])
    inputSize = inputs[0][0].shape[1]
    outputSize = outputs[0][0].shape[1]
    print("\n\nDimensions:")
    print("states: ", dataX[0][0].shape[1])
    print("actions: ", dataY[0][0].shape[1])
    print("inputs to NN: ", inputSize)
    print("outputs of NN: ", outputSize)

    #calc mean/std on full dataset
    mean_inp = np.expand_dims(np.mean(inputs,axis=(0,1,2)), axis=0)
    std_inp = np.expand_dims(np.std(inputs,axis=(0,1,2)), axis=0)
    mean_outp = np.expand_dims(np.mean(outputs,axis=(0,1,2)), axis=0)
    std_outp = np.expand_dims(np.std(outputs,axis=(0,1,2)), axis=0)
    print("\n\nCalulated means and stds... ", mean_inp.shape, std_inp.shape, mean_outp.shape, std_outp.shape, "\n\n")

    ###########################################################
    ## CREATE regressor, policy, data generator, maml model
    ###########################################################

    # create regressor (NN dynamics model)
    regressor = DeterministicMLPRegressor(inputSize, outputSize, dim_obs=outputSize, tf_datatype=tf_datatype, **config['model'])

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
    model = MAML(regressor, inputSize, outputSize, config=config['training'])
    model.construct_model(input_tensors=None, prefix='metatrain_')
    model.summ_op = tf.summary.merge_all()

    # saving
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.InteractiveSession()

    # initialize tensorflow vars
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    # set the mean/std of regressor according to mean/std of the data we have so far
    regressor.update_params_data_dist(mean_inp, std_inp, mean_outp, std_outp, 1)

    ###########################################################
    ## TRAIN THE DYNAMICS MODEL
    ###########################################################

    #restore model
    if(previous_dynamics_model is None):
        junk=1
    else:
        print("\n\nRESTORING PREVIOUS DYNAMICS MODEL FROM ", previous_dynamics_model)
        saver.restore(sess, previous_dynamics_model)

    #train on the given full dataset, for max_epochs
    train(inputs, outputs, curr_agg_iter, model, saver, sess, config)

    predicted_traj = regressor.do_forward_sim(dataX_full[0][7][27:45], dataY[0][7][27:45], state_representation)
    #np.save(save_dir + '/forwardsim_true.npy', dataX_full[0][7][27:45])
    #np.save(save_dir + '/forwardsim_pred.npy', predicted_traj)

    ###########################################################
    ## RUN THE MPC CONTROLLER
    ###########################################################

    #create controller node
    controller_node = GBAC_Controller(policy=policy, state_representation=state_representation, use_pid_mode=use_pid_mode, default_addrs=default_addrs, **config['roach'])

    #do 1 rollout
    print("\n\n\nPAUSING... right before a controller run... RESET THE ROBOT TO A GOOD LOCATION BEFORE CONTINUING...")
    #IPython.embed()
    resulting_x, selected_u, desired_seq, list_robot_info, list_mocap_info, old_saving_format_dict = controller_node.run(num_steps_per_rollout, desired_shape_for_rollout)
    
    #where to save this rollout
    pathStartName = save_dir + '/saved_rollouts/'+rollout_save_filename+ '_aggIter' +str(curr_agg_iter)

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

    #save the result of the run
    np.save(pathStartName + '/actions.npy', selected_u)
    np.save(pathStartName + '/states.npy', resulting_x)
    np.save(pathStartName + '/desired.npy', desired_seq)
    pickle.dump(list_robot_info,open(pathStartName + '_robotInfo.obj','w'))
    pickle.dump(list_mocap_info,open(pathStartName + '_mocapInfo.obj','w'))

    #stop roach
    print("killing robot")
    controller_node.kill_robot()

def main(config_path, extra_config):

    #################################
    ## INIT config and vars
    #################################

    #read in config vars
    config = yaml.load(open(config_path))
    config = replace_in_dict(config, extra_config)

    vg = VariantGenerator()
    vg.add('config', [config])
    ##vg.add('batch_size', [2000]) ######### to do: use this to decide how much data to read in from disk
    vg.add('meta_batch_size', [40]) ##################
    vg.add('update_batch_size', [20]) ############# 8 
    vg.add('update_lr', [0.001])
    vg.add('max_epochs', [1]) ###########################
    vg.add('horizon', [5])
    vg.add('curr_agg_iter', [0])


    ##print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('MAML', vg.size))

    for v in vg.variants():

        time.sleep(1.)

        _v = v.copy(); del _v['config'], _v['_hidden_keys']
        v['config'] = replace_in_dict(v['config'], _v)
        #v['exp_name'] = exp_name = v['config']['logging']['log_dir'] + '__'.join(
        #    [v['config']['experiment_type']] + [k + '_' + str(v) for k,v in _v.items() if k not in ['name', 'experiment_type', 'dim_hidden']])


        v['exp_name'] = exp_name = v['config']['logging']['log_dir'] + v['config']['experiment_type'] + '__max_epochs_5__meta_batch_size_40__batch_size_2000__update_batch_size_20__horizon_5'

        run_experiment_lite(
            run,
            sync_s3_pkl=True,
            periodic_sync=True,
            variant=v,
            snapshot_mode="all",
            mode="local",
            use_cloudpickle=True,
            exp_name=exp_name,
            use_gpu=False,
            pre_commands=[#"yes | pip install --upgrade pip",
                          "yes | pip install tensorflow=='1.4.1'",
                          "yes | pip install --upgrade cloudpickle"],
            seed=v['config']['seed']
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'config.yaml')
    parser.add_argument('-p', '--config_path', type=str, default=default_path,
                  help='directory for the config yaml file.')
    parser.add_argument('-c', '--config', type=dict, default=dict())
    args = parser.parse_args()
    main(config_path=args.config_path, extra_config=args.config)

