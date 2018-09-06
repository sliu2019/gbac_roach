from sandbox.rocky.tf.policies.base import Policy
from rllab.core.serializable import Serializable
import numpy as np
from sandbox.ignasi.maml.utils import PlotRegressor
import tensorflow as tf
import time
import matplotlib
from rllab.envs.env_spec import EnvSpec
from rllab.spaces.box import Box
import IPython
import time

#roach stuff
from nn_dynamics_roach.msg import velroach_msg
import rospy
from trajectories import make_trajectory
from rospy.exceptions import ROSException
from threading import Condition
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from roach_utils import *
from utils import *


class GBAC_Controller(object):
    def __init__(
            self,
            sess,
            policy,
            model,
            use_pid_mode=True,
            state_representation='all',
            frequency_value= 10,
            serial_port= '/dev/ttyUSB0',
            baud_rate= 57600,
            default_addrs= None,
            update_batch_size= 0,
            num_updates=1,
            #anything after this is unused by this code
            x_index= 0,
            y_index= 1,
            yaw_cos_index= 10,
            yaw_sin_index= 11,
            visualize_rviz= True,
    ):
        self.sess = sess
        self.policy = policy
        self.model = model
        self.state_representation = state_representation
        self.use_pid_mode = use_pid_mode
        self.frequency_value = frequency_value
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.default_addrs = default_addrs
        self.update_batch_size = update_batch_size
        self.num_updates = num_updates

        self.lock = Condition()
        self.mocap_info = PoseStamped()

        #init vars
        self.x_index = 0
        self.y_index = 1
        
        #init node
        rospy.init_node('MAML_roach_controller_node', anonymous=True)
        self.callback_mocap(rospy.wait_for_message("/mocap/pose", PoseStamped))
        self.rate = rospy.Rate(self.frequency_value) 

        self.xb, self.robots, shared.imu_queues = setup_roach(self.serial_port, self.baud_rate, self.default_addrs, self.use_pid_mode, 1)
        
        #set PID gains
        for robot in self.robots:
            if(self.use_pid_mode):
                robot.setMotorGains([1800,200,100,0,0, 1800,200,100,0,0])

        #make subscribers
        self.sub_mocap = rospy.Subscriber('/mocap/pose', PoseStamped, self.callback_mocap)

        #make publishers
        self.publish_robotinfo= rospy.Publisher('/robot0/robotinfo', velroach_msg, queue_size=5)
        self.policy.publish_markers = rospy.Publisher('visualize_selected', MarkerArray, queue_size=5)
        self.policy.publish_markers_desired = rospy.Publisher('visualize_desired', MarkerArray, queue_size=5)

    def callback_mocap(self,data):
        self.mocap_info = data

    def kill_robot(self):
        stop_and_exit_roach(self.xb, self.lock, self.robots, self.use_pid_mode)
        #Prevents sys.exit(1) from being called at end
        #stop_and_exit_roach_special(self.xb, self.lock, self.robots, self.use_pid_mode)

    def run(self,num_steps_for_rollout,desired_shape_for_rollout):

        #save theta*
        thetaStar = self.model.regressor.get_params()

        #lists for return vals
        self.traj_taken=[]
        self.actions_taken=[]
        list_robot_info=[]
        list_mocap_info=[]

        #lists for intermediate vals
        self.save_perp_dist=[]
        self.save_forward_dist=[]
        self.saved_old_forward_dist=[]
        self.save_moved_to_next=[]
        self.save_desired_heading=[]
        self.save_curr_heading=[]
        list_best_action_sequences = []
        curr_line_segment = 0
        old_curr_forward=0

        # Remove after testing
        curr_line_segment_test = 0
        old_curr_forward_test =0

        #init vars
        step=0
        optimal_action=[0,0]

        # init the "past k" window
        # The dimensions are hard-coded
        inputa = np.empty((0, 24))
        labela = np.empty((0, 24))
        K = self.update_batch_size

        while True:

            if(step%10==0):
                print("     step #: ", step)

            ########################
            ##### SEND COMMAND #####
            ########################

            self.lock.acquire()
            for robot in self.robots:
              send_action = np.copy(optimal_action)
              print("\nsent action: ", send_action[0], send_action[1])
              if(self.use_pid_mode):
                  robot.setVelGetTelem(send_action[0], send_action[1])
              else:
                  robot.setThrustGetTelem(send_action[0], send_action[1])
            self.lock.release()

            ########################
            #### RECEIVE STATE #####
            ########################

            got_data=False
            start_time = time.time()

            if step == 0:
                big_loop_start = time.time()
            while(got_data==False):
                #print("DT: ", time.time() - big_loop_start)
                #big_loop_start = time.time()
                if (time.time() - start_time)%5 == 0:
                    print("Controller is waiting to receive data from robot")
                if (time.time() - start_time) > 10:
                    # Unsuccessful run; roach stopped communicating with xbee
                    stop_roach(self.lock, self.robots, self.use_pid_mode)
                    sys.exit()
                for q in shared.imu_queues.values():
                    #while loop, because sometimes, you get multiple things from robot
                    #but they're all same, so just use the last one
                    while not q.empty():
                        d = q.get()
                        got_data=True

            if(got_data):
                robotinfo=velroach_msg()
                robotinfo.stamp = rospy.Time.now()
                robotinfo.curLeft = optimal_action[0]
                robotinfo.curRight = optimal_action[1]
                robotinfo.posL = d[2]
                robotinfo.posR = d[3]
                robotinfo.gyroX = d[8]
                robotinfo.gyroY = d[9]
                robotinfo.gyroZ = d[10]
                robotinfo.bemfL = d[14]
                robotinfo.bemfR = d[15]
                robotinfo.vBat = d[16]
                self.publish_robotinfo.publish(robotinfo)
                #print("got state")

            #collect info to save for later
            list_robot_info.append(robotinfo)
            list_mocap_info.append(self.mocap_info)

            if(step==0):
                old_time= -7
                old_pos= self.mocap_info.pose.position #curr pos
                old_al= robotinfo.posL/math.pow(2,16)*2*math.pi #curr al
                old_ar= robotinfo.posR/math.pow(2,16)*2*math.pi #curr ar

            #check dt of controller
            if(step>0):
                step_dt = (robotinfo.stamp.secs-old_time.secs) + (robotinfo.stamp.nsecs-old_time.nsecs)*0.000000001
                print("DT: ", step_dt)

            #create state from the info
            full_curr_state, _, _, _, _ = singlestep_to_state(robotinfo, self.mocap_info, old_time, old_pos, old_al, old_ar, "all")
            abbrev_curr_state, old_time, old_pos, old_al, old_ar = singlestep_to_state(robotinfo, self.mocap_info, old_time, old_pos, old_al, old_ar, self.state_representation)

            ########################
            ##### DESIRED TRAJ #####
            ########################

            if(step==0):
                print("starting x position: ", full_curr_state[self.x_index])
                print("starting y position: ", full_curr_state[self.y_index])
                self.policy.desired_states = make_trajectory(desired_shape_for_rollout, np.copy(full_curr_state), self.x_index, self.y_index)

            #########################
            ## CHECK STOPPING COND ##
            #########################

            if(step>num_steps_for_rollout):
                
                print("DONE TAKING ", step, " STEPS.")

                #stop roach
                stop_roach(self.lock, self.robots, self.use_pid_mode)

                #return stuff to save
                old_saving_format_dict={
                'actions_taken': self.actions_taken,
                'desired_states': self.policy.desired_states,
                'traj_taken': self.traj_taken,
                'save_perp_dist': self.save_perp_dist,
                'save_forward_dist': self.save_forward_dist,
                'saved_old_forward_dist': self.saved_old_forward_dist,
                'save_moved_to_next': self.save_moved_to_next,
                'save_desired_heading': self.save_desired_heading,
                'save_curr_heading': self.save_curr_heading,
                }

                return(self.traj_taken, self.actions_taken, self.policy.desired_states, list_robot_info, list_mocap_info, old_saving_format_dict, list_best_action_sequences)

            ########################
            ### UPDATE REGRESSOR ###
            ########################
            # if step >= 2:
            #     s = create_nn_input_using_staterep(self.traj_taken[-2], self.state_representation)
            #     a = self.actions_taken[-2]
            #     ds = self.traj_taken[-1]-self.traj_taken[-2]
            #     if step < 2 + K: 
            #         #IPython.embed() # check s, a shapes for concat
            #         inputa = np.append(inputa, np.expand_dims(np.concatenate((s, a)), axis=1).T, axis=0)
            #         labela = np.append(labela, np.expand_dims(ds, axis=1).T, axis=0)
            #         #IPython.embed() #check the resulting shapes

            #         # Tile, then get rid of excess
            #         IPython.embed()
            #         reps = int(np.ceil(K/float(inputa.shape[0])))
            #         inputa_temp = np.tile(inputa, (reps,1))
            #         labela_temp = np.tile(labela, (reps,1))

            #         inputa_temp = np.delete(inputa_temp, range(inputa.shape[0]*reps % K), axis=0)
            #         labela_temp = np.delete(labela_temp, range(inputa.shape[0]*reps % K), axis=0)

            #         inputa_temp = np.expand_dims(inputa_temp, axis=0)
            #         labela_temp = np.expand_dims(labela_temp, axis=0)

            #         feed_dict = {self.model.inputa: inputa_temp, self.model.labela: labela_temp}
            #     else: 
            #         inputa = np.delete(inputa, 0, axis=0)
            #         labela = np.delete(labela, 0, axis=0)

            #         #IPython.embed() # check s, a shapes for concat
            #         inputa = np.append(inputa, np.expand_dims(np.concatenate((s, a)), axis=1).T, axis=0)
            #         labela = np.append(labela, np.expand_dims(ds, axis=1).T, axis=0)
            #         #IPython.embed() #check the resulting shapes

            #         inputa = np.expand_dims(inputa, axis=0)
            #         labela = np.expand_dims(labela, axis=0)

            #         feed_dict = {self.model.inputa: inputa, self.model.labela: labela}

            #     #reset weights of regressor to theta*
            #     self.model.regressor.set_params(thetaStar)
            #     print("....done resetting to theta*")
            #     #self.model.regressor.set_params(thetaCurrent)

            #     #take gradient step on theta* using the past K points
            #     for _ in range(self.num_updates):
            #         print("taking a gradient step")
            #         self.sess.run([self.model.test_op], feed_dict=feed_dict)

            # get the past K points (s,a,ds)
            length= len(self.traj_taken)
            K = self.update_batch_size
            if length >= 2:
                if length <= K:
                    list_of_s=[]
                    list_of_a=[]
                    list_of_ds=[]

                    for i in range(length-1):
                        s = create_nn_input_using_staterep(self.traj_taken[i], self.state_representation)
                        a = self.actions_taken[i]
                        ds = self.traj_taken[i+1]-self.traj_taken[i]

                        list_of_s.append(s)
                        list_of_a.append(a)
                        list_of_ds.append(ds)

                    # count = 0
                    # for j in range(K-(length-1)):
                    #     list_of_s.append(list_of_s[i])
                    #     list_of_a.append(list_of_a[i])
                    #     list_of_ds.append(list_of_ds[i])
                    #     count+=1
                    #     if count >= (length-1):
                    #         count = 0

                    for j in range(K-(length-1)):
                        list_of_s.append(s)
                        list_of_a.append(a)
                        list_of_ds.append(ds)

                if(length>K):
                    i= length-1-K
                    list_of_s=[]
                    list_of_a=[]
                    list_of_ds=[]
                    while(i<(length-1)): # You can implement this faster by using a queue
                        list_of_s.append(create_nn_input_using_staterep(self.traj_taken[i], self.state_representation))
                        list_of_a.append(self.actions_taken[i])
                        list_of_ds.append(self.traj_taken[i+1]-self.traj_taken[i])
                        i+=1
                list_of_s= np.array(list_of_s)
                list_of_a= np.array(list_of_a)
                list_of_ds= np.array(list_of_ds)

                #to do: speed this section up by doing fewer conversions/list-making
                # Couldn't you use numpy and vector operations to parallelize

                #organize the points into what the regressor wants
                k_labels = (list_of_ds).reshape(1, K, -1)
                k_inputs = np.concatenate([list_of_s, list_of_a], axis=-1).reshape(1, K, -1)
                feed_dict = {self.model.inputa: k_inputs, self.model.labela: k_labels}

                #reset weights of regressor to theta*
                self.model.regressor.set_params(thetaStar)
                print("....done resetting to theta*")
                #self.model.regressor.set_params(thetaCurrent)

                #take gradient step on theta* using the past K points
                for _ in range(self.num_updates):
                    print("taking a gradient step")
                    self.sess.run([self.model.test_op], feed_dict=feed_dict)

            ########################
            #### COMPUTE ACTION ####
            ########################
            old_curr_forward_before = old_curr_forward
            curr_line_segment_before = curr_line_segment
            optimal_action_before = optimal_action
            #get the best action
            optimal_action, curr_line_segment, old_curr_forward, info_for_saving, best_sequence_of_actions, best_sequence_of_states = self.policy.get_best_action(np.copy(full_curr_state), np.copy(optimal_action), curr_line_segment, old_curr_forward, "red")            

            #reset weights of regressor to theta*
            self.model.regressor.set_params(thetaStar)
            # For debugging only: get best action for theta* and also plot it
            optimal_action_test, curr_line_segment_test, old_curr_forward_test, info_for_saving_test, best_sequence_of_actions_test, best_sequence_of_states_test = self.policy.get_best_action(np.copy(full_curr_state), np.copy(optimal_action_before), curr_line_segment_before, old_curr_forward_before, "green")          

            ########################
            ######## SAVING ########
            ########################

            #save (s,a) from robot execution (use next state in list as s')
            self.traj_taken.append(full_curr_state)
            self.actions_taken.append(optimal_action)

            #append vars for later saving
            self.save_perp_dist.append(info_for_saving['save_perp_dist'])
            self.save_forward_dist.append(info_for_saving['save_forward_dist'])
            self.saved_old_forward_dist.append(info_for_saving['saved_old_forward_dist'])
            self.save_moved_to_next.append(info_for_saving['save_moved_to_next'])
            self.save_desired_heading.append(info_for_saving['save_desired_heading'])
            self.save_curr_heading.append(info_for_saving['save_curr_heading'])
            list_best_action_sequences.append(best_sequence_of_actions)

            #keep looping
            self.rate.sleep()
            step+=1
