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

#roach stuff
import rospy
from trajectories import make_trajectory
from rospy.exceptions import ROSException
from threading import Condition
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from roach_utils import *

class GBAC_Controller(object):
    def __init__(
            self,
            policy,
            use_pid_mode=True,
            frequency_value= 10,
            serial_port= '/dev/ttyUSB0',
            baud_rate= 57600,
            default_addrs= ['\x00\x01'],#anything after this is unused by this code
            x_index= 0,
            y_index= 1,
            yaw_cos_index= 10,
            yaw_sin_index= 11,
            visualize_rviz= True,
    ):
        self.policy = policy
        self.use_pid_mode = use_pid_mode
        self.frequency_value = frequency_value
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.default_addrs = default_addrs

        self.lock = Condition()
        self.mocap_info = PoseStamped()

        #init vars
        self.x_index = 0
        self.y_index = 1
        
        #init node
        rospy.init_node('MAML_roach_controller_node', anonymous=True)
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
        self.publish_markers= self.policy.publish_markers = rospy.Publisher('visualize_selected', MarkerArray, queue_size=5)
        self.publish_markers_desired= self.policy.publish_markers_desired = rospy.Publisher('visualize_desired', MarkerArray, queue_size=5)

    def callback_mocap(self,data):
        self.mocap_info = data

    def kill_robot(self):
        stop_and_exit_roach(self.xb, self.lock, self.robots, self.use_pid_mode)
        #Prevents sys.exit(1) from being called at end
        #stop_and_exit_roach_special(self.xb, self.lock, self.robots, self.use_pid_mode)

    def run(self,num_steps_for_rollout,desired_shape_for_rollout):

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
        curr_line_segment = 0
        old_curr_forward=0

        #init vars
        step=0
        optimal_action=[0,0]

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
            while(got_data==False):
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
                'desired_states': self.desired_states,
                'traj_taken': self.traj_taken,
                'save_perp_dist': self.save_perp_dist,
                'save_forward_dist': self.save_forward_dist,
                'saved_old_forward_dist': self.saved_old_forward_dist,
                'save_moved_to_next': self.save_moved_to_next,
                'save_desired_heading': self.save_desired_heading,
                'save_curr_heading': self.save_curr_heading,
                }

                return(self.traj_taken, self.actions_taken, self.desired_states, list_robot_info, list_mocap_info, old_saving_format_dict)

            ########################
            #### COMPUTE ACTION ####
            ########################

            if(step==0):
                #create desired trajectory
                print("starting x position: ", full_curr_state[self.x_index])
                print("starting y position: ", full_curr_state[self.y_index])
                self.desired_states = self.policy.desired_states = make_trajectory(desired_shape_for_rollout, np.copy(full_curr_state), self.x_index, self.y_index)


            #save traj taken
            self.traj_taken.append(full_curr_state)

            #get the best action
            optimal_action, curr_line_segment, old_curr_forward, info_for_saving = self.policy.get_best_action(np.copy(full_curr_state), np.copy(optimal_action), curr_line_segment, old_curr_forward)            

            #append vars for later saving
            self.actions_taken.append(optimal_action)
            self.save_perp_dist.append(info_for_saving['save_perp_dist'])
            self.save_forward_dist.append(info_for_saving['save_forward_dist'])
            self.saved_old_forward_dist.append(info_for_saving['saved_old_forward_dist'])
            self.save_moved_to_next.append(info_for_saving['save_moved_to_next'])
            self.save_desired_heading.append(info_for_saving['save_desired_heading'])
            self.save_curr_heading.append(info_for_saving['save_curr_heading'])

            #keep looping
            self.rate.sleep()
            step+=1
