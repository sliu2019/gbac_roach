#!/usr/bin/env python

import math
import rospy
import numpy as np
import numpy.random as npr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
import copy 
import sys
import tensorflow as tf
import signal
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import time
import IPython
import matplotlib.pyplot as plt

#mine
from dynamics_model import Dyn_Model
from trajectories import make_trajectory
from dynamics_model import Dyn_Model
from compute_action import Actions

#others
from threading import Condition
import thread
from Queue import Queue
from collections import OrderedDict

# Roach Imports
import command
import time, traceback
import serial
from velociroach import *
from roach_dynamics_learning.msg import velroach_msg
import shared_multi as shared
import math


class DiffDriveController(object):

    def __init__(self, dt_steps, min_motor_gain, max_motor_gain, nominal, weight_horiz, weight_heading, traj_save_path, frequency_value=10, actionSize=2):

      self.do_telem=False
      self.traj_save_path = traj_save_path
      self.desired_shape_for_traj = traj_save_path[:-1] #straight, left, right, circle_left, zigzag, figure8

      self.save_dir = "../run_907"
      if not os.path.exists(self.save_dir):
        os.makedirs(self.save_dir)
      if not os.path.exists(self.save_dir+'/'+traj_save_path):
        os.makedirs(self.save_dir+'/'+traj_save_path)

      self.dt_steps = dt_steps
      self.x_index=0
      self.y_index=1
      self.yaw_index=5

      #vars
      self.frequency_value = frequency_value
      self.DEFAULT_ADDRS = ['\x00\x01']
      self.lock = Condition()
      self.mocap_info = PoseStamped()
      self.command_from_joystick = [0,0]
      self.action_shape = (actionSize,)

      #actions
      self.nominal = nominal
      self.min_motor_gain= min_motor_gain
      self.max_motor_gain= max_motor_gain

      #these no longer exist (for new controller)
      self.forward_encouragement_factor=0
      self.perp_dist_threshold = 0

      #params
      self.horiz = weight_horiz
      self.heading = weight_heading

      self.setup()

    def setup(self):

      #init node
      rospy.init_node('diffdrive_controller_node', anonymous=True)
      self.rate = rospy.Rate(self.frequency_value)
      
      #setting up serial
      self.xb = None
      try:
        self.xb = setupSerial('/dev/ttyUSB2',57600)
      except:
        print('Failed to set up serial, exiting')

      #setting up roach bridge
      if self.xb is not None:

        #setup roach
        shared.xb = self.xb
        self.robots = [Velociroach(addr,self.xb) for addr in self.DEFAULT_ADDRS]
        self.n_robots=len(self.robots)
        for r in self.robots:
          r.running = False
          r.VERBOSE = False
        shared.ROBOTS = self.robots
        for robot in self.robots:
          #self.STOP_ROBOT(robot)
          robot.setMotorGains([1800,200,100,0,0, 1800,200,100,0,0])
          robot.zeroPosition()
          

          if(self.do_telem):
            #new telem stuff
            robot.setupTelemetryDataNum(10000)
            time.sleep(1)
            robot.zeroPosition()

            time.sleep(1)
            robot.eraseFlashMem()
            time.sleep(1)
            robot.startTelemetrySave()
            #time.sleep(2)
          self.START_ROBOT(robot)

        #setup the info receiving... although not using right now
        shared.imu_queues = OrderedDict()
        shared.imu_queues[self.robots[0].DEST_ADDR_int] = Queue()

      #make publisher/subscriber
      self.sub_mocap = rospy.Subscriber('/mocap/pose', PoseStamped, self.callback_mocap)
      self.publish_markers_desired= rospy.Publisher('visualize_desired', MarkerArray, queue_size=5)

    def callback_mocap(self,data):
      self.mocap_info = data


    def run(self,num_steps_for_rollout):

      #init values for the loop below
      self.traj_taken=[]
      self.actions_taken=[]
      self.save_perp_dist=[]
      self.save_forward_dist=[]
      self.saved_old_forward_dist=[]
      self.save_moved_to_next=[]
      self.save_desired_heading=[]
      self.save_curr_heading=[]

      old_curr_forward=0
      self.curr_line_segment = 0
      take_steps=True
      num_iters=-1

      old_state = 0
      dt=1
      optimal_action=[0,0]

      while(take_steps==True):

        if(num_iters%10==0):
          print "\n", "****** step #: ", num_iters

        ########################
        ##### SEND COMMAND #####
        ########################

        self.lock.acquire()
        for robot in self.robots:
          ##send_action = [0, 0]
          send_action = np.copy(optimal_action)
          if(num_iters>(num_steps_for_rollout-2)):
            send_action=[0,0]
          print "\nsent action: ", send_action[0], send_action[1]
          robot.setVelGetTelem(send_action[0], send_action[1]) 
        self.lock.release()

        ########################
        #### UPDATE STATE #####
        ########################

        num_iters+=1

        #calculate dt
        curr_time=rospy.Time.now()
        if(num_iters>0):
          dt = (curr_time.secs-old_time.secs) + (curr_time.nsecs-old_time.nsecs)*0.000000001

        #create state vector from the received data
        x= self.mocap_info.pose.position.x
        y= self.mocap_info.pose.position.y
        z= self.mocap_info.pose.position.z
        if(num_iters==0):
          vx= 0
          vy= 0
          vz= 0
        else:
          vx= (x-old_state[0])/dt
          vy= (y-old_state[1])/dt 
          vz= (z-old_state[2])/dt
        angles = self.quat_to_euler(self.mocap_info.pose.orientation)
        r= angles[0]
        p= angles[1]
        yw= angles[2]

        #convert to rad
        r=r*np.pi/180.0
        p=p*np.pi/180.0
        yw=yw*np.pi/180.0

        curr_state=[x,y,z,r,p,yw,vx,vy,vz] #angles in radians

        #print("created state")

        #########################
        ## CHECK STOPPING COND ##
        #########################

        if(num_iters>=num_steps_for_rollout):

          print("DONE TAKING ", num_iters, " STEPS.")

          for robot in self.robots:
            robot.PIDStopMotors()
            time.sleep(.5)
            robot.setThrustGetTelem(0,0)

            robot.running = False
            time.sleep(.5)

            if(self.do_telem):
              robot.downloadTelemetry()

          #save
          take_steps=False
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_actions.npy', self.actions_taken)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_desired.npy', self.desired_states)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_executed.npy', self.traj_taken)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_perp.npy', self.save_perp_dist)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_forward.npy', self.save_forward_dist)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_oldforward.npy', self.saved_old_forward_dist)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_movedtonext.npy', self.save_moved_to_next)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_desheading.npy', self.save_desired_heading)
          np.save(self.save_dir +'/'+ self.traj_save_path +'/diffdrive_currheading.npy', self.save_curr_heading)

          time.sleep(1)

      

          print('RoachBridge exiting.')
          xb_safe_exit(self.xb)
          return

        ########################
        #### COMPUTE ACTION ####
        ########################

        if(num_iters==0):
          #create desired trajectory
          print("starting x position: ", curr_state[self.x_index])
          print("starting y position: ", curr_state[self.y_index])
          self.desired_states = make_trajectory(self.desired_shape_for_traj, np.copy(curr_state), self.x_index, self.y_index)

        if(num_iters%self.dt_steps == 0):
          self.traj_taken.append(curr_state)
          optimal_action, save_perp_dist, save_forward_dist, save_moved_to_next, save_desired_heading, save_curr_heading = self.compute_optimal_diffdrive_action(np.copy(curr_state), np.copy(optimal_action), num_iters)
          
          
          self.actions_taken.append(optimal_action)
          self.save_perp_dist.append(save_perp_dist)
          self.save_forward_dist.append(save_forward_dist)
          self.saved_old_forward_dist.append(old_curr_forward)
          self.save_moved_to_next.append(save_moved_to_next)
          self.save_desired_heading.append(save_desired_heading)
          self.save_curr_heading.append(save_curr_heading)

          old_curr_forward = np.copy(save_forward_dist)

        #print("computed action")

        ########################
        ## WAIT FOR SOME TIME ##
        ########################

        old_time= copy.deepcopy(curr_time)
        old_state=np.copy(curr_state)
        self.rate.sleep()

    def STOP_ROBOT(self,r):
      r.PIDStopMotors()
      r.running = False

    def START_ROBOT(self,r):
      r.PIDStartMotors()
      r.running = True

    def quat_to_euler(self,orientation):
      x=orientation.x
      y=orientation.y
      z=orientation.z
      w=orientation.w

      ysqr = y*y
      
      t0 = +2.0 * (w * x + y*z)
      t1 = +1.0 - 2.0 * (x*x + ysqr)
      X = math.degrees(math.atan2(t0, t1))
      
      t2 = +2.0 * (w*y - z*x)
      t2 =  1 if t2 > 1 else t2
      t2 = -1 if t2 < -1 else t2
      Y = math.degrees(math.asin(t2))
      
      t3 = +2.0 * (w * z + x*y)
      t4 = +1.0 - 2.0 * (ysqr + z*z)
      Z = math.degrees(math.atan2(t3, t4))
      
      return [X,Y,Z]


    def compute_optimal_diffdrive_action(self,curr_state, currently_executing_action, step):

      x_index = self.x_index
      y_index = self.y_index
      yaw_index = self.yaw_index

      move_to_next= 0
      curr_seg = int (np.copy(self.curr_line_segment))
      moved_to_next = 0

      #array of "the point"... for each sim
      pt = np.copy(curr_state) # N x state

      #arrays of line segment points... for each sim
      curr_start = self.desired_states[curr_seg]
      curr_end = self.desired_states[curr_seg+1]
      next_start = self.desired_states[curr_seg+1]
      next_end = self.desired_states[curr_seg+2]

      ############ closest distance from point to curr line segment

      #vars
      a = pt[x_index]- curr_start[0]
      b = pt[y_index]- curr_start[1]
      c = curr_end[0]- curr_start[0]
      d = curr_end[1]- curr_start[1]

      #project point onto line segment
      which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

      #point on line segment that's closest to the pt
      closest_pt_x = np.copy(which_line_section)
      closest_pt_y = np.copy(which_line_section)
      if(which_line_section<0):
        closest_pt_x=curr_start[0]
        closest_pt_y=curr_start[1]
      elif(which_line_section>1):
        closest_pt_x=curr_end[0]
        closest_pt_y=curr_end[1]
      else:
        closest_pt_x= curr_start[0] + np.multiply(which_line_section,c)
        closest_pt_y= curr_start[1] + np.multiply(which_line_section,d)

      #min dist from pt to that closest point (ie closest dist from pt to line segment)
      min_perp_dist = np.sqrt((pt[x_index]-closest_pt_x)*(pt[x_index]-closest_pt_x) + (pt[y_index]-closest_pt_y)*(pt[y_index]-closest_pt_y))

      #"forward-ness" of the pt... for each sim
      curr_forward = which_line_section

      ############ closest distance from point to next line segment

      #vars
      a = pt[x_index]- next_start[0]
      b = pt[y_index]- next_start[1]
      c = next_end[0]- next_start[0]
      d = next_end[1]- next_start[1]

      #project point onto line segment
      which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

      #point on line segment that's closest to the pt
      closest_pt_x = np.copy(which_line_section)
      closest_pt_y = np.copy(which_line_section)
      if(which_line_section<0):
        closest_pt_x=next_start[0]
        closest_pt_y=next_start[1]
      elif(which_line_section>1):
        closest_pt_x=next_end[0]
        closest_pt_y=next_end[1]
      else:
        closest_pt_x= next_start[0] + np.multiply(which_line_section,c)
        closest_pt_y= next_start[1] + np.multiply(which_line_section,d)

      #min dist from pt to that closest point (ie closes dist from pt to line segment)
      dist = np.sqrt((pt[x_index]-closest_pt_x)*(pt[x_index]-closest_pt_x) + (pt[y_index]-closest_pt_y)*(pt[y_index]-closest_pt_y))

      #pick which line segment it's closest to, and update vars accordingly
      if(dist<=min_perp_dist):
        curr_seg += 1
        moved_to_next = 1
        curr_forward = which_line_section
        min_perp_dist=dist

      ################## publish the desired traj
      markerArray2 = MarkerArray()
      marker_id=0
      for des_pt_num in range(8): #5 for all, 8 for circle, 4 for zigzag
        marker = Marker()
        marker.id=marker_id
        marker.header.frame_id = "/world"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.pose.position.x = self.desired_states[des_pt_num,0]
        marker.pose.position.y = self.desired_states[des_pt_num,1]
        marker.pose.position.z = 0
        markerArray2.markers.append(marker)
        marker_id+=1
      self.publish_markers_desired.publish(markerArray2)

      ################## advance which line segment we are on
      if(moved_to_next==1):
          self.curr_line_segment+=1
          print("**************************** MOVED ONTO NEXT LINE SEG")

      ################## given the current state, the perpendicular distance, and desired heading, calculate the velocities to output for each motor

      #what side of the line segment is the point on
      angle_to_point = np.arctan2(pt[y_index]-curr_start[1], pt[x_index]-curr_start[0])
      angle_of_line = np.arctan2(curr_end[1]-curr_start[1], curr_end[0]-curr_start[0])

      if(moving_distance(angle_of_line, angle_to_point)>0):
        on_right = True
      else:
        on_right = False

      #angle for moving robot toward desired heading
      heading_turn_amount = moving_distance(angle_of_line, pt[yaw_index])

      if(on_right):
        heading_turn_amount += self.horiz*min_perp_dist
      else:
        heading_turn_amount -= self.horiz*min_perp_dist

      #angle --> velocities
      left_vel = self.nominal - heading_turn_amount*self.heading
      right_vel = self.nominal + heading_turn_amount*self.heading

      #clip
      if(left_vel>self.max_motor_gain):
        left_vel=self.max_motor_gain
      if(left_vel<self.min_motor_gain):
        left_vel=self.min_motor_gain
      if(right_vel>self.max_motor_gain):
        right_vel=self.max_motor_gain
      if(right_vel<self.min_motor_gain):
        right_vel=self.min_motor_gain

      action_to_take = [left_vel*math.pow(2,16)*0.001, right_vel*math.pow(2,16)*0.001]

      save_perp_dist= min_perp_dist
      save_forward_dist= curr_forward
      save_moved_to_next= moved_to_next
      save_desired_heading= angle_of_line
      save_curr_heading= pt[yaw_index]
      return action_to_take, save_perp_dist, save_forward_dist, save_moved_to_next, save_desired_heading, save_curr_heading

#distance needed for unit 2 to go toward unit1
def moving_distance(unit1, unit2):
  phi = (unit2-unit1) % (2*np.pi)
  sign = -1
  # used to calculate sign
  if not ((phi >= 0 and phi <= np.pi) or (
          phi <= -np.pi and phi >= -2*np.pi)):
      sign = 1
  if phi > np.pi:
      result = 2*np.pi-phi
  else:
      result = phi
  return result*sign

def main():

  is_sty=True
  
  #parameters to tune
  min_motor_gain= 2
  max_motor_gain= 9
  nominal = 7

  weight_horiz= 3 #6
  weight_heading= 5 #10

  traj_save_path = str(sys.argv[1]) #ex. straight0
  dc = DiffDriveController(dt_steps=1, min_motor_gain=min_motor_gain, max_motor_gain=max_motor_gain, nominal=nominal, weight_horiz=weight_horiz, weight_heading=weight_heading, traj_save_path=traj_save_path, frequency_value=10, actionSize=2)

  if(traj_save_path[:-1]=='straight'):
    numSteps=70
  elif(traj_save_path[:-1]=='left'):
    numSteps=120
    if(is_sty):
      numSteps=150
  elif(traj_save_path[:-1]=='right'):
    numSteps=150
  elif(traj_save_path[:-1]=='zigzag'):
    numSteps=200
  else:
    numSteps=0


  # numSteps=40

  time.sleep(.5)
  dc.run(numSteps)


if __name__ == '__main__':
    main()


