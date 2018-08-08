#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

import threading

N_ROBOT = 1
N_STICK = 2
pubs=[]

def vo_to_twist(vo):
  twist = Twist()
  twist.linear.x = vo[0][0]
  twist.linear.y = vo[2][0]
  twist.linear.z = vo[2][1]
  twist.angular.x = vo[2][2]
  twist.angular.y = vo[2][3]
  twist.angular.z = vo[1][0]
  return twist

class ManJoyState():
  def __init__(self):
    self.cmd_in = (0,0)
    self.joy_in = [(0,0)]*N_STICK
    self.buttons = [0, 0, 0, 0]
    self.lock = threading.Condition()

  def joy_callback(self, joy_msg):
    self.lock.acquire()
    # update joy_in based on analog axes
    self.joy_in[0] = (joy_msg.axes[1], joy_msg.axes[0])
    self.joy_in[1] = (joy_msg.axes[4], joy_msg.axes[3])
    
    # update desired destinations based on buttons
    for i in range(0,4):
       self.buttons[i] = joy_msg.buttons[i]
    self.lock.release()

    ##### ONLY PUBLISH ONTO CMD_VEL WHEN YOU GET SOMETING FROM JOYSTICK
    '''vo_cmd = [self.joy_in[0], self.joy_in[1], self.buttons]
    pubs[0].publish(vo_to_twist(vo_cmd))
    self.lock.release()'''

  def cmd_callback(self, which, tw_msg):
    self.lock.acquire()
    # update cmd_in
    self.cmd_in = (tw_msg.linear.x, tw_msg.angular.z)
    rospy.loginfo('callback cmd_in[%d] = (%f,%f)' % 
      (which, self.cmd_in[0], self.cmd_in[1]))
    self.lock.release()
    
def talker():
  global pubs 

  state = ManJoyState()  
  rospy.init_node('man_joy_override', anonymous=True)

  pubs = []
  for i in range(N_ROBOT):
    pubs.append(rospy.Publisher('robot%d/cmd_vel' % i, Twist, queue_size = 1))
    
    # need to create a new functional scope to callback based on topic number
    def curried_callback(j):
      return lambda m: state.cmd_callback(j,m)

    rospy.Subscriber('robot%d/cmd_vel_in' % i, Twist, curried_callback(i))
  
  rospy.Subscriber('joy', Joy, state.joy_callback)
  
  r = rospy.Rate(200) 
  #^^^ make this fast, this just updates the /robot0/cmd_vel command, which updates the "state" var in roach_bridge... doesnt affect any comm rates or anything

  while not rospy.is_shutdown():

    for i in range(N_ROBOT):
      state.lock.acquire()
      which = i

      #######PUBLISH AT CONSTANT RATE
      vo_cmd = [state.joy_in[0], state.joy_in[1], state.buttons]
      pubs[i].publish(vo_to_twist(vo_cmd))
      state.lock.release()

    r.sleep()

if __name__ == '__main__':
  try:
    talker()
  except rospy.ROSInterruptException: pass
