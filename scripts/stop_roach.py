#!/usr/bin/env python

import math
import rospy
import numpy as np
import numpy.random as npr
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
import copy 
import sys
import tensorflow as tf
import signal
from threading import Condition
import thread
from Queue import Queue
from collections import OrderedDict
import time, sys, os, traceback
import serial
import math

#add nn_dynamics_roach to sys.path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#my imports
from utils import *
import command

#####################################################
#####################################################

#vars to specify

DEFAULT_ADDRS = ['\x00\x01']
serial_port = '/dev/ttyUSB1'
baud_rate = 57600

#####################################################
#####################################################

def run():

  #init node
  rospy.init_node('stop_node', anonymous=True)
  lock = Condition()

  #setting up serial and roach bridge
  xb, robots, _ = setup_roach(serial_port, baud_rate, DEFAULT_ADDRS, False)

  #stop the roach
  stop_and_exit_roach(xb, lock, robots, False)


if __name__ == '__main__':
  run()