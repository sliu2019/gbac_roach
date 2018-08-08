import rospy
import os, sys
#add nn_dynamics_roach to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#my imports
import time
import command
import shared_multi as shared
from velociroach import *
from nn_dynamics_roach.msg import velroach_msg


publish_robotinfo= rospy.Publisher('/robot0/robotinfo', velroach_msg, queue_size=5)

while True:
	got_data = False
	start_time= time.time()
	while(got_data==False):
		  if (time.time() - start_time)%5 == 0:
			print("Controller is waiting to receive data from robot")
		  for q in shared.imu_queues.values():
			#while loop, because sometimes, you get multiple things from robot
			#but they're all same, so just use the last one
			while not q.empty():
			  d = q.get()
			  got_data=True

	if(got_data):
		publish_robotinfo.publish(robotinfo)