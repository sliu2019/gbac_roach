
import shared_multi as shared
from velociroach import *
from utils import *
from collections import OrderedDict
#from queue import Queue #for python3
from Queue import Queue

def setup_roach(serial_port, baud_rate, default_addrs, use_pid_mode, top):

	#setup serial
	xb = None
	try:
		xb = setupSerial(serial_port, baud_rate)
		print("Done setting up serial.\n")
	except:
		print('Failed to set up serial, exiting')

	#setup the roach
	if xb is not None:
		shared.xb = xb
		robots = [Velociroach(addr,xb) for addr in default_addrs]
		n_robots=len(robots)

		for r in robots:
			r.running = False
			r.VERBOSE = False
			r.setPIDOutputChannel(1)
			if(use_pid_mode):
				r.PIDStartMotors()
				r.running = True
			r.zeroPosition() 
		shared.ROBOTS = robots

		#setup the info receiving
		shared.imu_queues = OrderedDict()
		shared.imu_queues[robots[0].DEST_ADDR_int] = Queue()

		print("Done setting up RoachBridge.\n")
	print(xb)
	return xb, robots, shared.imu_queues

def start_fans(lock, robot):
	lock.acquire()
	for i in range(3):
		robot.startFans()
		time.sleep(1)
	lock.release()

def stop_fans(lock, robot):
	lock.acquire()
	for i in range(3):
		robot.stopFans()
		time.sleep(1)
	lock.release()

def start_roach(xb, lock, robots, use_pid_mode):
	print("starting roach")

	#set thrust for both motors to 0
	lock.acquire()
	for robot in robots:
		if(use_pid_mode):
			robot.PIDStartMotors()
			robot.running = True
		robot.setThrustGetTelem(0, 0) 

		
	lock.release()
	return

def stop_roach(lock, robots, use_pid_mode):
	#set thrust for both motors to 0
	lock.acquire()
	for robot in robots:
		if(use_pid_mode):
			robot.setVelGetTelem(0,0)
			# robot.PIDStopMotors()
			# robot.running = False
		else:
			robot.setThrustGetTelem(0, 0)
			#robot.downloadTelemetry() 
	lock.release()
	#IPython.embed()
	return

def stop_and_exit_roach(xb, lock, robots, use_pid_mode):
	#set thrust for both motors to 0
	lock.acquire()
	for robot in robots:
		if(use_pid_mode):
			#robot.setVelGetTelem(0,0)
			#print("before stoping motors")
			#IPython.embed()
			robot.PIDStopMotors()
			robot.running = False
			#IPython.embed()
		else:
			robot.setThrustGetTelem(0, 0) 
			### robot.downloadTelemetry()
	lock.release()

	#IPython.embed()
	#exit RoachBridge
	xb_safe_exit(xb)
	return
	
def stop_and_exit_roach_special(xb, lock, robots, use_pid_mode):
	# Same as above, except calling xb_safe_exitCollect prevents sys.exit(1) from occurring at end
	#set thrust for both motors to 0
	lock.acquire()
	for robot in robots:
		if(use_pid_mode):
			#robot.setVelGetTelem(0,0)
			robot.PIDStopMotors()
			robot.running = False
			#IPython.embed()
		else:
			robot.setThrustGetTelem(0, 0) 
			### robot.downloadTelemetry()
	lock.release()

	#exit RoachBridge
	xb_safe_exitCollect(xb)
	return