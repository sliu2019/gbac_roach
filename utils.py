
import math
import copy
import tensorflow as tf
import numpy as np
import IPython
import os
import pickle

def getDataFromDisk(itr, experiment_type, use_one_hot, use_camera, cheaty_training, state_representation):

  list_of_pathLists, num_training_rollouts = whichFiles(itr, experiment_type)
  print("\n\nNumber of training rollouts: ", num_training_rollouts)
  return getData(list_of_pathLists, num_training_rollouts, use_one_hot, use_camera, cheaty_training, state_representation)

def whichFiles(itr, experiment_type):

  ##############################
  #### SELECT DIRECTORY
  ##############################
  
  if(itr==0):
    #random data
    #data_path = "/media/camera_training_data"
    data_path = "/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/camera_training_data"
  else:
    print("\n\nNOT IMPLEMENTED: getting data (for anything other than random)...")
    IPython.embed()
  dirs = os.listdir(data_path)

  ##############################
  ### WHICH FILES FROM THAT DIR
  ##############################

  if(experiment_type=='terrain_types'):

    #task_list=['turf', 'carpet', 'styrofoam', 'gravel'] #################### this decides what to read in
    task_list=['carpet']
    months = ['all']
    max_runs_per_surface = 400 ###############################################################
    path_lsts = {"turf": [], "carpet":[], "styrofoam": [], "gravel": []}
    runs_per_surface = {"turf": 0, "carpet":0, "styrofoam":0, "gravel":0}
    num_training_rollouts=0

    for directory in dirs:  
      lst = directory.split("_")
      surface = lst[0]
      month = lst[2]
      day = lst[3]

      if(day in ['91']): #'23', '24'
        junk=1
      else:
        if (surface in task_list or ("all" in task_list and surface!="joystick")) and ((month in months) or ("all" in months)):
          directory_path = os.path.join(data_path, directory)
          dir_files = [x for x in os.listdir(directory_path) if not os.path.isdir(os.path.join(directory_path, x))]
          dir_files = [x for x in dir_files if (os.path.splitext(x)[1] == ".obj")]
          dir_files.sort()
          if (len(dir_files) % 3) != 0:
            print("Something wacky in folder: ", directory_path)
          for i in range(int(len(dir_files)/3)):
            if runs_per_surface[surface] < max_runs_per_surface:
              image_file = dir_files[3*i]
              mocap_file = dir_files[3*i +1]
              robot_file = dir_files[3*i + 2]
              num_training_rollouts+=1

              if image_file[0] == mocap_file[0] == robot_file[0]:
                path_lsts[surface].append(os.path.join(directory_path, image_file))
                path_lsts[surface].append(os.path.join(directory_path, mocap_file))
                path_lsts[surface].append(os.path.join(directory_path, robot_file))
                runs_per_surface[surface] = runs_per_surface[surface] + 1
              else: 
                print("error w reading initial random data")
                IPython.embed()

  else:
    print("\n\nNOT IMPLEMENTED: getting data (for experiment other than terrain_types)...")
    IPython.embed()

  list_of_pathLists = []
  for surf in task_list:
    list_of_pathLists.append(path_lsts[surf]) #looks sorted...
  #IPython.embed()
  return list_of_pathLists, num_training_rollouts

#each output of this should be
  #list of tasks
  #each task should have rollouts from that task
  #each rollout should have its points
def getData(list_of_pathLists, num_training_rollouts, use_one_hot, use_camera, cheaty_training, state_representation):

  dataX=[]
  dataX_full=[]
  dataY=[]
  dataZ=[]

  for path_lst in list_of_pathLists:

    dataX_curr = []
    dataX_curr_full = []
    dataY_curr = []
    dataZ_curr = []
    i=0
    while((3*i+2) < len(path_lst)):

      #read in data from 1 rollout (grouping of 3 files)
      camera_file = path_lst[3*i] #### ignoring this
      mocap_file = path_lst[3*i + 1]
      robot_file = path_lst[3*i+2]

      robot_info = pickle.load(open(robot_file,'rb'))
      mocap_info = pickle.load(open(mocap_file,'rb'))

      #turn saved rollout into s
      full_states_for_dataX, actions_for_dataY= rollout_to_states(robot_info, mocap_info, "all")
      abbrev_states_for_dataX, actions_for_dataY = rollout_to_states(robot_info, mocap_info, state_representation)
        #states_for_dataX: (length-1)x24 cuz ignore 1st one (no deriv)
        #actions_for_dataY: (length-1)x2

      #use s to create ds
      states_for_dataZ = full_states_for_dataX[1:,:]-full_states_for_dataX[:-1,:]

      #s,a,ds
      dataX_curr.append(abbrev_states_for_dataX[:-1,:]) #the last one doesnt have a corresponding next state
      dataX_curr_full.append(full_states_for_dataX[:-1,:])
      dataY_curr.append(actions_for_dataY[:-1,:])
      dataZ_curr.append(states_for_dataZ)

      #read the next rollout 
      i+=1

    dataX.append(dataX_curr)
    dataX_full.append(dataX_curr_full)
    dataY.append(dataY_curr)
    dataZ.append(dataZ_curr)
  
  return dataX, dataY, dataZ, dataX_full

def quat_to_eulerDegrees(orientation):
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

#datatypes
tf_datatype= tf.float32
np_datatype= np.float32

def create_onehot(curr_surface, use_camera = False, mappings= None):
    curr_onehot = None

    if (use_camera):

        index = 0
        if(curr_surface=='carpet'):
            index = 0
        if(curr_surface=='gravel'):
            index = 20
        if(curr_surface=='turf'):
            index = 30
        if(curr_surface=='styrofoam'):
            index = 10
        index += np.random.randint(10)
        curr_onehot = mappings[index]
        curr_onehot=np.array(list(curr_onehot) + [1])


        #mean vec (used to do this with subtracting old mean, rather than new mean in myalexnet)
        '''mean_carpet = np.mean(mappings[0:10,:], axis=0)
        mean_gravel = np.mean(mappings[10:20,:], axis=0)
        mean_turf = np.mean(mappings[20:30,:], axis=0)
        mean_sty = np.mean(mappings[30:40,:], axis=0)

        if(curr_surface=='carpet'):
            curr_onehot = mean_carpet
        if(curr_surface=='gravel'):
            curr_onehot = mean_gravel
        if(curr_surface=='turf'):
            curr_onehot = mean_turf
        if(curr_surface=='styrofoam'):
            curr_onehot = mean_sty
        curr_onehot=np.array(list(curr_onehot) + [1])'''
        
    else:
        #use one hot
        curr_onehot = np.zeros((1,4)).astype(np_datatype)
        if(curr_surface=='carpet'):
            curr_onehot[0,0]=1
        if(curr_surface=='gravel'):
            curr_onehot[0,1]=1
        if(curr_surface=='turf'):
            curr_onehot[0,2]=1
        if(curr_surface=='styrofoam'):
            curr_onehot[0,3]=1

    return curr_onehot

def singlestep_to_state(robot_info, mocap_info, old_time, old_pos, old_al, old_ar, state_representation):

	#dt
	curr_time = robot_info.stamp
	if(old_time==-7):
		dt=1
	else:
		dt = (curr_time.secs-old_time.secs) + (curr_time.nsecs-old_time.nsecs)*0.000000001

	#mocap position
	curr_pos= mocap_info.pose.position

	#mocap pose
	angles = quat_to_eulerDegrees(mocap_info.pose.orientation)
	r= angles[0]
	p= angles[1]
	yw= angles[2]
	#convert r,p,y to rad
	r=r*np.pi/180.0
	p=p*np.pi/180.0
	yw=yw*np.pi/180.0

	#gyro angular velocity
	wx= robot_info.gyroX
	wy= robot_info.gyroY
	wz= robot_info.gyroZ

	#encoders
	al= robot_info.posL/math.pow(2,16)*2*math.pi
	ar= robot_info.posR/math.pow(2,16)*2*math.pi

	#com vel
	vel_x = (curr_pos.x-old_pos.x)/dt
	vel_y = (curr_pos.y-old_pos.y)/dt
	vel_z = (curr_pos.z-old_pos.z)/dt

	#motor vel
	vel_al = (al-old_al)/dt
	vel_ar = (ar-old_ar)/dt

	#create the state
	if(state_representation=="all"):
		state = np.array([curr_pos.x, curr_pos.y, curr_pos.z, 
									vel_x, vel_y, vel_z, 
									np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw), 
									wx, wy, wz, 
									np.cos(al), np.sin(al), np.cos(ar), np.sin(ar), 
									vel_al, vel_ar, 
									robot_info.bemfL, robot_info.bemfR, robot_info.vBat])
	elif(state_representation == "exclude_x_y"):
		state = np.array([ curr_pos.z, vel_x, vel_y, vel_z, 
								np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw), 
								wx, wy, wz, 
								np.cos(al), np.sin(al), np.cos(ar), np.sin(ar), 
								vel_al, vel_ar, 
								robot_info.bemfL, robot_info.bemfR, robot_info.vBat])	

	#save curr as old
	old_time= copy.deepcopy(curr_time)
	old_pos=copy.deepcopy(curr_pos)
	old_al=np.copy(al)
	old_ar=np.copy(ar)

	return state, old_time, old_pos, old_al, old_ar

def rollout_to_states(robot_info, mocap_info, state_representation):
	#print(type(robot_info[0]))
	#print(state_representation)

	list_states=[]
	list_actions=[]

	for step in range(0,len(robot_info)):

		if(step==0):
			old_time= robot_info[step].stamp
			old_pos= mocap_info[step].pose.position
			old_al= robot_info[step].posL/math.pow(2,16)*2*math.pi
			old_ar= robot_info[step].posR/math.pow(2,16)*2*math.pi
		else:
			#dt
			curr_time = robot_info[step].stamp
			dt = (curr_time.secs-old_time.secs) + (curr_time.nsecs-old_time.nsecs)*0.000000001

			#mocap position
			curr_pos= mocap_info[step].pose.position

			#mocap pose
			angles = quat_to_eulerDegrees(mocap_info[step].pose.orientation)
			r= angles[0]
			p= angles[1]
			yw= angles[2]
			#convert r,p,y to rad
			r=r*np.pi/180.0
			p=p*np.pi/180.0
			yw=yw*np.pi/180.0

			#gyro angular velocity
			wx= robot_info[step].gyroX
			wy= robot_info[step].gyroY
			wz= robot_info[step].gyroZ

			#encoders
			al= robot_info[step].posL/math.pow(2,16)*2*math.pi
			ar= robot_info[step].posR/math.pow(2,16)*2*math.pi

			#com vel
			vel_x = (curr_pos.x-old_pos.x)/dt
			vel_y = (curr_pos.y-old_pos.y)/dt
			vel_z = (curr_pos.z-old_pos.z)/dt

			#motor vel
			vel_al = (al-old_al)/dt
			vel_ar = (ar-old_ar)/dt

			#create the state
			if(state_representation=="all"):
				states = np.array([curr_pos.x, curr_pos.y, curr_pos.z, 
											vel_x, vel_y, vel_z, 
											np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw), 
											wx, wy, wz, 
											np.cos(al), np.sin(al), np.cos(ar), np.sin(ar), 
											vel_al, vel_ar, 
											robot_info[step].bemfL, robot_info[step].bemfR, robot_info[step].vBat])
			elif(state_representation == "exclude_x_y"):
				states = np.array([ curr_pos.z, vel_x, vel_y, vel_z, 
											np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw), 
											wx, wy, wz, 
											np.cos(al), np.sin(al), np.cos(ar), np.sin(ar), 
											vel_al, vel_ar, 
											robot_info[step].bemfL, robot_info[step].bemfR, robot_info[step].vBat])				
			list_states.append(states)


			#create the action
			action=np.array([robot_info[step].curLeft, robot_info[step].curRight])
			list_actions.append(action)

			#save curr as old
			old_time=copy.deepcopy(curr_time)
			old_pos=copy.deepcopy(curr_pos)
			old_al=copy.deepcopy(al)
			old_ar=copy.deepcopy(ar)
	return np.array(list_states), np.array(list_actions)