import IPython
import numpy as np 
import matplotlib.pyplot as plt
from moving_distance import moving_distance


def main():

	################################################################
	###################### VARS TO SPECIFY #########################

	is_diffDrive = False

	#run_1: gravel model on gravel
	run_num = 'run_30'
	#traj_save_path = ['zigzag0', 'zigzag1', 'zigzag2', 'zigzag3', 'zigzag4']
	traj_save_path = ['straight0', 'straight1', 'straight2', 'straight3', 'straight4']
	#traj_save_path = ['left0', 'left1', 'left2', 'left3', 'left4']
	#traj_save_path = ['right0', 'right1', 'right2', 'right3', 'right4']


	horiz_penalty_factor= 30
	backward_discouragement= 10
	heading_penalty_factor= 5

	################################################################
	################################################################

	list_dist=[]
	list_cost=[]
	list_avg_cost=[]

	for traj_path in traj_save_path:

		#read in traj info
		if(is_diffDrive):
			curr_dir = '../' + run_num + '/' + traj_path + '/' + 'diffdrive_'
		else:
			curr_dir = '../' + run_num + '/' + traj_path + '/'

		actions_taken = np.load(curr_dir +'actions.npy')
		desired_states = np.load(curr_dir +'desired.npy')
		traj_taken = np.load(curr_dir +'executed.npy')
		save_perp_dist = np.load(curr_dir +'perp.npy')
		save_forward_dist = np.load(curr_dir +'forward.npy')
		saved_old_forward_dist = np.load(curr_dir +'oldforward.npy')
		save_moved_to_next = np.load(curr_dir +'movedtonext.npy')
		save_desired_heading = np.load(curr_dir +'desheading.npy')
		save_curr_heading = np.load(curr_dir +'currheading.npy')

		#calculate cost
		cost_per_step = []
		total_dist = 0
		length = actions_taken.shape[0]

		for i in range(length):
			p = save_perp_dist[i]
			ND = save_forward_dist[i]
			OD = saved_old_forward_dist[i]
			moved_to_next = save_moved_to_next[i]
			a = save_desired_heading[i]
			h = save_curr_heading[i]
			diff = np.abs(moving_distance(a, h))

			#write this as desired
			cost = p*horiz_penalty_factor
			cost += diff*heading_penalty_factor
			if(moved_to_next==0):
				cost += (OD - ND)*backward_discouragement

			cost_per_step.append(cost)
			if(i==0):
				total_dist=0
			else:
				x_diff = traj_taken[i][0]-traj_taken[i-1][0]
				y_diff = traj_taken[i][1]-traj_taken[i-1][1]
				total_dist+= np.sqrt(x_diff*x_diff + y_diff*y_diff)

		#save
		total_cost = np.sum(np.array(cost_per_step))
		list_dist.append(total_dist)
		list_cost.append(total_cost)
		list_avg_cost.append(total_cost/length)

		#plt.plot(desired_states[:5,0], desired_states[:5,1], 'ro')
		#plt.plot(traj_taken[:,0], traj_taken[:,1])
		#plt.show()

	print()
	print()
	print("costs: ", list_cost)
	print("mean: ", np.mean(list_cost), " ... std: ", np.std(list_cost))
	print("mean: ", np.mean(list_avg_cost), " ... std: ", np.std(list_avg_cost))
	print(list_avg_cost)
	print()
	print()

	return


if __name__ == '__main__':
    main()
