import IPython
import numpy as np 
import matplotlib.pyplot as plt
from moving_distance import moving_distance
import yaml


def main():

	################################################################
	###################### VARS TO SPECIFY #########################

	is_diffDrive = False

	#run_1: gravel model on gravel
	#rollouts_path = '/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/terrain_types__regularization_weight_0.001__use_reg_True__meta_batch_size_250__meta_lr_0.001__horizon_5__max_epochs_80__update_lr_0.1__curr_agg_iter_0__update_batch_size_16/saved_rollouts'
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/terrain_types__regularization_weight_0.001__use_reg_True__meta_batch_size_250__meta_lr_0.001__horizon_5__max_epochs_80__update_lr_0.1__curr_agg_iter_0__update_batch_size_16_NON_GBAC/saved_rollouts"
	#rollouts_path = "/home/anagabandi/roach_workspace/src/nn_dynamics_roach/run_36"
	rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach_copy/Wednesday_optimization/ulr_5_num_update_1/_ubs_8_ulr_2.0num_updates1_layers_1_x100_task_list_all/foam_fixed_leg/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach_copy/Wednesday_optimization/NON_GBAC/foam/saved_rollouts"

	#traj_save_path = ['zigzag0', 'zigzag1', 'zigzag2', 'zigzag3', 'zigzag4']
	#traj_save_path = ['left0_aggIter0','left1_aggIter0', 'left2_aggIter0', 'left3_aggIter0', 'left4_aggIter0']
	traj_save_path = ['left0_aggIter0','left1_aggIter0', 'left2_aggIter0','left3_aggIter0', 'left4_aggIter0']
	#traj_save_path = ['straight0_aggIter0','straight1_aggIter0', 'straight2_aggIter0','straight3_aggIter0', 'straight4_aggIter0']
	#traj_save_path = ['right0_aggIter0','right1_aggIter0', 'right2_aggIter0','right3_aggIter0', 'right4_aggIter0']
	#traj_save_path = ['left7_aggIter0','left8_aggIter0', 'left9_aggIter0']
	#traj_save_path = ['right0_aggIter0','right1_aggIter0']
	#'right2_aggIter0','right3_aggIter0', 'right4_aggIter0', 'right5_aggIter0', 'right6_aggIter0', 'right7_aggIter0','right9_aggIter0']
	#traj_save_path = ['right3_aggIter0', 'right4_aggIter0', 'right5_aggIter0', 'right6_aggIter0', 'right7_aggIter0']
	#traj_save_path = ['left0', 'left1', 'left2', 'left3', 'left4']
	#traj_save_path = ['right0', 'right1', 'right2', 'right3', 'right4']


	

	################################################################
	################################################################

	list_dist=[]
	list_cost=[]
	list_avg_cost=[]

	for traj_path in traj_save_path:

		#read in traj info
		if(is_diffDrive):
			curr_dir = rollouts_path + '/' + traj_path + '/' + 'diffdrive_'
		else:
			curr_dir = rollouts_path + '/' + traj_path + '/'

		config = yaml.load(open(curr_dir + "saved_config.yaml"))
		horiz_penalty_factor= config['policy']['horiz_penalty_factor']
		backward_discouragement= config['policy']['backward_discouragement']
		heading_penalty_factor= config['policy']['heading_penalty_factor']

		actions_taken = np.load(curr_dir +'actions.npy')
		desired_states = np.load(curr_dir +'desired.npy')
		traj_taken = np.load(curr_dir +'oldFormat_executed.npy')
		save_perp_dist = np.load(curr_dir +'oldFormat_perp.npy')
		save_forward_dist = np.load(curr_dir +'oldFormat_forward.npy')
		saved_old_forward_dist = np.load(curr_dir +'oldFormat_oldforward.npy')
		save_moved_to_next = np.load(curr_dir +'oldFormat_movedtonext.npy')
		save_desired_heading = np.load(curr_dir +'oldFormat_desheading.npy')
		save_curr_heading = np.load(curr_dir +'oldFormat_currheading.npy')

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

		if "straight" in traj_path:
			plt.plot(desired_states[:3,0], desired_states[:3,1], 'ro')			
		else:
			plt.plot(desired_states[:5,0], desired_states[:5,1], 'ro')
		plt.plot(traj_taken[:,0], traj_taken[:,1])
		plt.savefig(rollouts_path + "/" + traj_path + "/x_y_visualization.png")
		plt.show()
		plt.clf()

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
