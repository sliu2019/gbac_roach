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
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Tuesday_optimization/_ubs_8_ulr_0.1num_updates1_layers_2_x500_task_list_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001/carpet/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Sunday_favorites/_ubs_16_ulr_0.01num_updates1_layers_1_x400_task_list_all_mlr_0.001_mbs_64/styrofoam/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/8_30_optimization/_ubs_16_ulr_0.5num_updates1_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_0/styrofoam/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/Wednesday_optimization/_ubs_16_ulr_1.0num_updates1_layers_2_x500_task_list_styrofoam_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001/styrofoam/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/8_31_optimization/_ubs_23_ulr_2.0num_updates3_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/carpet_long_trajectory/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/8_31_optimization/_ubs_23_ulr_2.0num_updates3_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/carpet_momentum/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/8_31_optimization/_ubs_16_ulr_0.1num_updates1_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/carpet/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/8_31_optimization/_ubs_23_ulr_2.0num_updates3_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/styrofoam_momentum_continual/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_4_optimization/_ubs_23_ulr_3.0num_updates2_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/turf/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_4_optimization/_ubs_23_ulr_3.0num_updates1_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/turf/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_2_optimization/_ubs_23_ulr_2.0num_updates2_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/carpet/saved_rollouts"

	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_2_optimization/_ubs_23_ulr_2.0num_updates2_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/carpet/saved_rollouts"
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_2_optimization/_ubs_23_ulr_0.0num_updates1_layers_2_x500_task_list_all_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5/carpet/saved_rollouts"
	rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_5_anusha/trythese/_ubs_23_ulr_2.0num_updates2_layers_2_x500_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_learninnerloss_True_usingULR0.1/carpet/saved_rollouts"

	traj_save_path = ['zigzag1_aggIter0']
	#traj_save_path = ['straight0_aggIter0']
	#traj_save_path = ['zigzag0', 'zigzag1', 'zigzag2', 'zigzag3', 'zigzag4']
	#traj_save_path = ['left0_aggIter0', 'left1_aggIter0', 'left2_aggIter0'] #'left3_aggIter0', 'left4_aggIter0', 'left5_aggIter0'
	#traj_save_path = ['left0_aggIter0','left1_aggIter0', 'left2_aggIter0', 'left3_aggIter0']
	#traj_save_path = ['straight0_aggIter0','straight1_aggIter0', 'straight2_aggIter0', 'straight3_aggIter0']
	#traj_save_path = ['straight3_aggIter0','straight4_aggIter0', 'straight5_aggIter0']
	#traj_save_path = ['right0_aggIter0','right1_aggIter0', 'right2_aggIter0']
	#traj_save_path = ['right9_aggIter0','right10_aggIter0', 'right11_aggIter0', 'right12_aggIter0']
	
	#traj_save_path = ['left7_aggIter0','left8_aggIter0', 'left9_aggIter0']
	#traj_save_path = ['right0_aggIter0','right1_aggIter0']
	
	#traj_save_path=['straight0_aggIter1','straight1_aggIter1', 'straight2_aggIter1', 'straight3_aggIter1', 'straight4_aggIter1']
	#traj_save_path = ['left9_aggIter0','left10_aggIter0', 'left11_aggIter0']
	#traj_save_path = ['right0_aggIter0','right1_aggIter0', 'right2_aggIter0']
	#traj_save_path = ['right6_aggIter0']
	#traj_save_path = ['uturn0_aggIter0','uturn1_aggIter0','uturn2_aggIter0']
	#traj_save_path = ['uturn0_aggIter0']

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


		if "straight" in traj_path:
			plt.ylim((min(traj_taken[:,1])-0.5, max(traj_taken[:,1])+1))
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
