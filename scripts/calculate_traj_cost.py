import IPython
import numpy as np 
import matplotlib.pyplot as plt
from moving_distance import moving_distance
import yaml


def main(traj_save_path):

	################################################################
	###################### VARS TO SPECIFY #########################
	traj_save_path = ['straight0_aggIter0','straight1_aggIter0', 'straight2_aggIter0','straight3_aggIter0', 'straight4_aggIter0']
	#traj_save_path = ['straight5_aggIter0','straight6_aggIter0','straight7_aggIter0','straight8_aggIter0','straight9_aggIter0']
	#traj_save_path = ['straight0_aggIter0','straight1_aggIter0','straight2_aggIter0','straight3_aggIter0','straight4_aggIter0', 'straight5_aggIter0', 'straight6_aggIter0', 'straight7_aggIter0']

	#traj_save_path = ['straight0_aggIter0','straight1_aggIter0','straight2_aggIter0','straight3_aggIter0', 'straight4_aggIter0', 'straight5_aggIter0']
	#traj_save_path = ['straight5_aggIter0', 'straight6_aggIter0', 'straight7_aggIter0', 'straight8_aggIter0', 'straight9_aggIter0']
	#traj_save_path = ['left0_aggIter0', 'left1_aggIter0']
	
	#traj_save_path = ['left0_aggIter0','left1_aggIter0', 'left2_aggIter0']
	#traj_save_path = ['circle0_aggIter0']
	
	#traj_save_path = ['right0_aggIter0','right1_aggIter0', 'right2_aggIter0', 'right3_aggIter0','right4_aggIter0']
	#traj_save_path = ['zigzag0_aggIter0','zigzag1_aggIter0','zigzag2_aggIter0','zigzag3_aggIter0','zigzag4_aggIter0', 'zigzag5_aggIter0', 'zigzag6_aggIter0', 'zigzag7_aggIter0', 'zigzag8_aggIter0', 'zigzag9_aggIter0']
	#traj_save_path = ['zigzag0_aggIter0','zigzag1_aggIter0','zigzag2_aggIter0','zigzag3_aggIter0','zigzag4_aggIter0']
	
	#traj_save_path = ['zigzag10_aggIter0','zigzag11_aggIter0','zigzag12_aggIter0','zigzag13_aggIter0','zigzag14_aggIter0']

	#traj_save_path = ['zigzag0_aggIter0','zigzag1_aggIter0']
	
	#traj_save_path = ['figure80_aggIter0', 'figure81_aggIter0', 'figure82_aggIter0']
	#traj_save_path = ['figure80_aggIter0', 'figure81_aggIter0', 'figure82_aggIter0', 'figure83_aggIter0', 'figure84_aggIter0']

	#traj_save_path = ['straight0_aggIter0','straight1_aggIter0','straight2_aggIter0']
	#traj_save_path = ['snake0_aggIter0','snake1_aggIter0', 'snake2_aggIter0', 'snake3_aggIter0', 'snake4_aggIter0']

	is_diffDrive = False

	#gbac agg1
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_7_optimization/_ubs_23_ulr_2.0num_updates2_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False_agg_0.9/rand/saved_rollouts"
	
	#gbac agg0
	rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_7_optimization/_ubs_23_ulr_2.0num_updates2_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False/shell_shift/saved_rollouts"
	
	#nongbac agg0
	#rollouts_path = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach/9_11_optimization/_ubs_23_ulr_0.0num_updates1_layers_2_x500_task_list_turf_styrofoam_carpet_mlr_0.001_mbs_64_num-sgd-steps_1_reg_weight_0.001_dim_bias_5_metatrain_lr_False/shell_shift_dyneval/saved_rollouts"

	#14.2 grams weight on the shell
	#5 cubes of foam on each corner, except 4 on the bottom right. For the shell shifting

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
		#plt.show()
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
    #main()
	'''traj_save_path = ['figure80_aggIter0', 'figure81_aggIter0', 'figure82_aggIter0', 'figure83_aggIter0', 'figure84_aggIter0']
	main(traj_save_path)
	traj_save_path = ['zigzag0_aggIter0', 'zigzag1_aggIter0', 'zigzag2_aggIter0', 'zigzag3_aggIter0', 'zigzag4_aggIter0']
	main(traj_save_path)
	
	traj_save_path = ['left0_aggIter0','left1_aggIter0', 'left2_aggIter0','left3_aggIter0','left4_aggIter0']#,
	main(traj_save_path)
	
	traj_save_path = ['straight0_aggIter0','straight3_aggIter0','straight2_aggIter0','straight3_aggIter0', 'straight4_aggIter0']
	main(traj_save_path)
	
	traj_save_path = ['right0_aggIter0','right1_aggIter0', 'right2_aggIter0', 'right3_aggIter0','right4_aggIter0']
	main(traj_save_path)'''

	traj_save_path = ['straight0_aggIter0','straight3_aggIter0','straight2_aggIter0','straight3_aggIter0', 'straight4_aggIter0']
	main(traj_save_path)