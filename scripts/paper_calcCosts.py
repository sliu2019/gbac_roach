import IPython
import numpy as np 
import matplotlib.pyplot as plt
from moving_distance import moving_distance


def main():

	do_print=False

	horiz_penalty_factor= 30
	backward_discouragement= 10
	heading_penalty_factor= 5

	carpet_runs = [16,20,15,47,100]
	gravel_runs = [13,12,14,49,104]
	turf_runs = [18,19,21,48,103]
	styrofoam_runs = [45,7,46,40,102]

	########################################################################
	########################################################################
	########################################################################

	model_names = ['separate', 'combined', 'onehot', 'camera', 'diffdrive']
	
	traj_zigzag = ['zigzag0', 'zigzag1', 'zigzag2', 'zigzag3', 'zigzag4']
	traj_straight = ['straight0', 'straight1', 'straight2', 'straight3', 'straight4']
	traj_left = ['left0', 'left1', 'left2', 'left3', 'left4']
	traj_right = ['right0', 'right1', 'right2', 'right3', 'right4']

	surface_names = ['styrofoam', 'carpet', 'gravel', 'turf']
	all_runs = [styrofoam_runs, carpet_runs, gravel_runs, turf_runs]

	all_trajecs = [traj_straight, traj_left, traj_right, traj_zigzag]

	########################################################################
	########################################################################
	########################################################################


	surface_counter=0
	all_normal_model=[]
	all_combined_model=[]
	all_onehot_model=[]
	all_camera_model=[]
	all_dd_model=[]

	for material_runs in all_runs:
		if(do_print):
			print '\nMATERIAL: ', surface_names[surface_counter]
		surface_counter+=1
		model_counter =0 
		normal_model=[]
		combined_model=[]
		onehot_model=[]

		for run_num in material_runs:
			if(do_print):
				print '    MODEL TYPE: ', model_names[model_counter]
			model_counter+=1

			for traj_save_path in all_trajecs:

				list_dist=[]
				list_cost=[]
				list_avg_cost=[]

				for traj_path in traj_save_path:

					model_type = model_names[model_counter-1]

					#read in traj info
					if(model_type=='diffdrive'):
						curr_dir = '../run_' + str(run_num) + '/' + traj_path + '/'
						actions_taken = np.load(curr_dir +'diffdrive_actions.npy')
						desired_states = np.load(curr_dir +'diffdrive_desired.npy')
						traj_taken = np.load(curr_dir +'diffdrive_executed.npy')
						save_perp_dist = np.load(curr_dir +'diffdrive_perp.npy')
						save_forward_dist = np.load(curr_dir +'diffdrive_forward.npy')
						saved_old_forward_dist = np.load(curr_dir +'diffdrive_oldforward.npy')
						save_moved_to_next = np.load(curr_dir +'diffdrive_movedtonext.npy')
						save_desired_heading = np.load(curr_dir +'diffdrive_desheading.npy')
						save_curr_heading = np.load(curr_dir +'diffdrive_currheading.npy')
					else:
						curr_dir = '../run_' + str(run_num) + '/' + traj_path + '/'
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

				if(do_print):
					print '        ' , traj_save_path[0][:-1], ' mean: ', np.mean(list_avg_cost)
				data_mean=np.mean(list_avg_cost)
				data=np.array([data_mean, np.std(list_avg_cost)])
				

				if(model_counter==1):
					normal_model.append(data)
					all_normal_model.append(data)
				if(model_counter==2):
					combined_model.append(data)
					all_combined_model.append(data)
				if(model_counter==3):
					onehot_model.append(data)
					all_onehot_model.append(data)
				if(model_counter==4):
					all_camera_model.append(data)
				if(model_counter==5):
					all_dd_model.append(data)


		'''plt.title(surface_names[surface_counter-1])
		normal_model=np.array(normal_model)
		combined_model=np.array(combined_model)
		onehot_model=np.array(onehot_model)

		# plot
		x = np.arange(4)
		normal   = plt.errorbar(x, normal_model[:,0], normal_model[:,1], 
								linestyle='None', marker='^', color='b', ecolor='b', label='normal')
		combined = plt.errorbar(x+0.1, combined_model[:,0], combined_model[:,1], 
								linestyle='None', marker='^', color='g', ecolor='g', label='combined')
		onehot   = plt.errorbar(x+0.2, onehot_model[:,0], onehot_model[:,1], 
								linestyle='None', marker='^', color='r', ecolor='r', label='one hot')
		plt.legend(handles=[normal, combined, onehot], loc=2)
		min_loss = min(np.amin(normal_model[:,0]-normal_model[:,1]),
					   np.amin(combined_model[:,0]-combined_model[:,1]),
					   np.amin(onehot_model[:,0]-onehot_model[:,1]))
		max_loss = max(np.amax(normal_model[:,0]+normal_model[:,1]),
					   np.amax(combined_model[:,0]+combined_model[:,1]),
					   np.amax(onehot_model[:,0]+onehot_model[:,1]))
		plt.axis([x[0]-0.5, x[-1]+0.5, min_loss-1, max_loss+1])
		plt.show()'''



	#PLOT
	#all_normal_model... by surface, list all trajecs

	plt.figure(figsize=(24,10))
	index = np.array([0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18])
	bar_width = 0.15
	opacity = 0.8

	all_normal_model=np.array(all_normal_model)
	all_combined_model=np.array(all_combined_model)
	all_onehot_model=np.array(all_onehot_model)
	all_dd_model=np.array(all_dd_model)
	all_camera_model=np.array(all_camera_model)


	rects4 = plt.bar(index + 0*bar_width, all_camera_model[:,0], bar_width,
		 			 yerr=all_camera_model[:,1],
	                 alpha=opacity,
	                 color='c',
	                 ecolor='c',
	                 label='Image-Conditioned')

	rects3 = plt.bar(index + 1*bar_width, all_onehot_model[:,0], bar_width,
		 			 yerr=all_onehot_model[:,1],
	                 alpha=opacity,
	                 color='b',
	                 ecolor='b',
	                 label='One-hot vector')

	rects1 = plt.bar(index + 2*bar_width, all_normal_model[:,0], bar_width,
					 yerr=all_normal_model[:,1],
	                 alpha=opacity,
	                 color='r',
	                 ecolor='r',
	                 label='Individual')

	rects2 = plt.bar(index + 3*bar_width, all_combined_model[:,0], bar_width,
		 			 yerr=all_combined_model[:,1],
	                 alpha=opacity,
	                 color='g',
	                 ecolor='g',
	                 label='Combined/Avg')

	rects5 = plt.bar(index + 4*bar_width, all_dd_model[:,0], bar_width,
		 			 yerr=all_dd_model[:,1],
	                 alpha=opacity,
	                 color='k',
	                 ecolor='k',
	                 label='Differential Drive')
	 
	plt.xlabel('STYROFOAM                                            CARPET                                                    GRAVEL                                                    TURF', fontsize = 18, fontweight='bold', labelpad=20)
	plt.title('Trajectory Following Costs', fontsize=20) 
	plt.xticks(index + bar_width, ('S', 'L', 'R', 'Z', 'S', 'L', 'R', 'Z', 'S', 'L', 'R', 'Z', 'S', 'L', 'R', 'Z'), fontsize=14)
	plt.legend(loc='upper right', fontsize=20)
	plt.tight_layout()
	plt.show()




	return


if __name__ == '__main__':
    main()
