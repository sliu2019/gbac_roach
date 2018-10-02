from sandbox.rocky.tf.policies.base import Policy
from rllab.core.serializable import Serializable
import numpy as np
from sandbox.ignasi.maml.utils import PlotRegressor
import tensorflow as tf
import time
import matplotlib
from rllab.envs.env_spec import EnvSpec
from rllab.spaces.box import Box
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from utils import *
import IPython
import random

class NaiveMPCController(Policy, Serializable):
    def __init__(
            self,
            regressor,
            inputSize,
            outputSize,
            left_min, 
            right_min, 
            left_max, 
            right_max,
            state_representation='all',
            n_candidates=1000,
            horizon=10,
            horiz_penalty_factor=30,
            backward_discouragement=10,
            heading_penalty_factor=5,
            visualize_rviz=True,
            x_index=0,
            y_index=1,
            yaw_cos_index=10,
            yaw_sin_index=11,
            test_regressor=False,
            frequency_value=10,
            serial_port=None,
            baud_rate=None,
    ):
        self.regressor = regressor
        self.visualize_rviz = visualize_rviz
        self.state_representation=state_representation

        #from config['policy']
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.horiz_penalty_factor = horiz_penalty_factor
        self.backward_discouragement = backward_discouragement
        self.heading_penalty_factor = heading_penalty_factor

        #from config['roach']
        self.x_index = x_index
        self.y_index = y_index
        self.yaw_cos_index = yaw_cos_index
        self.yaw_sin_index = yaw_sin_index

        #useless
        self.random = False
        self.multi_input = False
        self.is_onehot = False

        #initialized by the gbac_controller_node
        self.desired_states = None
        self.publish_markers_desired = None
        self.publish_markers = None

        '''if test_regressor:
            self.plot = PlotRegressor('RoachEnv')
            #NOTE: setting this to True doesnt give you an optimal rollout
                #it plans only evey horizon-steps, and doesnt replan at each step
                    #because it's trying to verify accuracy of regressor'''
        Serializable.quick_init(self, locals())

        #define action space for roach
        my_obs = Box(low=-1*np.ones(outputSize,), high=np.ones(outputSize,))
        my_ac = Box(low=np.array([left_min, right_min]), high=np.array([left_max, right_max]))
        super(NaiveMPCController, self).__init__(env_spec=EnvSpec(observation_space=my_obs, action_space=my_ac))

    @property
    def vectorized(self):
        return True

    #distance needed for unit 2 to go toward unit1.... NOT THE CORRECT SIGN
    def moving_distance(self, unit1, unit2):
      phi = (unit2-unit1) % (2*np.pi)
      phi[phi > np.pi] = (2*np.pi-phi)[phi > np.pi]
      return phi

    def get_reward(observation, next_observation, acs):
        print("\n\nNOT IMPLEMENTED get_reward...\n\n")
        import IPython
        IPython.embed()

    def get_random_action(self, n):
        return self.action_space.sample_n(n)

    '''def get_best_action_OLD(self, observation, actions=None):
        n = self.n_candidates
        m = len(observation)
        R = np.zeros((n*m,))

        #randomly sample n sequences of length-h actions (for multiple envs in parallel)
            #[h, N, ac_dim]
        a = self.get_random_action(self.horizon*n*m).reshape((self.horizon, n*m, -1))

        #simulate the action sequences
        for h in range(self.horizon):
            if h == 0:
                cand_a = a[h].reshape((m,n,-1)) #[1, N, ac_dim]
                observation = np.array(observation).reshape((m,1,-1)) #goes from [1,obs_dim] --> [1, 1, obs_dim]
                observation = np.tile(observation, (1,n,1)).reshape((m*n, -1)) # tiled observations, so [N, obs_dim]
            if(self.state_representation=='exclude_x_y'):
                next_observation = self.regressor.predict(np.concatenate([observation[:,2:], a[h]], axis=1)) + observation
            else:
                next_observation = self.regressor.predict(np.concatenate([observation, a[h]], axis=1)) + observation
                #observation    [1000 x 21]
                #a[h]           [1000 x 5]
                #after concat   [1000 x 26]
                #output of predict  [1000x 21]
            r = self.get_reward(observation, next_observation, a[h])
            R += r
            observation = next_observation
        R = R.reshape(m,n)
        #return the action from the sequence that resulted in highest reward
        return cand_a[range(m), np.argmax(R, axis=1)]'''

    ############################################
    ##### GET BEST ACTION
    ############################################

    def get_best_action(self, observation, currently_executing_action, curr_line_segment, old_curr_forward, color):
        n = self.n_candidates
        full_curr_state = np.copy(observation)

        #randomly sample n sequences of length-h actions (for multiple envs in parallel)
            #[h, N, ac_dim]
        a = self.get_random_action(self.horizon*n).reshape((self.horizon, n, -1))
        #print(a[:5, :15, :])

        #make the 1st one be the currently executing action
        a[0,:,0]=currently_executing_action[0]
        a[0,:,1]=currently_executing_action[1]

        #simulate the action sequences
        resulting_states = [] ##this will be [horizon+1, N, statesize]
        for h in range(self.horizon):
            if h == 0:
                cand_a = a[h].reshape((1,n,-1)) #[1, N, ac_dim]
                observation = np.array(observation).reshape((1,1,-1)) #goes from [1,obs_dim] --> [1, 1, obs_dim]
                observation = np.tile(observation, (1,n,1)).reshape((n, -1)) # tiled observations, so [N, obs_dim]
                resulting_states.append(observation)
            use_this_obs = create_nn_input_using_staterep(observation, self.state_representation, multiple=True)
            next_observation = self.regressor.predict(np.concatenate([use_this_obs, a[h]], axis=1)) + observation #[1000 x 21]
            resulting_states.append(next_observation)
            observation = next_observation

        #evaluate all options and pick the best one
        optimal_action, curr_line_segment, old_curr_forward, info_for_saving, best_sequence_of_actions, best_sequence_of_states = self.select_the_best_one_for_traj_follow(full_curr_state, a, np.array(resulting_states), curr_line_segment, old_curr_forward, color)

        return optimal_action, curr_line_segment, old_curr_forward, info_for_saving, best_sequence_of_actions, best_sequence_of_states

    #################################################
    ##### ACTION SELECTION when trajectory following
    #################################################

    def select_the_best_one_for_traj_follow(self, full_curr_state, all_samples, resulting_states, curr_line_segment, old_curr_forward, color):

        desired_states= self.desired_states
        
        ###################################################################
        ### check if curr point is closest to curr_line_segment or if it moved on to next one
        ###################################################################

        curr_start = desired_states[curr_line_segment]
        curr_end = desired_states[curr_line_segment+1]
        next_start = desired_states[curr_line_segment+1]
        next_end = desired_states[curr_line_segment+2]
        
        ############ closest distance from point to current line segment
        #vars
        a = full_curr_state[self.x_index]- curr_start[0]
        b = full_curr_state[self.y_index]- curr_start[1]
        c = curr_end[0]- curr_start[0]
        d = curr_end[1]- curr_start[1]
        #project point onto line segment
        which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))
        #point on line segment that's closest to the pt
        if(which_line_section<0):
            closest_pt_x = curr_start[0]
            closest_pt_y = curr_start[1]
        elif(which_line_section>1):
            closest_pt_x = curr_end[0]
            closest_pt_y = curr_end[1]
        else:
            closest_pt_x= curr_start[0] + np.multiply(which_line_section,c)
            closest_pt_y= curr_start[1] + np.multiply(which_line_section,d)
        #min dist from pt to that closest point (ie closes dist from pt to line segment)
        min_perp_dist = np.sqrt((full_curr_state[self.x_index]-closest_pt_x)*(full_curr_state[self.x_index]-closest_pt_x) + (full_curr_state[self.y_index]-closest_pt_y)*(full_curr_state[self.y_index]-closest_pt_y))
        #"forward-ness" of the pt... for each sim
        curr_forward = which_line_section
        
        ############ closest distance from point to next line segment
        #vars
        a = full_curr_state[self.x_index]- next_start[0]
        b = full_curr_state[self.y_index]- next_start[1]
        c = next_end[0]- next_start[0]
        d = next_end[1]- next_start[1]
        #project point onto line segment
        which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))
        #point on line segment that's closest to the pt
        if(which_line_section<0):
            closest_pt_x = next_start[0]
            closest_pt_y = next_start[1]
        elif(which_line_section>1):
            closest_pt_x = next_end[0]
            closest_pt_y = next_end[1]
        else:
            closest_pt_x= next_start[0] + np.multiply(which_line_section,c)
            closest_pt_y= next_start[1] + np.multiply(which_line_section,d)
        #min dist from pt to that closest point (ie closes dist from pt to line segment)
        dist = np.sqrt((full_curr_state[self.x_index]-closest_pt_x)*(full_curr_state[self.x_index]-closest_pt_x) + (full_curr_state[self.y_index]-closest_pt_y)*(full_curr_state[self.y_index]-closest_pt_y))
        
        #pick which line segment it's closest to, and update vars accordingly
        moved_to_next = False
        if(dist<min_perp_dist):
            print(" **************************** MOVED ONTO NEXT LINE SEG")
            curr_line_segment+=1
            curr_forward= which_line_section
            min_perp_dist = np.copy(dist)
            moved_to_next = True

        ########################################
        #### headings
        ########################################

        curr_start = desired_states[curr_line_segment]
        curr_end = desired_states[curr_line_segment+1]
        desired_yaw = np.arctan2(curr_end[1]-curr_start[1], curr_end[0]-curr_start[0])
        curr_yaw = np.arctan2(full_curr_state[self.yaw_sin_index],full_curr_state[self.yaw_cos_index])

        ########################################
        #### save vars
        ########################################

        save_perp_dist = np.copy(min_perp_dist)
        save_forward_dist = np.copy(curr_forward)
        saved_old_forward_dist = np.copy(old_curr_forward)
        save_moved_to_next = np.copy(moved_to_next)
        save_desired_heading = np.copy(desired_yaw)
        save_curr_heading = np.copy(curr_yaw)

        old_curr_forward = np.copy(curr_forward)

        ########################################
        #### evaluate scores for each option 
        ########################################
        
        #init vars for evaluating the trajectories
        scores=np.zeros((self.n_candidates,))
        done_forever=np.zeros((self.n_candidates,))
        move_to_next=np.zeros((self.n_candidates,))
        curr_seg = np.tile(curr_line_segment,(self.n_candidates,))
        curr_seg = curr_seg.astype(int)
        prev_forward = np.zeros((self.n_candidates,))
        moved_to_next = np.zeros((self.n_candidates,))

        prev_pt = resulting_states[0]

        for pt_number in range(resulting_states.shape[0]):

            #array of "the point"... for each sim
            pt = resulting_states[pt_number] # N x state

            #arrays of line segment points... for each sim
            curr_start = desired_states[curr_seg]
            curr_end = desired_states[curr_seg+1]
            next_start = desired_states[curr_seg+1]
            next_end = desired_states[curr_seg+2]

            #vars... for each sim
            min_perp_dist = np.ones((self.n_candidates, ))*5000

            ############ closest distance from point to current line segment

            #vars
            a = pt[:,self.x_index]- curr_start[:,0]
            b = pt[:,self.y_index]- curr_start[:,1]
            c = curr_end[:,0]- curr_start[:,0]
            d = curr_end[:,1]- curr_start[:,1]

            #project point onto line segment
            which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

            #point on line segment that's closest to the pt
            closest_pt_x = np.copy(which_line_section)
            closest_pt_y = np.copy(which_line_section)
            closest_pt_x[which_line_section<0] = curr_start[:,0][which_line_section<0]
            closest_pt_y[which_line_section<0] = curr_start[:,1][which_line_section<0]
            closest_pt_x[which_line_section>1] = curr_end[:,0][which_line_section>1]
            closest_pt_y[which_line_section>1] = curr_end[:,1][which_line_section>1]
            closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,0] + np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
            closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,1] + np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

            #min dist from pt to that closest point (ie closes dist from pt to line segment)
            min_perp_dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + (pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

            #"forward-ness" of the pt... for each sim
            curr_forward = which_line_section

            ############ closest distance from point to next line segment

            #vars
            a = pt[:,self.x_index]- next_start[:,0]
            b = pt[:,self.y_index]- next_start[:,1]
            c = next_end[:,0]- next_start[:,0]
            d = next_end[:,1]- next_start[:,1]

            #project point onto line segment
            which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

            #point on line segment that's closest to the pt
            closest_pt_x = np.copy(which_line_section)
            closest_pt_y = np.copy(which_line_section)
            closest_pt_x[which_line_section<0] = next_start[:,0][which_line_section<0]
            closest_pt_y[which_line_section<0] = next_start[:,1][which_line_section<0]
            closest_pt_x[which_line_section>1] = next_end[:,0][which_line_section>1]
            closest_pt_y[which_line_section>1] = next_end[:,1][which_line_section>1]
            closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,0] + np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
            closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,1] + np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

            #min dist from pt to that closest point (ie closes dist from pt to line segment)
            dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + (pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

            #pick which line segment it's closest to, and update vars accordingly
            curr_seg[dist<=min_perp_dist] += 1
            moved_to_next[dist<=min_perp_dist] = 1
            curr_forward[dist<=min_perp_dist] = which_line_section[dist<=min_perp_dist]#### np.clip(which_line_section,0,1)[dist<=min_perp_dist]
            min_perp_dist = np.min([min_perp_dist, dist], axis=0)

            ########################################
            #### scoring
            ########################################

            #penalize horiz dist
            scores += min_perp_dist*self.horiz_penalty_factor

            #penalize moving backward
            scores[moved_to_next==0] += (prev_forward - curr_forward)[moved_to_next==0]*self.backward_discouragement

            #penalize heading away from angle of line
            desired_yaw = np.arctan2(curr_end[:,1]-curr_start[:,1], curr_end[:,0]-curr_start[:,0])
            curr_yaw = np.arctan2(pt[:,self.yaw_sin_index],pt[:,self.yaw_cos_index])
            diff = np.abs(self.moving_distance(desired_yaw, curr_yaw))
            scores += diff*self.heading_penalty_factor

            #update
            prev_forward = np.copy(curr_forward)
            prev_pt = np.copy(pt)

        ########################################
        #### pick best one 
        ########################################
        #print(scores[:15])
        best_score = np.min(scores)
        best_sim_number = np.argmin(scores) 
        best_sequence_of_actions = all_samples[:,best_sim_number,:]

        if(self.visualize_rviz):
            #publish the desired traj
            markerArray2 = MarkerArray()
            marker_id=0
            for des_pt_num in range(desired_states.shape[0]): #5 for all, 8 for circle, 4 for zigzag
                marker = Marker()
                marker.id=marker_id
                marker.header.frame_id = "/world"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.15
                marker.scale.y = 0.15
                marker.scale.z = 0.15
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.pose.position.x = desired_states[des_pt_num,0]
                marker.pose.position.y = desired_states[des_pt_num,1]
                marker.pose.position.z = 0
                markerArray2.markers.append(marker)
                marker_id+=1
            self.publish_markers_desired.publish(markerArray2)

            #publish the best sequence selected
            best_sequence_of_states= resulting_states[:,best_sim_number,:] # (h+1)x(stateSize)
            markerArray = MarkerArray()
            marker_id=0

            #print("rviz red dot state shape: ", best_sequence_of_states[0, :].shape)
            for marker_num in range(resulting_states.shape[0]):
                marker = Marker()
                marker.id=marker_id
                marker.header.frame_id = "/world"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 1.0

                if color == "green":
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                elif color == "red":
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                marker.color.b = 0.0
                marker.pose.position.x = best_sequence_of_states[marker_num,0]
                marker.pose.position.y = best_sequence_of_states[marker_num,1]
                #print("rviz detects current roach pose to be: ", best_sequence_of_states[marker_num, :2])
                marker.pose.position.z = 0
                markerArray.markers.append(marker)
                marker_id+=1
            self.publish_markers.publish(markerArray)

            if color == "blue":
                # sample random resulting state sequences
                samples = np.random.randint(0, resulting_states.shape[1], size=7)
                #IPython.embed() #what size?

                for sample in samples:
                    red_val = random.uniform(0, 1)
                    blue_val = random.uniform(0, 1)
                    green_val = random.uniform(0, 1)

                    sequence_of_states = resulting_states[:, sample, :]
                    for marker_num in range(sequence_of_states.shape[0]):
                        marker = Marker()
                        marker.id=marker_id
                        marker.header.frame_id = "/world"
                        marker.type = marker.SPHERE
                        marker.action = marker.ADD
                        marker.scale.x = 0.05
                        marker.scale.y = 0.05
                        marker.scale.z = 0.05
                        marker.color.a = 1.0

                        
                        marker.color.r = red_val
                        marker.color.g = blue_val
                        marker.color.b = green_val
                        marker.pose.position.x = sequence_of_states[marker_num,0]
                        marker.pose.position.y = sequence_of_states[marker_num,1]
                        #print("rviz detects current roach pose to be: ", best_sequence_of_states[marker_num, :2])
                        marker.pose.position.z = 0
                        markerArray.markers.append(marker)
                        marker_id+=1
                    self.publish_markers.publish(markerArray)

        #info for saving
        info_for_saving={
        'save_perp_dist': save_perp_dist,
        'save_forward_dist': save_forward_dist,
        'saved_old_forward_dist': saved_old_forward_dist,
        'save_moved_to_next': save_moved_to_next,
        'save_desired_heading': save_desired_heading,
        'save_curr_heading': save_curr_heading,
        }

        #the 0th entry is the currently executing action... so the 1st entry is the optimal action to take
        action_to_take = np.copy(best_sequence_of_actions[1])

        return action_to_take, curr_line_segment, old_curr_forward, info_for_saving, best_sequence_of_actions, best_sequence_of_states



