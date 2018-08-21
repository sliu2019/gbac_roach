import pickle
import numpy as np

path_list = ["/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/MAML_roach_copy/terrain_types_gravel_model_on_gravel", "/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/MAML_roach_copy/terrain_types_turf_model_on_turf", "/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/MAML_roach_copy/terrain_types_styrofoam_model_on_styrofoam", "/media/anagabandi/f1e71f04-dc4b-4434-ae4c-fcb16447d5b3/MAML_roach_copy/terrain_types_carpet_model_on_carpet"]


# for i in range(len(path_list)):
# 	for j in range(len(path_list)):
# 		i_weights = pickle.load(open(path_list[i] + "/weights.pickle", "r")) 
# 		j_weights = pickle.load(open(path_list[j] + "/weights.pickle", "r")) 

# 		for k in range(len(i_weights)):

ref = pickle.load(open(path_list[0] + "/weights.pickle", "r")) 

for i in range(len(path_list)):
	i_weights = pickle.load(open(path_list[i] + "/weights.pickle", "r")) 

	print(np.linalg.norm(i_weights[1] - ref[1]))
	print(np.linalg.norm(i_weights[1]))

	print(i_weights[1])