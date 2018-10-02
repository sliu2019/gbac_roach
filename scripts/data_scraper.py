import os, sys
import yaml
import IPython
# Script to scrape on-pol data from gbac_model folders and the like 
# Find all the saved_rollouts folders, then go inside the subfolders. Within each subfolder, if it has both a robotinfo and a mocapinfo, then collect it. 
# Make sure you save it all at the end! \

# You also need to know which surface this is for, so I guess you can only use the surface-labeled folder
# Also a few points in here may be bogus because the roach can fall off the mat, etc. 

def main():
	# save the numpy under MAML_roach
	# save list of filenames or dict as a yampl
	master_folder = "/home/anagabandi/rllab-private/data/local/experiment/MAML_roach"
	surfaces = ["carpet", "turf", "styrofoam"]

	path_lsts = {"carpet": [], "turf":[], "styrofoam":[]}


	for root, subdirs, files in os.walk(master_folder):
		#IPython.embed()
		folders = root.split("/")
		if folders[-2] == "saved_rollouts" and folders[-3] in surfaces:
			# check for mocap and robot file
			surface = folders[-3]
			if "mocapInfo.obj" in files and "robotInfo.obj" in files:
				print("adding: " + root + "/robotInfo.obj")
				path_lsts[surface].append(os.path.join(root, "imageInfo.obj"))
				path_lsts[surface].append(os.path.join(root, "mocapInfo.obj"))
				path_lsts[surface].append(os.path.join(root, "robotInfo.obj"))

	list_of_pathLists = []
	list_of_pathLists.append(path_lsts["turf"])
	list_of_pathLists.append(path_lsts["carpet"])
	list_of_pathLists.append(path_lsts["styrofoam"])

	IPython.embed()
	with open(os.path.join(master_folder, "path_list_dictionary.yaml"), 'w+') as outfile:
		yaml.dump(path_lsts, outfile)
	#list_of_pathLists.append(path_lsts["gravel"])


if __name__ == "__main__":
	main()
