# gbac_roach

Dependencies:
 * Python **2.7**
 * Numpy version **1.15.0**
 * TensorFlow (GPU) version **1.4.1**
 * Rospy
 * ROS Kinetic
 * rllab at https://github.com/rll/rllab
 * ipython version **5.8.0**

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.



The only files you need to modify are launch_maml_train.py.