import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/home/luanzhang/Hiwi_Ye_20210918/iiwa_14_RL_Path_Tracking-main/iiwa_rl/results/step_episode_reward_error_200epi_isdone10.csv')
x_mean_error_list = data['X Mean Error List']
frame_itr_lst = data['Frame Iteration List']
plt.axis([min(frame_itr_lst),max(frame_itr_lst),min(x_mean_error_list),max(x_mean_error_list)])
plt.title("Mean X Error For Each Steps")
plt.xlabel('Frame Iterations (100 steps for 1 Episode)')
plt.ylabel('X Mean Error / m')
plt.plot(frame_itr_lst,x_mean_error_list)
plt.savefig('/home/luanzhang/Hiwi_Ye_20210918/iiwa_14_RL_Path_Tracking-main/iiwa_rl/1017_sac_mean_error_x.png')