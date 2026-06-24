import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 3d coordinates from 12 joints
motion_data_df = pd.read_csv(r"F:\studium\Masterarbeit\projectcode\12.csv")
names =['x','y','z']
motion_data_df.columns = names
df1 = motion_data_df.iloc[0:1247,:].reset_index(drop=True)
df2 = motion_data_df.iloc[1247:2494,:].reset_index(drop=True)
df3 = motion_data_df.iloc[2494:3741,:].reset_index(drop=True)
df4 = motion_data_df.iloc[3741:4988,:].reset_index(drop=True)
df5 = motion_data_df.iloc[4988:6235,:].reset_index(drop=True)
df6 = motion_data_df.iloc[6235:7482,:].reset_index(drop=True)
df7 = motion_data_df.iloc[7482:8729,:].reset_index(drop=True)
df8 = motion_data_df.iloc[8729:9976,:].reset_index(drop=True)
df9 = motion_data_df.iloc[9976:11223,:].reset_index(drop=True)
df10 = motion_data_df.iloc[11223:12470,:].reset_index(drop=True)
df11 = motion_data_df.iloc[12470:13717,:].reset_index(drop=True)
df12 = motion_data_df.iloc[13717:14964,:].reset_index(drop=True)
motion_data_df =pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12],axis=1)
# m = [0, 1246, 2493, 3740, 4987, 6234 ,7481 , 8728 , 9975 , 11222 , 12469 , 13716 , 14963]
motion_data_df.insert(0, 'index', range(len(motion_data_df)))
# Extract the time and joint positions from the DataFrame
time_steps = motion_data_df['index'].values
motion_data = motion_data_df.iloc[:, 1:].values  # Exclude the 'Time' column

# Reshape the motion_data to (num_frames, 33 * 43) shape
num_frames = len(time_steps)
motion_data = motion_data.reshape(num_frames, 12, 3)

def update(frame):
    ax.clear()

    # Extract the 3D positions of the joints at the current frame/time step
    joint_positions = motion_data[frame].reshape(-1, 3)

    # Plot each joint as a point in 3D space
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], c='b', marker='o')

    # Connect the joints to form the skeleton
    # Define the bones (connections between joints)
    bones = [
        # (0, 1),(1, 2),(2, 3),(3, 4),(0, 15),(0, 19),(7, 8),(8, 9), (9, 10), (11, 12),(12, 13), (13, 14), (15, 16), (16, 17), (17, 18),(19, 20), (20, 21),(21, 22)  # Example connections; update these based on your joint indices
        # ... and so on for the other bones
        (0, 1),(1, 2),(3, 4),(4, 5),(6, 7),(7, 8),(9, 10),(10, 11),(0, 6),(3, 9),(6, 9),(0, 3)
    ]

    # Plot the bones
    for bone in bones:
        j1, j2 = bone
        ax.plot([joint_positions[j1, 0], joint_positions[j2, 0]],
                [joint_positions[j1, 1], joint_positions[j2, 1]],
                [joint_positions[j1, 2], joint_positions[j2, 2]], c='b')

    # Set axis labels and plot limits based on the range of joint positions
    min_range = np.min(joint_positions) -0.2 # Adjust the offset as needed
    max_range = np.max(joint_positions) +0.2
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(2.5, 3)
    ax.set_ylim(2.2, 2.6)
    ax.set_zlim(0, 0.6)

    # You can also add a title or other annotations to the plot if desired
    # ax.set_title(f"Frame {frame}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

num_frames = motion_data.shape[0]  # Get the number of time steps from the data
animation = FuncAnimation(fig, update, frames=num_frames, interval=100)

plt.show()
# Define the filename for the saved animation
output_filename = 'motion_animation.mp4'

# # Save the animation as an .mp4 file
animation.save(output_filename, writer='ffmpeg', dpi=300)