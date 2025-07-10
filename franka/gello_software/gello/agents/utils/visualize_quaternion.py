import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def plot_coordinate_frame(ax, quat, label, colors, origin=[0, 0, 0]):
    """ Plot a quaternion as a rotated coordinate frame with distinct colors for X, Y, Z axes """
    r = R.from_quat(quat)  # Convert quaternion to rotation matrix
    rot_matrix = r.as_matrix()
    
    origin = np.array(origin)
    
    for i in range(3):
        ax.quiver(*origin, *rot_matrix[:, i], color=colors[i], length=0.8, linewidth=2)

    ax.text(*origin, label, color=colors[0], fontsize=12, fontweight="bold")

def plot_orientation_vector(ax, quat, color, origin=[0, 0, 0]):
    """ Plot a single vector representing the orientation of the quaternion """
    r = R.from_quat(quat)  # Convert quaternion to rotation matrix
    direction = r.as_matrix()[:, 2]  # Extract the Z-axis direction
    origin = np.array(origin)
    
    ax.quiver(*origin, *direction, color=color, length=1.2, linewidth=3, linestyle="dashed")

# Define two quaternions, (x, y, z, w)
quat1 = [0.00215866671902961, 0.99985926041098, 0.00908918152251996, 0.0137612721807273]  # Robot
quat2 = [-0.0196726339020651, 0.53094031186603, -0.0956643793015703, 0.841762267638255]  # Hand

# Set up the 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

# Plot the original coordinate system (identity)
plot_coordinate_frame(ax, [0, 0, 0, 1], '', ['gray', 'gray', 'gray'])

# Define custom colors for each axis of the coordinate frames
robot_colors = ['red', 'orange', 'yellow']  # X, Y, Z
hand_colors = ['green', 'blue', 'purple']  # X, Y, Z

# Plot the two quaternions with new color schemes
plot_coordinate_frame(ax, quat1, '', robot_colors)
plot_coordinate_frame(ax, quat2, '', hand_colors)

# Plot single orientation vectors for both quaternions
plot_orientation_vector(ax, quat1, 'magenta')  # Robot Orientation
plot_orientation_vector(ax, quat2, 'cyan')  # Hand Orientation

# Create legend elements (scatter plots for legend)
legend_elements = [
    plt.Line2D([0], [0], color='red', lw=2, label='Robot X-Axis'),
    plt.Line2D([0], [0], color='orange', lw=2, label='Robot Y-Axis'),
    plt.Line2D([0], [0], color='yellow', lw=2, label='Robot Z-Axis'),
    plt.Line2D([0], [0], color='green', lw=2, label='Hand X-Axis'),
    plt.Line2D([0], [0], color='blue', lw=2, label='Hand Y-Axis'),
    plt.Line2D([0], [0], color='purple', lw=2, label='Hand Z-Axis'),
    plt.Line2D([0], [0], color='magenta', lw=2, linestyle="dashed", label='Robot Orientation Vector'),
    plt.Line2D([0], [0], color='blue', lw=2, linestyle="dashed", label='Hand Orientation Vector')
]

# Add legend to the plot
ax.legend(handles=legend_elements, loc='upper right')

# Labels and grid
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quaternion Visualization with Distinct Axis Colors')

plt.savefig("/mnt/data2/dexmimic/workspace/franka/gello_software/gello/agents/utils/quaternion_visualization.png", dpi=300)
