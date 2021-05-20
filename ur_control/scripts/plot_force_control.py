from ur_control import transformations
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


data_actual = np.load("/root/o2ac-ur/catkin_ws/src/o2ac_routines/scripts/plt/actual.npy")
orientation_actual = np.array([list(transformations.euler_from_quaternion(t[3:], axes="rxyz")) for t in data_actual])
# orientation_actual = orientation_actual / np.linalg.norm(orientation_actual)

data_target = np.load("/root/o2ac-ur/catkin_ws/src/o2ac_routines/scripts/plt/target.npy")
orientation_target = np.array([list(transformations.euler_from_quaternion(t[3:], axes="rxyz")) for t in data_target])
# orientation_target = orientation_target / np.linalg.norm(orientation_target)

data_target2 = np.load("/root/o2ac-ur/catkin_ws/src/o2ac_routines/scripts/plt/target2.npy")
orientation_target2 = np.array([list(transformations.euler_from_quaternion(t[3:], axes="rxyz")) for t in data_target2])
# orientation_target2 = orientation_target2 / np.linalg.norm(orientation_target2)

trajectory = np.load("/root/o2ac-ur/catkin_ws/src/o2ac_routines/scripts/plt/trajectory.npy")
orientation_traj = np.array([list(transformations.euler_from_quaternion(t[3:], axes="rxyz")) for t in trajectory])
# orientation_traj = orientation_traj / np.linalg.norm(orientation_traj)

data_dxf = np.load("/root/o2ac-ur/catkin_ws/src/o2ac_routines/scripts/plt/data_dxf.npy")

data = data_target
data = trajectory
data = data_target2
data = data_dxf
data = data_actual

print(data.shape)

fig = plt.figure(figsize=(8, 8))

cm = plt.get_cmap("RdYlGn")
col = np.arange(data.shape[0])

dir = 5

def plot_orientation():
    ax = fig.add_subplot(311)
    ax.plot(np.arange(0, data.shape[0]), orientation_target2[:, 0], label="ax_target")
    ax.plot(np.arange(0, data.shape[0]), orientation_actual[:, 0], label="ax_actual")
    ax.plot(np.linspace(0, data.shape[0], num=orientation_traj.shape[0]), orientation_traj[:, 0], label="ax_trajectory")
    ax.legend()
    ax = fig.add_subplot(312)
    ax.plot(np.arange(0, data.shape[0]), orientation_target2[:, 1], label="ax_target")
    ax.plot(np.arange(0, data.shape[0]), orientation_actual[:, 1], label="ax_actual")
    ax.plot(np.linspace(0, data.shape[0], num=orientation_traj.shape[0]), orientation_traj[:, 1], label="ax_trajectory")
    ax = fig.add_subplot(313)
    ax.plot(np.arange(0, data.shape[0]), orientation_target2[:, 2], label="ax_target")
    ax.plot(np.arange(0, data.shape[0]), orientation_actual[:, 2], label="ax_actual")
    ax.plot(np.linspace(0, data.shape[0], num=orientation_traj.shape[0]), orientation_traj[:, 2], label="ax_trajectory")

def plot_3d():
    radius = 0.003
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.zeros_like(data[:,0]), data[:,1], data[:,2], s=10, c=col, marker='o')
    ax.plot(np.zeros_like(data_target2[:,0]), data_target2[:,1], data_target2[:,2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_ylim(0.57-radius,0.57+radius)
    ax.set_zlim(0.225-radius,0.225+radius)
    # ax = fig.add_subplot(212)
    # ax.plot(np.arange(0, data.shape[0]), data_dxf[:, dir])
    # ax.set_xlim(100,200)

# plot_3d()
# plt.show()
plot_orientation()
plt.show()
