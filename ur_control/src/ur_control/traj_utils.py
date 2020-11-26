import numpy as np

def spiral(radius, theta_offset, revolutions, steps):
    theta = np.linspace(0, 2*np.pi*revolutions, steps) + theta_offset
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    return x, y

def get_conical_helix_trajectory(p1, p2, steps, revolutions=5.0):
    """ Compute Cartesian conical helix between 2 points"""
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    radius = np.linspace(euclidean_dist, 0, steps)
    theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))

    x, y = spiral(radius, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.linspace(p1[2]-(p1[2]-p2[2])/2, p2[2], steps)
    return concat_vec(x, y, z, steps)

def circunference(p1, p2, steps, axes=[0,1,2]):
    euclidean_dist = np.linalg.norm(np.array(p2[:axes[2]])-np.array(p1[:axes[2]]))
    theta_offset = np.arctan2((p1[axes[1]] - p2[axes[1]]), (p1[axes[0]]-p2[axes[0]]))
    x,y = spiral(euclidean_dist, theta_offset, 1.0, steps)
    x += p2[axes[0]]
    y += p2[axes[1]]
    z = np.zeros(steps)+p1[axes[2]]
    return concat_vec([x, y, z], axes, steps)

def concat_vec(xyz, axes, steps):
    x = xyz[axes[0]].reshape(-1, steps)
    y = xyz[axes[1]].reshape(-1, steps)
    z = xyz[axes[2]].reshape(-1, steps)
    return np.concatenate((x, y, z), axis=0).T
