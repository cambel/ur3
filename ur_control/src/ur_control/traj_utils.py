import numpy as np

def spiral(radius, theta_offset, revolutions, steps):
    theta = np.linspace(0, 2*np.pi*revolutions, steps) + theta_offset
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    return x, y

def get_conical_helix_trajectory(p1, p2, steps, revolutions=5.0, axes="XYZ"):
    """ Compute Cartesian conical helix between 2 points"""
    p1 = convert_point(np.copy(p1), axes)
    p2 = convert_point(np.copy(p2), axes)
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    radius = np.linspace(euclidean_dist, 0, steps)
    theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))

    x, y = spiral(radius, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.linspace(p1[2]-(p1[2]-p2[2])/2, p2[2], steps)
    return concat_vec(x, y, z, steps)

def get_spiral_trajectory(p1, p2, steps, revolutions=5.0, from_center=False, axes="XYZ"):
    """ Compute Cartesian conical helix between 2 points"""
    p1 = convert_point(np.copy(p1), axes)
    p2 = convert_point(np.copy(p2), axes)
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    if from_center: # start the spiral as if p1 is the center and p2 is the farthest point
        radius = np.linspace(0, euclidean_dist, steps)
        theta_offset = 0.0
    else: 
        # Compute the distance from p1 to p2 and start the spiral as if p2 is the center
        radius = np.linspace(euclidean_dist, 0, steps)
        theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))

    x, y = spiral(radius, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.linspace(p1[2]-(p1[2]-p2[2])/2, p2[2], steps)
    return concat_vec(x, y, z, steps)

def get_circular_trajectory(p1, p2, steps, revolutions=1.0, axes="XYZ"):
    p1 = convert_point(np.copy(p1), axes)
    p2 = convert_point(np.copy(p2), axes)
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))
    x,y = spiral(euclidean_dist, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.zeros(steps)+p1[2]
    return concat_vec(x, y, z, steps)

def concat_vec(x, y, z, steps):
    x = x.reshape(-1, steps)
    y = y.reshape(-1, steps)
    z = z.reshape(-1, steps)
    return np.concatenate((x, y, z), axis=0).T

def convert_point(point, axes):
    if axes == "XYZ":
        return point
    elif axes == "XZY":
        return np.array(point[0], point[2], point[1])
    elif axes == "YZX":
        return np.roll(point, -1)
    elif axes == "YXZ":
        return np.array(point[1], point[0], point[2])
    elif axes == "ZXY":
        return np.roll(point, 1)
    elif axes == "ZYX":
        return np.array(point[2], point[1], point[0])
    else:
        raise Exception("Invalid sequence: %s" % axes)
