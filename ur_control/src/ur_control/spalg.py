#! /usr/bin/env python
import rospy
import numpy as np
import scipy.optimize
import ur_control.transformations as tr
from pyquaternion import Quaternion

X_AXIS = np.array([1., 0., 0.])
Y_AXIS = np.array([0., 1., 0.])
Z_AXIS = np.array([0., 0., 1.])


class Plane(object):
    """Represents a plane defined by a normal vector through the given point."""

    def __init__(self, normal=None, point=None, equation=None):
        if equation is None:
            self.point = np.array(point)
            self.normal = tr.unit_vector(normal)
            self.offset = -np.dot(self.normal, self.point)
        else:
            norm = tr.vector_norm(equation[:3])
            self.normal = tr.unit_vector(equation[:3])
            self.offset = equation[3] / norm
            self.point = -self.offset*self.normal
        # Plane origin
        self.origin = np.array(self.point)

    def __repr__(self):
        printoptions = np.get_printoptions()
        np.set_printoptions(precision=4, suppress=True)
        text = '<Plane(equation: {0} origin: {1})>'.format(self.coefficients, self.origin)
        np.set_printoptions(**printoptions)
        return text

    def __str__(self):
        return self.__repr__()

    @property
    def coefficients(self):
        return np.hstack((self.normal, self.offset))

    def distance(self, point):
        """
        Calculates distance from a point to the plane.
        @type  point: np.array
        @param point: The input point
        @rtype: float
        @return: The distance from the point to the plane
        """
        dist = np.dot(self.normal, point) + self.offset
        return dist

    def generate_grid(self, cells=10, side_length=1.0):
        """
        Generates a 3D grid with the required number of C{cells}.
        The grid is a square with the given C{side_length}
        @type  cells: int
        @param cells: Number of cells for the grid
        @type  size: float
        @param size: The grid size in meters
        @rtype: np.array
        @return: The grid representation of the plane with shape (cells, 3)
        """
        # First create the grid in the XY plane
        linspace = np.linspace(-0.5, 0.5, num=cells) * side_length
        xx, yy = np.meshgrid(linspace, linspace)
        grid = []
        for i in range(cells):
            # Vertical lines
            grid.append(np.array([xx[0, i], yy[0, i], 0]))
            grid.append(np.array([xx[-1, i], yy[-1, i], 0]))
            # Horizontal lines
            grid.append(np.array([xx[i, 0], yy[i, 0], 0]))
            grid.append(np.array([xx[i, -1], yy[i, -1], 0]))
        grid = np.array(grid)
        # Second, project the grid onto the plane
        # The equation of the XY plane is z = 0
        T = transformation_between_planes(self.coefficients, [0, 0, 1, 0])
        R = T[:3, :3]
        t = T[:3, 3]
        aligned_grid = np.dot(R, grid.T).T + t
        return aligned_grid

    def generate_mesh(self, side_length=1.0, thickness=0.001):
        """
        Generates a mesh representation of the plane. It is obtained by
        extruding a square with the given C{side_length} to reach the
        specified C{thickness}
        The grid is a square with the given C{side_length}
        @type  side_length: float
        @param side_length: The square side length (meters)
        @type  thickness: float
        @param thickness: The cuboid thickness
        @rtype: np.array, np.array
        @return: vertices,faces of the mesh
        """
        grid = self.generate_grid(cells=2, side_length=side_length)
        lower_point = self.origin - self.normal*thickness
        lower_plane = Plane(normal=self.normal, point=lower_point)
        lower_grid = lower_plane.generate_grid(cells=2, side_length=side_length)
        vertices = np.vstack((grid, lower_grid))
        hull = scipy.spatial.ConvexHull(vertices)
        counterclockwise_hull(hull)
        faces = hull.simplices
        offset_origin = np.mean(vertices, axis=0) - self.origin
        R = self.get_transform()[:3, :3]
        offset_thickness = np.dot(R.T, [0, 0, thickness])
        vertices -= offset_origin + offset_thickness
        return vertices, faces

    def get_ray_intersection(self, ray_origin, ray_dir, epsilon=1e-6):
        """
        Returns the point where the given ray intersects with this plane
        @type  ray_origin: Ray origin
        @param ray_dir: Ray direction. Must be unit vector
        @param epsilon: Epsilon to avoid 0 division
        @rtype: np.array
        @return: The intersection point
        """
        dot = np.dot(self.normal, ray_dir)
        if abs(dot) > epsilon:
            w = ray_origin - self.origin
            fac = -np.dot(self.normal, w) / dot
            return ray_origin + (ray_dir * fac)
        else:
            return None

    def get_transform(self):
        """
        Returns the plane transform
        @rtype: np.array
        @return: The plane transform
        """
        T = rotation_matrix_from_axes(self.normal, oldaxis=Z_AXIS)
        T[:3, 3] = self.origin
        return T

    def project(self, point):
        """
        Projects a point onto the plane.
        @type  point: np.array
        @param point: The input point
        @rtype: np.array
        @return: The projected 3D point
        """
        distance = self.distance(point)
        return (point - distance*self.normal)


def counterclockwise_hull(hull):
    """
    Make the edges counterclockwise order
    @type  hull: scipy.spatial.ConvexHull
    @param hull: Convex hull to be re-ordered.
    """
    midpoint = np.sum(hull.points, axis=0) / hull.points.shape[0]
    for i, simplex in enumerate(hull.simplices):
        x, y, z = hull.points[simplex]
        voutward = (x + y + z) / 3 - midpoint
        vccw = np.cross((y - x), (z - y))
        if np.inner(vccw, voutward) < 0:
            hull.simplices[i] = [simplex[0], simplex[2], simplex[1]]


def fit_plane_lstsq(XYZ):
    """
    Fits a plane to a point cloud.
    Where z=a.x+b.y+c; Rearranging: a.x+b.y-z+c=0
    @type  XYZ: list
    @param XYZ: list of points
    @rtype: np.array
    @return: normalized normal vector of the plane in the form C{(a,b,-1)}
    """
    [rows, cols] = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  # X
    G[:, 1] = XYZ[:, 1]  # Y
    Z = XYZ[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return normal


def fit_plane_optimize(points, seed=None):
    """
    Fits a plane to a point cloud using C{scipy.optimize.leastsq}
    @type  points: list
    @param points: list of points
    @rtype: np.array
    @return: normalized normal vector of the plane
    """
    if seed is None:
        seed = np.zeros(4)
        seed[:3] = tr.unit_vector(tr.random_vector(3))
    # Optimization functions

    def f_min(X, p):
        normal = p[0:3]
        d = p[3]
        result = ((normal*X.T).sum(axis=1) + d) / np.linalg.norm(normal)
        return result

    def residuals(params, signal, X):
        return f_min(X, params)
    # Optimize
    XYZ = np.array(points).T
    p0 = np.array(seed)
    sol = scipy.optimize.leastsq(residuals, p0, args=(None, XYZ))[0]
    nn = np.linalg.norm(sol[:3])
    sol /= nn
    seed_error = (f_min(XYZ, p0)**2).sum()
    fit_error = (f_min(XYZ, sol)**2).sum()
    return sol, seed_error, fit_error


def fit_plane_solve(XYZ):
    """
    Fits a plane to a point cloud using C{np.linalg.solve}
    @type  XYZ: list
    @param XYZ: list of points
    @rtype: np.array
    @return: normalized normal vector of the plane
    """
    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]
    npts = len(X)
    A = np.array([[sum(X*X), sum(X*Y), sum(X)],
                  [sum(X*Y), sum(Y*Y), sum(Y)],
                  [sum(X),   sum(Y), npts]])
    B = np.array([[sum(X*Z), sum(Y*Z), sum(Z)]])
    normal = np.linalg.solve(A, B.T)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return normal.ravel()


def fit_plane_svd(XYZ):
    """
    Fits a plane to a point cloud using C{np.linalg.svd}
    @type  XYZ: list
    @param XYZ: list of points
    @rtype: np.array
    @return: normalized normal vector of the plane
    """
    [rows, cols] = XYZ.shape
    # Set up constraint equations of the form  AB = 0,
    # where B is a column vector of the plane coefficients
    # in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
    p = (np.ones((rows, 1)))
    AB = np.hstack([XYZ, p])
    [u, d, v] = np.linalg.svd(AB, 0)
    B = v[3, :]                    # Solution is last column of v.
    nn = np.linalg.norm(B[0:3])
    B = B / nn
    return B[0:3]


def force_frame_transform(bTa):
    """
    Calculates the coordinate transformation for force vectors.
    The force vectors obey special transformation rules.
    B{Reference:} Handbook of robotics page 39, equation 2.9
    @type bTa: array, shape (4,4)
    @param bTa: Homogeneous transformation that represents the position
    and orientation of frame M{A} relative to frame M{B}
    @rtype: array, shape (6,6)
    @return: The coordinate transformation from M{A} to M{B} for force
    vectors
    """
    aTb = transform_inv(bTa)
    return motion_frame_transform(aTb).T


def inertia_matrix_from_vector(i):
    """
    Returns the inertia matrix from its vectorized form.
    @type i: array, shape (6,1)
    @param i: The inertia parameters in its vectorized form.
    @rtype: array, shape (3,3)
    @return: The resulting inertia matrix.
    """
    I11 = i[0]
    I12 = i[1]
    I13 = i[2]
    I22 = i[3]
    I23 = i[4]
    I33 = i[5]
    return np.array([[I11, I12, I13],
                     [I11, I22, I23],
                     [I13, I23, I33]])


def L_matrix(w):
    """
    Returns the 3x6 matrix of angular velocity elements.
    @type w: array
    @param w: The angular velocity array
    @rtype: array, shape (3,6)
    @return: The resulting numpy array
    """
    res = np.zeros((3, 6))
    res[0, :3] = w.flatten()
    res[1, 1:5] = np.insert(w.flatten(), 1, 0)
    res[2, 2:] = np.insert(w.flatten(), 1, 0)
    return res


def motion_frame_transform(bTa):
    """
    Calculates the coordinate transformation for motion vectors.
    The motion vectors obey special transformation rules.
    B{Reference:} Handbook of robotics page 39, equation 2.9
    @type bTa: array, shape (4,4)
    @param bTa: Homogeneous transformation that represents the position
    and orientation of frame M{A} relative to frame M{B}
    @rtype: array, shape (6,6)
    @return: The coordinate transformation from M{A} to M{B} for motion
    vectors
    """
    bRa = bTa[:3, :3]
    bPa = bTa[:3, 3]
    bXa = np.zeros((6, 6))
    bXa[:3, :3] = bRa
    bXa[3:, :3] = np.dot(skew(bPa), bRa)
    bXa[3:, 3:] = bRa
    return bXa


def perpendicular_vector(v):
    """
    Finds an arbitrary perpendicular vector to B{v}
    @type  v: np.array
    @param v: The input vector
    @rtype: np.array
    @return: The perpendicular vector.
    """
    v = tr.unit_vector(v)
    if np.allclose(v[:2], np.zeros(2)):
        if np.isclose(v[2], 0.):
            # v is (0, 0, 0)
            raise ValueError('zero vector')
        # v is (0, 0, Z)
        return Y_AXIS
    return np.array([-v[1], v[0], 0])


def polygon_area(points, plane=None):
    if plane is None:
        plane_eq = fit_plane_optimize(points)
        plane = Plane(equation=plane_eq)
    total = np.zeros(3)
    for i in range(len(points)):
        vi1 = points[i]
        if i is len(points)-1:
            vi2 = points[0]
        else:
            vi2 = points[i+1]
        prod = np.cross(vi1, vi2)
        total += prod
    result = np.dot(total, plane.normal)
    return abs(result/2.)


def rotation_matrix_from_axes(newaxis, oldaxis=Z_AXIS, point=None):
    """
    Returns the rotation matrix that aligns two vectors.
    @type  newaxis: np.array
    @param newaxis: The goal axis
    @type  oldaxis: np.array
    @param oldaxis: The initial axis
    @rtype: array, shape (4,4)
    @return: The resulting rotation matrix that aligns the old to the new axis.
    """
    oldaxis = tr.unit_vector(oldaxis)
    newaxis = tr.unit_vector(newaxis)
    c = np.dot(oldaxis, newaxis)
    angle = np.arccos(c)
    if np.isclose(c, -1.0) or np.allclose(newaxis, oldaxis):
        v = perpendicular_vector(newaxis)
    else:
        v = tr.unit_vector(np.cross(oldaxis, newaxis))
    return tr.rotation_matrix(angle, v, point)


def skew(v):
    """
    Returns the 3x3 skew matrix.
    The skew matrix is a square matrix M{A} whose transpose is also its
    negative; that is, it satisfies the condition M{-A = A^T}.
    @type v: array
    @param v: The input array
    @rtype: array, shape (3,3)
    @return: The resulting skew matrix
    """
    skv = np.roll(np.roll(np.diag(np.asarray(v).flatten()), 1, 1), -1, 0)
    return (skv - skv.T)


def transformation_estimation_svd(A, B):
    """
    This method implements SVD-based estimation of the transformation
    aligning the given correspondences.
    Estimate a rigid transformation between a source and a target matrices
    using SVD.
    For further information please check:
      - U{http://dx.doi.org/10.1109/TPAMI.1987.4767965}
      - U{http://nghiaho.com/?page_id=671}
    @type A: numpy.array
    @param A: Points expressed in the reference frame A
    @type B: numpy.array
    @param B: Points expressed in the reference frame B
    @rtype: R (3x3), t (3x1)
    @return: (R) rotation matrix and (t) translation vector of the rigid
    transformation.
    """
    assert A.shape == B.shape
    assert A.shape[1] == 3
    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = -np.dot(R, centroid_A.T) + centroid_B.T
    return R, t


def transformation_between_planes(newplane, oldplane):
    """
    Returns the transformation matrix that aligns two planes.
    @type  newplane: np.array
    @param newplane: The goal plane in the form [a,b,c,d] where a.x+b.y+c.z+d=0
    @type  oldplane: np.array
    @param oldplane: The initial plane in the form [a,b,c,d] where a.x+b.y+c.z+d=0
    @rtype: np.array, shape (4,4)
    @return: The resulting homogeneous transformation that converts C{oldplane} to C{newplane}.
    """
    newaxis = np.array(newplane[:3])
    Rnew = rotation_matrix_from_axes(newaxis, Z_AXIS)[:3, :3]
    newpoint = np.dot(Rnew, np.array([0, 0, newplane[3]]))
    oldaxis = np.array(oldplane[:3])
    Rold = rotation_matrix_from_axes(oldaxis, Z_AXIS)[:3, :3]
    oldpoint = np.dot(Rold, np.array([0, 0, oldplane[3]]))
    T = rotation_matrix_from_axes(newaxis, oldaxis)
    T[:3, 3] = -newpoint + np.dot(T[:3, :3], oldpoint)
    return T


def transform_inv(T):
    """
    Calculates the inverse of the input homogeneous transformation.

    This method is more efficient than using C{numpy.linalg.inv}, given
    the special properties of the homogeneous transformations.
    @type T: array, shape (4,4)
    @param T: The input homogeneous transformation
    @rtype: array, shape (4,4)
    @return: The inverse of the input homogeneous transformation
    """
    R = T[:3, :3].T
    p = T[:3, 3]
    T_inv = np.identity(4)
    T_inv[:3, :3] = R
    T_inv[:3, 3] = np.dot(-R, p)
    return T_inv


def quaternions_orientation_error(Qd, Qc):
    """
    Calculates the orientation error between to quaternions
    Qd is the desired orientation
    Qc is the current orientation
    both with respect to the same fixed frame

    return vector part
    """
    if isinstance(Qd, Quaternion) and isinstance(Qd, Quaternion):
        ne = Qc.scalar*Qd.scalar + np.dot(np.array(Qc.vector).T, Qd.vector)
        ee = Qc.scalar*np.array(Qd.vector) - Qd.scalar*np.array(Qc.vector) + np.dot(skew(Qc.vector), Qd.vector)
        ee *= np.sign(ne)  # disambiguate the sign of the quaternion
        return ee
    else:
        assert isinstance(Qd, (list, np.ndarray))
        assert isinstance(Qc, (list, np.ndarray))
        q1 = tr.vector_to_pyquaternion(Qd)
        q2 = tr.vector_to_pyquaternion(Qc)
        return quaternions_orientation_error(q1, q2)


def translation_rotation_error(to_pose, from_pose):
    position_error = to_pose[:3] - from_pose[:3]
    orientation_error = quaternions_orientation_error(to_pose[3:], from_pose[3:])
    return np.concatenate((position_error, orientation_error))


def convert_wrench(wrench_force, pose):
    ee_transform = tr.pose_to_transform(pose)

    # # # Wrench force transformation
    wFtS = force_frame_transform(ee_transform)
    wrench = np.dot(wFtS, wrench_force)

    return wrench


def face_towards(target_position, current_pose, up_vector=[0, 0, 1]):
    """
        Compute orientation to "face towards" a point in space 
        given the current position and the initial vector representing "up"
        default is z as is the outward direction from the end-effector
    """
    cposition = current_pose[:3]
    direction = tr.unit_vector(target_position-cposition)

    cmd_rot = look_rotation(direction, up=up_vector)
    target_quat = tr.vector_from_pyquaternion(cmd_rot)

    return np.concatenate([cposition, target_quat])


def look_rotation(forward, up=[0, 0, 1]):
    forward = tr.unit_vector(forward)
    right = tr.unit_vector(np.cross(up, forward))
    up = np.cross(forward, right)
    m00 = right[0]
    m01 = right[1]
    m02 = right[2]
    m10 = up[0]
    m11 = up[1]
    m12 = up[2]
    m20 = forward[0]
    m21 = forward[1]
    m22 = forward[2]

    num8 = (m00 + m11) + m22
    quaternion = Quaternion()
    if (num8 > 0.0):

        num = np.sqrt(num8 + 1)
        quaternion[0] = num * 0.5
        num = 0.5 / num
        quaternion[1] = (m12 - m21) * num
        quaternion[2] = (m20 - m02) * num
        quaternion[3] = (m01 - m10) * num
        return quaternion

    if ((m00 >= m11) and (m00 >= m22)):

        num7 = np.sqrt(((1 + m00) - m11) - m22)
        num4 = 0.5 / num7
        quaternion[1] = 0.5 * num7
        quaternion[2] = (m01 + m10) * num4
        quaternion[3] = (m02 + m20) * num4
        quaternion[0] = (m12 - m21) * num4
        return quaternion

    if (m11 > m22):

        num6 = np.sqrt(((1 + m11) - m00) - m22)
        num3 = 0.5 / num6
        quaternion[1] = (m10 + m01) * num3
        quaternion[2] = 0.5 * num6
        quaternion[3] = (m21 + m12) * num3
        quaternion[0] = (m20 - m02) * num3
        return quaternion

    num5 = np.sqrt(((1 + m22) - m00) - m11)
    num2 = 0.5 / num5
    quaternion[1] = (m20 + m02) * num2
    quaternion[2] = (m21 + m12) * num2
    quaternion[3] = 0.5 * num5
    quaternion[0] = (m01 - m10) * num2
    return quaternion


def jump_threshold(trajectory, dt, threshold):
    traj = np.copy(trajectory)
    speed = np.abs((traj - np.roll(traj, -1)) / dt)
    speed[-1] = 0.0  # ignore last point

    mean = np.mean(speed[:-1], 0)
    std = np.std(speed[:-1], 0)

    # print("mean:", np.round(mean, 2))
    # print("std:", np.round(std, 2))
    for i, s in enumerate(speed[:-1]):
        if np.any(s > mean + threshold*std):
            # print("### spike:", i, np.round(s-mean,2))
            # TODO(cambel): fix for cases where spikes are consecutive 
            traj[i] = (traj[i-1] + traj[i+1]) / 2.0
        # else:
        #     print("usual:", i, np.round(s-mean,2))

    return traj
